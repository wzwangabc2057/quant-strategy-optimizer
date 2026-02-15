"""
PIT Universe 构建器
================================================================================
构建历史时点可交易的动态股票池，消除幸存者偏差。
================================================================================
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import os
import json

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetcher import DataFetcher

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理numpy类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        return super().default(obj)


@dataclass
class UniverseConfig:
    """Universe配置"""
    # 上市门槛
    min_list_days: int = 60              # 最小上市天数（交易日）

    # 上市日期可靠性
    listing_reliability_threshold: float = 0.8  # 数据连续性阈值（80%）
    listing_reliability_window: int = 10        # 检测窗口（交易日）

    # 流动性门槛
    min_adv_cny: float = 2000.0          # 最小20日成交额（万元）

    # 涨跌停检测
    limit_threshold: float = 0.095       # 涨跌停检测阈值（9.5%）
    use_limit_detection: bool = True     # 是否启用涨跌停检测

    # 持仓限制
    max_positions: int = 50              # 最大持仓数量

    # 参与率
    max_participation_rate: float = 0.01  # 最大参与率（默认1%）


@dataclass
class UniverseRecord:
    """单只股票的Universe记录"""
    symbol: str
    date: str
    is_tradable: bool
    reason_flags: List[str]
    close: float
    volume: float
    adv20: float          # 20日平均成交额（万元）
    list_days: int        # 上市天数
    prev_close: float = 0.0
    daily_return: float = 0.0
    is_limit_up: bool = False
    is_limit_down: bool = False


class UniverseBuilder:
    """
    PIT Universe 构建器

    功能：
    - 构建历史时点可交易的股票池
    - 过滤停牌、涨跌停、低流动性股票
    - 输出剔除原因便于审计
    - 上市日期可靠性检测
    """

    # 剔除原因标志
    FLAG_NO_DATA = 'NO_DATA'
    FLAG_SUSPENDED = 'SUSPENDED'
    FLAG_NEW_LISTING = 'NEW_LISTING'
    FLAG_LIMIT_UP = 'LIMIT_UP'
    FLAG_LIMIT_DOWN = 'LIMIT_DOWN'
    FLAG_LOW_LIQUIDITY = 'LOW_LIQUIDITY'
    FLAG_LISTING_UNRELIABLE = 'LISTING_ESTIMATION_UNRELIABLE'  # 新增

    def __init__(self, config: UniverseConfig = None, fetcher: DataFetcher = None,
                 output_dir: str = None):
        self.config = config or UniverseConfig()
        self.fetcher = fetcher or DataFetcher()
        self.output_dir = output_dir

        # 缓存
        self._price_cache = None
        self._listing_dates = {}  # 上市日期缓存
        self._listing_reliability = {}  # 上市日期可靠性缓存

        # 统计
        self.stats = {
            'total_dates': 0,
            'avg_universe_size': 0,
            'exclusion_counts': {},
            'listing_unreliable_count': 0,  # 新增
        }

    def _load_price_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """加载价格数据（带缓存）"""
        if self._price_cache is None or self._price_cache['end'] < end_date:
            logger.info(f"加载价格数据: {start_date} ~ {end_date}")

            # 获取所有A股数据
            result = self.fetcher.client.query(f"""
                SELECT toString(date) as date, code, close, vol, high, low
                FROM stock_data_qfq
                WHERE date >= '{start_date}'
                  AND date <= '{end_date}'
                ORDER BY date, code
            """)

            if not result.result_rows:
                logger.warning(f"未找到价格数据: {start_date} ~ {end_date}")
                return pd.DataFrame()

            df = pd.DataFrame(result.result_rows,
                            columns=['date', 'code', 'close', 'volume', 'high', 'low'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.drop_duplicates(subset=['date', 'code'], keep='last')

            self._price_cache = {
                'df': df,
                'start': start_date,
                'end': end_date
            }

        return self._price_cache['df']

    def _estimate_listing_date(self, code: str, price_df: pd.DataFrame) -> Optional[pd.Timestamp]:
        """估算上市日期（基于首次出现日期）"""
        if code in self._listing_dates:
            return self._listing_dates[code]

        code_data = price_df[price_df['code'] == code]
        if len(code_data) == 0:
            return None

        first_date = code_data['date'].min()
        self._listing_dates[code] = first_date
        return first_date

    def _check_listing_reliability(self, code: str, date_ts: pd.Timestamp,
                                   price_df: pd.DataFrame) -> Tuple[bool, float]:
        """
        检查上市日期估算的可靠性

        Args:
            code: 股票代码
            date_ts: 当前日期
            price_df: 价格数据

        Returns:
            (is_reliable, reliability_score)
        """
        if code in self._listing_reliability:
            return self._listing_reliability[code]

        # 获取最近N天的数据
        window = self.config.listing_reliability_window
        code_data = price_df[(price_df['code'] == code) &
                            (price_df['date'] <= date_ts)].tail(window)

        if len(code_data) < window * 0.5:
            # 数据太少，可靠性低
            self._listing_reliability[code] = (False, 0.0)
            return False, 0.0

        # 计算数据连续性（有效行情天数 / 总天数）
        valid_days = 0
        for _, row in code_data.iterrows():
            if pd.notna(row['close']) and row['close'] > 0 and pd.notna(row['volume']) and row['volume'] > 0:
                valid_days += 1

        reliability_score = valid_days / len(code_data)
        is_reliable = reliability_score >= self.config.listing_reliability_threshold

        self._listing_reliability[code] = (is_reliable, reliability_score)
        return is_reliable, reliability_score

    def _calculate_adv20(self, code: str, date: pd.Timestamp,
                         price_df: pd.DataFrame) -> float:
        """计算20日平均成交额（万元）"""
        code_data = price_df[(price_df['code'] == code) &
                            (price_df['date'] <= date)].tail(20)

        if len(code_data) < 5:  # 至少5天数据
            return 0

        # 成交额 = 收盘价 × 成交量 / 10000 (转为万元)
        adv20 = (code_data['close'] * code_data['volume']).mean() / 10000
        return adv20

    def _detect_limit(self, close: float, prev_close: float,
                      volume: float, adv20_vol: float) -> Tuple[bool, bool]:
        """
        检测涨跌停

        Returns:
            (is_limit_up, is_limit_down)
        """
        if not self.config.use_limit_detection or prev_close <= 0:
            return False, False

        daily_return = (close - prev_close) / prev_close

        # 计算成交量相对正常水平的比例
        vol_ratio = volume / adv20_vol if adv20_vol > 0 else 1.0

        # 涨停特征：涨幅接近10% + 成交萎缩（封板）
        is_limit_up = daily_return > self.config.limit_threshold and vol_ratio < 0.5

        # 跌停特征：跌幅接近-10% + 成交萎缩（封板）
        is_limit_down = daily_return < -self.config.limit_threshold and vol_ratio < 0.5

        return is_limit_up, is_limit_down

    def build_universe(self, date: str,
                       include_reasons: bool = True) -> pd.DataFrame:
        """
        构建单日Universe

        Args:
            date: 日期 (YYYY-MM-DD)
            include_reasons: 是否包含剔除原因

        Returns:
            DataFrame with columns: [symbol, is_tradable, reason_flags, adv20, list_days, close]
        """
        date_ts = pd.to_datetime(date)

        # 加载前后各30天数据用于计算ADV和上市天数
        lookback_start = (date_ts - timedelta(days=90)).strftime('%Y-%m-%d')
        lookback_end = (date_ts + timedelta(days=1)).strftime('%Y-%m-%d')

        price_df = self._load_price_data(lookback_start, lookback_end)

        assert len(price_df) > 0, f"[SANITY CHECK] 价格数据为空: {date}"

        # 获取当日数据
        today_data = price_df[price_df['date'] == date_ts].copy()

        if len(today_data) == 0:
            logger.warning(f"日期 {date} 无交易数据（可能为非交易日）")
            return pd.DataFrame()

        # 获取前一交易日数据（用于计算收益和涨跌停）
        prev_dates = price_df[price_df['date'] < date_ts]['date'].unique()
        prev_date = prev_dates[-1] if len(prev_dates) > 0 else None

        records = []

        for _, row in today_data.iterrows():
            code = row['code']
            close = row['close']
            volume = row['volume']

            reasons = []
            is_tradable = True

            # 1. 检查成交（停牌检测）
            if pd.isna(volume) or volume <= 0:
                reasons.append(self.FLAG_SUSPENDED)
                is_tradable = False

            # 2. 计算上市天数
            list_date = self._estimate_listing_date(code, price_df)
            if list_date is not None:
                list_days = (date_ts - list_date).days
            else:
                list_days = 9999  # 未知则默认足够长

            if list_days < self.config.min_list_days:
                reasons.append(self.FLAG_NEW_LISTING)
                is_tradable = False

            # 2.1 检查上市日期可靠性（新增）
            listing_reliable, listing_score = self._check_listing_reliability(code, date_ts, price_df)
            if not listing_reliable and is_tradable:
                reasons.append(self.FLAG_LISTING_UNRELIABLE)
                is_tradable = False
                self.stats['listing_unreliable_count'] = self.stats.get('listing_unreliable_count', 0) + 1

            # 3. 计算ADV20
            adv20 = self._calculate_adv20(code, date_ts, price_df)

            if is_tradable and adv20 < self.config.min_adv_cny:
                reasons.append(self.FLAG_LOW_LIQUIDITY)
                is_tradable = False

            # 4. 涨跌停检测
            prev_close = 0
            daily_return = 0
            is_limit_up = False
            is_limit_down = False

            if prev_date is not None and is_tradable:
                prev_data = price_df[(price_df['date'] == prev_date) &
                                    (price_df['code'] == code)]
                if len(prev_data) > 0:
                    prev_close = prev_data.iloc[0]['close']
                    if prev_close > 0:
                        daily_return = (close - prev_close) / prev_close

                        # 计算ADV20的成交量
                        adv20_vol = adv20 * 10000 / close if close > 0 else 0
                        is_limit_up, is_limit_down = self._detect_limit(
                            close, prev_close, volume, adv20_vol
                        )

                        if is_limit_up:
                            reasons.append(self.FLAG_LIMIT_UP)
                            # 涨停仍然可以持有，只是不能买入
                            # 这里标记但不设为不可交易

            record = UniverseRecord(
                symbol=code,
                date=date,
                is_tradable=is_tradable,
                reason_flags=reasons if include_reasons else [],
                close=close,
                volume=volume,
                adv20=adv20,
                list_days=list_days,
                prev_close=prev_close,
                daily_return=daily_return,
                is_limit_up=is_limit_up,
                is_limit_down=is_limit_down,
            )
            records.append(record)

        result = pd.DataFrame([
            {
                'symbol': r.symbol,
                'date': r.date,
                'is_tradable': r.is_tradable,
                'reason_flags': ','.join(r.reason_flags) if r.reason_flags else '',
                'close': r.close,
                'volume': r.volume,
                'adv20': r.adv20,
                'list_days': r.list_days,
                'daily_return': r.daily_return,
                'is_limit_up': r.is_limit_up,
                'is_limit_down': r.is_limit_down,
            }
            for r in records
        ])

        # 断言检查
        assert len(result) > 0, f"[SANITY CHECK] Universe为空: {date}"
        assert 'symbol' in result.columns, "[SANITY CHECK] 缺少symbol列"
        assert 'is_tradable' in result.columns, "[SANITY CHECK] 缺少is_tradable列"

        tradable_count = result['is_tradable'].sum()
        assert tradable_count >= 100, f"[SANITY CHECK] 可交易股票过少: {tradable_count}"

        return result

    def build_universe_range(self, start_date: str, end_date: str,
                            freq: str = 'B') -> Dict[str, pd.DataFrame]:
        """
        构建日期范围内的Universe

        Args:
            start_date: 开始日期
            end_date: 结束日期
            freq: 频率 ('B' = 工作日, 'W-MON' = 每周一)

        Returns:
            {date_str: universe_df}
        """
        logger.info(f"构建Universe: {start_date} ~ {end_date}")

        # 先加载所有需要的价格数据
        lookback_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
        self._load_price_data(lookback_start, end_date)

        # 生成日期序列
        dates = pd.date_range(start_date, end_date, freq=freq)

        universes = {}
        exclusion_stats = {}

        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            universe = self.build_universe(date_str)

            if len(universe) > 0:
                universes[date_str] = universe

                # 统计剔除原因
                for _, row in universe.iterrows():
                    if row['reason_flags']:
                        for flag in row['reason_flags'].split(','):
                            exclusion_stats[flag] = exclusion_stats.get(flag, 0) + 1

        # 更新统计
        self.stats['total_dates'] = len(universes)
        if universes:
            self.stats['avg_universe_size'] = np.mean(
                [df['is_tradable'].sum() for df in universes.values()]
            )
        self.stats['exclusion_counts'] = exclusion_stats

        logger.info(f"构建完成: {len(universes)} 个交易日, "
                   f"平均可交易 {self.stats['avg_universe_size']:.0f} 只股票")

        return universes

    def get_tradable_symbols(self, date: str) -> List[str]:
        """获取当日可交易股票列表"""
        universe = self.build_universe(date)
        if len(universe) == 0:
            return []

        tradable = universe[universe['is_tradable'] & ~universe['is_limit_up']]
        return tradable['symbol'].tolist()

    def get_universe_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()

    def save_universe_evidence(self, universes: Dict[str, pd.DataFrame],
                               sample_dates: List[str] = None):
        """保存Universe证据文件"""
        if self.output_dir is None:
            return

        os.makedirs(self.output_dir, exist_ok=True)

        # 1. 保存统计
        stats_df = []
        for date, universe in universes.items():
            tradable = universe[universe['is_tradable']]
            stats_df.append({
                'date': date,
                'total_stocks': len(universe),
                'tradable_stocks': len(tradable),
                'avg_adv20': tradable['adv20'].mean(),
                'median_adv20': tradable['adv20'].median(),
            })

        stats_path = os.path.join(self.output_dir, 'universe_stats.csv')
        pd.DataFrame(stats_df).to_csv(stats_path, index=False)
        logger.info(f"Universe统计已保存: {stats_path}")

        # 2. 保存剔除原因汇总
        reasons_df = []
        for reason, count in self.stats.get('exclusion_counts', {}).items():
            reasons_df.append({
                'reason': reason,
                'total_occurrences': count,
                'avg_per_day': count / max(len(universes), 1),
            })

        reasons_path = os.path.join(self.output_dir, 'exclusion_reasons.csv')
        pd.DataFrame(reasons_df).to_csv(reasons_path, index=False)
        logger.info(f"剔除原因汇总已保存: {reasons_path}")

        # 3. 保存抽样日期证据
        if sample_dates is None:
            # 随机抽取5个日期
            all_dates = list(universes.keys())
            n_samples = min(5, len(all_dates))
            sample_dates = np.random.choice(all_dates, n_samples, replace=False).tolist()

        for date in sample_dates:
            if date in universes:
                sample_path = os.path.join(self.output_dir, f'sample_{date.replace("-", "")}.csv')
                universes[date].to_csv(sample_path, index=False)

        logger.info(f"抽样证据已保存: {len(sample_dates)} 个日期")

        # 4. 保存统计摘要
        summary_path = os.path.join(self.output_dir, 'universe_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(self.stats, f, indent=2, cls=NumpyEncoder)
        logger.info(f"统计摘要已保存: {summary_path}")


def test_universe_builder():
    """测试UniverseBuilder"""
    print("="*60)
    print("测试 UniverseBuilder")
    print("="*60)

    config = UniverseConfig(
        min_list_days=60,
        min_adv_cny=2000,
    )

    builder = UniverseBuilder(config=config, output_dir='./results/universe_test')

    # 测试单日
    print("\n测试 build_universe('2024-01-02'):")
    universe = builder.build_universe('2024-01-02')

    if len(universe) > 0:
        print(f"  总股票数: {len(universe)}")
        print(f"  可交易股票数: {universe['is_tradable'].sum()}")
        print(f"  平均ADV20: {universe[universe['is_tradable']]['adv20'].mean():.0f} 万元")

        # 剔除原因分布
        print("\n  剔除原因分布:")
        reasons = universe[universe['reason_flags'] != '']['reason_flags'].str.split(',')
        all_reasons = [r for reasons_list in reasons.dropna() for r in reasons_list]
        from collections import Counter
        for reason, count in Counter(all_reasons).most_common():
            print(f"    {reason}: {count}")
    else:
        print("  无数据（可能为非交易日）")

    # 测试范围
    print("\n测试 build_universe_range('2024-01-01', '2024-01-31'):")
    universes = builder.build_universe_range('2024-01-01', '2024-01-31')
    print(f"  生成的交易日数: {len(universes)}")
    print(f"  平均可交易股票数: {builder.stats['avg_universe_size']:.0f}")

    # 保存证据
    builder.output_dir = './results/universe_test'
    builder.save_universe_evidence(universes)

    print("\n" + "="*60)
    print("测试完成")
    print("="*60)


if __name__ == '__main__':
    test_universe_builder()
