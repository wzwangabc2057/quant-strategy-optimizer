"""
全市场多因子选股引擎
从 MCP 服务获取全市场数据，进行因子选股
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mcp_client import get_mcp_client


class StockSelector:
    """全市场选股器"""

    # 冻结参数
    FROZEN_PARAMS = {
        'lag_days': 60,
        'participation_rate': 0.01,
        'max_turnover': 0.30,
        'industry_cap': 0.25,
        'single_cap': 0.08,
        'min_list_days': 60,
        'min_adv': 2000,  # 万元
    }

    # R4 稳健型因子权重
    R4_FACTOR_WEIGHTS = {
        'quality': 0.30,      # 质量因子
        'growth': 0.15,       # 成长因子
        'momentum': 0.10,     # 动量因子
        'value': 0.20,        # 价值因子
        'small_cap': 0.05,    # 小盘因子
        'low_volatility': 0.20, # 低波因子
    }

    # R5 进取型因子权重
    R5_FACTOR_WEIGHTS = {
        'quality': 0.15,
        'growth': 0.30,
        'momentum': 0.20,
        'value': 0.10,
        'small_cap': 0.15,
        'low_volatility': 0.10,
    }

    def __init__(self):
        self.client = get_mcp_client()
        self._stock_basic_cache = None
        self._industry_cache = {}

    def get_all_stocks(self) -> pd.DataFrame:
        """
        获取全市场股票基本信息

        Returns:
            DataFrame with columns: [code, name, industry, list_date, exchange]
        """
        print("获取全市场股票列表...")

        # 从 MCP 获取股票基本信息
        stocks = self.client.get_stock_basic(list_status='L', page_size=10000)

        if not stocks:
            print("警告: 未能获取股票列表")
            return pd.DataFrame()

        records = []
        for s in stocks:
            ts_code = s.get('tsCode', '')
            code = ts_code.split('.')[0] if '.' in ts_code else ts_code
            exchange = ts_code.split('.')[1] if '.' in ts_code else ''

            records.append({
                'code': code,
                'name': s.get('name', ''),
                'industry': s.get('industry', ''),
                'list_date': s.get('listDate', ''),
                'exchange': exchange,
                'market': s.get('market', ''),
            })

        self._stock_basic_cache = pd.DataFrame(records)
        print(f"获取到 {len(self._stock_basic_cache)} 只股票")
        return self._stock_basic_cache

    def get_daily_basics(self, codes: List[str], trade_date: str = None) -> pd.DataFrame:
        """
        批量获取每日指标

        Args:
            codes: 股票代码列表
            trade_date: 交易日期 (YYYYMMDD)

        Returns:
            DataFrame with PE, PB, market cap, turnover etc.
        """
        print(f"获取 {len(codes)} 只股票的每日指标...")

        all_records = []

        # 分批获取 (每批100只)
        batch_size = 100
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i+batch_size]

            for code in batch:
                # 转换代码格式
                ts_code = self._convert_code(code)

                basics = self.client.get_daily_basic(
                    ts_code=ts_code,
                    end_date=trade_date,  # 使用 end_date 而不是 trade_date
                    page_size=1
                )

                if basics:
                    b = basics[0]
                    all_records.append({
                        'code': code,
                        'trade_date': b.get('tradeDate', ''),
                        'close': float(b.get('close', 0) or 0),
                        'pe': float(b.get('pe', 0) or 0),
                        'pe_ttm': float(b.get('peTtm', 0) or 0),
                        'pb': float(b.get('pb', 0) or 0),
                        'ps': float(b.get('ps', 0) or 0),
                        'dv_ratio': float(b.get('dvRatio', 0) or 0),  # 股息率
                        'total_mv': float(b.get('totalMv', 0) or 0),   # 总市值(万元)
                        'circ_mv': float(b.get('circMv', 0) or 0),     # 流通市值
                        'turnover_rate': float(b.get('turnoverRate', 0) or 0),
                        'volume_ratio': float(b.get('volumeRatio', 0) or 0),
                    })

        return pd.DataFrame(all_records)

    def get_daily_quotes(self, codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        批量获取日线行情 (用于计算动量)

        Args:
            codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame with OHLCV data
        """
        print(f"获取 {len(codes)} 只股票的历史行情...")

        all_records = []

        start_dt = start_date.replace('-', '')
        end_dt = end_date.replace('-', '')

        for code in codes[:200]:  # 限制数量
            ts_code = self._convert_code(code)
            quotes = self.client.get_daily_quotes(
                ts_code=ts_code,
                start_date=start_dt,
                end_date=end_dt
            )

            for q in quotes:
                all_records.append({
                    'code': code,
                    'trade_date': q.get('tradeDate', ''),
                    'open': float(q.get('open', 0) or 0),
                    'high': float(q.get('high', 0) or 0),
                    'low': float(q.get('low', 0) or 0),
                    'close': float(q.get('close', 0) or 0),
                    'volume': float(q.get('vol', 0) or 0),
                    'amount': float(q.get('amount', 0) or 0),
                    'pct_change': float(q.get('pctChange', 0) or 0),
                })

        return pd.DataFrame(all_records)

    def calculate_factors(self,
                          stock_basic: pd.DataFrame,
                          daily_basics: pd.DataFrame,
                          quotes: pd.DataFrame = None) -> pd.DataFrame:
        """
        计算多因子得分

        Args:
            stock_basic: 股票基本信息
            daily_basics: 每日指标
            quotes: 历史行情 (可选，用于动量计算)

        Returns:
            DataFrame with factor scores
        """
        print("计算因子得分...")

        if daily_basics.empty:
            return pd.DataFrame()

        df = daily_basics.copy()

        # ========== 1. 价值因子 ==========
        # PE越低越好
        df['value_score'] = df['pe'].apply(
            lambda x: max(0, 100 - min(x, 100)) if x > 0 else 50
        )

        # PB越低越好
        pb_score = df['pb'].apply(
            lambda x: max(0, 100 - min(x * 10, 100)) if x > 0 else 50
        )
        df['value_score'] = df['value_score'] * 0.6 + pb_score * 0.4

        # 股息率加分
        df['value_score'] = df['value_score'] + df['dv_ratio'] * 5

        # ========== 2. 质量因子 ==========
        # 基于市值和换手率的代理
        # 大市值 + 稳定换手 = 高质量
        if 'total_mv' in df.columns:
            max_mv = df['total_mv'].max()
            df['quality_score'] = 50 + (df['total_mv'] / max_mv * 30)
        else:
            df['quality_score'] = 50

        # 换手率适中为佳
        turnover_score = df['turnover_rate'].apply(
            lambda x: 100 - abs(x - 5) * 5 if 0 < x < 20 else 30
        )
        df['quality_score'] = df['quality_score'] * 0.5 + turnover_score * 0.5

        # ========== 3. 成长因子 ==========
        # 使用量比的代理
        df['growth_score'] = df['volume_ratio'].apply(
            lambda x: min(100, 50 + x * 20) if x > 0 else 50
        )

        # ========== 4. 动量因子 ==========
        if quotes is not None and not quotes.empty:
            # 计算20日收益率
            momentum_data = []
            for code in df['code'].unique():
                code_quotes = quotes[quotes['code'] == code].sort_values('trade_date')
                if len(code_quotes) >= 20:
                    returns = code_quotes['pct_change'].tail(20).sum()
                    momentum_data.append({'code': code, 'momentum_20d': returns})

            if momentum_data:
                momentum_df = pd.DataFrame(momentum_data)
                df = df.merge(momentum_df, on='code', how='left')
                df['momentum_score'] = df['momentum_20d'].apply(
                    lambda x: 50 + min(x * 2, 50) if pd.notna(x) else 50
                )
            else:
                df['momentum_score'] = 50
        else:
            # 使用换手率作为动量代理
            df['momentum_score'] = df['turnover_rate'].apply(
                lambda x: min(100, 50 + x * 5)
            )

        # ========== 5. 小盘因子 ==========
        if 'circ_mv' in df.columns:
            # 流通市值越小分数越高
            max_circ = df['circ_mv'].max()
            df['small_cap_score'] = 100 * (1 - df['circ_mv'] / max_circ)
        else:
            df['small_cap_score'] = 50

        # ========== 6. 低波因子 ==========
        # 换手率低 = 低波动
        df['low_volatility_score'] = df['turnover_rate'].apply(
            lambda x: max(20, 100 - x * 3)
        )

        return df

    def calculate_composite_score(self,
                                   df: pd.DataFrame,
                                   profile: str = 'R4') -> pd.DataFrame:
        """
        计算综合得分

        Args:
            df: 因子得分 DataFrame
            profile: R4 / R5

        Returns:
            DataFrame with composite score
        """
        weights = self.R4_FACTOR_WEIGHTS if profile == 'R4' else self.R5_FACTOR_WEIGHTS

        df['composite_score'] = (
            df['quality_score'] * weights['quality'] +
            df['growth_score'] * weights['growth'] +
            df['momentum_score'] * weights['momentum'] +
            df['value_score'] * weights['value'] +
            df['small_cap_score'] * weights['small_cap'] +
            df['low_volatility_score'] * weights['low_volatility']
        )

        return df.sort_values('composite_score', ascending=False)

    def apply_filters(self,
                      df: pd.DataFrame,
                      stock_basic: pd.DataFrame,
                      min_list_days: int = 60,
                      min_adv: float = 2000) -> pd.DataFrame:
        """
        应用过滤条件

        Args:
            df: 数据 DataFrame
            stock_basic: 股票基本信息
            min_list_days: 最小上市天数
            min_adv: 最小流通市值 (万元)

        Returns:
            过滤后的 DataFrame
        """
        print("应用过滤条件...")

        # 合并基本信息
        df = df.merge(stock_basic[['code', 'list_date', 'industry', 'name']], on='code', how='left')

        # 过滤条件1: 上市天数
        today = datetime.now()
        df['list_date_dt'] = pd.to_datetime(df['list_date'], errors='coerce')
        df['list_days'] = (today - df['list_date_dt']).dt.days
        df = df[df['list_days'] >= min_list_days]

        # 过滤条件2: 最小流通市值
        if 'circ_mv' in df.columns:
            df = df[df['circ_mv'] >= min_adv]

        # 过滤条件3: PE有效
        df = df[(df['pe'] > 0) & (df['pe'] < 200)]

        # 过滤条件4: 剔除 ST
        df = df[~df['name'].str.contains('ST', na=False)]

        # 过滤条件5: 剔除科创板、北交所 (可选)
        # df = df[~df['code'].str.startswith(('688', '8', '4'))]

        print(f"过滤后剩余 {len(df)} 只股票")
        return df

    def select(self,
               capital: float = 1000000,
               profile: str = 'R4',
               n_positions: int = 30,
               min_list_days: int = 60,
               min_adv: float = 2000) -> List[Dict]:
        """
        执行选股

        Args:
            capital: 资金规模
            profile: R4稳健 / R5进取
            n_positions: 持仓数量
            min_list_days: 最小上市天数
            min_adv: 最小流通市值 (万元)

        Returns:
            持仓列表
        """
        print(f"\n{'='*50}")
        print(f"全市场选股 - {profile} 稳健型" if profile == 'R4' else f"全市场选股 - {profile} 进取型")
        print(f"资金规模: {capital:,.0f}")
        print(f"{'='*50}")

        # 1. 获取全市场股票
        stock_basic = self.get_all_stocks()
        if stock_basic.empty:
            return []

        # 2. 获取每日指标
        all_codes = stock_basic['code'].tolist()

        # 取最近一个交易日的数据
        trade_date = datetime.now().strftime('%Y%m%d')
        daily_basics = self.get_daily_basics(all_codes[:500], trade_date)  # 限制500只

        if daily_basics.empty:
            print("警告: 未获取到每日指标数据")
            return []

        # 3. 计算因子
        df = self.calculate_factors(stock_basic, daily_basics)

        if df.empty:
            return []

        # 4. 应用过滤
        df = self.apply_filters(df, stock_basic, min_list_days, min_adv)

        # 5. 计算综合得分
        df = self.calculate_composite_score(df, profile)

        # 6. 取 Top N
        df = df.head(n_positions)

        # 7. 分配权重
        positions = self._assign_weights(df, capital)

        print(f"\n选股完成: {len(positions)} 只股票")
        return positions

    def _assign_weights(self, df: pd.DataFrame, capital: float) -> List[Dict]:
        """
        分配权重

        Args:
            df: 排序后的股票 DataFrame
            capital: 资金规模

        Returns:
            持仓列表
        """
        n = len(df)
        if n == 0:
            return []

        # 权重档位
        def get_weight_mult(rank, total):
            pct = rank / total
            if pct <= 0.1:
                return 2.0
            elif pct <= 0.3:
                return 1.5
            elif pct <= 0.5:
                return 1.2
            elif pct <= 0.7:
                return 1.0
            else:
                return 0.8

        positions = []
        for i, (_, row) in enumerate(df.iterrows()):
            weight_mult = get_weight_mult(i + 1, n)

            code = row['code']
            price = row.get('close', 10)
            base_weight = 1.0 / n
            weight = base_weight * weight_mult

            amount = capital * weight
            shares = int(amount / price / 100) * 100

            positions.append({
                'code': code,
                'name': row.get('name', ''),
                'weight': round(weight, 4),
                'shares': shares,
                'price': round(price, 2),
                'amount': round(shares * price, 2),
                'industry': row.get('industry', ''),
                'score': round(row.get('composite_score', 0), 2),
                'pe': round(row.get('pe', 0), 2),
                'pb': round(row.get('pb', 0), 2),
                'circ_mv': round(row.get('circ_mv', 0), 0),
            })

        # 归一化权重
        total_weight = sum(p['weight'] for p in positions)
        for p in positions:
            p['weight'] = round(p['weight'] / total_weight, 4)
            p['amount'] = round(capital * p['weight'], 2)

        return positions

    def _convert_code(self, code: str) -> str:
        """转换股票代码格式"""
        code = str(code).zfill(6)
        if code.startswith(('60', '68')):
            return f"{code}.SH"
        elif code.startswith(('00', '30')):
            return f"{code}.SZ"
        elif code.startswith(('82', '83', '87', '88')):
            return f"{code}.BJ"
        return f"{code}.SZ"


# 测试
if __name__ == '__main__':
    selector = StockSelector()
    positions = selector.select(
        capital=1000000,
        profile='R4',
        n_positions=10
    )

    print("\n选股结果:")
    for p in positions:
        print(f"  {p['code']} {p['name']}: 权重 {p['weight']*100:.2f}%, 得分 {p['score']:.1f}")
