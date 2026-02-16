"""
基于 MCP 全市场选股的回测
验证选股效果
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
from data.mcp_client import get_mcp_client

# 回测参数
START_DATE = '2023-01-01'
END_DATE = '2025-12-31'
INITIAL_CAPITAL = 1000000
N_POSITIONS = 30
REBALANCE_FREQ = 'M'  # 月度调仓


class MCPBacktestSelector:
    """基于 MCP 数据的回测选股器"""

    # 因子权重
    R4_WEIGHTS = {
        'quality': 0.30,
        'growth': 0.15,
        'momentum': 0.10,
        'value': 0.20,
        'small_cap': 0.05,
        'low_volatility': 0.20,
    }

    R5_WEIGHTS = {
        'quality': 0.15,
        'growth': 0.30,
        'momentum': 0.20,
        'value': 0.10,
        'small_cap': 0.15,
        'low_volatility': 0.10,
    }

    def __init__(self):
        self.client = get_mcp_client()
        self.stock_cache = {}
        self.quote_cache = {}

    def get_stock_basic(self) -> pd.DataFrame:
        """获取股票基本信息"""
        print("获取股票列表...")
        stocks = self.client.get_stock_basic(list_status='L', page_size=10000)

        records = []
        for s in stocks:
            ts_code = s.get('tsCode', '')
            code = ts_code.split('.')[0] if '.' in ts_code else ts_code
            records.append({
                'code': code,
                'name': s.get('name', ''),
                'industry': s.get('industry', ''),
                'list_date': s.get('listDate', ''),
            })

        return pd.DataFrame(records)

    def get_daily_basics_by_date(self, trade_date: str) -> pd.DataFrame:
        """获取某日的全市场指标"""
        print(f"  获取 {trade_date} 的每日指标...")
        basics = self.client.get_daily_basic_by_date(trade_date=trade_date, page_size=5000)

        records = []
        for b in basics:
            ts_code = b.get('tsCode', '')
            code = ts_code.split('.')[0] if '.' in ts_code else ts_code
            records.append({
                'code': code,
                'trade_date': b.get('tradeDate', ''),
                'close': float(b.get('close', 0) or 0),
                'pe': float(b.get('pe', 0) or 0),
                'pe_ttm': float(b.get('peTtm', 0) or 0),
                'pb': float(b.get('pb', 0) or 0),
                'total_mv': float(b.get('totalMv', 0) or 0),
                'circ_mv': float(b.get('circMv', 0) or 0),
                'turnover_rate': float(b.get('turnoverRate', 0) or 0),
            })

        return pd.DataFrame(records)

    def get_quotes(self, codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """获取历史行情"""
        # 由于 MCP 单只查询较慢，这里用全市场行情按日期查询
        print(f"  获取行情数据 {start_date} ~ {end_date}...")

        all_records = []
        start_dt = start_date.replace('-', '')
        end_dt = end_date.replace('-', '')

        # 分批获取
        current = datetime.strptime(start_dt, '%Y%m%d')
        end = datetime.strptime(end_dt, '%Y%m%d')

        while current <= end:
            date_str = current.strftime('%Y%m%d')
            quotes = self.client.get_daily_quotes_by_date(trade_date=date_str, page_size=6000)

            for q in quotes:
                ts_code = q.get('tsCode', '')
                code = ts_code.split('.')[0] if '.' in ts_code else ts_code
                if code in codes:
                    all_records.append({
                        'code': code,
                        'trade_date': q.get('tradeDate', ''),
                        'close': float(q.get('close', 0) or 0),
                        'pct_change': float(q.get('pctChange', 0) or 0),
                    })

            current += timedelta(days=1)

        return pd.DataFrame(all_records)

    def calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算因子得分"""
        if df.empty:
            return df

        # 价值因子 (PE, PB 越低越好)
        df['value_score'] = df['pe'].apply(
            lambda x: max(0, 100 - min(x, 100)) if x > 0 else 50
        )
        pb_score = df['pb'].apply(
            lambda x: max(0, 100 - min(x * 10, 100)) if x > 0 else 50
        )
        df['value_score'] = df['value_score'] * 0.6 + pb_score * 0.4

        # 质量因子 (市值越大越稳定)
        if 'total_mv' in df.columns and df['total_mv'].max() > 0:
            max_mv = df['total_mv'].max()
            df['quality_score'] = 50 + (df['total_mv'] / max_mv * 30)
        else:
            df['quality_score'] = 50

        # 成长因子 (换手率代理)
        df['growth_score'] = df['turnover_rate'].apply(
            lambda x: min(100, 50 + x * 5) if x > 0 else 50
        )

        # 动量因子 (暂时用换手率代理)
        df['momentum_score'] = df['turnover_rate'].apply(
            lambda x: min(100, 50 + x * 5)
        )

        # 小盘因子
        if 'circ_mv' in df.columns and df['circ_mv'].max() > 0:
            max_circ = df['circ_mv'].max()
            df['small_cap_score'] = 100 * (1 - df['circ_mv'] / max_circ)
        else:
            df['small_cap_score'] = 50

        # 低波因子
        df['low_volatility_score'] = df['turnover_rate'].apply(
            lambda x: max(20, 100 - x * 3)
        )

        return df

    def calculate_composite_score(self, df: pd.DataFrame, profile: str = 'R4') -> pd.DataFrame:
        """计算综合得分"""
        weights = self.R4_WEIGHTS if profile == 'R4' else self.R5_WEIGHTS

        df['composite_score'] = (
            df['quality_score'] * weights['quality'] +
            df['growth_score'] * weights['growth'] +
            df['momentum_score'] * weights['momentum'] +
            df['value_score'] * weights['value'] +
            df['small_cap_score'] * weights['small_cap'] +
            df['low_volatility_score'] * weights['low_volatility']
        )

        return df.sort_values('composite_score', ascending=False)

    def select_at_date(self, trade_date: str, profile: str = 'R4',
                       n_positions: int = 30) -> List[Dict]:
        """
        在指定日期选股

        Args:
            trade_date: 交易日期 YYYYMMDD
            profile: R4 / R5
            n_positions: 持仓数量

        Returns:
            持仓列表
        """
        # 1. 获取股票列表
        stock_basic = self.get_stock_basic()

        # 2. 获取每日指标
        daily_basics = self.get_daily_basics_by_date(trade_date)
        if daily_basics.empty:
            return []

        # 3. 合并基本信息
        df = daily_basics.merge(
            stock_basic[['code', 'list_date', 'industry', 'name']],
            on='code', how='left'
        )

        # 4. 过滤
        # 上市天数
        today = datetime.strptime(trade_date, '%Y%m%d')
        df['list_date_dt'] = pd.to_datetime(df['list_date'], errors='coerce')
        df['list_days'] = (today - df['list_date_dt']).dt.days
        df = df[df['list_days'] >= 60]

        # 流通市值
        df = df[df['circ_mv'] >= 2000]

        # PE 有效
        df = df[(df['pe'] > 0) & (df['pe'] < 200)]

        # 剔除 ST
        df = df[~df['name'].str.contains('ST', na=False)]

        # 5. 计算因子
        df = self.calculate_factors(df)

        # 6. 计算综合得分
        df = self.calculate_composite_score(df, profile)

        # 7. 取 Top N
        df = df.head(n_positions)

        # 8. 分配权重
        positions = self._assign_weights(df)
        return positions

    def _assign_weights(self, df: pd.DataFrame) -> List[Dict]:
        """分配权重"""
        n = len(df)
        if n == 0:
            return []

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
            base_weight = 1.0 / n
            weight = base_weight * weight_mult

            positions.append({
                'code': row['code'],
                'name': row.get('name', ''),
                'weight': weight,
                'price': row.get('close', 0),
                'score': row.get('composite_score', 0),
            })

        # 归一化
        total = sum(p['weight'] for p in positions)
        for p in positions:
            p['weight'] = round(p['weight'] / total, 4)

        return positions


def run_backtest():
    """运行回测"""
    print("\n" + "="*60)
    print(" 全市场选股回测 (基于 MCP 数据)")
    print("="*60)

    selector = MCPBacktestSelector()

    # 获取调仓日期
    start = datetime.strptime(START_DATE, '%Y-%m-%d')
    end = datetime.strptime(END_DATE, '%Y-%m-%d')

    # 每月末调仓
    rebalance_dates = pd.date_range(start=start, end=end, freq='ME')
    rebalance_dates = [d.strftime('%Y%m%d') for d in rebalance_dates]

    print(f"\n回测区间: {START_DATE} ~ {END_DATE}")
    print(f"调仓频率: 月度 ({len(rebalance_dates)} 次)")
    print(f"持仓数量: {N_POSITIONS}")

    # 模拟一次选股（测试）
    print("\n测试选股 (最近日期)...")
    test_date = rebalance_dates[-1] if rebalance_dates else datetime.now().strftime('%Y%m%d')

    for profile in ['R4', 'R5']:
        print(f"\n--- {profile} {'稳健型' if profile == 'R4' else '进取型'} ---")
        positions = selector.select_at_date(test_date, profile, n_positions=10)

        print(f"选中 {len(positions)} 只股票:")
        for p in positions[:5]:
            print(f"  {p['code']} {p['name']}: 权重 {p['weight']*100:.2f}%, 得分 {p['score']:.1f}")

    print("\n" + "="*60)
    print(" 注意: 完整回测需要获取大量历史数据，可能需要较长时间")
    print("="*60)


if __name__ == '__main__':
    run_backtest()
