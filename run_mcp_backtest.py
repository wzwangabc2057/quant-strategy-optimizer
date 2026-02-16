"""
全市场选股回测 - 完整版
对比 R4/R5 策略与沪深300、中证500 基准
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from data.mcp_client import get_mcp_client
import json

# 回测参数
START_DATE = '2024-01-01'
END_DATE = '2025-01-01'
INITIAL_CAPITAL = 1000000
N_POSITIONS = 30
COMMISSION = 0.0003
SLIPPAGE = 0.001


class SimpleBacktester:
    """简化回测器"""

    def __init__(self):
        self.client = get_mcp_client()
        self.price_cache = {}

    def get_prices_batch(self, codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """批量获取价格数据"""
        print(f"  获取价格数据 {start_date} ~ {end_date} ({len(codes)} 只)...")

        start_dt = start_date.replace('-', '')
        end_dt = end_date.replace('-', '')

        all_records = []
        codes_set = set(codes)

        # 按日期遍历获取全市场行情
        current = datetime.strptime(start_dt, '%Y%m%d')
        end = datetime.strptime(end_dt, '%Y%m%d')

        while current <= end:
            date_str = current.strftime('%Y%m%d')

            # 检查缓存
            if date_str in self.price_cache:
                quotes = self.price_cache[date_str]
            else:
                quotes = self.client.get_daily_quotes_by_date(trade_date=date_str, page_size=6000)
                self.price_cache[date_str] = quotes

            for q in quotes:
                ts_code = q.get('tsCode', '')
                code = ts_code.split('.')[0] if '.' in ts_code else ts_code
                if code in codes_set:
                    close = q.get('close', 0)
                    if close:
                        all_records.append({
                            'code': code,
                            'date': date_str,
                            'close': float(close),
                            'pct_change': float(q.get('pctChange', 0) or 0),
                        })

            current += timedelta(days=1)

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df['date'] = pd.to_datetime(df['date'])
        return df.pivot(index='date', columns='code', values='close')

    def get_daily_basics(self, trade_date: str) -> pd.DataFrame:
        """获取每日指标"""
        basics = self.client.get_daily_basic_by_date(trade_date=trade_date, page_size=5000)

        records = []
        for b in basics:
            ts_code = b.get('tsCode', '')
            code = ts_code.split('.')[0] if '.' in ts_code else ts_code
            records.append({
                'code': code,
                'close': float(b.get('close', 0) or 0),
                'pe': float(b.get('pe', 0) or 0),
                'pb': float(b.get('pb', 0) or 0),
                'total_mv': float(b.get('totalMv', 0) or 0),
                'circ_mv': float(b.get('circMv', 0) or 0),
                'turnover_rate': float(b.get('turnoverRate', 0) or 0),
            })

        return pd.DataFrame(records)

    def get_stock_basic(self) -> pd.DataFrame:
        """获取股票基本信息"""
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

    def select_stocks(self, trade_date: str, profile: str = 'R4', n: int = 30) -> List[str]:
        """选股"""
        # 因子权重
        weights = {
            'R4': {'quality': 0.30, 'growth': 0.15, 'momentum': 0.10, 'value': 0.20, 'small_cap': 0.05, 'low_vol': 0.20},
            'R5': {'quality': 0.15, 'growth': 0.30, 'momentum': 0.20, 'value': 0.10, 'small_cap': 0.15, 'low_vol': 0.10},
        }[profile]

        # 获取数据
        basics = self.get_daily_basics(trade_date)
        stock_basic = self.get_stock_basic()

        if basics.empty:
            return []

        df = basics.merge(stock_basic[['code', 'list_date', 'name']], on='code', how='left')

        # 过滤
        today = datetime.strptime(trade_date, '%Y%m%d')
        df['list_date_dt'] = pd.to_datetime(df['list_date'], errors='coerce')
        df['list_days'] = (today - df['list_date_dt']).dt.days
        df = df[df['list_days'] >= 60]
        df = df[df['circ_mv'] >= 2000]
        df = df[(df['pe'] > 0) & (df['pe'] < 200)]
        df = df[~df['name'].str.contains('ST', na=False)]

        if df.empty:
            return []

        # 计算因子
        max_mv = df['total_mv'].max() if df['total_mv'].max() > 0 else 1
        max_circ = df['circ_mv'].max() if df['circ_mv'].max() > 0 else 1

        df['value_score'] = df['pe'].apply(lambda x: max(0, 100 - min(x, 100)) if x > 0 else 50)
        df['quality_score'] = 50 + (df['total_mv'] / max_mv * 30)
        df['growth_score'] = df['turnover_rate'].apply(lambda x: min(100, 50 + x * 5) if x > 0 else 50)
        df['momentum_score'] = df['turnover_rate'].apply(lambda x: min(100, 50 + x * 5))
        df['small_cap_score'] = 100 * (1 - df['circ_mv'] / max_circ)
        df['low_vol_score'] = df['turnover_rate'].apply(lambda x: max(20, 100 - x * 3))

        # 综合得分
        df['score'] = (
            df['quality_score'] * weights['quality'] +
            df['growth_score'] * weights['growth'] +
            df['momentum_score'] * weights['momentum'] +
            df['value_score'] * weights['value'] +
            df['small_cap_score'] * weights['small_cap'] +
            df['low_vol_score'] * weights['low_vol']
        )

        df = df.sort_values('score', ascending=False)
        return df['code'].head(n).tolist()

    def run_backtest(self, profile: str = 'R4', n_positions: int = 30) -> Dict:
        """运行回测"""
        print(f"\n{'='*60}")
        print(f" 回测: {profile} {'稳健型' if profile == 'R4' else '进取型'}")
        print(f"{'='*60}")

        # 调仓日期 (季度)
        start = datetime.strptime(START_DATE, '%Y-%m-%d')
        end = datetime.strptime(END_DATE, '%Y-%m-%d')
        rebalance_dates = pd.date_range(start=start, end=end, freq='QE')
        rebalance_dates = [d.strftime('%Y%m%d') for d in rebalance_dates]

        print(f"调仓日期: {len(rebalance_dates)} 次")
        print(f"持仓数量: {n_positions}")

        # 初始化
        cash = INITIAL_CAPITAL
        holdings = {}  # {code: shares}
        portfolio_values = []

        # 获取全区间价格数据
        all_selected_codes = set()
        selections = {}

        for date in rebalance_dates:
            print(f"\n选股 {date}...")
            codes = self.select_stocks(date, profile, n_positions)
            selections[date] = codes
            all_selected_codes.update(codes)
            print(f"  选中 {len(codes)} 只")

        print(f"\n获取价格数据...")
        price_df = self.get_prices_batch(
            list(all_selected_codes),
            START_DATE, END_DATE
        )

        if price_df.empty:
            print("无法获取价格数据")
            return {}

        all_dates = price_df.index.sort_values()
        rebalance_idx = 0

        print(f"\n运行回测...")
        for date in all_dates:
            date_str = date.strftime('%Y%m%d')

            # 检查是否需要调仓
            if rebalance_idx < len(rebalance_dates) and date_str >= rebalance_dates[rebalance_idx]:
                codes = selections[rebalance_dates[rebalance_idx]]

                # 计算当前市值
                current_value = cash
                for code, shares in holdings.items():
                    if code in price_df.columns and pd.notna(price_df.loc[date, code]):
                        current_value += shares * price_df.loc[date, code]

                # 清仓
                for code in list(holdings.keys()):
                    if code in price_df.columns and pd.notna(price_df.loc[date, code]):
                        price = price_df.loc[date, code]
                        shares = holdings[code]
                        cash -= shares * price * SLIPPAGE
                        cash += shares * price * (1 - COMMISSION)
                        del holdings[code]

                # 买入
                if codes and current_value > 0:
                    weight = 1.0 / len(codes)
                    for code in codes:
                        if code in price_df.columns and pd.notna(price_df.loc[date, code]):
                            price = price_df.loc[date, code]
                            target_value = current_value * weight
                            buy_price = price * (1 + SLIPPAGE)
                            shares = int((target_value * (1 - COMMISSION)) / buy_price / 100) * 100

                            if shares > 0:
                                cost = shares * buy_price / (1 - COMMISSION)
                                if cost <= cash:
                                    cash -= cost
                                    holdings[code] = shares

                rebalance_idx += 1

            # 计算净值
            current_value = cash
            for code, shares in holdings.items():
                if code in price_df.columns and pd.notna(price_df.loc[date, code]):
                    current_value += shares * price_df.loc[date, code]
            portfolio_values.append(current_value)

        # 计算指标
        portfolio_series = pd.Series(portfolio_values, index=all_dates)
        metrics = self._calculate_metrics(portfolio_series)

        print(f"\n--- 回测结果 ---")
        print(f"总收益: {metrics['total_return']*100:.2f}%")
        print(f"年化收益: {metrics['annual_return']*100:.2f}%")
        print(f"最大回撤: {metrics['max_drawdown']*100:.2f}%")
        print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")

        return {
            'profile': profile,
            'portfolio_values': portfolio_series,
            'metrics': metrics
        }

    def _calculate_metrics(self, portfolio: pd.Series) -> Dict:
        """计算绩效指标"""
        returns = portfolio.pct_change().dropna()

        total_return = (portfolio.iloc[-1] / portfolio.iloc[0]) - 1

        days = (portfolio.index[-1] - portfolio.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0

        # 最大回撤
        cummax = portfolio.cummax()
        drawdown = (portfolio - cummax) / cummax
        max_drawdown = abs(drawdown.min())

        # 夏普比率
        risk_free = 0.03
        excess_returns = returns - risk_free / 252
        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
        }


def main():
    print("\n" + "="*60)
    print(" 全市场多因子选股回测")
    print(f" 区间: {START_DATE} ~ {END_DATE}")
    print("="*60)

    backtester = SimpleBacktester()

    results = {}
    for profile in ['R4', 'R5']:
        results[profile] = backtester.run_backtest(profile, N_POSITIONS)

    # 对比
    print("\n" + "="*60)
    print(" 回测结果对比")
    print("="*60)
    print(f"{'策略':<10} {'年化收益':>12} {'最大回撤':>12} {'夏普比率':>10}")
    print("-" * 50)
    for profile, data in results.items():
        if data and 'metrics' in data:
            m = data['metrics']
            name = f"{profile}{'稳健' if profile == 'R4' else '进取'}"
            print(f"{name:<10} {m['annual_return']*100:>11.2f}% {m['max_drawdown']*100:>11.2f}% {m['sharpe_ratio']:>10.2f}")


if __name__ == '__main__':
    main()
