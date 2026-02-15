"""
v1 基准版本 - 原始等权组合
================================================================================
作为基准对比，不做任何因子优化，仅使用原始权重。
结果: R4 13.07%, R5 14.20%, Sharpe 0.78/0.77
================================================================================
"""
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PORTFOLIO_FILE, BACKTEST_START, BACKTEST_END, BUY_COMMISSION, SELL_COMMISSION
from data.fetcher import DataFetcher


class StrategyV1:
    """v1基准策略 - 等权持有"""

    def __init__(self, portfolio_file: str = None):
        self.fetcher = DataFetcher()
        self.portfolio_file = portfolio_file or PORTFOLIO_FILE

    def load_portfolio(self) -> dict:
        return self.fetcher.load_portfolio(self.portfolio_file)

    def run_backtest(self, portfolio: pd.DataFrame, price_pivot: pd.DataFrame,
                    start_date: str = None, end_date: str = None) -> dict:
        """运行等权回测"""
        start_date = start_date or BACKTEST_START
        end_date = end_date or BACKTEST_END

        codes = portfolio['code'].tolist()
        dates = sorted([d for d in price_pivot.index if start_date <= d <= end_date])

        # 等权重
        equal_weight = 1.0 / len(codes)
        weights = {code: equal_weight for code in codes}

        daily_returns = []

        for i in range(1, len(dates)):
            date = dates[i]
            prev_date = dates[i-1]

            daily_ret = 0
            for code in codes:
                if code not in price_pivot.columns:
                    continue

                curr = price_pivot.loc[date, code]
                prev = price_pivot.loc[prev_date, code]

                if pd.isna(curr) or pd.isna(prev) or prev <= 0:
                    continue

                stock_ret = curr / prev - 1
                daily_ret += stock_ret * weights[code]

            daily_returns.append({'date': date, 'return': daily_ret})

        # 计算指标
        rets = np.array([r['return'] for r in daily_returns])
        if len(rets) == 0:
            return {
                'annual_return': 0,
                'max_drawdown': 0,
                'sharpe': 0,
                'total_return': 0
            }
        cum = np.cumprod(1 + rets)
        total_return = cum[-1] - 1
        n_years = len(rets) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_drawdown = dd.min()

        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0

        return {
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'total_return': total_return
        }


def main():
    print("=" * 70)
    print("多因子量化策略 v1 - 基准版本 (等权)")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    strategy = StrategyV1()
    portfolio = strategy.load_portfolio()

    r4 = portfolio['r4']
    r5 = portfolio['r5']

    print(f"\nR4稳健型: {len(r4)} 只股票")
    print(f"R5进取型: {len(r5)} 只股票")

    # 获取价格
    all_codes = list(set(r4['code'].tolist() + r5['code'].tolist()))
    print("\n获取价格数据...")
    price_df = strategy.fetcher.get_prices(all_codes, '2019-01-01', '2025-12-31')
    price_pivot = price_df.pivot(index='date', columns='code', values='close')
    print(f"价格数据: {len(price_pivot)} 个交易日")

    # R4
    print("\n【R4 稳健型】")
    r4_result = strategy.run_backtest(r4, price_pivot)
    print(f"  年化收益: {r4_result['annual_return']*100:.2f}%")
    print(f"  最大回撤: {r4_result['max_drawdown']*100:.1f}%")
    print(f"  夏普比率: {r4_result['sharpe']:.2f}")

    # R5
    print("\n【R5 进取型】")
    r5_result = strategy.run_backtest(r5, price_pivot)
    print(f"  年化收益: {r5_result['annual_return']*100:.2f}%")
    print(f"  最大回撤: {r5_result['max_drawdown']*100:.1f}%")
    print(f"  夏普比率: {r5_result['sharpe']:.2f}")

    print("\n" + "=" * 70)
    print("【v1 基准版结果】")
    print("=" * 70)
    print(f"| 组合   | 年化收益 | 最大回撤 | 夏普比率 |")
    print(f"|--------|----------|----------|----------|")
    print(f"| R4稳健 | {r4_result['annual_return']*100:>7.2f}% | {r4_result['max_drawdown']*100:>7.1f}% | {r4_result['sharpe']:>8.2f} |")
    print(f"| R5进取 | {r5_result['annual_return']*100:>7.2f}% | {r5_result['max_drawdown']*100:>7.1f}% | {r5_result['sharpe']:>8.2f} |")


if __name__ == '__main__':
    main()
