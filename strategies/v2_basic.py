"""
v2 基础优化版 - 因子加权
================================================================================
在基准版本基础上，使用因子加权优化。
结果: R4 17.23%, R5 18.10%, Sharpe 0.99/0.96
================================================================================
"""
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    PORTFOLIO_FILE, BACKTEST_START, BACKTEST_END,
    BUY_COMMISSION, SELL_COMMISSION, STABLE_WEIGHTS, AGGRESSIVE_WEIGHTS
)
from data.fetcher import DataFetcher


class StrategyV2:
    """v2基础优化策略 - 因子加权"""

    def __init__(self, portfolio_file: str = None):
        self.fetcher = DataFetcher()
        self.portfolio_file = portfolio_file or PORTFOLIO_FILE

    def load_portfolio(self) -> dict:
        return self.fetcher.load_portfolio(self.portfolio_file)

    def build_factor_data(self, codes: list, price_pivot: pd.DataFrame,
                         eval_date: str) -> pd.DataFrame:
        """构建因子数据"""
        factor_df = pd.DataFrame({'code': codes})

        # 财务数据
        fin_df = self.fetcher.get_financial_data(codes)
        factor_df = factor_df.merge(fin_df, on='code', how='left')

        # 价格因子
        available_dates = [d for d in price_pivot.index if d <= eval_date]
        if len(available_dates) >= 120:
            recent = price_pivot.loc[available_dates].tail(120)

            # 动量
            if len(recent) >= 60:
                ret_60 = recent.iloc[-1] / recent.iloc[-60] - 1
                factor_df['momentum'] = factor_df['code'].map(ret_60.to_dict()).fillna(0)

            # 回撤
            cummax = recent.cummax()
            drawdown = ((recent - cummax) / cummax).min() * 100
            factor_df['drawdown'] = factor_df['code'].map(drawdown.to_dict()).fillna(-10)

            # 波动率
            vol = recent.pct_change().tail(60).std() * np.sqrt(252) * 100
            factor_df['volatility'] = factor_df['code'].map(vol.to_dict()).fillna(30)

        factor_df = factor_df.fillna({'roe': 10, 'eps': 1, 'momentum': 0, 'drawdown': -10, 'volatility': 30})
        return factor_df

    def calculate_scores(self, factor_df: pd.DataFrame, factor_weights: dict) -> pd.DataFrame:
        """计算因子得分"""
        result = factor_df.copy()

        def percentile_score(series, ascending=True):
            return series.rank(pct=True) * 100 if ascending else (1 - series.rank(pct=True)) * 100

        if 'roe' in factor_df.columns:
            result['roe_score'] = percentile_score(factor_df['roe'], True)
        if 'momentum' in factor_df.columns:
            result['momentum_score'] = percentile_score(factor_df['momentum'], True)
        if 'drawdown' in factor_df.columns:
            result['reversal_score'] = percentile_score(-factor_df['drawdown'], True)
        if 'volatility' in factor_df.columns:
            result['low_volatility_score'] = percentile_score(factor_df['volatility'], False)

        # 综合得分
        composite = pd.Series(50.0, index=result.index)
        for factor, weight in factor_weights.items():
            score_col = f'{factor}_score'
            if score_col in result.columns:
                composite += (result[score_col].fillna(50) - 50) * weight * 2

        result['composite_score'] = composite.clip(0, 100)
        return result

    def run_backtest(self, portfolio: pd.DataFrame, price_pivot: pd.DataFrame,
                    strategy_type: str = 'stable',
                    start_date: str = None, end_date: str = None) -> dict:
        """运行回测"""
        factor_weights = STABLE_WEIGHTS if strategy_type == 'stable' else AGGRESSIVE_WEIGHTS
        start_date = start_date or BACKTEST_START
        end_date = end_date or BACKTEST_END

        codes = portfolio['code'].tolist()
        dates = sorted([d for d in price_pivot.index if start_date <= d <= end_date])

        # 月度调仓
        rebalance_dates = []
        current_month = None
        for d in dates:
            ym = d[:7]
            if ym != current_month:
                current_month = ym
                rebalance_dates.append(d)

        # 初始权重
        current_weights = dict(zip(portfolio['code'], portfolio['weight']))
        current_weights = {k: v/sum(current_weights.values()) for k, v in current_weights.items()}
        original_weights = current_weights.copy()

        daily_returns = []
        original_returns = []

        for i in range(1, len(dates)):
            date = dates[i]
            prev_date = dates[i-1]

            # 月度调仓
            if date in rebalance_dates and i > 1:
                factor_df = self.build_factor_data(codes, price_pivot, date)
                scored_df = self.calculate_scores(factor_df, factor_weights)

                # 基于得分调整权重
                scores = dict(zip(scored_df['code'], scored_df['composite_score']))
                total_score = sum(scores.values())
                new_weights = {code: scores.get(code, 50) / total_score * len(codes) * current_weights.get(code, 0)
                              for code in codes}

                # 归一化
                total_w = sum(new_weights.values())
                if total_w > 0:
                    new_weights = {k: v/total_w for k, v in new_weights.items()}

                # 计算换手成本
                turnover = sum(abs(new_weights.get(c, 0) - current_weights.get(c, 0)) for c in codes) / 2
                trade_cost = turnover * (BUY_COMMISSION + SELL_COMMISSION)

                current_weights = new_weights

            # 计算收益
            daily_ret = 0
            orig_ret = 0

            for code in codes:
                if code not in price_pivot.columns:
                    continue

                curr = price_pivot.loc[date, code]
                prev = price_pivot.loc[prev_date, code]

                if pd.isna(curr) or pd.isna(prev) or prev <= 0:
                    continue

                stock_ret = curr / prev - 1
                daily_ret += stock_ret * current_weights.get(code, 0)
                orig_ret += stock_ret * original_weights.get(code, 0)

            if date in rebalance_dates and i > 1:
                daily_ret -= trade_cost

            daily_returns.append({'date': date, 'return': daily_ret})
            original_returns.append({'date': date, 'return': orig_ret})

        # 计算指标
        def calc_metrics(daily_rets):
            rets = np.array([r['return'] for r in daily_rets])
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
            }

        return {
            'enhanced': calc_metrics(daily_returns),
            'original': calc_metrics(original_returns),
        }


def main():
    print("=" * 70)
    print("多因子量化策略 v2 - 基础优化版")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    strategy = StrategyV2()
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
    r4_result = strategy.run_backtest(r4, price_pivot, 'stable')
    r4_enh = r4_result['enhanced']
    print(f"  年化收益: {r4_enh['annual_return']*100:.2f}%")
    print(f"  最大回撤: {r4_enh['max_drawdown']*100:.1f}%")
    print(f"  夏普比率: {r4_enh['sharpe']:.2f}")

    # R5
    print("\n【R5 进取型】")
    r5_result = strategy.run_backtest(r5, price_pivot, 'aggressive')
    r5_enh = r5_result['enhanced']
    print(f"  年化收益: {r5_enh['annual_return']*100:.2f}%")
    print(f"  最大回撤: {r5_enh['max_drawdown']*100:.1f}%")
    print(f"  夏普比率: {r5_enh['sharpe']:.2f}")

    print("\n" + "=" * 70)
    print("【v2 基础优化版结果】")
    print("=" * 70)
    print(f"| 组合   | 年化收益 | 最大回撤 | 夏普比率 |")
    print(f"|--------|----------|----------|----------|")
    print(f"| R4稳健 | {r4_enh['annual_return']*100:>7.2f}% | {r4_enh['max_drawdown']*100:>7.1f}% | {r4_enh['sharpe']:>8.2f} |")
    print(f"| R5进取 | {r5_enh['annual_return']*100:>7.2f}% | {r5_enh['max_drawdown']*100:>7.1f}% | {r5_enh['sharpe']:>8.2f} |")


if __name__ == '__main__':
    main()
