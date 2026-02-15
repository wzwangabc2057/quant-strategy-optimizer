"""
回测引擎模块
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from data.fetcher import get_price_pivot
from config import BUY_COMMISSION, SELL_COMMISSION
from .metrics import calculate_performance_metrics, print_metrics


class BacktestEngine:
    """回测引擎"""

    def __init__(self, buy_commission: float = BUY_COMMISSION,
                 sell_commission: float = SELL_COMMISSION,
                 slippage: float = 0.001):
        self.buy_commission = buy_commission
        self.sell_commission = sell_commission
        self.slippage = slippage

    def run_backtest(self, price_df: pd.DataFrame,
                     rebalance_weights: Dict[str, pd.DataFrame],
                     initial_capital: float = 1000000) -> Dict:
        """
        运行回测

        Args:
            price_df: 价格数据 (date, code, close)
            rebalance_weights: 调仓权重字典 {'r4': DataFrame, 'r5': DataFrame}
            initial_capital: 初始资金

        Returns:
            回测结果字典
        """
        price_pivot = get_price_pivot(price_df)

        results = {}
        for portfolio_type, weights_df in rebalance_weights.items():
            portfolio_values = self._run_single_backtest(
                price_pivot, weights_df, initial_capital
            )
            metrics = calculate_performance_metrics(portfolio_values)
            results[portfolio_type] = {
                'portfolio_values': portfolio_values,
                'metrics': metrics
            }

        return results

    def _run_single_backtest(self, price_pivot: pd.DataFrame,
                            weights_df: pd.DataFrame,
                            initial_capital: float) -> pd.Series:
        """运行单个组合的回测"""
        all_dates = price_pivot.index.sort_values()
        rebalance_dates = weights_df.index.sort_values()

        # 初始化
        cash = initial_capital
        holdings = {}  # {code: shares}
        portfolio_values = []

        rebalance_idx = 0

        for date in all_dates:
            # 检查是否需要调仓
            if rebalance_idx < len(rebalance_dates) and date >= rebalance_dates[rebalance_idx]:
                weights = weights_df.loc[rebalance_dates[rebalance_idx]]

                # 计算目标持仓
                current_value = cash + sum(
                    holdings.get(code, 0) * price_pivot.loc[date, code]
                    for code in holdings if code in price_pivot.columns
                )

                # 卖出所有持仓
                for code in list(holdings.keys()):
                    if code in price_pivot.columns and pd.notna(price_pivot.loc[date, code]):
                        price = price_pivot.loc[date, code]
                        shares = holdings[code]
                        cash -= shares * price * self.slippage  # 滑点
                        cash += shares * price * (1 - self.sell_commission)
                        del holdings[code]

                # 买入新持仓
                for code in weights.index:
                    if code in price_pivot.columns and pd.notna(price_pivot.loc[date, code]):
                        target_weight = weights[code]
                        if target_weight > 0:
                            price = price_pivot.loc[date, code]
                            target_value = current_value * target_weight
                            # 考虑滑点和佣金
                            buy_price = price * (1 + self.slippage)
                            shares = (target_value * (1 - self.buy_commission)) / buy_price
                            shares = int(shares / 100) * 100  # 整手
                            if shares > 0:
                                cost = shares * buy_price / (1 - self.buy_commission)
                                if cost <= cash:
                                    cash -= cost
                                    holdings[code] = shares

                rebalance_idx += 1

            # 计算当日净值
            current_value = cash + sum(
                holdings.get(code, 0) * price_pivot.loc[date, code]
                for code in holdings if code in price_pivot.columns
            )
            portfolio_values.append(current_value)

        return pd.Series(portfolio_values, index=all_dates)

    def run_backtest_with_stop_loss(self, price_df: pd.DataFrame,
                                    rebalance_weights: Dict[str, pd.DataFrame],
                                    initial_capital: float = 1000000,
                                    stop_loss: float = -0.15,
                                    take_profit: float = 0.30,
                                    trailing_stop: float = 0.10) -> Dict:
        """
        带止损止盈的回测

        Args:
            price_df: 价格数据
            rebalance_weights: 调仓权重
            initial_capital: 初始资金
            stop_loss: 止损线 (默认-15%)
            take_profit: 止盈线 (默认+30%)
            trailing_stop: 移动止损回撤 (默认10%)

        Returns:
            回测结果
        """
        price_pivot = get_price_pivot(price_df)
        all_dates = price_pivot.index.sort_values()
        rebalance_dates = list(rebalance_weights.values())[0].index.sort_values()

        results = {}
        for portfolio_type, weights_df in rebalance_weights.items():
            # 初始化
            cash = initial_capital
            holdings = {}  # {code: {'shares': n, 'cost_price': p, 'high_price': h}}
            portfolio_values = []
            rebalance_idx = 0

            for date in all_dates:
                # 检查止损止盈
                for code in list(holdings.keys()):
                    if code in price_pivot.columns and pd.notna(price_pivot.loc[date, code]):
                        current_price = price_pivot.loc[date, code]
                        holding = holdings[code]
                        cost_price = holding['cost_price']

                        # 更新最高价
                        holding['high_price'] = max(holding['high_price'], current_price)

                        pnl_pct = (current_price - cost_price) / cost_price

                        # 检查是否触发止损止盈
                        should_sell = False
                        if pnl_pct <= stop_loss:
                            should_sell = True  # 止损
                        elif pnl_pct >= take_profit:
                            should_sell = True  # 止盈
                        elif pnl_pct > 0.10:  # 盈利10%后启动移动止损
                            drawdown_from_high = (holding['high_price'] - current_price) / holding['high_price']
                            if drawdown_from_high >= trailing_stop:
                                should_sell = True  # 移动止损

                        if should_sell:
                            shares = holding['shares']
                            cash -= shares * current_price * self.slippage
                            cash += shares * current_price * (1 - self.sell_commission)
                            del holdings[code]

                # 检查是否需要调仓
                if rebalance_idx < len(rebalance_dates) and date >= rebalance_dates[rebalance_idx]:
                    weights = weights_df.loc[rebalance_dates[rebalance_idx]]

                    # 清仓不在目标中的股票
                    for code in list(holdings.keys()):
                        if code not in weights.index or weights[code] == 0:
                            if code in price_pivot.columns and pd.notna(price_pivot.loc[date, code]):
                                price = price_pivot.loc[date, code]
                                shares = holdings[code]['shares']
                                cash -= shares * price * self.slippage
                                cash += shares * price * (1 - self.sell_commission)
                                del holdings[code]

                    # 计算当前总市值
                    current_value = cash + sum(
                        h['shares'] * price_pivot.loc[date, code]
                        for code, h in holdings.items()
                        if code in price_pivot.columns
                    )

                    # 买入新持仓
                    for code in weights.index:
                        if code not in holdings and code in price_pivot.columns:
                            if pd.notna(price_pivot.loc[date, code]):
                                target_weight = weights[code]
                                if target_weight > 0:
                                    price = price_pivot.loc[date, code]
                                    target_value = current_value * target_weight
                                    buy_price = price * (1 + self.slippage)
                                    shares = (target_value * (1 - self.buy_commission)) / buy_price
                                    shares = int(shares / 100) * 100
                                    if shares > 0:
                                        cost = shares * buy_price / (1 - self.buy_commission)
                                        if cost <= cash:
                                            cash -= cost
                                            holdings[code] = {
                                                'shares': shares,
                                                'cost_price': price,
                                                'high_price': price
                                            }

                    rebalance_idx += 1

                # 计算当日净值
                current_value = cash + sum(
                    h['shares'] * price_pivot.loc[date, code]
                    for code, h in holdings.items()
                    if code in price_pivot.columns
                )
                portfolio_values.append(current_value)

            portfolio_series = pd.Series(portfolio_values, index=all_dates)
            metrics = calculate_performance_metrics(portfolio_series)
            results[portfolio_type] = {
                'portfolio_values': portfolio_series,
                'metrics': metrics
            }

        return results


def print_backtest_results(results: Dict, title: str = "回测结果"):
    """打印回测结果"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

    for portfolio_type, data in results.items():
        type_name = "R4稳健型" if portfolio_type == 'r4' else "R5进取型"
        print_metrics(data['metrics'], f"{type_name}组合")
