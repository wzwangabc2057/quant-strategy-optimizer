"""
回测指标计算模块
"""
import numpy as np
import pandas as pd
from typing import Dict, List


def calculate_returns(prices: pd.Series) -> pd.Series:
    """计算收益率序列"""
    return prices.pct_change().fillna(0)


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """计算累计收益率"""
    return (1 + returns).cumprod()


def calculate_annual_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """计算年化收益率"""
    if len(returns) == 0:
        return 0.0
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    return annual_return * 100


def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """计算最大回撤"""
    if len(cumulative_returns) == 0:
        return 0.0
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min() * 100


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.03,
                          periods_per_year: int = 252) -> float:
    """计算夏普比率"""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()


def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """计算年化波动率"""
    if len(returns) == 0:
        return 0.0
    return returns.std() * np.sqrt(periods_per_year) * 100


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.03,
                           periods_per_year: int = 252) -> float:
    """计算索提诺比率"""
    if len(returns) == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return float('inf')
    downside_std = downside_returns.std()
    if downside_std == 0:
        return float('inf')
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std


def calculate_win_rate(returns: pd.Series) -> float:
    """计算胜率"""
    if len(returns) == 0:
        return 0.0
    winning_periods = (returns > 0).sum()
    return winning_periods / len(returns) * 100


def calculate_calmar_ratio(annual_return: float, max_drawdown: float) -> float:
    """计算卡玛比率"""
    if max_drawdown == 0:
        return float('inf')
    return annual_return / abs(max_drawdown)


def calculate_performance_metrics(portfolio_values: pd.Series,
                                  benchmark_values: pd.Series = None) -> Dict:
    """
    计算完整的绩效指标

    Args:
        portfolio_values: 投资组合净值序列
        benchmark_values: 基准净值序列（可选）

    Returns:
        包含各项指标的字典
    """
    returns = calculate_returns(portfolio_values)
    cumulative = calculate_cumulative_returns(returns)

    metrics = {
        'annual_return': calculate_annual_return(returns),
        'max_drawdown': calculate_max_drawdown(cumulative),
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'volatility': calculate_volatility(returns),
        'sortino_ratio': calculate_sortino_ratio(returns),
        'win_rate': calculate_win_rate(returns),
    }

    metrics['calmar_ratio'] = calculate_calmar_ratio(
        metrics['annual_return'],
        metrics['max_drawdown']
    )

    # 如果有基准数据，计算相对指标
    if benchmark_values is not None and len(benchmark_values) == len(portfolio_values):
        benchmark_returns = calculate_returns(benchmark_values)
        excess_returns = returns - benchmark_returns
        metrics['excess_return'] = calculate_annual_return(excess_returns)
        metrics['tracking_error'] = calculate_volatility(excess_returns)
        if metrics['tracking_error'] > 0:
            metrics['information_ratio'] = metrics['excess_return'] / metrics['tracking_error']

    return metrics


def print_metrics(metrics: Dict, title: str = "绩效指标"):
    """打印绩效指标"""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    print(f"  年化收益率:  {metrics['annual_return']:>8.2f}%")
    print(f"  最大回撤:    {metrics['max_drawdown']:>8.2f}%")
    print(f"  夏普比率:    {metrics['sharpe_ratio']:>8.2f}")
    print(f"  索提诺比率:  {metrics['sortino_ratio']:>8.2f}")
    print(f"  卡玛比率:    {metrics['calmar_ratio']:>8.2f}")
    print(f"  年化波动率:  {metrics['volatility']:>8.2f}%")
    print(f"  胜率:        {metrics['win_rate']:>8.2f}%")
    if 'excess_return' in metrics:
        print(f"  超额收益:    {metrics['excess_return']:>8.2f}%")
        print(f"  信息比率:    {metrics.get('information_ratio', 0):>8.2f}")
    print(f"{'='*50}\n")


def compare_strategies(results: List[Dict]) -> pd.DataFrame:
    """
    对比多个策略的绩效

    Args:
        results: 策略结果列表，每个元素包含 name 和 metrics

    Returns:
        对比表格
    """
    comparison = []
    for result in results:
        row = {
            '策略': result['name'],
            '年化收益(%)': result['metrics']['annual_return'],
            '最大回撤(%)': result['metrics']['max_drawdown'],
            '夏普比率': result['metrics']['sharpe_ratio'],
            '卡玛比率': result['metrics']['calmar_ratio'],
        }
        if 'excess_return' in result['metrics']:
            row['超额收益(%)'] = result['metrics']['excess_return']
        comparison.append(row)

    return pd.DataFrame(comparison)
