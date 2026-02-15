"""
动量因子模块
"""
import pandas as pd
import numpy as np
from typing import Dict


def calculate_single_period_momentum(prices: pd.Series, period: int) -> float:
    """计算单周期动量"""
    if len(prices) < period + 1:
        return np.nan
    return prices.iloc[-1] / prices.iloc[-period - 1] - 1


def calculate_multi_period_momentum(prices: pd.Series,
                                    periods: Dict[str, dict] = None) -> float:
    """
    计算多周期加权动量

    Args:
        prices: 价格序列
        periods: 周期配置 {'1w': {'weight': 0.15, 'days': 5}, ...}

    Returns:
        加权动量值
    """
    if periods is None:
        periods = {
            '1w': {'weight': 0.15, 'days': 5},
            '1m': {'weight': 0.25, 'days': 20},
            '3m': {'weight': 0.35, 'days': 60},
            '6m': {'weight': 0.25, 'days': 120},
        }

    total_momentum = 0
    total_weight = 0

    for period_name, config in periods.items():
        days = config['days']
        weight = config['weight']
        momentum = calculate_single_period_momentum(prices, days)
        if not np.isnan(momentum):
            total_momentum += momentum * weight
            total_weight += weight

    if total_weight == 0:
        return np.nan

    return total_momentum / total_weight


def calculate_volatility_adjusted_momentum(prices: pd.Series,
                                           lookback: int = 60) -> float:
    """
    计算波动率调整动量 (夏普式)

    动量 = 收益率 / 波动率

    Args:
        prices: 价格序列
        lookback: 回看天数

    Returns:
        风险调整后动量
    """
    if len(prices) < lookback + 1:
        return np.nan

    returns = prices.pct_change().dropna()
    if len(returns) < lookback:
        return np.nan

    recent_returns = returns.iloc[-lookback:]
    mean_return = recent_returns.mean()
    volatility = recent_returns.std()

    if volatility == 0:
        return np.nan

    # 年化
    ann_return = mean_return * 252
    ann_vol = volatility * np.sqrt(252)

    return ann_return / ann_vol


def calculate_momentum_score(price_series: pd.DataFrame,
                             code: str,
                             periods: Dict[str, dict] = None) -> float:
    """
    计算股票的综合动量得分

    Args:
        price_series: 价格数据透视表 (date x code)
        code: 股票代码
        periods: 周期配置

    Returns:
        动量得分 (0-1)
    """
    if code not in price_series.columns:
        return np.nan

    prices = price_series[code].dropna()
    if len(prices) < 10:
        return np.nan

    # 多周期动量
    multi_momentum = calculate_multi_period_momentum(prices, periods)

    # 波动率调整动量
    vol_adj_momentum = calculate_volatility_adjusted_momentum(prices)

    if np.isnan(multi_momentum) and np.isnan(vol_adj_momentum):
        return np.nan

    # 综合得分 (60% 多周期 + 40% 波动率调整)
    if np.isnan(multi_momentum):
        score = vol_adj_momentum
    elif np.isnan(vol_adj_momentum):
        score = multi_momentum * 10  # 简单缩放
    else:
        score = 0.6 * multi_momentum + 0.4 * vol_adj_momentum * 0.1

    return score


def calculate_all_momentum_scores(price_series: pd.DataFrame,
                                  codes: list,
                                  periods: Dict[str, dict] = None) -> pd.Series:
    """
    计算所有股票的动量得分

    Args:
        price_series: 价格数据透视表
        codes: 股票代码列表
        periods: 周期配置

    Returns:
        动量得分序列
    """
    scores = {}
    for code in codes:
        scores[code] = calculate_momentum_score(price_series, code, periods)

    return pd.Series(scores)
