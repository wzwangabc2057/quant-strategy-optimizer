"""
反转因子模块
"""
import pandas as pd
import numpy as np
from typing import Dict


def calculate_price_reversal(prices: pd.Series, lookback: int = 20) -> float:
    """
    计算价格反转得分

    过去N日跌幅越大，反转潜力越大

    Args:
        prices: 价格序列
        lookback: 回看天数

    Returns:
        反转得分 (跌幅的负值，即跌幅越大得分越高)
    """
    if len(prices) < lookback + 1:
        return np.nan

    ret = prices.iloc[-1] / prices.iloc[-lookback - 1] - 1
    # 反转得分 = -跌幅 (跌幅越大，得分越高)
    return -ret


def calculate_drawdown_reversal(prices: pd.Series, lookback: int = 60) -> float:
    """
    基于回撤的反转得分

    从近期最高点回撤幅度越大，反转潜力越大

    Args:
        prices: 价格序列
        lookback: 回看天数

    Returns:
        回撤反转得分
    """
    if len(prices) < lookback:
        return np.nan

    recent_prices = prices.iloc[-lookback:]
    high = recent_prices.max()
    current = recent_prices.iloc[-1]

    if high == 0:
        return np.nan

    drawdown = (high - current) / high
    return drawdown  # 回撤越大得分越高


def calculate_rsi_reversal(prices: pd.Series, period: int = 14) -> float:
    """
    基于RSI的反转得分

    RSI超卖时反转潜力大

    Args:
        prices: 价格序列
        period: RSI周期

    Returns:
        RSI反转得分 (RSI越低得分越高)
    """
    if len(prices) < period + 1:
        return np.nan

    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))

    current_rsi = rsi.iloc[-1]
    if np.isnan(current_rsi):
        return np.nan

    # RSI越低，得分越高 (反转潜力)
    return (50 - current_rsi) / 50  # 归一化到 [-1, 1]


def calculate_reversal_score(price_series: pd.DataFrame,
                             code: str,
                             quality_score: float = 0.5) -> float:
    """
    计算综合反转得分

    只对优质股票给予反转加分

    Args:
        price_series: 价格数据透视表
        code: 股票代码
        quality_score: 质量得分 (ROE等)

    Returns:
        反转得分
    """
    if code not in price_series.columns:
        return np.nan

    prices = price_series[code].dropna()
    if len(prices) < 30:
        return np.nan

    # 计算各反转指标
    price_rev = calculate_price_reversal(prices, 20)
    drawdown_rev = calculate_drawdown_reversal(prices, 60)
    rsi_rev = calculate_rsi_reversal(prices, 14)

    if np.isnan(price_rev):
        return np.nan

    # 基础反转得分
    base_score = 0.4 * price_rev + 0.4 * drawdown_rev + 0.2 * (rsi_rev if not np.isnan(rsi_rev) else 0)

    # 只有优质股票才给予反转加分
    if quality_score > 0.5:
        # 质量越好，反转权重越高
        quality_boost = quality_score * 0.5
        base_score *= (1 + quality_boost)

    return base_score


def calculate_all_reversal_scores(price_series: pd.DataFrame,
                                  codes: list,
                                  quality_scores: Dict[str, float] = None) -> pd.Series:
    """
    计算所有股票的反转得分

    Args:
        price_series: 价格数据透视表
        codes: 股票代码列表
        quality_scores: 质量得分字典

    Returns:
        反转得分序列
    """
    if quality_scores is None:
        quality_scores = {}

    scores = {}
    for code in codes:
        quality = quality_scores.get(code, 0.5)
        scores[code] = calculate_reversal_score(price_series, code, quality)

    return pd.Series(scores)
