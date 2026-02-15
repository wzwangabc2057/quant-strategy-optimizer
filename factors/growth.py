"""
成长因子模块
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional


def calculate_profit_growth_score(profit_growth: float) -> float:
    """
    计算利润增长得分

    Args:
        profit_growth: 净利润增长率 (%)

    Returns:
        增长得分 (0-1)
    """
    if pd.isna(profit_growth):
        return np.nan

    if profit_growth >= 50:
        return 1.0
    elif profit_growth >= 30:
        return 0.9
    elif profit_growth >= 20:
        return 0.8
    elif profit_growth >= 10:
        return 0.6
    elif profit_growth >= 0:
        return 0.4
    elif profit_growth >= -10:
        return 0.2
    else:
        return 0


def calculate_revenue_growth_score(revenue_growth: float) -> float:
    """
    计算营收增长得分

    Args:
        revenue_growth: 营收增长率 (%)

    Returns:
        增长得分 (0-1)
    """
    if pd.isna(revenue_growth):
        return np.nan

    if revenue_growth >= 30:
        return 1.0
    elif revenue_growth >= 20:
        return 0.8
    elif revenue_growth >= 10:
        return 0.6
    elif revenue_growth >= 0:
        return 0.4
    elif revenue_growth >= -10:
        return 0.2
    else:
        return 0


def calculate_peg_score(pe: float, profit_growth: float) -> float:
    """
    计算PEG得分

    PEG = PE / 利润增长率
    PEG < 1 低估，PEG > 1 高估

    Args:
        pe: 市盈率
        profit_growth: 利润增长率 (%)

    Returns:
        PEG得分 (0-1)
    """
    if pd.isna(pe) or pd.isna(profit_growth) or profit_growth <= 0 or pe <= 0:
        return np.nan

    peg = pe / profit_growth

    if peg <= 0.5:
        return 1.0
    elif peg <= 0.8:
        return 0.9
    elif peg <= 1.0:
        return 0.8
    elif peg <= 1.2:
        return 0.6
    elif peg <= 1.5:
        return 0.4
    else:
        return 0.2


def calculate_small_cap_score(market_cap: float,
                              all_caps: pd.Series = None) -> float:
    """
    计算小市值因子得分

    Args:
        market_cap: 市值 (亿元)
        all_caps: 所有股票市值 (用于相对排名)

    Returns:
        小市值得分 (0-1)
    """
    if pd.isna(market_cap) or market_cap <= 0:
        return np.nan

    if all_caps is not None:
        # 基于排名的得分
        rank = (all_caps <= market_cap).sum()
        total = len(all_caps.dropna())
        if total == 0:
            return np.nan
        # 市值越小，得分越高
        return 1 - rank / total
    else:
        # 绝对值得分
        if market_cap < 50:
            return 1.0
        elif market_cap < 100:
            return 0.8
        elif market_cap < 200:
            return 0.6
        elif market_cap < 500:
            return 0.4
        else:
            return 0.2


def calculate_low_volatility_score(returns: pd.Series) -> float:
    """
    计算低波动率因子得分

    Args:
        returns: 收益率序列

    Returns:
        低波动得分 (0-1)
    """
    if len(returns) < 20:
        return np.nan

    volatility = returns.std() * np.sqrt(252)  # 年化波动率

    if volatility <= 0.15:
        return 1.0
    elif volatility <= 0.20:
        return 0.8
    elif volatility <= 0.30:
        return 0.6
    elif volatility <= 0.40:
        return 0.4
    else:
        return 0.2


def calculate_growth_score(financial_data: pd.DataFrame,
                          code: str,
                          price_series: pd.Series = None,
                          all_caps: pd.Series = None) -> Dict[str, float]:
    """
    计算股票的综合成长得分

    Args:
        financial_data: 财务数据
        code: 股票代码
        price_series: 价格序列 (用于计算波动率)
        all_caps: 所有股票市值

    Returns:
        成长因子得分字典
    """
    row = financial_data[financial_data['code'] == code]
    if len(row) == 0:
        return {
            'profit_growth': np.nan,
            'revenue_growth': np.nan,
            'peg': np.nan,
            'small_cap': np.nan,
            'low_volatility': np.nan,
        }

    row = row.iloc[0]

    scores = {
        'profit_growth': calculate_profit_growth_score(row.get('profit_growth', np.nan)),
        'revenue_growth': calculate_revenue_growth_score(row.get('revenue_growth', np.nan)),
        'peg': calculate_peg_score(row.get('pe', np.nan), row.get('profit_growth', np.nan)),
        'small_cap': calculate_small_cap_score(row.get('market_cap', np.nan), all_caps),
    }

    # 波动率因子需要价格数据
    if price_series is not None and len(price_series) > 20:
        returns = price_series.pct_change().dropna()
        scores['low_volatility'] = calculate_low_volatility_score(returns)
    else:
        scores['low_volatility'] = np.nan

    return scores


def calculate_all_growth_scores(financial_data: pd.DataFrame,
                                codes: list,
                                price_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    计算所有股票的成长得分

    Args:
        financial_data: 财务数据
        codes: 股票代码列表
        price_data: 价格数据透视表

    Returns:
        成长得分 DataFrame
    """
    # 获取所有市值
    all_caps = financial_data.set_index('code')['market_cap'] if 'market_cap' in financial_data.columns else None

    scores = []
    for code in codes:
        price_series = price_data[code] if price_data is not None and code in price_data.columns else None
        code_scores = calculate_growth_score(financial_data, code, price_series, all_caps)
        code_scores['code'] = code
        scores.append(code_scores)

    return pd.DataFrame(scores).set_index('code')
