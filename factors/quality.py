"""
质量因子模块
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional


def calculate_roe_score(roe: float) -> float:
    """
    计算ROE得分

    ROE > 20%: 满分
    ROE 15-20%: 0.8
    ROE 10-15%: 0.6
    ROE 5-10%: 0.4
    ROE < 5%: 0.2
    """
    if pd.isna(roe) or roe <= 0:
        return 0
    if roe >= 20:
        return 1.0
    elif roe >= 15:
        return 0.8
    elif roe >= 10:
        return 0.6
    elif roe >= 5:
        return 0.4
    else:
        return 0.2


def calculate_roe_stability(roe_history: pd.Series) -> float:
    """
    计算ROE稳定性

    基于ROE历史数据的波动性

    Args:
        roe_history: ROE历史序列

    Returns:
        稳定性得分 (0-1)
    """
    if len(roe_history) < 4:
        return np.nan

    mean_roe = roe_history.mean()
    if mean_roe <= 0:
        return 0

    # 变异系数
    cv = roe_history.std() / abs(mean_roe)

    # 转换为得分 (CV越小越好)
    if cv <= 0.1:
        return 1.0
    elif cv <= 0.2:
        return 0.8
    elif cv <= 0.3:
        return 0.6
    elif cv <= 0.5:
        return 0.4
    else:
        return 0.2


def calculate_cash_flow_quality(operating_cash: float, net_profit: float) -> float:
    """
    计算现金流质量

    经营现金流 / 净利润

    Args:
        operating_cash: 经营现金流
        net_profit: 净利润

    Returns:
        现金流质量得分
    """
    if pd.isna(operating_cash) or pd.isna(net_profit) or net_profit <= 0:
        return np.nan

    ratio = operating_cash / net_profit

    if ratio >= 1.2:
        return 1.0
    elif ratio >= 1.0:
        return 0.9
    elif ratio >= 0.8:
        return 0.7
    elif ratio >= 0.5:
        return 0.5
    elif ratio > 0:
        return 0.3
    else:
        return 0


def calculate_dividend_score(dividend_yield: float) -> float:
    """
    计算股息率得分

    Args:
        dividend_yield: 股息率 (%)

    Returns:
        股息得分 (0-1)
    """
    if pd.isna(dividend_yield) or dividend_yield <= 0:
        return 0

    if dividend_yield >= 5:
        return 1.0
    elif dividend_yield >= 3:
        return 0.8
    elif dividend_yield >= 2:
        return 0.6
    elif dividend_yield >= 1:
        return 0.4
    else:
        return 0.2


def calculate_pe_value_score(pe: float) -> float:
    """
    计算PE估值得分

    PE越低越好 (在合理范围内)

    Args:
        pe: 市盈率

    Returns:
        估值得分 (0-1)
    """
    if pd.isna(pe) or pe <= 0:
        return np.nan

    if pe <= 8:
        return 1.0
    elif pe <= 15:
        return 0.8
    elif pe <= 25:
        return 0.6
    elif pe <= 40:
        return 0.4
    else:
        return 0.2


def calculate_quality_score(financial_data: pd.DataFrame,
                            code: str) -> Dict[str, float]:
    """
    计算股票的综合质量得分

    Args:
        financial_data: 财务数据 DataFrame
        code: 股票代码

    Returns:
        质量因子得分字典
    """
    row = financial_data[financial_data['code'] == code]
    if len(row) == 0:
        return {
            'roe': np.nan,
            'dividend_yield': np.nan,
            'pe_value': np.nan,
            'cash_flow_quality': np.nan,
        }

    row = row.iloc[0]

    return {
        'roe': calculate_roe_score(row.get('roe', np.nan)),
        'dividend_yield': calculate_dividend_score(row.get('dividend_yield', np.nan)),
        'pe_value': calculate_pe_value_score(row.get('pe', np.nan)),
        'cash_flow_quality': calculate_cash_flow_quality(
            row.get('operating_cash', np.nan),
            row.get('net_profit', np.nan)
        ),
    }


def calculate_all_quality_scores(financial_data: pd.DataFrame,
                                 codes: list) -> pd.DataFrame:
    """
    计算所有股票的质量得分

    Args:
        financial_data: 财务数据
        codes: 股票代码列表

    Returns:
        质量得分 DataFrame
    """
    scores = []
    for code in codes:
        code_scores = calculate_quality_score(financial_data, code)
        code_scores['code'] = code
        scores.append(code_scores)

    return pd.DataFrame(scores).set_index('code')
