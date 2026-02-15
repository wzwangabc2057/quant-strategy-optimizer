"""
全局配置
"""
import os

# ClickHouse配置
CLICKHOUSE_HOST = '127.0.0.1'
CLICKHOUSE_PORT = 8123

# 回测配置
BACKTEST_START = '2020-01-01'
BACKTEST_END = '2025-12-31'

# 交易成本
BUY_COMMISSION = 0.00026   # 买入0.026%
SELL_COMMISSION = 0.00126  # 卖出0.126% (含印花税)

# 持仓文件路径
PORTFOLIO_FILE = '/Users/kangbing/Downloads/四大类策略持仓20251231.xlsx'

# 输出目录
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

# 因子配置
FACTOR_CONFIG = {
    # 动量周期权重
    'momentum_periods': {
        '1w': {'weight': 0.15, 'days': 5},
        '1m': {'weight': 0.25, 'days': 20},
        '3m': {'weight': 0.35, 'days': 60},
        '6m': {'weight': 0.25, 'days': 120},
    },

    # ATR止损参数
    'atr_period': 20,
    'atr_multiplier': 2.0,

    # 相关性阈值
    'max_correlation': 0.7,
}

# 策略因子权重
STABLE_WEIGHTS = {
    'dividend_yield': 0.20,
    'pe_value': 0.06,
    'roe': 0.18,
    'roe_stability': 0.12,
    'cash_flow_quality': 0.08,
    'profit_growth': 0.10,
    'revenue_growth': 0.04,
    'peg': 0.06,
    'small_cap': 0.04,
    'momentum': 0.06,
    'reversal': 0.04,
    'low_volatility': 0.02,
}

AGGRESSIVE_WEIGHTS = {
    'dividend_yield': 0.02,
    'pe_value': 0.02,
    'roe': 0.12,
    'roe_stability': 0.06,
    'cash_flow_quality': 0.04,
    'profit_growth': 0.24,
    'revenue_growth': 0.14,
    'peg': 0.16,
    'small_cap': 0.08,
    'momentum': 0.08,
    'reversal': 0.02,
    'low_volatility': 0.02,
}
