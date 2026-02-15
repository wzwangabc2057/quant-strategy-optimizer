"""
全局配置
"""
import os

# ClickHouse配置
CLICKHOUSE_HOST = '192.168.0.74'
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

# ==================== Universe 配置 ====================

UNIVERSE_CONFIG = {
    # 上市门槛
    'min_list_days': 60,               # 最小上市天数（交易日）

    # 流动性门槛
    'min_adv_cny': 2000,               # 最小20日成交额（万元）

    # 涨跌停检测
    'limit_threshold': 0.095,          # 涨跌停检测阈值（9.5%）
    'use_limit_detection': True,       # 是否启用涨跌停检测

    # 持仓限制
    'max_positions': 50,               # 最大持仓数量

    # 参与率
    'max_participation_rate': 0.10,    # 最大参与率
}

# ==================== 入场门槛配置 ====================

# R4 稳健型入场门槛
R4_ENTRY_GATE = {
    'composite_score_pct': 90.0,       # 综合分位数 >= 90%（Top 10%）
    'quality_min': 60.0,               # 质量因子得分 >= 60
    'growth_min': 0.0,                 # 不设成长门槛
    'momentum_min': 40.0,              # 动量因子得分 >= 40（允许反转）
    'value_min': 0.0,                  # 不设价值门槛
    'enable_trend_filter': False,      # 不启用趋势过滤
    'allow_reversal': True,            # 允许反转
    'small_cap_boost': False,          # 不加小市值分
    'max_positions': 50,               # 最大持仓数
    'min_positions': 20,               # 最小持仓数
}

# R5 进取型入场门槛
R5_ENTRY_GATE = {
    'composite_score_pct': 85.0,       # 综合分位数 >= 85%（Top 15%）
    'quality_min': 0.0,                # 不设质量门槛
    'growth_min': 50.0,                # 成长因子得分 >= 50
    'momentum_min': 50.0,              # 动量因子得分 >= 50
    'value_min': 0.0,                  # 不设价值门槛
    'enable_trend_filter': False,      # 不启用趋势过滤
    'allow_reversal': False,           # 不追求反转
    'small_cap_boost': True,           # 加小市值分
    'max_positions': 60,               # 最大持仓数
    'min_positions': 25,               # 最小持仓数
}

# ==================== 调仓配置 ====================

REBALANCE_CONFIG = {
    'frequency': 'monthly',            # 调仓频率: monthly / weekly / daily
    'turnover_limit': 0.30,            # 单次换手上限（30%）
    'min_trade_value': 10000,          # 最小交易金额（元）
}

# ==================== 财务可用日延迟配置 ====================
# 由于 ClickHouse 无 announce_date，使用 report_date + lag_days

FINANCIAL_LAG_PRESETS = {
    'base': 45,      # 基准延迟（估算最小值）
    'paper': 60,     # Paper Trading 默认（保守）
    'stress': 90,    # 压力测试延迟（极端情况）
}

# 默认 Paper Trading 使用的延迟天数
DEFAULT_LAG_DAYS = 60

# ==================== 执行配置 ====================

EXECUTION_CONFIG = {
    # 参与率上限
    'participation_rate_default': 0.01,     # 默认 1%（保守）
    'participation_rate_aggressive': 0.02,  # 进取 2%

    # 资金规模
    'paper_capital': 1_000_000,             # Paper Trading 资金规模

    # 容量裁剪
    'enable_capacity_clip': True,           # 是否启用容量裁剪
    'capacity_clip_mode': 'redistribute',   # redistribute(重分配) 或 cash(留现金)

    # 换手约束
    'max_turnover': 0.30,                   # 单次最大换手
    'enable_turnover_cap': True,            # 是否启用换手上限
}

# ==================== 治理约束配置 ====================

GOVERNANCE_CONFIG = {
    # 单票权重上限
    'R4': {
        'max_single_weight': 0.08,          # 单票最大权重 8%
        'max_industry_weight': 0.25,        # 行业最大权重 25%
        'min_list_days_reliability': 0.8,   # 上市日期可靠性阈值
    },
    'R5': {
        'max_single_weight': 0.10,          # 单票最大权重 10%
        'max_industry_weight': 0.30,        # 行业最大权重 30%
        'min_list_days_reliability': 0.7,   # 上市日期可靠性阈值
    },
}

# ==================== 行业分类配置 ====================

INDUSTRY_CONFIG = {
    'source_table': 'stock_block_em',       # 行业数据来源表
    'block_type': 'industry',               # 行业类型字段值
    'enable_industry_constraint': True,     # 是否启用行业约束
}

# ==================== Gate v2 验收配置 ====================

GATE_V2_CONFIG = {
    'R4': {
        'annual_return_p25_stress1': 18.0,  # Stress1下P25年化≥18%
        'max_drawdown_p75': 20.0,           # P75回撤≤20%
        'sharpe_p50': 1.0,                   # P50夏普≥1.0
        'max_turnover': 3.0,                 # 年换手≤300%
        'min_holding_days': 20,              # 或 平均持仓≥20天
        'max_cost_ratio': 35.0,              # 成本占毛收益≤35%
    },
    'R5': {
        'annual_return_p25_stress1': 20.0,  # Stress1下P25年化≥20%
        'max_drawdown_p75': 25.0,           # P75回撤≤25%
        'sharpe_p50': 1.0,                   # P50夏普≥1.0
        'max_turnover': 5.0,                 # 年换手≤500%
        'min_holding_days': 10,              # 或 平均持仓≥10天
        'max_cost_ratio': 45.0,              # 成本占毛收益≤45%
    }
}
