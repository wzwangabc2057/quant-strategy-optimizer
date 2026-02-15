"""
回测模块
"""
from .engine import BacktestEngine, print_backtest_results
from .metrics import calculate_performance_metrics, print_metrics, compare_strategies
from .data_alignment import DataAlignment, AsofJoin, SurvivorshipBiasHandler
from .cost_model import TransactionCostModel, CostConfig, CostAnalyzer, StressTestCostModel
from .validation import (
    WalkForwardValidator, ThreeSplitValidator,
    PerturbationTester, RobustnessAnalyzer, ValidationConfig
)
from .redteam import RedTeamAuditor, RedTeamConfig
