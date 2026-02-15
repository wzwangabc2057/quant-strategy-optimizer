"""
组合策略核心类
================================================================================
整合因子计算、风控、订单生成的主策略类
支持动态Universe和入场逻辑，不再依赖静态名单。
================================================================================
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
import os
import warnings

from .risk_control import RiskController, RiskConfig
from .order_generator import OrderGenerator, OrderConfig, Order

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """策略配置"""
    name: str = "MultiFactorStrategy"

    # 因子权重
    factor_weights: Dict[str, float] = field(default_factory=lambda: {
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
    })

    # 权重调整
    weight_tiers: Dict[float, float] = field(default_factory=lambda: {
        0.95: 2.5,   # Top 5%
        0.90: 2.0,
        0.80: 1.6,
        0.70: 1.3,
        0.60: 1.1,
        0.40: 1.0,
        0.30: 0.9,
        0.20: 0.75,
        0.10: 0.5,
        0.00: 0.3,
    })

    # 基础参数
    max_weight: float = 0.08
    min_weight: float = 0.005
    base_tilt_strength: float = 1.0

    # 功能开关
    enable_reversal_boost: bool = True
    enable_market_regime: bool = True
    enable_risk_control: bool = True
    use_dynamic_universe: bool = True  # 是否使用动态Universe

    # 风控参数
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    order_config: OrderConfig = field(default_factory=OrderConfig)


class PortfolioStrategy:
    """
    组合策略

    支持两种模式：
    1. 动态模式：从Universe自动选股（推荐）
    2. 静态模式：使用外部名单（存在幸存者偏差风险）
    """

    def __init__(self, config: StrategyConfig = None,
                 universe_builder=None,
                 entry_logic=None,
                 governance=None):
        self.config = config or StrategyConfig()
        self.risk_controller = RiskController(self.config.risk_config)
        self.order_generator = OrderGenerator(self.config.order_config)

        # 动态模式组件
        self.universe_builder = universe_builder
        self.entry_logic = entry_logic
        self.governance = governance

        # 状态
        self.current_weights = {}
        self.current_holdings = {}
        self.current_nav = 1.0
        self.market_regime = 'neutral'

        # 日志
        self.rebalance_history = []
        self.performance_log = []
        self.universe_warning_shown = False

    def detect_market_regime(self, market_prices: pd.Series) -> Tuple[str, float, float]:
        """
        检测市场环境

        Returns:
            (regime, tilt_mult, max_weight_mult)
        """
        if not self.config.enable_market_regime:
            return 'neutral', 1.0, 1.0

        if len(market_prices) < 60:
            return 'volatile', 0.8, 0.8

        ret_20 = market_prices.iloc[-1] / market_prices.iloc[-20] - 1
        ret_60 = market_prices.iloc[-1] / market_prices.iloc[-60] - 1
        volatility = market_prices.pct_change().tail(20).std() * np.sqrt(252)

        ma20 = market_prices.tail(20).mean()
        ma60 = market_prices.tail(60).mean()
        trend = (ma20 / ma60 - 1)

        # 强牛市
        if ret_60 > 0.15 and ret_20 > 0.05 and volatility < 0.25:
            return 'strong_bull', 1.4, 1.5

        # 温和牛市
        if ret_60 > 0.08 and trend > 0.02:
            return 'bull', 1.2, 1.2

        # 熊市
        if ret_60 < -0.15 and ret_20 < -0.05:
            return 'strong_bear', 0.4, 0.6

        if ret_60 < -0.08 or trend < -0.02:
            return 'bear', 0.6, 0.8

        # 高波动
        if volatility > 0.35:
            return 'high_vol', 0.5, 0.7

        return 'neutral', 0.9, 0.9

    def calculate_factor_scores(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """计算因子得分"""
        result = factor_data.copy()

        def percentile_score(series, ascending=True):
            return series.rank(pct=True) * 100 if ascending else (1 - series.rank(pct=True)) * 100

        # 各因子得分
        score_mappings = {
            'roe': ('roe', True),
            'momentum': ('momentum', True),
            'reversal': ('drawdown_3m', False),  # 回撤越大，反转潜力越大
            'low_volatility': ('volatility', False),
            'profit_growth': ('net_profit_yoy', True),
            'revenue_growth': ('revenue_yoy', True),
            'pe_value': ('pe', False),
            'dividend_yield': ('dividend_yield', True),
        }

        for factor_name, (col_name, ascending) in score_mappings.items():
            if col_name in factor_data.columns:
                result[f'{factor_name}_score'] = percentile_score(factor_data[col_name], ascending)

        # 综合得分
        composite = pd.Series(50.0, index=result.index)
        for factor, weight in self.config.factor_weights.items():
            score_col = f'{factor}_score'
            if score_col in result.columns:
                composite += (result[score_col].fillna(50) - 50) * weight * 2

        result['composite_score'] = composite.clip(0, 100)

        return result

    def apply_reversal_boost(self, scores: pd.DataFrame, factor_data: pd.DataFrame) -> pd.DataFrame:
        """绩优股回撤加成"""
        if not self.config.enable_reversal_boost:
            return scores

        result = scores.copy()

        is_quality = (factor_data['roe'] > 6) & (factor_data.get('net_profit_yoy', 0) > -5)

        if 'drawdown_3m' not in factor_data.columns:
            return result

        drawdown = -factor_data['drawdown_3m']
        boost = pd.Series(0, index=result.index)

        # 分层加成
        boost[is_quality & (drawdown > 30)] = 25
        boost[is_quality & (drawdown > 25) & (drawdown <= 30)] = 20
        boost[is_quality & (drawdown > 20) & (drawdown <= 25)] = 15
        boost[is_quality & (drawdown > 15) & (drawdown <= 20)] = 10
        boost[is_quality & (drawdown > 10) & (drawdown <= 15)] = 6
        boost[is_quality & (drawdown > 5) & (drawdown <= 10)] = 3

        result['composite_score'] = (result['composite_score'] + boost).clip(0, 100)

        return result

    def adjust_weights(self, portfolio: pd.DataFrame, scores: pd.DataFrame,
                      tilt_mult: float = 1.0, max_weight_mult: float = 1.0) -> pd.DataFrame:
        """
        调整权重

        Args:
            portfolio: 原始持仓 [code, weight]
            scores: 因子得分 [code, composite_score]
            tilt_mult: 倾斜系数
            max_weight_mult: 最大权重系数

        Returns:
            调整后的持仓
        """
        # 处理列名差异：portfolio可能是'code'，scores可能是'symbol'
        portfolio_adj = portfolio.copy()
        if 'code' in portfolio_adj.columns and 'symbol' not in portfolio_adj.columns:
            portfolio_adj['symbol'] = portfolio_adj['code']

        if 'symbol' in scores.columns and 'code' not in scores.columns:
            scores_adj = scores.copy()
            scores_adj['code'] = scores_adj['symbol']
        else:
            scores_adj = scores

        # 检查是否有symbol列
        merge_col = 'symbol' if 'symbol' in portfolio_adj.columns else 'code'
        score_merge_col = 'symbol' if 'symbol' in scores_adj.columns else 'code'

        result = portfolio_adj[[merge_col, 'weight']].copy() if 'weight' in portfolio_adj.columns else portfolio_adj[[merge_col]].copy()

        if 'weight' not in result.columns:
            result['weight'] = 1.0 / len(result)  # 等权

        result = result.merge(scores_adj[[score_merge_col, 'composite_score']],
                              left_on=merge_col, right_on=score_merge_col, how='left')
        result['composite_score'] = result['composite_score'].fillna(50)

        # 计算百分位
        result['percentile'] = result['composite_score'].rank(pct=True)

        # 获取权重调整系数
        def get_multiplier(pct):
            for threshold, mult in sorted(self.config.weight_tiers.items(), reverse=True):
                if pct >= threshold:
                    return mult * tilt_mult
            return self.config.weight_tiers[0.0] * tilt_mult

        result['multiplier'] = result['percentile'].apply(get_multiplier)
        result['adjusted_weight'] = result['weight'] * result['multiplier']

        # 应用权重限制
        max_w = self.config.max_weight * max_weight_mult
        result['adjusted_weight'] = result['adjusted_weight'].clip(
            lower=self.config.min_weight,
            upper=max_w
        )

        # 归一化
        total = result['adjusted_weight'].sum()
        if total > 0:
            result['adjusted_weight'] = result['adjusted_weight'] / total

        # 保留原始列名
        if merge_col == 'symbol' and 'code' not in result.columns:
            result['code'] = result['symbol']

        return result

    def select_from_universe(self, date: str, factor_data: pd.DataFrame,
                            profile: str = 'R4') -> Tuple[pd.DataFrame, Dict]:
        """
        从动态Universe选股

        Args:
            date: 日期
            factor_data: 因子数据
            profile: 'R4' 或 'R5'

        Returns:
            (portfolio, selection_log)
        """
        if self.universe_builder is None or self.entry_logic is None:
            raise ValueError("UniverseBuilder或EntryLogic未初始化")

        # 1. 构建Universe
        universe = self.universe_builder.build_universe(date)

        if len(universe) == 0:
            logger.warning(f"日期 {date} 无Universe数据")
            return pd.DataFrame(), {'status': 'no_universe'}

        # 2. 执行选股
        target_portfolio, selection_log = self.entry_logic.select_stocks(
            universe, factor_data, self.current_holdings
        )

        selection_log['date'] = date
        selection_log['profile'] = profile

        return target_portfolio, selection_log

    def rebalance(self,
                 portfolio: pd.DataFrame,
                 factor_data: pd.DataFrame,
                 market_prices: pd.Series,
                 current_prices: Dict[str, float],
                 total_value: float,
                 date: str = None,
                 use_dynamic: bool = None) -> Tuple[List[Order], List[Order], dict]:
        """
        执行调仓

        Args:
            portfolio: 原始持仓（如果use_dynamic=True则忽略）
            factor_data: 因子数据
            market_prices: 市场指数价格
            current_prices: 当前股票价格
            total_value: 总市值
            date: 当前日期（动态模式需要）
            use_dynamic: 是否使用动态Universe（None则使用配置）

        Returns:
            (sell_orders, buy_orders, summary)
        """
        # 确定模式
        if use_dynamic is None:
            use_dynamic = self.config.use_dynamic_universe

        # 动态模式警告
        if not use_dynamic and not self.universe_warning_shown:
            warnings.warn(
                "使用静态名单模式，存在幸存者偏差风险。"
                "建议设置 use_dynamic_universe=True",
                UserWarning
            )
            self.universe_warning_shown = True

        # 1. 检测市场环境
        regime, tilt_mult, weight_mult = self.detect_market_regime(market_prices)
        self.market_regime = regime

        # 2. 风控检查
        risk_position_ratio = 1.0
        if self.config.enable_risk_control:
            market_triggered, market_ratio = self.risk_controller.check_market_stop_loss(market_prices)
            drawdown_triggered, drawdown_ratio = self.risk_controller.check_drawdown_stop(self.current_nav)
            risk_position_ratio = min(market_ratio, drawdown_ratio)

        # 3. 动态选股或使用给定持仓
        if use_dynamic and date and self.universe_builder and self.entry_logic:
            # 动态模式
            profile = 'R4' if 'stable' in str(factor_data.get('profile', 'stable')) else 'R5'
            portfolio, selection_log = self.select_from_universe(date, factor_data, profile)

            if len(portfolio) == 0:
                logger.warning("动态选股无结果，跳过调仓")
                return [], [], {'status': 'no_selection'}

        # 4. 计算因子得分
        scores = self.calculate_factor_scores(factor_data)

        # 5. 反转加成
        scores = self.apply_reversal_boost(scores, factor_data)

        # 6. 调整权重
        adjusted_portfolio = self.adjust_weights(portfolio, scores, tilt_mult, weight_mult)

        # 7. 应用风控仓位比例
        target_weights = dict(zip(
            adjusted_portfolio['code'] if 'code' in adjusted_portfolio.columns else adjusted_portfolio['symbol'],
            adjusted_portfolio['adjusted_weight']
        ))

        if risk_position_ratio < 1.0:
            for code in target_weights:
                target_weights[code] *= risk_position_ratio

        # 8. 应用治理约束（如果有）
        if self.governance:
            target_weights, governance_log = self.governance.apply_constraints(
                target_weights, factor_data
            )
        else:
            governance_log = {}

        # 9. 生成订单
        sell_orders, buy_orders, summary = self.order_generator.generate_rebalance_orders(
            current_holdings=self.current_holdings,
            target_weights=target_weights,
            prices=current_prices,
            total_value=total_value,
        )

        # 10. 更新状态
        self.current_weights = target_weights
        self.rebalance_history.append({
            'timestamp': datetime.now().isoformat(),
            'date': date,
            'regime': regime,
            'tilt_mult': tilt_mult,
            'weight_mult': weight_mult,
            'risk_position_ratio': risk_position_ratio,
            'use_dynamic': use_dynamic,
            'sell_count': len(sell_orders),
            'buy_count': len(buy_orders),
            'turnover': summary['turnover'],
        })

        # 合并日志
        summary['regime'] = regime
        summary['risk_position_ratio'] = risk_position_ratio
        summary['governance'] = governance_log

        return sell_orders, buy_orders, summary

    def update_nav(self, daily_return: float):
        """更新净值"""
        self.current_nav *= (1 + daily_return)

    def get_status(self) -> dict:
        """获取策略状态"""
        return {
            'name': self.config.name,
            'current_nav': self.current_nav,
            'market_regime': self.market_regime,
            'rebalance_count': len(self.rebalance_history),
            'use_dynamic_universe': self.config.use_dynamic_universe,
            'risk_status': self.risk_controller.get_status(),
        }

    def save_state(self, filepath: str):
        """保存状态"""
        state = {
            'config': {
                'name': self.config.name,
                'max_weight': self.config.max_weight,
                'min_weight': self.config.min_weight,
                'use_dynamic_universe': self.config.use_dynamic_universe,
            },
            'current_nav': self.current_nav,
            'market_regime': self.market_regime,
            'rebalance_history': self.rebalance_history[-100:],
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"策略状态已保存: {filepath}")

    def load_state(self, filepath: str):
        """加载状态"""
        if not os.path.exists(filepath):
            return

        with open(filepath, 'r') as f:
            state = json.load(f)

        self.current_nav = state.get('current_nav', 1.0)
        self.market_regime = state.get('market_regime', 'neutral')
        self.rebalance_history = state.get('rebalance_history', [])
        logger.info(f"策略状态已加载: {filepath}")


def create_strategy(profile: str = 'R4',
                   use_dynamic_universe: bool = True,
                   universe_builder=None,
                   entry_logic=None,
                   governance=None) -> PortfolioStrategy:
    """
    工厂函数：创建策略实例

    Args:
        profile: 'R4' 稳健型 或 'R5' 进取型
        use_dynamic_universe: 是否使用动态Universe
        universe_builder: Universe构建器
        entry_logic: 入场逻辑
        governance: 治理模块

    Returns:
        PortfolioStrategy实例
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import STABLE_WEIGHTS, AGGRESSIVE_WEIGHTS

    factor_weights = STABLE_WEIGHTS if profile == 'R4' else AGGRESSIVE_WEIGHTS

    config = StrategyConfig(
        name=f"MultiFactorStrategy_{profile}",
        factor_weights=factor_weights,
        use_dynamic_universe=use_dynamic_universe,
    )

    return PortfolioStrategy(
        config=config,
        universe_builder=universe_builder,
        entry_logic=entry_logic,
        governance=governance,
    )
