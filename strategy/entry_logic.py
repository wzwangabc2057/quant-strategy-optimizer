"""
入场逻辑模块
================================================================================
实现"自动找符合标准的逻辑进场"，不依赖外部名单。
================================================================================
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class EntryGateConfig:
    """入场门槛配置"""
    # 综合得分门槛
    composite_score_pct: float = 90.0    # 综合分位数门槛 (Top X%)

    # 因子最低分
    quality_min: float = 0.0             # 质量因子最低分
    growth_min: float = 0.0              # 成长因子最低分
    momentum_min: float = 0.0            # 动量因子最低分
    value_min: float = 0.0               # 价值因子最低分

    # 趋势过滤
    enable_trend_filter: bool = False    # 是否启用趋势过滤
    market_regime_filter: bool = False   # 市场环境过滤

    # 特殊规则
    allow_reversal: bool = True          # 允许反转入场
    small_cap_boost: bool = False        # 小市值加分

    # 数量限制
    max_positions: int = 50              # 最大持仓数
    min_positions: int = 20              # 最小持仓数


# R4 稳健型默认配置
R4_ENTRY_CONFIG = EntryGateConfig(
    composite_score_pct=90.0,    # Top 10%
    quality_min=60.0,            # 质量因子 >= 60
    momentum_min=40.0,           # 允许反转
    enable_trend_filter=False,
    allow_reversal=True,
    small_cap_boost=False,
    max_positions=50,
    min_positions=20,
)

# R5 进取型默认配置
R5_ENTRY_CONFIG = EntryGateConfig(
    composite_score_pct=85.0,    # Top 15%
    growth_min=50.0,             # 成长因子 >= 50
    momentum_min=50.0,           # 动量 >= 50
    enable_trend_filter=False,
    allow_reversal=False,        # 不追求反转
    small_cap_boost=True,        # 小市值加分
    max_positions=60,
    min_positions=25,
)


@dataclass
class FactorScores:
    """因子得分"""
    symbol: str
    composite_score: float = 50.0
    quality_score: float = 50.0
    growth_score: float = 50.0
    momentum_score: float = 50.0
    value_score: float = 50.0
    small_cap_score: float = 50.0
    volatility_score: float = 50.0


class EntryLogic:
    """
    入场逻辑

    功能：
    - 基于因子得分筛选可入场股票
    - 应用入场门槛（Entry Gate）
    - 生成目标持仓权重
    """

    def __init__(self, config: EntryGateConfig = None,
                 factor_weights: Dict[str, float] = None):
        self.config = config or R4_ENTRY_CONFIG
        self.factor_weights = factor_weights or {
            'quality': 0.30,
            'growth': 0.20,
            'momentum': 0.15,
            'value': 0.15,
            'small_cap': 0.10,
            'low_volatility': 0.10,
        }

        # 日志
        self.entry_log = []
        self.rejection_log = []

    def calculate_composite_score(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算综合得分

        Args:
            factor_data: 包含因子数据的DataFrame

        Returns:
            添加了composite_score的DataFrame
        """
        result = factor_data.copy()

        # 各因子列名映射
        factor_columns = {
            'quality': ['roe', 'roe_stability', 'cash_flow_quality'],
            'growth': ['profit_growth', 'revenue_growth', 'peg'],
            'momentum': ['momentum', 'momentum_1m', 'momentum_3m'],
            'value': ['pe_value', 'dividend_yield'],
            'small_cap': ['small_cap', 'market_cap_inv'],
            'low_volatility': ['low_volatility', 'volatility_inv'],
        }

        # 计算各因子类别得分
        for category, cols in factor_columns.items():
            available_cols = [c for c in cols if c in factor_data.columns]
            if available_cols:
                # 等权平均
                result[f'{category}_score'] = factor_data[available_cols].mean(axis=1)
            else:
                result[f'{category}_score'] = 50.0

        # 综合得分
        composite = pd.Series(50.0, index=result.index)

        weight_sum = sum(self.factor_weights.values())
        for category, weight in self.factor_weights.items():
            score_col = f'{category}_score'
            if score_col in result.columns:
                normalized_weight = weight / weight_sum
                composite += (result[score_col].fillna(50) - 50) * normalized_weight * 2

        result['composite_score'] = composite.clip(0, 100)

        # 小市值加分（R5专用）
        if self.config.small_cap_boost and 'small_cap_score' in result.columns:
            small_cap_boost = (result['small_cap_score'] - 50) * 0.1
            result['composite_score'] = (result['composite_score'] + small_cap_boost).clip(0, 100)

        return result

    def apply_entry_gate(self, universe: pd.DataFrame,
                        scores: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        应用入场门槛

        Args:
            universe: 可交易股票池
            scores: 因子得分

        Returns:
            (passed_stocks, gate_stats)
        """
        # 合并数据
        merged = universe.merge(scores, on='symbol', how='left')

        # 只考虑可交易的股票
        tradable = merged[merged['is_tradable'] == True].copy()

        assert len(tradable) > 0, "[SANITY CHECK] 可交易股票为空"

        # 计算分位数
        tradable['score_percentile'] = tradable['composite_score'].rank(pct=True) * 100

        # 入场门槛检查
        passed_mask = pd.Series(True, index=tradable.index)

        # 1. 综合得分分位数
        score_pct_threshold = 100 - self.config.composite_score_pct
        passed_mask &= tradable['score_percentile'] >= self.config.composite_score_pct
        tradable['reject_score_pct'] = ~passed_mask

        # 2. 质量因子最低分
        if self.config.quality_min > 0 and 'quality_score' in tradable.columns:
            quality_check = tradable['quality_score'] >= self.config.quality_min
            tradable['reject_quality'] = ~quality_check & passed_mask
            passed_mask &= quality_check

        # 3. 成长因子最低分
        if self.config.growth_min > 0 and 'growth_score' in tradable.columns:
            growth_check = tradable['growth_score'] >= self.config.growth_min
            tradable['reject_growth'] = ~growth_check & passed_mask
            passed_mask &= growth_check

        # 4. 动量因子最低分
        if self.config.momentum_min > 0 and 'momentum_score' in tradable.columns:
            momentum_check = tradable['momentum_score'] >= self.config.momentum_min
            tradable['reject_momentum'] = ~momentum_check & passed_mask
            passed_mask &= momentum_check

        # 5. 趋势过滤（如果启用）
        if self.config.enable_trend_filter and 'momentum_score' in tradable.columns:
            # 动量为负则禁入
            trend_check = tradable['momentum_score'] >= 50
            tradable['reject_trend'] = ~trend_check & passed_mask
            passed_mask &= trend_check

        # 通过入场门槛的股票
        passed = tradable[passed_mask].copy()

        # 按综合得分排序，取Top N
        passed = passed.sort_values('composite_score', ascending=False)
        passed = passed.head(self.config.max_positions)

        # 统计
        gate_stats = {
            'total_tradable': len(tradable),
            'passed_gate': len(passed),
            'reject_score_pct': tradable['reject_score_pct'].sum() if 'reject_score_pct' in tradable else 0,
            'reject_quality': tradable['reject_quality'].sum() if 'reject_quality' in tradable else 0,
            'reject_growth': tradable['reject_growth'].sum() if 'reject_growth' in tradable else 0,
            'reject_momentum': tradable['reject_momentum'].sum() if 'reject_momentum' in tradable else 0,
        }

        return passed, gate_stats

    def assign_weights(self, passed: pd.DataFrame,
                      weight_tiers: Dict[float, float] = None) -> pd.DataFrame:
        """
        分配权重

        Args:
            passed: 通过入场门槛的股票
            weight_tiers: 权重档位 {percentile_threshold: multiplier}

        Returns:
            添加了target_weight的DataFrame
        """
        if len(passed) == 0:
            return pd.DataFrame()

        # 默认权重档位
        if weight_tiers is None:
            weight_tiers = {
                0.95: 2.5,   # Top 5%
                0.90: 2.0,
                0.80: 1.6,
                0.70: 1.3,
                0.60: 1.1,
                0.50: 1.0,
                0.40: 0.9,
                0.30: 0.75,
                0.20: 0.5,
                0.00: 0.3,
            }

        result = passed.copy()

        # 计算在通过股票中的分位数
        result['tier_percentile'] = result['composite_score'].rank(pct=True)

        # 分配权重乘数
        def get_multiplier(pct):
            for threshold, mult in sorted(weight_tiers.items(), reverse=True):
                if pct >= threshold:
                    return mult
            return weight_tiers[0.0]

        result['weight_multiplier'] = result['tier_percentile'].apply(get_multiplier)

        # 基础权重（等权）
        base_weight = 1.0 / len(result)
        result['base_weight'] = base_weight

        # 应用乘数
        result['target_weight'] = result['base_weight'] * result['weight_multiplier']

        # 归一化
        total = result['target_weight'].sum()
        if total > 0:
            result['target_weight'] = result['target_weight'] / total

        return result

    def select_stocks(self, universe: pd.DataFrame,
                     factor_data: pd.DataFrame,
                     current_holdings: Dict[str, float] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        完整的选股流程

        Args:
            universe: 可交易股票池
            factor_data: 因子数据
            current_holdings: 当前持仓 {symbol: weight}

        Returns:
            (target_portfolio, selection_log)
        """
        logger.info(f"执行选股: Universe {len(universe)} 只股票")

        # 1. 计算综合得分
        scores = self.calculate_composite_score(factor_data)

        # 2. 应用入场门槛
        passed, gate_stats = self.apply_entry_gate(universe, scores)

        logger.info(f"入场门槛: {gate_stats['passed_gate']}/{gate_stats['total_tradable']} 通过")

        # 3. 分配权重
        target_portfolio = self.assign_weights(passed)

        if len(target_portfolio) == 0:
            logger.warning("无股票通过入场门槛")
            return pd.DataFrame(), {'gate_stats': gate_stats}

        # 4. 合并当前持仓信息（如果有）
        if current_holdings:
            target_portfolio['current_weight'] = target_portfolio['symbol'].map(
                lambda x: current_holdings.get(x, 0)
            )
            target_portfolio['weight_change'] = (
                target_portfolio['target_weight'] - target_portfolio['current_weight'].fillna(0)
            )

        # 5. 生成日志
        selection_log = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'composite_score_pct': self.config.composite_score_pct,
                'quality_min': self.config.quality_min,
                'growth_min': self.config.growth_min,
                'momentum_min': self.config.momentum_min,
            },
            'gate_stats': gate_stats,
            'selected_count': len(target_portfolio),
            'top_scores': target_portfolio.nlargest(5, 'composite_score')[
                ['symbol', 'composite_score', 'target_weight']
            ].to_dict('records'),
        }

        self.entry_log.append(selection_log)

        assert len(target_portfolio) >= self.config.min_positions or gate_stats['passed_gate'] < self.config.min_positions, \
            f"[SANITY CHECK] 选股数量不足: {len(target_portfolio)} < {self.config.min_positions}"

        return target_portfolio, selection_log

    def get_entry_log(self) -> List[Dict]:
        """获取入场日志"""
        return self.entry_log.copy()

    def save_entry_log(self, filepath: str):
        """保存入场日志"""
        with open(filepath, 'w') as f:
            json.dump(self.entry_log, f, indent=2, default=str)
        logger.info(f"入场日志已保存: {filepath}")


class PortfolioRebalancer:
    """
    组合调仓器

    整合 Universe -> Factors -> Score -> EntryGate -> Weighting -> Governance 流程
    """

    def __init__(self, entry_logic: EntryLogic, governance=None):
        self.entry_logic = entry_logic
        self.governance = governance

        self.rebalance_log = []

    def rebalance(self,
                 universe: pd.DataFrame,
                 factor_data: pd.DataFrame,
                 current_holdings: Dict[str, float],
                 prices: Dict[str, float],
                 total_value: float) -> Tuple[Dict[str, float], Dict]:
        """
        执行调仓

        Args:
            universe: 当日Universe
            factor_data: 因子数据
            current_holdings: 当前持仓
            prices: 当前价格
            total_value: 总市值

        Returns:
            (new_weights, rebalance_info)
        """
        # 1. 选股
        target_portfolio, selection_log = self.entry_logic.select_stocks(
            universe, factor_data, current_holdings
        )

        if len(target_portfolio) == 0:
            return {}, {'status': 'no_stocks_passed'}

        # 2. 提取目标权重
        target_weights = dict(zip(
            target_portfolio['symbol'],
            target_portfolio['target_weight']
        ))

        # 3. 应用治理约束（如果有）
        if self.governance:
            target_weights, governance_log = self.governance.apply_constraints(
                target_weights, factor_data
            )
        else:
            governance_log = {}

        # 4. 计算换手
        if current_holdings:
            turnover = sum(
                abs(target_weights.get(s, 0) - current_holdings.get(s, 0))
                for s in set(target_weights.keys()) | set(current_holdings.keys())
            ) / 2
        else:
            turnover = sum(target_weights.values())

        # 5. 生成调仓日志
        rebalance_info = {
            'timestamp': datetime.now().isoformat(),
            'selection': selection_log,
            'governance': governance_log,
            'turnover': turnover,
            'n_stocks': len(target_weights),
            'target_weights': target_weights,
        }

        self.rebalance_log.append(rebalance_info)

        return target_weights, rebalance_info

    def get_rebalance_log(self) -> List[Dict]:
        """获取调仓日志"""
        return self.rebalance_log.copy()


def create_entry_logic(profile: str = 'R4',
                       factor_weights: Dict[str, float] = None) -> EntryLogic:
    """
    工厂函数：创建入场逻辑

    Args:
        profile: 'R4' 稳健型 或 'R5' 进取型
        factor_weights: 自定义因子权重

    Returns:
        EntryLogic实例
    """
    if profile == 'R4':
        config = R4_ENTRY_CONFIG
        default_weights = {
            'quality': 0.30,
            'growth': 0.15,
            'momentum': 0.10,
            'value': 0.20,
            'small_cap': 0.05,
            'low_volatility': 0.20,
        }
    else:  # R5
        config = R5_ENTRY_CONFIG
        default_weights = {
            'quality': 0.15,
            'growth': 0.30,
            'momentum': 0.20,
            'value': 0.10,
            'small_cap': 0.15,
            'low_volatility': 0.10,
        }

    weights = factor_weights or default_weights

    return EntryLogic(config=config, factor_weights=weights)


def test_entry_logic():
    """测试入场逻辑"""
    print("="*60)
    print("测试 EntryLogic")
    print("="*60)

    # 创建模拟数据
    np.random.seed(42)
    n_stocks = 100

    universe = pd.DataFrame({
        'symbol': [f'{i:06d}' for i in range(1, n_stocks + 1)],
        'is_tradable': np.random.choice([True, False], n_stocks, p=[0.9, 0.1]),
        'adv20': np.random.uniform(1000, 50000, n_stocks),
        'close': np.random.uniform(5, 100, n_stocks),
    })

    factor_data = pd.DataFrame({
        'symbol': universe['symbol'],
        'roe': np.random.uniform(0, 30, n_stocks),
        'profit_growth': np.random.uniform(-20, 50, n_stocks),
        'momentum': np.random.uniform(-30, 50, n_stocks),
        'pe_value': np.random.uniform(5, 50, n_stocks),
    })

    # 测试 R4
    print("\n测试 R4 稳健型入场逻辑:")
    entry_r4 = create_entry_logic('R4')

    target_r4, log_r4 = entry_r4.select_stocks(universe, factor_data)

    print(f"  通过入场门槛: {len(target_r4)} 只")
    print(f"  入场门槛统计: {log_r4['gate_stats']}")

    if len(target_r4) > 0:
        print(f"\n  Top 5 持仓:")
        for _, row in target_r4.head(5).iterrows():
            print(f"    {row['symbol']}: 得分 {row['composite_score']:.1f}, "
                  f"权重 {row['target_weight']*100:.2f}%")

    # 测试 R5
    print("\n测试 R5 进取型入场逻辑:")
    entry_r5 = create_entry_logic('R5')

    target_r5, log_r5 = entry_r5.select_stocks(universe, factor_data)

    print(f"  通过入场门槛: {len(target_r5)} 只")
    print(f"  入场门槛统计: {log_r5['gate_stats']}")

    print("\n" + "="*60)
    print("测试完成")
    print("="*60)


if __name__ == '__main__':
    test_entry_logic()
