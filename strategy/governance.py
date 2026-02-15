"""
组合治理模块
================================================================================
包含:
- 单票权重约束
- 行业权重约束（接入 stock_block_em）
- 流动性约束
- ADV20 容量裁剪（前置生效）
- 换手上限约束（前置生效）
- 风险预算
- 组合优化
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

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    GOVERNANCE_CONFIG, EXECUTION_CONFIG, INDUSTRY_CONFIG,
    FINANCIAL_LAG_PRESETS, DEFAULT_LAG_DAYS
)

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        return super().default(obj)


@dataclass
class GovernanceConfig:
    """治理配置"""
    # 权重约束
    max_single_weight: float = 0.08        # 单票最大权重 8%
    min_single_weight: float = 0.005       # 单票最小权重 0.5%
    max_industry_weight: float = 0.25      # 行业最大权重 25%

    # 流动性约束
    min_daily_turnover: float = 10000000   # 最小日成交额 1000万
    max_participation_rate: float = 0.01   # 最大参与率 1%（默认）

    # 容量裁剪
    enable_capacity_clip: bool = True      # 是否启用容量裁剪
    capacity_clip_mode: str = 'redistribute'  # redistribute 或 cash

    # 换手约束
    max_turnover: float = 0.30             # 单次最大换手 30%
    enable_turnover_cap: bool = True       # 是否启用换手上限

    # 风险预算
    target_volatility: float = 0.15        # 目标波动率 15%
    max_drawdown_budget: float = 0.20      # 最大回撤预算 20%
    risk_contribution_cap: float = 0.10    # 单资产风险贡献上限 10%

    # 上市日期可靠性
    min_list_days_reliability: float = 0.8  # 数据连续性阈值


def create_governance_config(profile: str = 'R4') -> GovernanceConfig:
    """创建治理配置"""
    config = GOVERNANCE_CONFIG.get(profile, GOVERNANCE_CONFIG['R4'])

    return GovernanceConfig(
        max_single_weight=config.get('max_single_weight', 0.08),
        max_industry_weight=config.get('max_industry_weight', 0.25),
        max_participation_rate=EXECUTION_CONFIG.get('participation_rate_default', 0.01),
        enable_capacity_clip=EXECUTION_CONFIG.get('enable_capacity_clip', True),
        capacity_clip_mode=EXECUTION_CONFIG.get('capacity_clip_mode', 'redistribute'),
        max_turnover=EXECUTION_CONFIG.get('max_turnover', 0.30),
        enable_turnover_cap=EXECUTION_CONFIG.get('enable_turnover_cap', True),
        min_list_days_reliability=config.get('min_list_days_reliability', 0.8),
    )


class PortfolioGovernance:
    """组合治理"""

    def __init__(self, config: GovernanceConfig = None, profile: str = 'R4'):
        self.config = config or create_governance_config(profile)
        self.profile = profile
        self.industry_map = {}  # {code: industry}

        # 裁剪日志
        self.clip_log = {
            'capacity': [],
            'industry': [],
            'turnover': [],
        }

    def load_industry_data(self, fetcher=None):
        """从ClickHouse加载行业数据"""
        if fetcher is None:
            from data.fetcher import DataFetcher
            fetcher = DataFetcher()

        industry_df = fetcher.get_industry_classification()
        if len(industry_df) > 0:
            self.industry_map = dict(zip(industry_df['code'], industry_df['industry_name']))
            logger.info(f"加载行业数据: {len(self.industry_map)} 只股票")
        else:
            logger.warning("未能加载行业数据")

        return self.industry_map

    def set_industry_map(self, industry_map: Dict[str, str]):
        """设置行业映射"""
        self.industry_map = industry_map

    def apply_capacity_clip(self,
                           weights: Dict[str, float],
                           adv20_map: Dict[str, float],
                           total_value: float,
                           prices: Dict[str, float] = None) -> Tuple[Dict[str, float], Dict]:
        """
        应用容量裁剪（ADV20 约束）- 前置生效

        Args:
            weights: 目标权重
            adv20_map: ADV20 数据 {code: adv20_万元}
            total_value: 组合总市值
            prices: 当前价格 {code: price}

        Returns:
            (调整后权重, 裁剪报告)
        """
        if not self.config.enable_capacity_clip:
            return weights, {'clipped': False}

        adjusted = weights.copy()
        clipped_stocks = []
        total_clipped = 0.0

        participation_rate = self.config.max_participation_rate

        for code, weight in adjusted.items():
            adv20 = adv20_map.get(code, 0)

            if adv20 <= 0:
                continue

            # 计算容量上限
            # ADV20 单位是万元，capacity 是元
            capacity = adv20 * 10000 * participation_rate

            # 目标持仓金额
            target_value = weight * total_value

            if target_value > capacity:
                # 需要裁剪
                clip_ratio = capacity / target_value
                new_weight = weight * clip_ratio
                clipped_amount = target_value - capacity

                adjusted[code] = new_weight
                clipped_stocks.append({
                    'code': code,
                    'original_weight': weight,
                    'new_weight': new_weight,
                    'clip_ratio': clip_ratio,
                    'adv20_wan': adv20,
                    'capacity': capacity,
                    'clipped_amount': clipped_amount,
                })
                total_clipped += clipped_amount

        # 处理裁剪后的权重分配
        if clipped_stocks and self.config.capacity_clip_mode == 'redistribute':
            # 按比例分配到未触顶股票
            total_weight = sum(adjusted.values())
            if total_weight > 0:
                scale = 1.0 / total_weight
                for code in adjusted:
                    adjusted[code] *= scale

        # 记录裁剪日志
        clip_report = {
            'timestamp': datetime.now().isoformat(),
            'clipped': len(clipped_stocks) > 0,
            'n_clipped': len(clipped_stocks),
            'total_clipped_amount': total_clipped,
            'clipped_stocks': clipped_stocks,
            'participation_rate': participation_rate,
        }

        self.clip_log['capacity'].append(clip_report)

        if clipped_stocks:
            logger.info(f"容量裁剪: {len(clipped_stocks)} 只股票, 总裁剪金额 {total_clipped:,.0f} 元")

        return adjusted, clip_report

    def apply_turnover_cap(self,
                          current_weights: Dict[str, float],
                          target_weights: Dict[str, float],
                          max_turnover: float = None) -> Tuple[Dict[str, float], Dict]:
        """
        应用换手上限约束 - 前置生效

        Args:
            current_weights: 当前权重
            target_weights: 目标权重
            max_turnover: 最大换手率（覆盖配置）

        Returns:
            (调整后权重, 裁剪报告)
        """
        if not self.config.enable_turnover_cap:
            return target_weights, {'capped': False}

        max_turn = max_turnover or self.config.max_turnover

        # 计算换手
        all_codes = set(current_weights.keys()) | set(target_weights.keys())
        total_turnover = 0.0

        for code in all_codes:
            current = current_weights.get(code, 0)
            target = target_weights.get(code, 0)
            total_turnover += abs(target - current)

        total_turnover /= 2  # 单边换手

        if total_turnover <= max_turn:
            return target_weights, {
                'capped': False,
                'turnover': total_turnover,
                'max_turnover': max_turn,
            }

        # 需要裁剪换手
        scale = max_turn / total_turnover
        adjusted = {}

        for code in all_codes:
            current = current_weights.get(code, 0)
            target = target_weights.get(code, 0)

            # 按比例缩减变动
            adjusted[code] = current + (target - current) * scale

        turnover_report = {
            'timestamp': datetime.now().isoformat(),
            'capped': True,
            'original_turnover': total_turnover,
            'capped_turnover': max_turn,
            'scale': scale,
            'max_turnover': max_turn,
        }

        self.clip_log['turnover'].append(turnover_report)

        logger.info(f"换手裁剪: {total_turnover:.2%} -> {max_turn:.2%}")

        return adjusted, turnover_report

    def apply_weight_constraints(self,
                                weights: Dict[str, float],
                                industry_map: Dict[str, str] = None) -> Tuple[Dict[str, float], Dict]:
        """
        应用权重约束

        Args:
            weights: 目标权重
            industry_map: 行业映射

        Returns:
            (调整后的权重, 裁剪报告)
        """
        industry_map = industry_map or self.industry_map
        adjusted = weights.copy()

        single_stock_clips = []

        # 1. 单票权重约束
        for code in adjusted:
            if adjusted[code] > self.config.max_single_weight:
                single_stock_clips.append({
                    'code': code,
                    'original_weight': adjusted[code],
                    'new_weight': self.config.max_single_weight,
                })
                adjusted[code] = self.config.max_single_weight

        # 2. 行业权重约束
        industry_clips = []
        if industry_map:
            industry_weights = self._calculate_industry_weights(adjusted, industry_map)

            for industry, ind_weight in industry_weights.items():
                if ind_weight > self.config.max_industry_weight:
                    scale = self.config.max_industry_weight / ind_weight

                    industry_clipped = []
                    for code, w in adjusted.items():
                        if industry_map.get(code) == industry:
                            original = w
                            adjusted[code] = w * scale
                            industry_clipped.append({
                                'code': code,
                                'original_weight': original,
                                'new_weight': adjusted[code],
                            })

                    industry_clips.append({
                        'industry': industry,
                        'original_weight': ind_weight,
                        'new_weight': self.config.max_industry_weight,
                        'scale': scale,
                        'stocks': industry_clipped,
                    })

                    logger.info(f"行业 {industry} 权重 {ind_weight:.2%} -> {self.config.max_industry_weight:.2%}")

        # 3. 归一化
        total = sum(adjusted.values())
        if total > 0:
            for k in adjusted:
                adjusted[k] /= total

        clip_report = {
            'timestamp': datetime.now().isoformat(),
            'single_stock_clips': single_stock_clips,
            'industry_clips': industry_clips,
            'max_single_weight': self.config.max_single_weight,
            'max_industry_weight': self.config.max_industry_weight,
        }

        self.clip_log['industry'].append(clip_report)

        return adjusted, clip_report

    def apply_all_constraints(self,
                             target_weights: Dict[str, float],
                             current_weights: Dict[str, float],
                             adv20_map: Dict[str, float],
                             total_value: float,
                             industry_map: Dict[str, str] = None) -> Tuple[Dict[str, float], Dict]:
        """
        应用所有约束（完整流程）

        执行顺序:
        1. 容量裁剪 (ADV20)
        2. 换手上限
        3. 单票权重约束
        4. 行业权重约束

        Returns:
            (最终权重, 完整报告)
        """
        industry_map = industry_map or self.industry_map

        # 1. 容量裁剪
        weights, capacity_report = self.apply_capacity_clip(
            target_weights, adv20_map, total_value
        )

        # 2. 换手上限
        weights, turnover_report = self.apply_turnover_cap(
            current_weights, weights
        )

        # 3. 权重约束（单票 + 行业）
        weights, weight_report = self.apply_weight_constraints(
            weights, industry_map
        )

        full_report = {
            'timestamp': datetime.now().isoformat(),
            'profile': self.profile,
            'capacity_clip': capacity_report,
            'turnover_cap': turnover_report,
            'weight_constraints': weight_report,
            'config': {
                'max_single_weight': self.config.max_single_weight,
                'max_industry_weight': self.config.max_industry_weight,
                'max_participation_rate': self.config.max_participation_rate,
                'max_turnover': self.config.max_turnover,
            },
        }

        return weights, full_report

    def _calculate_industry_weights(self,
                                   weights: Dict[str, float],
                                   industry_map: Dict[str, str]) -> Dict[str, float]:
        """计算行业权重"""
        industry_weights = {}
        for code, weight in weights.items():
            industry = industry_map.get(code, 'unknown')
            industry_weights[industry] = industry_weights.get(industry, 0) + weight
        return industry_weights

    def check_constraints(self,
                         weights: Dict[str, float],
                         industry_map: Dict[str, str] = None) -> List[str]:
        """检查约束违规"""
        violations = []
        industry_map = industry_map or self.industry_map

        for code, weight in weights.items():
            if weight > self.config.max_single_weight:
                violations.append(f"单票权重违规: {code} = {weight:.2%} > {self.config.max_single_weight:.2%}")

        if industry_map:
            industry_weights = self._calculate_industry_weights(weights, industry_map)
            for industry, weight in industry_weights.items():
                if weight > self.config.max_industry_weight:
                    violations.append(f"行业权重违规: {industry} = {weight:.2%} > {self.config.max_industry_weight:.2%}")

        return violations

    def save_clip_reports(self, output_dir: str):
        """保存裁剪报告"""
        os.makedirs(output_dir, exist_ok=True)

        # 容量裁剪报告
        if self.clip_log['capacity']:
            capacity_path = os.path.join(output_dir, 'capacity_clip_report.csv')
            rows = []
            for report in self.clip_log['capacity']:
                for stock in report.get('clipped_stocks', []):
                    rows.append({
                        'timestamp': report['timestamp'],
                        **stock,
                    })
            if rows:
                pd.DataFrame(rows).to_csv(capacity_path, index=False)
                logger.info(f"容量裁剪报告已保存: {capacity_path}")

        # 行业裁剪报告
        if self.clip_log['industry']:
            industry_path = os.path.join(output_dir, 'industry_clip_report.csv')
            rows = []
            for report in self.clip_log['industry']:
                for clip in report.get('industry_clips', []):
                    for stock in clip.get('stocks', []):
                        rows.append({
                            'timestamp': report['timestamp'],
                            'industry': clip['industry'],
                            **stock,
                        })
            if rows:
                pd.DataFrame(rows).to_csv(industry_path, index=False)
                logger.info(f"行业裁剪报告已保存: {industry_path}")

        # 换手裁剪报告
        if self.clip_log['turnover']:
            turnover_path = os.path.join(output_dir, 'turnover_clip_report.json')
            with open(turnover_path, 'w') as f:
                json.dump(self.clip_log['turnover'], f, indent=2, cls=NumpyEncoder)
            logger.info(f"换手裁剪报告已保存: {turnover_path}")


class RiskBudgeting:
    """风险预算"""

    def __init__(self, config: GovernanceConfig = None):
        self.config = config or GovernanceConfig()

    def calculate_risk_contribution(self,
                                   weights: Dict[str, float],
                                   covariance_matrix: np.ndarray,
                                   codes: List[str]) -> Dict[str, float]:
        """计算风险贡献"""
        w = np.array([weights.get(c, 0) for c in codes])
        portfolio_var = w @ covariance_matrix @ w
        portfolio_vol = np.sqrt(portfolio_var)
        mcr = covariance_matrix @ w
        rc = w * mcr / portfolio_var
        return {code: rc[i] for i, code in enumerate(codes)}

    def check_risk_budget(self,
                         weights: Dict[str, float],
                         covariance_matrix: np.ndarray,
                         codes: List[str]) -> Tuple[bool, Dict]:
        """检查风险预算"""
        w = np.array([weights.get(c, 0) for c in codes])
        portfolio_var = w @ covariance_matrix @ w
        portfolio_vol = np.sqrt(portfolio_var)

        violations = []
        risk_contrib = self.calculate_risk_contribution(weights, covariance_matrix, codes)
        for code, rc in risk_contrib.items():
            if rc > self.config.risk_contribution_cap:
                violations.append(f"{code} 风险贡献 {rc:.2%} 超过上限")

        metrics = {
            'portfolio_volatility': portfolio_vol,
            'target_volatility': self.config.target_volatility,
            'violations': violations,
        }

        return len(violations) == 0, metrics


class ComplianceChecker:
    """合规检查器"""

    def __init__(self, governance: PortfolioGovernance = None):
        self.governance = governance or PortfolioGovernance()
        self.compliance_log = []

    def check_pre_trade(self,
                       orders: List[Dict],
                       current_weights: Dict[str, float],
                       industry_map: Dict[str, str] = None) -> Tuple[bool, List[str]]:
        """交易前合规检查"""
        violations = []

        new_weights = current_weights.copy()
        for order in orders:
            code = order['code']
            delta_weight = order.get('weight_change', 0)
            if order.get('direction') == 'buy':
                new_weights[code] = new_weights.get(code, 0) + delta_weight
            else:
                new_weights[code] = new_weights.get(code, 0) - delta_weight

        violations = self.governance.check_constraints(new_weights, industry_map)

        self.compliance_log.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'pre_trade',
            'violations': violations,
        })

        return len(violations) == 0, violations

    def check_post_trade(self,
                        final_weights: Dict[str, float],
                        industry_map: Dict[str, str] = None) -> Dict:
        """交易后合规检查"""
        violations = self.governance.check_constraints(final_weights, industry_map)

        report = {
            'timestamp': datetime.now().isoformat(),
            'violations': violations,
            'compliant': len(violations) == 0,
            'max_weight': max(final_weights.values()) if final_weights else 0,
            'n_positions': len([w for w in final_weights.values() if w > 0.001]),
        }

        self.compliance_log.append(report)
        return report
