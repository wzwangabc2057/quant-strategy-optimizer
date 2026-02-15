"""
组合治理模块
================================================================================
包含:
- 单票权重约束
- 行业权重约束
- 流动性约束
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

logger = logging.getLogger(__name__)


@dataclass
class GovernanceConfig:
    """治理配置"""
    # 权重约束
    max_single_weight: float = 0.08        # 单票最大权重 8%
    min_single_weight: float = 0.005       # 单票最小权重 0.5%
    max_industry_weight: float = 0.25      # 行业最大权重 25%

    # 流动性约束
    min_daily_turnover: float = 10000000   # 最小日成交额 1000万
    max_participation_rate: float = 0.10   # 最大参与率 10%

    # 风险预算
    target_volatility: float = 0.15        # 目标波动率 15%
    max_drawdown_budget: float = 0.20      # 最大回撤预算 20%
    risk_contribution_cap: float = 0.10    # 单资产风险贡献上限 10%


class PortfolioGovernance:
    """组合治理"""

    def __init__(self, config: GovernanceConfig = None):
        self.config = config or GovernanceConfig()
        self.industry_map = {}  # {code: industry}

    def set_industry_map(self, industry_map: Dict[str, str]):
        """设置行业映射"""
        self.industry_map = industry_map

    def apply_weight_constraints(self,
                                weights: Dict[str, float],
                                industry_map: Dict[str, str] = None) -> Dict[str, float]:
        """
        应用权重约束

        Args:
            weights: 目标权重
            industry_map: 行业映射

        Returns:
            调整后的权重
        """
        industry_map = industry_map or self.industry_map
        adjusted = weights.copy()

        # 1. 单票权重约束
        for code in adjusted:
            if adjusted[code] > self.config.max_single_weight:
                logger.debug(f"{code} 权重 {adjusted[code]:.2%} 超上限，调整为 {self.config.max_single_weight:.2%}")
                adjusted[code] = self.config.max_single_weight

        # 2. 行业权重约束
        if industry_map:
            industry_weights = self._calculate_industry_weights(adjusted, industry_map)

            for industry, ind_weight in industry_weights.items():
                if ind_weight > self.config.max_industry_weight:
                    # 按比例缩减该行业股票权重
                    scale = self.config.max_industry_weight / ind_weight
                    for code, w in adjusted.items():
                        if industry_map.get(code) == industry:
                            adjusted[code] = w * scale
                    logger.info(f"行业 {industry} 权重 {ind_weight:.2%} 超上限，缩放 {scale:.2f}")

        # 3. 归一化
        total = sum(adjusted.values())
        if total > 0:
            for k in adjusted:
                adjusted[k] /= total

        return adjusted

    def apply_liquidity_constraints(self,
                                   weights: Dict[str, float],
                                   daily_turnovers: Dict[str, float],
                                   total_value: float) -> Dict[str, float]:
        """
        应用流动性约束

        Args:
            weights: 目标权重
            daily_turnovers: 日成交额 {code: turnover}
            total_value: 组合总市值

        Returns:
            调整后的权重
        """
        adjusted = weights.copy()

        for code, weight in adjusted.items():
            turnover = daily_turnovers.get(code, 0)

            # 检查最小流动性
            if turnover < self.config.min_daily_turnover:
                # 流动性不足，降低权重
                scale = turnover / self.config.min_daily_turnover
                adjusted[code] = weight * scale
                logger.debug(f"{code} 流动性不足，权重降低到 {adjusted[code]:.2%}")

            # 检查参与率约束
            position_value = weight * total_value
            if turnover > 0:
                participation = position_value / turnover / 5  # 假设5天完成建仓
                if participation > self.config.max_participation_rate:
                    max_weight = turnover * self.config.max_participation_rate * 5 / total_value
                    adjusted[code] = min(weight, max_weight)
                    logger.debug(f"{code} 参与率过高，权重限制到 {adjusted[code]:.2%}")

        # 归一化
        total = sum(adjusted.values())
        if total > 0:
            for k in adjusted:
                adjusted[k] /= total

        return adjusted

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
        """
        检查约束违规

        Returns:
            违规信息列表
        """
        violations = []
        industry_map = industry_map or self.industry_map

        # 检查单票权重
        for code, weight in weights.items():
            if weight > self.config.max_single_weight:
                violations.append(f"单票权重违规: {code} = {weight:.2%} > {self.config.max_single_weight:.2%}")

        # 检查行业权重
        if industry_map:
            industry_weights = self._calculate_industry_weights(weights, industry_map)
            for industry, weight in industry_weights.items():
                if weight > self.config.max_industry_weight:
                    violations.append(f"行业权重违规: {industry} = {weight:.2%} > {self.config.max_industry_weight:.2%}")

        return violations


class RiskBudgeting:
    """风险预算"""

    def __init__(self, config: GovernanceConfig = None):
        self.config = config or GovernanceConfig()

    def calculate_risk_contribution(self,
                                   weights: Dict[str, float],
                                   covariance_matrix: np.ndarray,
                                   codes: List[str]) -> Dict[str, float]:
        """
        计算风险贡献

        Args:
            weights: 权重
            covariance_matrix: 协方差矩阵
            codes: 股票代码列表

        Returns:
            风险贡献 {code: contribution}
        """
        w = np.array([weights.get(c, 0) for c in codes])

        # 组合波动率
        portfolio_var = w @ covariance_matrix @ w
        portfolio_vol = np.sqrt(portfolio_var)

        # 边际风险贡献
        mcr = covariance_matrix @ w

        # 风险贡献
        rc = w * mcr / portfolio_var

        return {code: rc[i] for i, code in enumerate(codes)}

    def allocate_risk_budget(self,
                            target_return: float,
                            covariance_matrix: np.ndarray,
                            expected_returns: np.ndarray,
                            codes: List[str]) -> Dict[str, float]:
        """
        风险预算分配（简化实现）

        Args:
            target_return: 目标收益
            covariance_matrix: 协方差矩阵
            expected_returns: 预期收益
            codes: 股票代码

        Returns:
            权重分配
        """
        n = len(codes)

        # 简化：使用等风险贡献
        # 实际应使用优化算法

        # 计算波动率倒数作为初始权重
        vols = np.sqrt(np.diag(covariance_matrix))
        inv_vols = 1 / vols
        inv_vols = np.nan_to_num(inv_vols, nan=1.0)

        weights = inv_vols / inv_vols.sum()
        weights = np.nan_to_num(weights, nan=1/n)

        # 调整到目标波动率
        current_vol = np.sqrt(weights @ covariance_matrix @ weights)
        if current_vol > 0:
            scale = self.config.target_volatility / current_vol
            weights = weights * min(scale, 1.0)  # 不能加杠杆

        # 归一化
        weights = weights / weights.sum()

        return {code: weights[i] for i, code in enumerate(codes)}

    def check_risk_budget(self,
                         weights: Dict[str, float],
                         covariance_matrix: np.ndarray,
                         codes: List[str]) -> Tuple[bool, Dict]:
        """
        检查风险预算

        Returns:
            (是否通过, 风险指标)
        """
        w = np.array([weights.get(c, 0) for c in codes])

        portfolio_var = w @ covariance_matrix @ w
        portfolio_vol = np.sqrt(portfolio_var)

        risk_contrib = self.calculate_risk_contribution(weights, covariance_matrix, codes)

        # 检查单资产风险贡献
        violations = []
        for code, rc in risk_contrib.items():
            if rc > self.config.risk_contribution_cap:
                violations.append(f"{code} 风险贡献 {rc:.2%} 超过上限 {self.config.risk_contribution_cap:.2%}")

        metrics = {
            'portfolio_volatility': portfolio_vol,
            'target_volatility': self.config.target_volatility,
            'vol_ratio': portfolio_vol / self.config.target_volatility if self.config.target_volatility > 0 else 0,
            'max_risk_contribution': max(risk_contrib.values()) if risk_contrib else 0,
            'violations': violations,
        }

        passed = len(violations) == 0 and portfolio_vol <= self.config.target_volatility * 1.1

        return passed, metrics


class PortfolioOptimizer:
    """组合优化器"""

    def __init__(self, governance: PortfolioGovernance = None):
        self.governance = governance or PortfolioGovernance()

    def optimize(self,
                target_weights: Dict[str, float],
                factor_scores: Dict[str, float],
                covariance_matrix: np.ndarray = None,
                industry_map: Dict[str, str] = None,
                daily_turnovers: Dict[str, float] = None,
                total_value: float = 1000000) -> Dict[str, float]:
        """
        组合优化

        Args:
            target_weights: 目标权重
            factor_scores: 因子得分
            covariance_matrix: 协方差矩阵
            industry_map: 行业映射
            daily_turnovers: 日成交额
            total_value: 总市值

        Returns:
            优化后的权重
        """
        # 1. 根据因子得分调整权重
        adjusted = {}
        for code, weight in target_weights.items():
            score = factor_scores.get(code, 50)
            # 得分越高，权重越高
            multiplier = 0.5 + score / 100  # 0.5 ~ 1.5
            adjusted[code] = weight * multiplier

        # 2. 归一化
        total = sum(adjusted.values())
        if total > 0:
            for k in adjusted:
                adjusted[k] /= total

        # 3. 应用治理约束
        adjusted = self.governance.apply_weight_constraints(adjusted, industry_map)

        # 4. 应用流动性约束
        if daily_turnovers:
            adjusted = self.governance.apply_liquidity_constraints(
                adjusted, daily_turnovers, total_value
            )

        return adjusted

    def generate_efficient_frontier(self,
                                   expected_returns: np.ndarray,
                                   covariance_matrix: np.ndarray,
                                   n_points: int = 10) -> List[Dict]:
        """
        生成有效前沿（简化实现）

        Returns:
            [{'return': x, 'volatility': y, 'weights': {...}}]
        """
        n = len(expected_returns)

        # 不同风险厌恶参数
        lambdas = np.linspace(0.1, 10, n_points)

        frontier = []
        for lam in lambdas:
            # 简化的均值方差优化
            # w = (1/lam) * inv(Σ) * μ
            try:
                inv_cov = np.linalg.inv(covariance_matrix)
                w = (1/lam) * inv_cov @ expected_returns

                # 归一化（只做多）
                w = np.maximum(w, 0)
                w = w / w.sum()

                ret = w @ expected_returns
                vol = np.sqrt(w @ covariance_matrix @ w)

                frontier.append({
                    'risk_aversion': lam,
                    'return': ret,
                    'volatility': vol,
                    'sharpe': ret / vol if vol > 0 else 0,
                })
            except:
                continue

        return frontier


class ComplianceChecker:
    """合规检查器"""

    def __init__(self, governance: PortfolioGovernance = None):
        self.governance = governance or PortfolioGovernance()
        self.compliance_log = []

    def check_pre_trade(self,
                       orders: List[Dict],
                       current_weights: Dict[str, float],
                       industry_map: Dict[str, str] = None) -> Tuple[bool, List[str]]:
        """
        交易前合规检查

        Args:
            orders: 订单列表
            current_weights: 当前权重
            industry_map: 行业映射

        Returns:
            (是否通过, 违规列表)
        """
        violations = []

        # 模拟交易后权重
        new_weights = current_weights.copy()
        for order in orders:
            code = order['code']
            direction = order['direction']
            delta_weight = order.get('weight_change', 0)

            if direction == 'buy':
                new_weights[code] = new_weights.get(code, 0) + delta_weight
            else:
                new_weights[code] = new_weights.get(code, 0) - delta_weight

        # 检查约束
        violations = self.governance.check_constraints(new_weights, industry_map)

        # 记录日志
        self.compliance_log.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'pre_trade',
            'n_orders': len(orders),
            'violations': violations,
        })

        return len(violations) == 0, violations

    def check_post_trade(self,
                        final_weights: Dict[str, float],
                        industry_map: Dict[str, str] = None) -> Dict:
        """
        交易后合规检查

        Returns:
            合规报告
        """
        violations = self.governance.check_constraints(final_weights, industry_map)

        report = {
            'timestamp': datetime.now().isoformat(),
            'type': 'post_trade',
            'violations': violations,
            'compliant': len(violations) == 0,
            'weights_summary': {
                'max_weight': max(final_weights.values()) if final_weights else 0,
                'min_weight': min(final_weights.values()) if final_weights else 0,
                'n_positions': len([w for w in final_weights.values() if w > 0.001]),
            },
        }

        self.compliance_log.append(report)

        return report
