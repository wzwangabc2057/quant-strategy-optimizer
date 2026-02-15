"""
风控模块
================================================================================
包含:
- 大盘止损
- 个股止损
- 最大回撤控制
- 行业集中度限制
================================================================================
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """风控配置"""
    market_stop_loss: float = 0.85       # 大盘止损线 (跌破年线15%)
    single_stock_stop_loss: float = 0.80  # 个股止损线 (-20%)
    single_stock_take_profit: float = 0.30  # 个股止盈线 (+30%)
    max_drawdown_limit: float = 0.15      # 最大回撤限制
    reduce_position_ratio: float = 0.7    # 触发风控后仓位比例
    max_single_weight: float = 0.08       # 单股最大权重
    max_industry_weight: float = 0.30     # 行业最大权重
    trailing_stop_trigger: float = 0.10   # 移动止损触发点 (盈利10%后启动)
    trailing_stop_pct: float = 0.10       # 移动止损回撤比例


class PositionTracker:
    """持仓跟踪器"""

    def __init__(self):
        self.positions: Dict[str, dict] = {}  # {code: position_info}

    def update_position(self, code: str, shares: int, cost_price: float, date: str):
        """更新或新建持仓"""
        self.positions[code] = {
            'shares': shares,
            'cost_price': cost_price,
            'high_price': cost_price,
            'entry_date': date,
            'holding_days': 0,
            'stop_loss_price': cost_price * 0.80,  # 默认-20%止损
            'trailing_active': False,
        }

    def update_price(self, code: str, current_price: float):
        """更新价格"""
        if code not in self.positions:
            return

        pos = self.positions[code]
        pos['holding_days'] += 1

        # 更新最高价
        if current_price > pos['high_price']:
            pos['high_price'] = current_price

        # 检查是否触发移动止损
        if not pos['trailing_active']:
            profit_pct = (current_price - pos['cost_price']) / pos['cost_price']
            if profit_pct >= 0.10:  # 盈利10%后启动
                pos['trailing_active'] = True

        # 更新移动止损价
        if pos['trailing_active']:
            pos['stop_loss_price'] = pos['high_price'] * 0.90  # 从最高点回撤10%

    def check_stop_loss(self, code: str, current_price: float,
                        config: RiskConfig) -> Optional[str]:
        """检查是否触发止损"""
        if code not in self.positions:
            return None

        pos = self.positions[code]
        profit_pct = (current_price - pos['cost_price']) / pos['cost_price']

        # 止盈
        if profit_pct >= config.single_stock_take_profit:
            return 'take_profit'

        # 固定止损
        if current_price <= pos['stop_loss_price']:
            return 'stop_loss'

        return None

    def remove_position(self, code: str):
        """移除持仓"""
        if code in self.positions:
            del self.positions[code]


class RiskController:
    """风控控制器"""

    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.position_tracker = PositionTracker()

        # 状态
        self.market_stop_triggered = False
        self.drawdown_stop_triggered = False
        self.current_drawdown = 0
        self.peak_nav = 1.0
        self.risk_log = []

    def check_market_stop_loss(self, market_prices: pd.Series) -> Tuple[bool, float]:
        """
        检查大盘止损

        Args:
            market_prices: 市场指数价格序列

        Returns:
            (是否触发, 建议仓位比例)
        """
        if len(market_prices) < 250:
            return False, 1.0

        ma250 = market_prices.rolling(250).mean().iloc[-1]
        current = market_prices.iloc[-1]

        # 跌破年线15%
        if current < ma250 * self.config.market_stop_loss:
            if not self.market_stop_triggered:
                self.market_stop_triggered = True
                self._log_risk_event('market_stop_loss', {
                    'current': current,
                    'ma250': ma250,
                    'ratio': current / ma250,
                })
            return True, self.config.reduce_position_ratio

        # 恢复条件: 超过年线2%
        if self.market_stop_triggered and current > ma250 * 1.02:
            self.market_stop_triggered = False
            self._log_risk_event('market_recovered', {
                'current': current,
                'ma250': ma250,
            })

        return self.market_stop_triggered, self.config.reduce_position_ratio if self.market_stop_triggered else 1.0

    def check_drawdown_stop(self, current_nav: float) -> Tuple[bool, float]:
        """
        检查最大回撤止损

        Args:
            current_nav: 当前净值

        Returns:
            (是否触发, 建议仓位比例)
        """
        if current_nav > self.peak_nav:
            self.peak_nav = current_nav

        self.current_drawdown = (current_nav - self.peak_nav) / self.peak_nav

        if self.current_drawdown < -self.config.max_drawdown_limit:
            if not self.drawdown_stop_triggered:
                self.drawdown_stop_triggered = True
                self._log_risk_event('drawdown_stop', {
                    'drawdown': self.current_drawdown,
                    'peak_nav': self.peak_nav,
                    'current_nav': current_nav,
                })
            return True, self.config.reduce_position_ratio

        # 恢复条件: 回撤恢复到-5%以内
        if self.drawdown_stop_triggered and self.current_drawdown > -0.05:
            self.drawdown_stop_triggered = False
            self._log_risk_event('drawdown_recovered', {
                'drawdown': self.current_drawdown,
            })

        return self.drawdown_stop_triggered, self.config.reduce_position_ratio if self.drawdown_stop_triggered else 1.0

    def check_single_stock_stop(self, current_weights: Dict[str, float],
                                stock_returns: Dict[str, float]) -> Dict[str, float]:
        """
        检查单股止损

        Args:
            current_weights: 当前权重
            stock_returns: 股票收益率

        Returns:
            调整后的权重
        """
        adjusted = current_weights.copy()

        for code, ret in stock_returns.items():
            if ret < -0.20:  # 单股跌幅超过20%
                adjusted[code] = current_weights.get(code, 0) * 0.5
                self._log_risk_event('single_stock_stop', {
                    'code': code,
                    'return': ret,
                    'action': 'reduce_50%',
                })

        # 归一化
        total = sum(adjusted.values())
        if total > 0:
            for k in adjusted:
                adjusted[k] /= total

        return adjusted

    def apply_weight_constraints(self, weights: Dict[str, float],
                                industry_map: Dict[str, str] = None) -> Dict[str, float]:
        """
        应用权重约束

        Args:
            weights: 目标权重
            industry_map: 股票-行业映射

        Returns:
            调整后的权重
        """
        # 单股权重约束
        for code in weights:
            if weights[code] > self.config.max_single_weight:
                weights[code] = self.config.max_single_weight

        # 行业权重约束
        if industry_map:
            industry_weights = {}
            for code, weight in weights.items():
                industry = industry_map.get(code, 'unknown')
                industry_weights[industry] = industry_weights.get(industry, 0) + weight

            for industry, ind_weight in industry_weights.items():
                if ind_weight > self.config.max_industry_weight:
                    # 按比例缩减该行业股票权重
                    scale = self.config.max_industry_weight / ind_weight
                    for code, w in weights.items():
                        if industry_map.get(code) == industry:
                            weights[code] = w * scale

        # 归一化
        total = sum(weights.values())
        if total > 0:
            for k in weights:
                weights[k] /= total

        return weights

    def _log_risk_event(self, event_type: str, details: dict):
        """记录风控事件"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'details': details,
        }
        self.risk_log.append(event)
        logger.warning(f"风控事件: {event_type} - {details}")

    def get_status(self) -> dict:
        """获取风控状态"""
        return {
            'market_stop_triggered': self.market_stop_triggered,
            'drawdown_stop_triggered': self.drawdown_stop_triggered,
            'current_drawdown': self.current_drawdown,
            'peak_nav': self.peak_nav,
            'risk_events_count': len(self.risk_log),
        }
