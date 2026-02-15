"""
订单生成模块
================================================================================
生成调仓订单，支持:
- 最小交易金额过滤
- 整手处理
- 涨跌停检测
- 交易成本计算
================================================================================
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class OrderConfig:
    """订单配置"""
    min_trade_value: float = 1000      # 最小交易金额
    lot_size: int = 100                # 每手股数
    max_order_value: float = 1000000   # 单笔最大交易金额
    buy_commission: float = 0.00026    # 买入佣金
    sell_commission: float = 0.00126   # 卖出佣金 (含印花税)
    slippage: float = 0.001            # 滑点


@dataclass
class Order:
    """订单"""
    code: str
    direction: str  # 'buy' or 'sell'
    shares: int
    price: float
    value: float
    reason: str
    commission: float
    timestamp: str


class OrderGenerator:
    """订单生成器"""

    def __init__(self, config: OrderConfig = None):
        self.config = config or OrderConfig()
        self.orders_log = []

    def generate_rebalance_orders(self,
                                  current_holdings: Dict[str, int],
                                  target_weights: Dict[str, float],
                                  prices: Dict[str, float],
                                  total_value: float,
                                  limit_up_codes: List[str] = None,
                                  limit_down_codes: List[str] = None) -> Tuple[List[Order], List[Order], dict]:
        """
        生成调仓订单

        Args:
            current_holdings: 当前持仓 {code: shares}
            target_weights: 目标权重 {code: weight}
            prices: 当前价格 {code: price}
            total_value: 总市值
            limit_up_codes: 涨停股票列表
            limit_down_codes: 跌停股票列表

        Returns:
            (sell_orders, buy_orders, summary)
        """
        limit_up_codes = limit_up_codes or []
        limit_down_codes = limit_down_codes or []

        sell_orders = []
        buy_orders = []

        # 计算当前权重
        current_weights = {}
        for code, shares in current_holdings.items():
            if code in prices and shares > 0:
                value = shares * prices[code]
                current_weights[code] = value / total_value

        all_codes = set(current_weights.keys()) | set(target_weights.keys())

        # 先处理卖出 (包括跌停股票)
        for code in all_codes:
            current_w = current_weights.get(code, 0)
            target_w = target_weights.get(code, 0)

            if current_w > target_w and code not in limit_down_codes:
                # 需要卖出
                diff_value = (current_w - target_w) * total_value

                if diff_value < self.config.min_trade_value:
                    continue

                if code not in prices:
                    continue

                price = prices[code]
                shares_to_sell = int(diff_value / price / self.config.lot_size) * self.config.lot_size

                if shares_to_sell > 0 and shares_to_sell <= current_holdings.get(code, 0):
                    order = Order(
                        code=code,
                        direction='sell',
                        shares=shares_to_sell,
                        price=price,
                        value=shares_to_sell * price,
                        reason=f'减仓 {current_w:.2%} -> {target_w:.2%}',
                        commission=shares_to_sell * price * self.config.sell_commission,
                        timestamp=datetime.now().isoformat(),
                    )
                    sell_orders.append(order)

        # 计算卖出后可用资金
        sell_proceeds = sum(o.value - o.commission for o in sell_orders)
        available_cash = total_value - sum(current_holdings.get(c, 0) * prices.get(c, 0)
                                          for c in current_holdings) + sell_proceeds

        # 再处理买入 (排除涨停股票)
        buy_candidates = []
        for code in all_codes:
            current_w = current_weights.get(code, 0)
            target_w = target_weights.get(code, 0)

            if target_w > current_w and code not in limit_up_codes:
                diff_value = (target_w - current_w) * total_value
                if diff_value >= self.config.min_trade_value and code in prices:
                    buy_candidates.append((code, diff_value, current_w, target_w))

        # 按金额排序，优先买入金额大的
        buy_candidates.sort(key=lambda x: -x[1])

        remaining_cash = available_cash
        for code, diff_value, current_w, target_w in buy_candidates:
            if remaining_cash < self.config.min_trade_value:
                break

            price = prices[code]
            max_value = min(diff_value, remaining_cash, self.config.max_order_value)

            shares_to_buy = int(max_value / price / self.config.lot_size) * self.config.lot_size

            if shares_to_buy > 0:
                order_value = shares_to_buy * price
                commission = order_value * self.config.buy_commission

                order = Order(
                    code=code,
                    direction='buy',
                    shares=shares_to_buy,
                    price=price,
                    value=order_value,
                    reason=f'加仓 {current_w:.2%} -> {target_w:.2%}',
                    commission=commission,
                    timestamp=datetime.now().isoformat(),
                )
                buy_orders.append(order)
                remaining_cash -= (order_value + commission)

        # 排序
        sell_orders.sort(key=lambda x: -x.value)
        buy_orders.sort(key=lambda x: -x.value)

        # 汇总
        summary = {
            'sell_count': len(sell_orders),
            'buy_count': len(buy_orders),
            'sell_value': sum(o.value for o in sell_orders),
            'buy_value': sum(o.value for o in buy_orders),
            'sell_commission': sum(o.commission for o in sell_orders),
            'buy_commission': sum(o.commission for o in buy_orders),
            'total_commission': sum(o.commission for o in sell_orders + buy_orders),
            'turnover': (sum(o.value for o in sell_orders) + sum(o.value for o in buy_orders)) / 2,
            'limit_up_skipped': len([c for c in buy_candidates if c[0] in limit_up_codes]),
            'limit_down_skipped': len([c for c in all_codes if c in limit_down_codes and current_weights.get(c, 0) > 0]),
        }

        # 记录日志
        self.orders_log.append({
            'timestamp': datetime.now().isoformat(),
            'sell_orders': len(sell_orders),
            'buy_orders': len(buy_orders),
            'summary': summary,
        })

        logger.info(f"生成订单: 卖出{len(sell_orders)}笔 ({summary['sell_value']:,.0f}元), "
                   f"买入{len(buy_orders)}笔 ({summary['buy_value']:,.0f}元), "
                   f"佣金{summary['total_commission']:,.0f}元")

        return sell_orders, buy_orders, summary

    def generate_close_order(self, code: str, shares: int, price: float,
                            reason: str = '') -> Order:
        """生成清仓订单"""
        return Order(
            code=code,
            direction='sell',
            shares=shares,
            price=price,
            value=shares * price,
            reason=reason or '清仓',
            commission=shares * price * self.config.sell_commission,
            timestamp=datetime.now().isoformat(),
        )

    def calculate_trade_cost(self, buy_value: float, sell_value: float) -> float:
        """计算交易成本"""
        buy_cost = buy_value * (self.config.buy_commission + self.config.slippage)
        sell_cost = sell_value * (self.config.sell_commission + self.config.slippage)
        return buy_cost + sell_cost

    def print_orders(self, sell_orders: List[Order], buy_orders: List[Order]):
        """打印订单"""
        print("\n" + "="*60)
        print("调仓订单")
        print("="*60)

        if sell_orders:
            print("\n【卖出订单】")
            print(f"{'代码':<8} {'股数':>8} {'价格':>8} {'金额':>12} {'原因'}")
            print("-"*60)
            for o in sell_orders[:10]:
                print(f"{o.code:<8} {o.shares:>8} {o.price:>8.2f} {o.value:>12,.0f} {o.reason}")

        if buy_orders:
            print("\n【买入订单】")
            print(f"{'代码':<8} {'股数':>8} {'价格':>8} {'金额':>12} {'原因'}")
            print("-"*60)
            for o in buy_orders[:10]:
                print(f"{o.code:<8} {o.shares:>8} {o.price:>8.2f} {o.value:>12,.0f} {o.reason}")

        total_sell = sum(o.value for o in sell_orders)
        total_buy = sum(o.value for o in buy_orders)
        total_commission = sum(o.commission for o in sell_orders + buy_orders)

        print("\n" + "-"*60)
        print(f"卖出: {len(sell_orders)}笔, {total_sell:,.0f}元")
        print(f"买入: {len(buy_orders)}笔, {total_buy:,.0f}元")
        print(f"佣金: {total_commission:,.0f}元")
        print("="*60)
