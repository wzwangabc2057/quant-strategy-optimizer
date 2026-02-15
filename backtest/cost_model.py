"""
交易成本模型
================================================================================
包含:
- 分层佣金
- 印花税
- 滑点模型
- 冲击成本
- 涨跌停过滤
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
class CostConfig:
    """成本配置"""
    # 佣金
    buy_commission_rate: float = 0.00026      # 买入佣金率 0.026%
    sell_commission_rate: float = 0.0005      # 卖出佣金率 0.05%（不含印花税）
    min_commission: float = 5.0               # 最低佣金 5元

    # 印花税
    stamp_duty_rate: float = 0.001            # 印花税 0.1%（仅卖出）

    # 滑点
    base_slippage: float = 0.001              # 基础滑点 0.1%
    slippage_by_volume: Dict[str, float] = field(default_factory=lambda: {
        'high': 0.0005,      # 高流动性: 0.05%
        'medium': 0.001,     # 中流动性: 0.1%
        'low': 0.002,        # 低流动性: 0.2%
    })

    # 冲击成本
    impact_coefficient: float = 0.0005        # 冲击成本系数
    impact_decay: float = 0.5                 # 冲击衰减因子

    # 涨跌停
    limit_up_filter: bool = True              # 涨停过滤买入
    limit_down_filter: bool = True            # 跌停过滤卖出


class TransactionCostModel:
    """交易成本模型"""

    def __init__(self, config: CostConfig = None):
        self.config = config or CostConfig()

    def calculate_commission(self,
                            value: float,
                            direction: str = 'buy') -> float:
        """
        计算佣金

        Args:
            value: 交易金额
            direction: 'buy' 或 'sell'

        Returns:
            佣金金额
        """
        if direction == 'buy':
            rate = self.config.buy_commission_rate
        else:
            rate = self.config.sell_commission_rate

        commission = value * rate
        return max(commission, self.config.min_commission)

    def calculate_stamp_duty(self, value: float) -> float:
        """
        计算印花税（仅卖出）

        Args:
            value: 卖出金额

        Returns:
            印花税金额
        """
        return value * self.config.stamp_duty_rate

    def calculate_slippage(self,
                          value: float,
                          volume: float = None,
                          avg_daily_volume: float = None) -> float:
        """
        计算滑点

        Args:
            value: 交易金额
            volume: 交易量
            avg_daily_volume: 平均日成交量

        Returns:
            滑点成本
        """
        # 基础滑点
        slippage = value * self.config.base_slippage

        # 根据流动性调整
        if volume and avg_daily_volume and avg_daily_volume > 0:
            volume_ratio = volume / avg_daily_volume

            if volume_ratio > 0.1:  # 交易量超过日均10%
                liquidity = 'low'
            elif volume_ratio > 0.05:
                liquidity = 'medium'
            else:
                liquidity = 'high'

            rate = self.config.slippage_by_volume.get(liquidity, self.config.base_slippage)
            slippage = value * rate

        return slippage

    def calculate_impact_cost(self,
                             value: float,
                             daily_turnover: float = None) -> float:
        """
        计算市场冲击成本

        Args:
            value: 交易金额
            daily_turnover: 日成交额

        Returns:
            冲击成本
        """
        if daily_turnover is None or daily_turnover <= 0:
            # 使用默认估算
            return value * self.config.impact_coefficient

        # 冲击成本与交易占比成正比
        participation_rate = value / daily_turnover
        impact = value * self.config.impact_coefficient * (participation_rate ** self.config.impact_decay)

        # 上限为交易金额的1%
        return min(impact, value * 0.01)

    def calculate_total_cost(self,
                            value: float,
                            direction: str = 'buy',
                            volume: float = None,
                            avg_daily_volume: float = None,
                            daily_turnover: float = None) -> Dict[str, float]:
        """
        计算总交易成本

        Args:
            value: 交易金额
            direction: 'buy' 或 'sell'
            volume: 交易量
            avg_daily_volume: 平均日成交量
            daily_turnover: 日成交额

        Returns:
            成本明细字典
        """
        costs = {
            'commission': self.calculate_commission(value, direction),
            'slippage': self.calculate_slippage(value, volume, avg_daily_volume),
            'impact': self.calculate_impact_cost(value, daily_turnover),
        }

        # 印花税（仅卖出）
        if direction == 'sell':
            costs['stamp_duty'] = self.calculate_stamp_duty(value)
        else:
            costs['stamp_duty'] = 0

        costs['total'] = sum(costs.values())
        costs['total_bps'] = costs['total'] / value * 10000 if value > 0 else 0

        return costs

    def get_effective_cost_rate(self, direction: str = 'sell') -> float:
        """
        获取有效成本率（用于快速估算）

        Args:
            direction: 'buy' 或 'sell'

        Returns:
            有效成本率
        """
        if direction == 'buy':
            return (self.config.buy_commission_rate +
                    self.config.base_slippage +
                    self.config.impact_coefficient)
        else:
            return (self.config.sell_commission_rate +
                    self.config.stamp_duty_rate +
                    self.config.base_slippage +
                    self.config.impact_coefficient)


class LimitUpDownFilter:
    """涨跌停过滤器"""

    def __init__(self,
                 limit_up_threshold: float = 0.095,
                 limit_down_threshold: float = -0.095):
        """
        Args:
            limit_up_threshold: 涨停阈值 (默认9.5%)
            limit_down_threshold: 跌停阈值 (默认-9.5%)
        """
        self.limit_up_threshold = limit_up_threshold
        self.limit_down_threshold = limit_down_threshold

    def detect_limit_stocks(self,
                           price_df: pd.DataFrame,
                           prev_close_col: str = 'prev_close',
                           close_col: str = 'close') -> Tuple[List[str], List[str]]:
        """
        检测涨跌停股票

        Args:
            price_df: 价格数据
            prev_close_col: 前收盘价列名
            close_col: 收盘价列名

        Returns:
            (涨停股票列表, 跌停股票列表)
        """
        if prev_close_col not in price_df.columns:
            # 计算前收盘价
            price_df = price_df.copy()
            price_df['prev_close'] = price_df.groupby('code')[close_col].shift(1)
            prev_close_col = 'prev_close'

        price_df['return'] = (price_df[close_col] - price_df[prev_close_col]) / price_df[prev_close_col]

        limit_up = price_df[price_df['return'] >= self.limit_up_threshold]['code'].tolist()
        limit_down = price_df[price_df['return'] <= self.limit_down_threshold]['code'].tolist()

        return limit_up, limit_down

    def filter_tradable(self,
                       codes: List[str],
                       direction: str,
                       limit_up_codes: List[str],
                       limit_down_codes: List[str]) -> List[str]:
        """
        过滤可交易股票

        Args:
            codes: 待交易股票列表
            direction: 'buy' 或 'sell'
            limit_up_codes: 涨停股票列表
            limit_down_codes: 跌停股票列表

        Returns:
            可交易股票列表
        """
        if direction == 'buy':
            # 涨停无法买入
            return [c for c in codes if c not in limit_up_codes]
        else:
            # 跌停无法卖出
            return [c for c in codes if c not in limit_down_codes]


class CostAnalyzer:
    """成本分析器"""

    def __init__(self, cost_model: TransactionCostModel = None):
        self.cost_model = cost_model or TransactionCostModel()
        self.trade_log = []

    def log_trade(self,
                 code: str,
                 direction: str,
                 shares: int,
                 price: float,
                 date: str,
                 volume: float = None,
                 avg_daily_volume: float = None):
        """记录交易"""
        value = shares * price
        costs = self.cost_model.calculate_total_cost(
            value, direction, volume, avg_daily_volume
        )

        self.trade_log.append({
            'date': date,
            'code': code,
            'direction': direction,
            'shares': shares,
            'price': price,
            'value': value,
            **costs
        })

    def analyze(self) -> Dict:
        """分析成本"""
        if not self.trade_log:
            return {}

        df = pd.DataFrame(self.trade_log)

        summary = {
            'total_trades': len(df),
            'buy_trades': len(df[df['direction'] == 'buy']),
            'sell_trades': len(df[df['direction'] == 'sell']),
            'total_value': df['value'].sum(),
            'buy_value': df[df['direction'] == 'buy']['value'].sum(),
            'sell_value': df[df['direction'] == 'sell']['value'].sum(),
            'total_cost': df['total'].sum(),
            'total_commission': df['commission'].sum(),
            'total_stamp_duty': df['stamp_duty'].sum(),
            'total_slippage': df['slippage'].sum(),
            'total_impact': df['impact'].sum(),
            'avg_cost_bps': df['total_bps'].mean(),
            'cost_ratio': df['total'].sum() / df['value'].sum() if df['value'].sum() > 0 else 0,
        }

        # 成本构成
        total = summary['total_cost']
        if total > 0:
            summary['cost_breakdown'] = {
                'commission': df['commission'].sum() / total,
                'stamp_duty': df['stamp_duty'].sum() / total,
                'slippage': df['slippage'].sum() / total,
                'impact': df['impact'].sum() / total,
            }

        return summary

    def print_summary(self):
        """打印成本摘要"""
        summary = self.analyze()

        print("\n" + "="*60)
        print("交易成本分析")
        print("="*60)
        print(f"  总交易次数: {summary['total_trades']}")
        print(f"  总交易金额: {summary['total_value']:,.0f}")
        print(f"  总成本: {summary['total_cost']:,.0f} ({summary['avg_cost_bps']:.1f} bps)")
        print(f"  成本占比: {summary['cost_ratio']*100:.2f}%")
        print("\n  成本构成:")
        if 'cost_breakdown' in summary:
            for name, pct in summary['cost_breakdown'].items():
                print(f"    {name}: {pct*100:.1f}%")


class StressTestCostModel:
    """压力测试成本模型"""

    def __init__(self, base_config: CostConfig = None):
        self.base_config = base_config or CostConfig()

    def create_stress_config(self, stress_factor: float) -> CostConfig:
        """
        创建压力测试配置

        Args:
            stress_factor: 压力系数 (1.0=正常, 2.0=加倍)

        Returns:
            压力测试配置
        """
        return CostConfig(
            buy_commission_rate=self.base_config.buy_commission_rate * stress_factor,
            sell_commission_rate=self.base_config.sell_commission_rate * stress_factor,
            min_commission=self.base_config.min_commission,
            stamp_duty_rate=self.base_config.stamp_duty_rate * stress_factor,
            base_slippage=self.base_config.base_slippage * stress_factor,
            slippage_by_volume={
                k: v * stress_factor for k, v in self.base_config.slippage_by_volume.items()
            },
            impact_coefficient=self.base_config.impact_coefficient * stress_factor,
            impact_decay=self.base_config.impact_decay,
        )

    def run_stress_test(self,
                       trade_log: List[Dict],
                       stress_factors: List[float] = None) -> pd.DataFrame:
        """
        运行压力测试

        Args:
            trade_log: 交易日志
            stress_factors: 压力系数列表

        Returns:
            压力测试结果
        """
        stress_factors = stress_factors or [1.0, 1.5, 2.0, 3.0]
        results = []

        for factor in stress_factors:
            config = self.create_stress_config(factor)
            model = TransactionCostModel(config)

            total_cost = 0
            total_value = 0

            for trade in trade_log:
                costs = model.calculate_total_cost(
                    trade['value'],
                    trade['direction'],
                    trade.get('volume'),
                    trade.get('avg_daily_volume'),
                )
                total_cost += costs['total']
                total_value += trade['value']

            results.append({
                'stress_factor': factor,
                'total_cost': total_cost,
                'total_value': total_value,
                'cost_ratio': total_cost / total_value if total_value > 0 else 0,
                'cost_bps': total_cost / total_value * 10000 if total_value > 0 else 0,
            })

        return pd.DataFrame(results)
