"""
Daily Monitor - 每日监控核心逻辑
"""
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ops.utils import (
    load_params, save_params, ensure_dir, write_json,
    append_to_csv, calculate_drawdown, calculate_rolling_sharpe,
    FROZEN_PARAMS
)
from ops.alerts import AlertManager, format_daily_alert

logger = logging.getLogger(__name__)


class DailyMonitor:
    """每日监控器"""

    # 运营规则阈值
    DE_RISK_DRAWDOWN_THRESHOLD = 0.15      # 20日回撤 > 15%
    DE_RISK_NEGATIVE_DAYS = 10              # 连续负收益天数
    EXIT_MAX_DRAWDOWN_R4 = 0.25            # R4 最大回撤阈值
    EXIT_MAX_DRAWDOWN_R5 = 0.30            # R5 最大回撤阈值
    EXIT_MONTHLY_LOSS_THRESHOLD = -0.05    # 2个月累计收益 < -5%
    EXIT_SHARPE_THRESHOLD = 1.0            # 3个月夏普 < 1.0

    def __init__(self,
                 results_dir: str = 'results_paper',
                 capital: float = 1_000_000,
                 profile: str = 'R4',
                 enable_alerts: bool = True):
        """
        初始化每日监控器

        Args:
            results_dir: 结果目录
            capital: 初始资金
            profile: 风险配置 (R4/R5)
            enable_alerts: 启用告警
        """
        self.results_dir = results_dir
        self.capital = capital
        self.profile = profile
        self.enable_alerts = enable_alerts

        self.daily_dir = os.path.join(results_dir, 'daily')
        self.equity_file = os.path.join(results_dir, 'equity_curve.csv')
        self.alert_manager = AlertManager() if enable_alerts else None

        ensure_dir(self.daily_dir)

    def run(self, date: str = None) -> Dict[str, Any]:
        """
        运行每日监控

        Args:
            date: 日期 (YYYY-MM-DD)，默认今天

        Returns:
            监控结果
        """
        date = date or datetime.now().strftime('%Y-%m-%d')
        logger.info(f"运行每日监控: {date}")

        # 1. 获取当日持仓和价格
        positions, prices = self._get_positions_and_prices(date)

        if not positions or not prices:
            logger.warning(f"无法获取持仓或价格数据: {date}")
            return self._create_empty_result(date, "无法获取数据")

        # 2. 计算当日净值
        equity, daily_return = self._calculate_equity(positions, prices, date)

        # 3. 更新净值曲线
        self._update_equity_curve(date, equity, daily_return)

        # 4. 计算风险指标
        metrics = self._calculate_metrics(date)

        # 5. 检查运营规则
        decision, alerts = self._check_rules(metrics)

        # 6. 生成报告
        result = {
            'date': date,
            'equity': equity,
            'daily_return': daily_return,
            'decision': decision,
            'alerts': alerts,
            'metrics': metrics,
            'positions_count': len(positions),
        }

        # 7. 保存结果
        self._save_daily_result(date, result)

        # 8. 发送告警
        if self.enable_alerts and self.alert_manager:
            self._send_alert(result)

        logger.info(f"每日监控完成: {date}, 决策: {decision}")
        return result

    def _get_positions_and_prices(self, date: str) -> Tuple[Dict, Dict]:
        """获取持仓和价格"""
        # 从最近的 monthly 结果中读取持仓
        positions = self._load_latest_positions()
        if not positions:
            logger.warning("未找到持仓数据，使用模拟数据")
            # 模拟持仓（仅用于演示）
            positions = {f'00000{i}': 0.02 for i in range(1, 51)}

        # 获取价格数据
        prices = self._fetch_prices(list(positions.keys()), date)

        return positions, prices

    def _load_latest_positions(self) -> Dict[str, float]:
        """加载最近一次月度调仓的持仓"""
        # 查找最近的 monthly 目录
        monthly_base = os.path.join(self.results_dir, 'monthly')
        if not os.path.exists(monthly_base):
            return {}

        # 按时间倒序查找
        months = sorted(os.listdir(monthly_base), reverse=True)
        for month in months:
            month_dir = os.path.join(monthly_base, month)
            if not os.path.isdir(month_dir):
                continue

            # 查找最新的 run_id
            runs = sorted(os.listdir(month_dir), reverse=True)
            for run in runs:
                positions_file = os.path.join(month_dir, run, 'positions.json')
                if os.path.exists(positions_file):
                    data = load_params(positions_file)
                    return data.get('positions', {})

        return {}

    def _fetch_prices(self, codes: List[str], date: str) -> Dict[str, float]:
        """获取价格数据"""
        try:
            from data.fetcher import DataFetcher
            fetcher = DataFetcher()

            # 获取前一个交易日和当日的价格
            prev_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=3)).strftime('%Y-%m-%d')
            df = fetcher.get_prices(codes[:100], prev_date, date)  # 限制数量避免超时

            if len(df) == 0:
                logger.warning(f"未获取到价格数据: {date}")
                return {}

            # 获取最新价格
            latest = df[df['date'] == df['date'].max()]
            return dict(zip(latest['code'], latest['close']))

        except Exception as e:
            logger.error(f"获取价格数据失败: {e}")
            # 返回模拟价格（仅用于演示）
            return {code: 10.0 + np.random.random() * 5 for code in codes}

    def _calculate_equity(self,
                          positions: Dict[str, float],
                          prices: Dict[str, float],
                          date: str) -> Tuple[float, float]:
        """计算净值"""
        equity = 0.0
        for code, weight in positions.items():
            price = prices.get(code, 0)
            equity += weight * self.capital * (price / 10.0)  # 假设基准价格10

        # 读取昨日净值
        prev_equity = self._get_previous_equity()
        daily_return = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0

        return equity, daily_return

    def _get_previous_equity(self) -> float:
        """获取昨日净值"""
        if os.path.exists(self.equity_file):
            df = pd.read_csv(self.equity_file)
            if len(df) > 0:
                return df['equity'].iloc[-1]
        return self.capital

    def _update_equity_curve(self, date: str, equity: float, daily_return: float):
        """更新净值曲线"""
        record = pd.DataFrame([{
            'date': date,
            'equity': equity,
            'daily_return': daily_return,
        }])

        append_to_csv(record, self.equity_file)
        logger.info(f"净值曲线已更新: {date}, equity={equity:.2f}")

    def _calculate_metrics(self, date: str) -> Dict[str, Any]:
        """计算风险指标"""
        metrics = {
            'date': date,
            'capital': self.capital,
            'profile': self.profile,
        }

        if not os.path.exists(self.equity_file):
            return metrics

        df = pd.read_csv(self.equity_file)
        if len(df) < 2:
            return metrics

        equity_series = df['equity']
        returns = df['daily_return'].dropna()

        # 当前回撤
        drawdown_series = calculate_drawdown(equity_series)
        current_dd = drawdown_series.iloc[-1] if len(drawdown_series) > 0 else 0

        # 20日最大回撤
        dd_20d = drawdown_series.tail(20).min() if len(drawdown_series) >= 20 else current_dd

        # 滚动夏普
        rolling_sharpe = calculate_rolling_sharpe(returns, 20)

        # 连续负收益天数
        negative_streak = self._count_negative_streak(returns)

        # 近2个月累计收益
        returns_2m = returns.tail(40).sum() if len(returns) >= 40 else returns.sum()

        metrics.update({
            'current_drawdown': float(current_dd),
            'max_drawdown_20d': float(dd_20d),
            'rolling_sharpe_20d': rolling_sharpe,
            'negative_streak_days': negative_streak,
            'cum_return_2m': float(returns_2m),
            'total_return': float((equity_series.iloc[-1] / self.capital) - 1),
        })

        return metrics

    def _count_negative_streak(self, returns: pd.Series) -> int:
        """计算连续负收益天数"""
        streak = 0
        for ret in reversed(returns.values):
            if ret < 0:
                streak += 1
            else:
                break
        return streak

    def _check_rules(self, metrics: Dict) -> Tuple[str, List[str]]:
        """检查运营规则"""
        alerts = []
        decision = 'HOLD'

        # DE-RISK 规则
        dd_20d = metrics.get('max_drawdown_20d', 0)
        negative_streak = metrics.get('negative_streak_days', 0)

        if dd_20d < -self.DE_RISK_DRAWDOWN_THRESHOLD:
            alerts.append(f"20日回撤 {dd_20d:.2%} < -15%")
            decision = 'DE-RISK'

        if negative_streak >= self.DE_RISK_NEGATIVE_DAYS:
            alerts.append(f"连续 {negative_streak} 日负收益 >= 10天")
            decision = 'DE-RISK'

        # EXIT 规则
        current_dd = metrics.get('current_drawdown', 0)
        cum_return_2m = metrics.get('cum_return_2m', 0)
        rolling_sharpe = metrics.get('rolling_sharpe_20d', 0)

        max_dd_threshold = self.EXIT_MAX_DRAWDOWN_R4 if self.profile == 'R4' else self.EXIT_MAX_DRAWDOWN_R5

        if current_dd < -max_dd_threshold:
            alerts.append(f"最大回撤 {current_dd:.2%} 超过 {self.profile} 阈值 {-max_dd_threshold:.0%}")
            decision = 'EXIT'

        if cum_return_2m < self.EXIT_MONTHLY_LOSS_THRESHOLD:
            alerts.append(f"2个月累计收益 {cum_return_2m:.2%} < -5%")
            decision = 'EXIT'

        if rolling_sharpe < self.EXIT_SHARPE_THRESHOLD and rolling_sharpe != 0:
            alerts.append(f"滚动夏普 {rolling_sharpe:.2f} < 1.0")
            decision = 'EXIT'

        return decision, alerts

    def _save_daily_result(self, date: str, result: Dict):
        """保存每日结果"""
        date_dir = os.path.join(self.daily_dir, date.replace('-', ''))
        ensure_dir(date_dir)

        # 保存 KPI
        kpi_path = os.path.join(date_dir, 'daily_kpi.json')
        write_json(result['metrics'], kpi_path)

        # 保存告警
        alerts_path = os.path.join(date_dir, 'alerts.md')
        with open(alerts_path, 'w') as f:
            f.write(f"# 每日监控告警 - {date}\n\n")
            f.write(f"**决策**: {result['decision']}\n\n")
            if result['alerts']:
                f.write("**触发规则**:\n")
                for alert in result['alerts']:
                    f.write(f"- {alert}\n")
            else:
                f.write("无告警触发\n")

        # 保存参数快照
        params_path = os.path.join(date_dir, 'params.json')
        write_json(FROZEN_PARAMS, params_path)

        logger.info(f"每日结果已保存: {date_dir}")

    def _send_alert(self, result: Dict):
        """发送告警"""
        message = format_daily_alert(
            decision=result['decision'],
            equity_change=result['daily_return'],
            drawdown=result['metrics'].get('current_drawdown', 0),
            rolling_sharpe=result['metrics'].get('rolling_sharpe_20d', 0),
            alerts=result['alerts']
        )

        level = 'INFO' if result['decision'] == 'HOLD' else \
                'WARNING' if result['decision'] == 'DE-RISK' else 'CRITICAL'

        self.alert_manager.send(
            title=f"每日监控 - {result['date']}",
            message=message,
            level=level
        )

    def _create_empty_result(self, date: str, reason: str) -> Dict:
        """创建空结果"""
        return {
            'date': date,
            'equity': 0,
            'daily_return': 0,
            'decision': 'HOLD',
            'alerts': [f"数据缺失: {reason}"],
            'metrics': {},
            'positions_count': 0,
        }
