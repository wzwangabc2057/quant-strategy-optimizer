"""
生产环境运行入口
================================================================================
完整的产品化策略执行入口，支持:
- 配置文件加载
- 数据获取
- 策略执行
- 风控
- 订单生成
- 结果输出
================================================================================
"""
import pandas as pd
import numpy as np
import sys
import os
import json
import logging
from datetime import datetime
from typing import Dict, List
import argparse

# 设置路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    PORTFOLIO_FILE, BACKTEST_START, BACKTEST_END,
    BUY_COMMISSION, SELL_COMMISSION, STABLE_WEIGHTS, AGGRESSIVE_WEIGHTS
)
from data.fetcher import DataFetcher
from strategy import (
    PortfolioStrategy, StrategyConfig,
    RiskController, RiskConfig,
    OrderGenerator, OrderConfig
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionRunner:
    """生产环境运行器"""

    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.fetcher = DataFetcher()
        self.strategies = {}

    def _load_config(self, config_path: str) -> dict:
        """加载配置"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}

    def initialize(self):
        """初始化"""
        logger.info("="*60)
        logger.info("多因子量化策略 - 生产环境")
        logger.info(f"初始化时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)

        # 初始化策略
        for strategy_name, strategy_type in [('R4', 'stable'), ('R5', 'aggressive')]:
            factor_weights = STABLE_WEIGHTS if strategy_type == 'stable' else AGGRESSIVE_WEIGHTS

            config = StrategyConfig(
                name=strategy_name,
                factor_weights=factor_weights,
                risk_config=RiskConfig(
                    buy_commission=BUY_COMMISSION,
                    sell_commission=SELL_COMMISSION,
                ),
                order_config=OrderConfig(
                    buy_commission=BUY_COMMISSION,
                    sell_commission=SELL_COMMISSION,
                ),
            )

            self.strategies[strategy_name] = {
                'strategy': PortfolioStrategy(config),
                'type': strategy_type,
            }

        logger.info(f"已初始化 {len(self.strategies)} 个策略")

    def load_portfolio(self) -> dict:
        """加载持仓"""
        portfolio = self.fetcher.load_portfolio(PORTFOLIO_FILE)
        logger.info(f"R4: {len(portfolio['r4'])} 只股票")
        logger.info(f"R5: {len(portfolio['r5'])} 只股票")
        return portfolio

    def run_backtest(self, portfolio: dict, start_date: str = None, end_date: str = None):
        """运行回测"""
        start_date = start_date or BACKTEST_START
        end_date = end_date or BACKTEST_END

        # 获取价格数据
        all_codes = list(set(portfolio['r4']['code'].tolist() + portfolio['r5']['code'].tolist()))
        logger.info(f"获取价格数据 ({len(all_codes)} 只股票)...")
        price_df = self.fetcher.get_prices(all_codes, '2019-01-01', end_date)
        price_pivot = price_df.pivot(index='date', columns='code', values='close')

        dates = sorted([d for d in price_pivot.index if start_date <= d <= end_date])
        logger.info(f"回测区间: {dates[0]} ~ {dates[-1]} ({len(dates)} 个交易日)")

        # 市场平均价格
        market_avg = price_pivot.mean(axis=1)

        # 月度调仓日期
        rebalance_dates = []
        current_month = None
        for d in dates:
            ym = d[:7]
            if ym != current_month:
                current_month = ym
                rebalance_dates.append(d)

        results = {}

        for strategy_name, strategy_info in self.strategies.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"运行策略: {strategy_name}")
            logger.info(f"{'='*50}")

            strategy = strategy_info['strategy']
            pf = portfolio['r4'] if strategy_name == 'R4' else portfolio['r5']

            # 初始权重
            current_weights = dict(zip(pf['code'], pf['weight']))
            current_weights = {k: v/sum(current_weights.values()) for k, v in current_weights.items()}
            original_weights = current_weights.copy()

            daily_returns = []
            original_returns = []

            for i in range(1, len(dates)):
                date = dates[i]
                prev_date = dates[i-1]

                # 调仓
                if date in rebalance_dates and i > 1:
                    # 构建因子数据
                    factor_data = self._build_factor_data(pf['code'].tolist(), price_pivot, date)

                    # 执行调仓
                    current_prices = {code: price_pivot.loc[date, code]
                                     for code in pf['code'] if code in price_pivot.columns and pd.notna(price_pivot.loc[date, code])}

                    market_up_to_now = market_avg.loc[:date]

                    # 计算目标权重
                    scores = strategy.calculate_factor_scores(factor_data)
                    scores = strategy.apply_reversal_boost(scores, factor_data)

                    regime, tilt_mult, weight_mult = strategy.detect_market_regime(market_up_to_now)
                    adjusted = strategy.adjust_weights(pf, scores, tilt_mult, weight_mult)

                    new_weights = dict(zip(adjusted['code'], adjusted['adjusted_weight']))

                    # 计算换手成本
                    turnover = sum(abs(new_weights.get(c, 0) - current_weights.get(c, 0)) for c in pf['code']) / 2
                    trade_cost = turnover * (BUY_COMMISSION + SELL_COMMISSION)

                    current_weights = new_weights

                    logger.debug(f"{date}: {regime}, 换手{turnover*100:.1f}%, 成本{trade_cost*100:.3f}%")

                # 计算收益
                daily_ret = 0
                orig_ret = 0

                for code in pf['code']:
                    if code not in price_pivot.columns:
                        continue

                    curr = price_pivot.loc[date, code]
                    prev = price_pivot.loc[prev_date, code]

                    if pd.isna(curr) or pd.isna(prev) or prev <= 0:
                        continue

                    stock_ret = curr / prev - 1
                    daily_ret += stock_ret * current_weights.get(code, 0)
                    orig_ret += stock_ret * original_weights.get(code, 0)

                if date in rebalance_dates and i > 1:
                    daily_ret -= trade_cost

                daily_returns.append({'date': date, 'return': daily_ret})
                original_returns.append({'date': date, 'return': orig_ret})

                # 更新净值
                strategy.update_nav(daily_ret)

            # 计算指标
            metrics = self._calculate_metrics(daily_returns)
            original_metrics = self._calculate_metrics(original_returns)

            results[strategy_name] = {
                'enhanced': metrics,
                'original': original_metrics,
                'improvement': metrics['annual_return'] - original_metrics['annual_return'],
            }

            logger.info(f"增强: 年化 {metrics['annual_return']*100:.2f}% | "
                       f"夏普 {metrics['sharpe']:.2f} | "
                       f"回撤 {metrics['max_drawdown']*100:.1f}%")
            logger.info(f"原始: 年化 {original_metrics['annual_return']*100:.2f}%")
            logger.info(f"提升: {results[strategy_name]['improvement']*100:+.2f}%")

        return results

    def _build_factor_data(self, codes: list, price_pivot: pd.DataFrame, eval_date: str) -> pd.DataFrame:
        """构建因子数据"""
        factor_df = pd.DataFrame({'code': codes})

        # 获取财务数据
        fin_df = self.fetcher.get_financial_data(codes)
        factor_df = factor_df.merge(fin_df, on='code', how='left')

        # 价格因子
        available_dates = [d for d in price_pivot.index if d <= eval_date]
        if len(available_dates) >= 120:
            recent = price_pivot.loc[available_dates].tail(120)

            # 动量
            if len(recent) >= 60:
                ret_60 = recent.iloc[-1] / recent.iloc[-60] - 1
                factor_df['momentum'] = factor_df['code'].map(ret_60.to_dict()).fillna(0)

            # 回撤
            cummax = recent.cummax()
            drawdown = ((recent - cummax) / cummax).min() * 100
            factor_df['drawdown_3m'] = factor_df['code'].map(drawdown.to_dict()).fillna(-10)

            # 波动率
            vol = recent.pct_change().tail(60).std() * np.sqrt(252) * 100
            factor_df['volatility'] = factor_df['code'].map(vol.to_dict()).fillna(30)

        # 填充默认值
        defaults = {
            'roe': 10, 'eps': 1, 'net_profit_yoy': 0, 'revenue_yoy': 0,
            'momentum': 0, 'drawdown_3m': -10, 'volatility': 30,
        }
        factor_df = factor_df.fillna(defaults)

        return factor_df

    def _calculate_metrics(self, daily_returns: list) -> dict:
        """计算绩效指标"""
        rets = np.array([r['return'] for r in daily_returns])
        cum = np.cumprod(1 + rets)
        total_return = cum[-1] - 1
        n_years = len(rets) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_drawdown = dd.min()

        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0

        return {
            'annual_return': annual_return,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
        }

    def print_summary(self, results: dict):
        """打印汇总"""
        print("\n" + "="*70)
        print("【策略汇总】")
        print("="*70)
        print(f"\n{'策略':<8} {'年化收益':>10} {'最大回撤':>10} {'夏普比率':>10} {'提升':>8}")
        print("-"*70)

        for name, data in results.items():
            enh = data['enhanced']
            imp = data['improvement']
            print(f"{name:<8} {enh['annual_return']*100:>9.2f}% "
                  f"{enh['max_drawdown']*100:>9.1f}% "
                  f"{enh['sharpe']:>10.2f} {imp*100:>+7.2f}%")


def main():
    parser = argparse.ArgumentParser(description='多因子量化策略')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--start', type=str, default='2020-01-01', help='开始日期')
    parser.add_argument('--end', type=str, default='2025-12-31', help='结束日期')
    args = parser.parse_args()

    runner = ProductionRunner(args.config)
    runner.initialize()

    portfolio = runner.load_portfolio()
    results = runner.run_backtest(portfolio, args.start, args.end)
    runner.print_summary(results)


if __name__ == '__main__':
    main()
