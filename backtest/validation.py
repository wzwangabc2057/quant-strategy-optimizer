"""
验证框架模块
================================================================================
包含:
- Walk-forward验证
- 三段式验证 (train/valid/OOS)
- 扰动测试 (因子权重、股票池)
- 稳定性分析
================================================================================
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """验证配置"""
    # Walk-forward参数
    train_window: int = 252 * 2     # 训练窗口 2年
    test_window: int = 63           # 测试窗口 3个月
    step_size: int = 21             # 步长 1个月

    # 三段式参数
    train_ratio: float = 0.6        # 训练集比例
    valid_ratio: float = 0.2        # 验证集比例
    oos_ratio: float = 0.2          # 样本外比例

    # 扰动测试参数
    weight_perturbation: float = 0.2   # 因子权重扰动幅度
    stock_drop_ratio: float = 0.1       # 股票池剔除比例
    n_perturbation_runs: int = 10       # 扰动测试次数


class WalkForwardValidator:
    """Walk-forward验证器"""

    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.results = []

    def split_dates(self, dates: List[str]) -> List[Tuple[List[str], List[str]]]:
        """
        生成walk-forward日期分割

        Args:
            dates: 所有日期列表

        Returns:
            [(train_dates, test_dates), ...]
        """
        n = len(dates)
        splits = []

        train_size = self.config.train_window
        test_size = self.config.test_window
        step = self.config.step_size

        start = 0
        while start + train_size + test_size <= n:
            train_end = start + train_size
            test_end = train_end + test_size

            train_dates = dates[start:train_end]
            test_dates = dates[train_end:test_end]

            splits.append((train_dates, test_dates))

            start += step

        return splits

    def validate(self,
                run_backtest_func: Callable,
                dates: List[str],
                **kwargs) -> Dict:
        """
        执行walk-forward验证

        Args:
            run_backtest_func: 回测函数
            dates: 日期列表
            **kwargs: 传递给回测函数的参数

        Returns:
            验证结果
        """
        splits = self.split_dates(dates)

        fold_results = []
        for i, (train_dates, test_dates) in enumerate(splits):
            logger.info(f"Fold {i+1}/{len(splits)}: "
                       f"train {train_dates[0]}~{train_dates[-1]}, "
                       f"test {test_dates[0]}~{test_dates[-1]}")

            # 在训练集上优化参数（简化：使用默认参数）
            # 在测试集上评估
            result = run_backtest_func(
                start_date=test_dates[0],
                end_date=test_dates[-1],
                **kwargs
            )

            fold_results.append({
                'fold': i + 1,
                'train_start': train_dates[0],
                'train_end': train_dates[-1],
                'test_start': test_dates[0],
                'test_end': test_dates[-1],
                'annual_return': result.get('annual_return', 0),
                'max_drawdown': result.get('max_drawdown', 0),
                'sharpe': result.get('sharpe', 0),
            })

        self.results = fold_results
        return self.summarize()

    def summarize(self) -> Dict:
        """汇总结果"""
        if not self.results:
            return {}

        df = pd.DataFrame(self.results)

        summary = {
            'n_folds': len(df),
            'avg_return': df['annual_return'].mean(),
            'std_return': df['annual_return'].std(),
            'avg_sharpe': df['sharpe'].mean(),
            'std_sharpe': df['sharpe'].std(),
            'avg_drawdown': df['max_drawdown'].mean(),
            'worst_drawdown': df['max_drawdown'].max(),
            'return_range': (df['annual_return'].min(), df['annual_return'].max()),
            'positive_ratio': (df['annual_return'] > 0).mean(),
            'consistency': 1 - df['annual_return'].std() / df['annual_return'].mean()
                          if df['annual_return'].mean() != 0 else 0,
        }

        return summary

    def print_results(self):
        """打印结果"""
        if not self.results:
            print("无结果")
            return

        df = pd.DataFrame(self.results)
        summary = self.summarize()

        print("\n" + "="*70)
        print("Walk-Forward 验证结果")
        print("="*70)
        print(f"\n{'Fold':<6} {'测试区间':<25} {'年化收益':>10} {'最大回撤':>10} {'夏普':>8}")
        print("-"*70)

        for _, row in df.iterrows():
            period = f"{row['test_start']} ~ {row['test_end']}"
            print(f"{row['fold']:<6} {period:<25} {row['annual_return']*100:>9.2f}% "
                  f"{row['max_drawdown']*100:>9.1f}% {row['sharpe']:>8.2f}")

        print("-"*70)
        print(f"{'平均':<6} {'':<25} {summary['avg_return']*100:>9.2f}% "
              f"{summary['avg_drawdown']*100:>9.1f}% {summary['avg_sharpe']:>8.2f}")
        print(f"{'标准差':<6} {'':<25} {summary['std_return']*100:>9.2f}% "
              f"{'':<10} {summary['std_sharpe']:>8.2f}")
        print(f"\n一致性得分: {summary['consistency']*100:.1f}%")
        print(f"正收益比例: {summary['positive_ratio']*100:.1f}%")
        print("="*70)


class ThreeSplitValidator:
    """三段式验证器 (train/valid/OOS)"""

    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.results = {}

    def split_dates(self, dates: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        三段式分割

        Returns:
            (train_dates, valid_dates, oos_dates)
        """
        n = len(dates)

        train_end = int(n * self.config.train_ratio)
        valid_end = train_end + int(n * self.config.valid_ratio)

        train_dates = dates[:train_end]
        valid_dates = dates[train_end:valid_end]
        oos_dates = dates[valid_end:]

        return train_dates, valid_dates, oos_dates

    def validate(self,
                run_backtest_func: Callable,
                dates: List[str],
                **kwargs) -> Dict:
        """执行三段式验证"""
        train_dates, valid_dates, oos_dates = self.split_dates(dates)

        results = {}

        # 训练集
        logger.info(f"训练集: {train_dates[0]} ~ {train_dates[-1]}")
        results['train'] = run_backtest_func(
            start_date=train_dates[0],
            end_date=train_dates[-1],
            **kwargs
        )

        # 验证集
        logger.info(f"验证集: {valid_dates[0]} ~ {valid_dates[-1]}")
        results['valid'] = run_backtest_func(
            start_date=valid_dates[0],
            end_date=valid_dates[-1],
            **kwargs
        )

        # 样本外
        logger.info(f"样本外: {oos_dates[0]} ~ {oos_dates[-1]}")
        results['oos'] = run_backtest_func(
            start_date=oos_dates[0],
            end_date=oos_dates[-1],
            **kwargs
        )

        self.results = results

        # 计算过拟合指标
        if results['train'] and results['oos']:
            train_ret = results['train'].get('annual_return', 0)
            oos_ret = results['oos'].get('annual_return', 0)
            results['overfit_ratio'] = (train_ret - oos_ret) / train_ret if train_ret != 0 else 0

        return results

    def print_results(self):
        """打印结果"""
        if not self.results:
            return

        print("\n" + "="*60)
        print("三段式验证结果")
        print("="*60)
        print(f"\n{'数据集':<10} {'年化收益':>10} {'最大回撤':>10} {'夏普':>8}")
        print("-"*60)

        for name, result in self.results.items():
            if name == 'overfit_ratio':
                continue
            ret = result.get('annual_return', 0) * 100
            dd = result.get('max_drawdown', 0) * 100
            sharpe = result.get('sharpe', 0)
            print(f"{name:<10} {ret:>9.2f}% {dd:>9.1f}% {sharpe:>8.2f}")

        if 'overfit_ratio' in self.results:
            print(f"\n过拟合比率: {self.results['overfit_ratio']*100:.1f}%")
            if self.results['overfit_ratio'] > 0.3:
                print("⚠️ 警告: 过拟合风险较高")
        print("="*60)


class PerturbationTester:
    """扰动测试器"""

    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.results = []

    def perturb_weights(self,
                       weights: Dict[str, float],
                       perturbation: float = 0.2) -> Dict[str, float]:
        """
        扰动因子权重

        Args:
            weights: 原始权重
            perturbation: 扰动幅度

        Returns:
            扰动后的权重
        """
        perturbed = {}
        for factor, weight in weights.items():
            # 在 [1-perturbation, 1+perturbation] 范围内随机扰动
            noise = 1 + np.random.uniform(-perturbation, perturbation)
            perturbed[factor] = weight * noise

        # 归一化
        total = sum(perturbed.values())
        if total > 0:
            for k in perturbed:
                perturbed[k] /= total

        return perturbed

    def perturb_stocks(self,
                      stocks: List[str],
                      drop_ratio: float = 0.1) -> List[str]:
        """
        扰动股票池

        Args:
            stocks: 原始股票列表
            drop_ratio: 剔除比例

        Returns:
            扰动后的股票列表
        """
        n_drop = int(len(stocks) * drop_ratio)
        stocks_copy = stocks.copy()
        np.random.shuffle(stocks_copy)
        return stocks_copy[n_drop:]

    def run_weight_perturbation(self,
                               run_backtest_func: Callable,
                               base_weights: Dict[str, float],
                               n_runs: int = None,
                               **kwargs) -> Dict:
        """
        因子权重扰动测试

        Args:
            run_backtest_func: 回测函数
            base_weights: 基础因子权重
            n_runs: 运行次数

        Returns:
            扰动测试结果
        """
        n_runs = n_runs or self.config.n_perturbation_runs
        results = []

        for i in range(n_runs):
            perturbed_weights = self.perturb_weights(
                base_weights,
                self.config.weight_perturbation
            )

            result = run_backtest_func(
                factor_weights=perturbed_weights,
                **kwargs
            )

            results.append({
                'run': i + 1,
                'annual_return': result.get('annual_return', 0),
                'max_drawdown': result.get('max_drawdown', 0),
                'sharpe': result.get('sharpe', 0),
            })

        return {
            'results': results,
            'avg_return': np.mean([r['annual_return'] for r in results]),
            'std_return': np.std([r['annual_return'] for r in results]),
            'return_range': (
                min(r['annual_return'] for r in results),
                max(r['annual_return'] for r in results)
            ),
        }

    def run_stock_perturbation(self,
                              run_backtest_func: Callable,
                              base_stocks: List[str],
                              n_runs: int = None,
                              **kwargs) -> Dict:
        """
        股票池扰动测试

        Args:
            run_backtest_func: 回测函数
            base_stocks: 基础股票列表
            n_runs: 运行次数

        Returns:
            扰动测试结果
        """
        n_runs = n_runs or self.config.n_perturbation_runs
        results = []

        for i in range(n_runs):
            perturbed_stocks = self.perturb_stocks(
                base_stocks,
                self.config.stock_drop_ratio
            )

            result = run_backtest_func(
                stocks=perturbed_stocks,
                **kwargs
            )

            results.append({
                'run': i + 1,
                'annual_return': result.get('annual_return', 0),
                'max_drawdown': result.get('max_drawdown', 0),
                'sharpe': result.get('sharpe', 0),
            })

        return {
            'results': results,
            'avg_return': np.mean([r['annual_return'] for r in results]),
            'std_return': np.std([r['annual_return'] for r in results]),
            'return_range': (
                min(r['annual_return'] for r in results),
                max(r['annual_return'] for r in results)
            ),
        }

    def run_all_perturbations(self,
                             run_backtest_func: Callable,
                             base_weights: Dict[str, float],
                             base_stocks: List[str],
                             **kwargs) -> Dict:
        """运行所有扰动测试"""
        logger.info("运行因子权重扰动测试...")
        weight_results = self.run_weight_perturbation(
            run_backtest_func, base_weights, **kwargs
        )

        logger.info("运行股票池扰动测试...")
        stock_results = self.run_stock_perturbation(
            run_backtest_func, base_stocks, **kwargs
        )

        return {
            'weight_perturbation': weight_results,
            'stock_perturbation': stock_results,
        }


class RobustnessAnalyzer:
    """鲁棒性分析器"""

    def __init__(self):
        self.results = {}

    def analyze_frequency_sensitivity(self,
                                     run_backtest_func: Callable,
                                     frequencies: List[str] = None,
                                     **kwargs) -> Dict:
        """
        调仓频率敏感性分析

        Args:
            run_backtest_func: 回测函数
            frequencies: 频率列表 ['daily', 'weekly', 'biweekly', 'monthly']
        """
        frequencies = frequencies or ['weekly', 'biweekly', 'monthly']

        results = {}
        for freq in frequencies:
            result = run_backtest_func(rebalance_freq=freq, **kwargs)
            results[freq] = {
                'annual_return': result.get('annual_return', 0),
                'sharpe': result.get('sharpe', 0),
                'max_drawdown': result.get('max_drawdown', 0),
            }

        self.results['frequency'] = results
        return results

    def check_robustness_gate(self,
                             base_return: float,
                             perturbed_results: Dict,
                             threshold: float = 0.15) -> Tuple[bool, str]:
        """
        检查鲁棒性门槛

        Args:
            base_return: 基准收益
            perturbed_results: 扰动测试结果
            threshold: 允许的波动阈值

        Returns:
            (是否通过, 信息)
        """
        avg_return = perturbed_results['avg_return']

        if base_return == 0:
            return False, "基准收益为0"

        deviation = abs(avg_return - base_return) / base_return

        if deviation > threshold:
            return False, f"扰动后收益偏离{deviation*100:.1f}%，超过阈值{threshold*100:.0f}%"

        return True, f"扰动后收益偏离{deviation*100:.1f}%，在可接受范围内"

    def print_summary(self):
        """打印鲁棒性分析摘要"""
        print("\n" + "="*60)
        print("鲁棒性分析摘要")
        print("="*60)

        if 'frequency' in self.results:
            print("\n调仓频率敏感性:")
            print(f"  {'频率':<10} {'年化收益':>10} {'夏普':>8}")
            for freq, result in self.results['frequency'].items():
                print(f"  {freq:<10} {result['annual_return']*100:>9.2f}% {result['sharpe']:>8.2f}")

        print("="*60)
