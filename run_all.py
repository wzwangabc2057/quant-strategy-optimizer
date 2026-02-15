"""
运行所有策略版本并对比结果
================================================================================
"""
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PORTFOLIO_FILE, BACKTEST_START, BACKTEST_END
from data.fetcher import DataFetcher


def run_version(strategy_class, name, price_pivot, portfolio):
    """运行单个策略版本"""
    strategy = strategy_class()

    r4_result = strategy.run_backtest(portfolio['r4'], price_pivot, 'stable')
    r5_result = strategy.run_backtest(portfolio['r5'], price_pivot, 'aggressive')

    return {
        'name': name,
        'r4_annual': r4_result['enhanced']['annual_return'] * 100,
        'r4_drawdown': r4_result['enhanced']['max_drawdown'] * 100,
        'r4_sharpe': r4_result['enhanced']['sharpe'],
        'r5_annual': r5_result['enhanced']['annual_return'] * 100,
        'r5_drawdown': r5_result['enhanced']['max_drawdown'] * 100,
        'r5_sharpe': r5_result['enhanced']['sharpe'],
    }


def main():
    print("=" * 80)
    print(" 多因子量化策略 - 版本对比")
    print(f" 运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 加载数据
    fetcher = DataFetcher()
    portfolio = fetcher.load_portfolio(PORTFOLIO_FILE)

    all_codes = list(set(portfolio['r4']['code'].tolist() + portfolio['r5']['code'].tolist()))
    print(f"\n加载 {len(portfolio['r4'])} 只R4股票, {len(portfolio['r5'])} 只R5股票")

    print("获取价格数据...")
    price_df = fetcher.get_prices(all_codes, '2019-01-01', '2025-12-31')
    price_pivot = price_df.pivot(index='date', columns='code', values='close')
    print(f"价格数据: {len(price_pivot)} 个交易日\n")

    # 导入策略
    from strategies.v1_benchmark import StrategyV1
    from strategies.v2_basic import StrategyV2
    from strategies.v3_aggressive import StrategyV3
    from strategies.v4_smart import StrategyV4

    results = []

    # 运行各版本
    versions = [
        (StrategyV1, "v1基准"),
        (StrategyV2, "v2基础"),
        (StrategyV3, "v3激进"),
        (StrategyV4, "v4智能"),
    ]

    for strategy_class, name in versions:
        print(f"运行 {name}...")
        try:
            result = run_version(strategy_class, name, price_pivot, portfolio)
            results.append(result)
            print(f"  R4: {result['r4_annual']:.2f}% | R5: {result['r5_annual']:.2f}%")
        except Exception as e:
            print(f"  错误: {e}")

    # 输出对比表
    print("\n" + "=" * 80)
    print(" 版本对比结果")
    print("=" * 80)
    print()
    print("| 版本   | R4年化 | R4夏普 | R5年化 | R5夏普 | 核心优化         |")
    print("|--------|--------|--------|--------|--------|------------------|")

    optimizations = {
        "v1基准": "原始等权",
        "v2基础": "因子加权",
        "v3激进": "非线性倾斜",
        "v4智能": "ATR止损+波动率调整",
    }

    for r in results:
        opt = optimizations.get(r['name'], "")
        print(f"| {r['name']:<6} | {r['r4_annual']:>5.2f}% | {r['r4_sharpe']:>6.2f} | {r['r5_annual']:>5.2f}% | {r['r5_sharpe']:>6.2f} | {opt:<16} |")

    print()

    # 保存结果
    df = pd.DataFrame(results)
    output_file = os.path.join(os.path.dirname(__file__), 'results', 'comparison.csv')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"结果已保存到: {output_file}")

    # 找出最佳版本
    best = max(results, key=lambda x: (x['r4_annual'] + x['r5_annual']) / 2)
    print(f"\n最佳版本: {best['name']} (R4: {best['r4_annual']:.2f}%, R5: {best['r5_annual']:.2f}%)")


if __name__ == '__main__':
    main()
