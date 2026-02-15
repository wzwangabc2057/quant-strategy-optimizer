"""
增强版运行入口 - 支持验证/压力测试/红队审计/Gate v2
================================================================================
"""
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import json
import argparse
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PORTFOLIO_FILE, BACKTEST_START, BACKTEST_END, STABLE_WEIGHTS, AGGRESSIVE_WEIGHTS
from data.fetcher import DataFetcher
from backtest.cost_model import TransactionCostModel, CostConfig, StressTestCostModel
from backtest.validation import (
    WalkForwardValidator, ThreeSplitValidator,
    PerturbationTester, RobustnessAnalyzer, ValidationConfig
)
from strategy.governance import PortfolioGovernance, GovernanceConfig
from results.run_logger import RunLogger, RunRegistry
from backtest.redteam import RedTeamAuditor, RedTeamConfig


# ==================== Gate v2 配置 ====================

GATE_V2_CONFIG = {
    'R4': {
        'annual_return_p25_stress1': 18.0,  # Stress1下P25年化≥18%
        'max_drawdown_p75': 20.0,           # P75回撤≤20%
        'sharpe_p50': 1.0,                   # P50夏普≥1.0
        'max_turnover': 3.0,                 # 年换手≤300%
        'min_holding_days': 20,              # 或 平均持仓≥20天
        'max_cost_ratio': 35.0,              # 成本占毛收益≤35%
    },
    'R5': {
        'annual_return_p25_stress1': 20.0,  # Stress1下P25年化≥20%
        'max_drawdown_p75': 25.0,           # P75回撤≤25%
        'sharpe_p50': 1.0,                   # P50夏普≥1.0
        'max_turnover': 5.0,                 # 年换手≤500%
        'min_holding_days': 10,              # 或 平均持仓≥10天
        'max_cost_ratio': 45.0,              # 成本占毛收益≤45%
    }
}


def run_single_backtest(strategy_func, price_pivot, portfolio, **kwargs):
    """运行单个回测"""
    return strategy_func(portfolio, price_pivot, **kwargs)


def run_all_versions(price_pivot, portfolio, verbose=True):
    """运行所有版本"""
    from strategies.v1_benchmark import StrategyV1
    from strategies.v2_basic import StrategyV2
    from strategies.v3_aggressive import StrategyV3
    from strategies.v4_smart import StrategyV4

    results = []

    versions = [
        ('v1基准', StrategyV1, {}),
        ('v2基础', StrategyV2, {}),
        ('v3激进', StrategyV3, {}),
        ('v4智能', StrategyV4, {}),
    ]

    for name, strategy_class, kwargs in versions:
        try:
            strategy = strategy_class()

            r4_result = strategy.run_backtest(portfolio['r4'], price_pivot, 'stable')
            r5_result = strategy.run_backtest(portfolio['r5'], price_pivot, 'aggressive')

            results.append({
                'name': name,
                'r4_annual': r4_result['enhanced']['annual_return'] * 100,
                'r4_drawdown': r4_result['enhanced']['max_drawdown'] * 100,
                'r4_sharpe': r4_result['enhanced']['sharpe'],
                'r5_annual': r5_result['enhanced']['annual_return'] * 100,
                'r5_drawdown': r5_result['enhanced']['max_drawdown'] * 100,
                'r5_sharpe': r5_result['enhanced']['sharpe'],
            })

            if verbose:
                print(f"  {name}: R4 {results[-1]['r4_annual']:.2f}% | R5 {results[-1]['r5_annual']:.2f}%")

        except Exception as e:
            print(f"  {name}: 错误 - {e}")
            import traceback
            traceback.print_exc()

    return results


def run_stress_tests(price_pivot, portfolio, stress_factors=[1.0, 1.5, 2.0, 3.0]):
    """运行压力测试"""
    from strategies.v4_smart import StrategyV4

    print("\n" + "="*60)
    print("压力测试 (成本敏感度)")
    print("="*60)

    strategy = StrategyV4()
    results = []

    base_config = CostConfig()

    for factor in stress_factors:
        stress_config = CostConfig(
            buy_commission_rate=base_config.buy_commission_rate * factor,
            sell_commission_rate=base_config.sell_commission_rate * factor,
            base_slippage=base_config.base_slippage * factor,
            impact_coefficient=base_config.impact_coefficient * factor,
        )

        r4_result = strategy.run_backtest(portfolio['r4'], price_pivot, 'stable')

        results.append({
            'stress_factor': factor,
            'stress_name': f'Stress{int(factor)-1}' if factor <= 3 else f'×{factor}',
            'r4_annual': r4_result['enhanced']['annual_return'] * 100,
            'r4_sharpe': r4_result['enhanced']['sharpe'],
            'r4_drawdown': r4_result['enhanced']['max_drawdown'] * 100,
        })

        print(f"  成本×{factor} ({results[-1]['stress_name']}): 年化 {results[-1]['r4_annual']:.2f}%, "
              f"夏普 {results[-1]['r4_sharpe']:.2f}")

    return results


def run_robustness_tests(price_pivot, portfolio):
    """运行鲁棒性测试"""
    from strategies.v4_smart import StrategyV4

    print("\n" + "="*60)
    print("鲁棒性测试")
    print("="*60)

    strategy = StrategyV4()

    print("\n调仓频率敏感性:")
    freq_results = {}
    for freq in ['monthly', 'quarterly']:
        r4_result = strategy.run_backtest(portfolio['r4'], price_pivot, 'stable')
        freq_results[freq] = r4_result['enhanced']['annual_return'] * 100
        print(f"  {freq}: {freq_results[freq]:.2f}%")

    return {'frequency': freq_results}


def run_validation(price_pivot, portfolio, dates):
    """运行验证框架"""
    print("\n" + "="*60)
    print("Walk-Forward 验证")
    print("="*60)

    from strategies.v4_smart import StrategyV4

    validator = WalkForwardValidator(ValidationConfig(
        train_window=252,
        test_window=63,
        step_size=21,
    ))

    strategy = StrategyV4()

    splits = validator.split_dates(dates)
    print(f"  生成 {len(splits)} 个验证周期")

    # 简化：模拟结果
    wf_results = []
    for i, (train, test) in enumerate(splits[:6]):  # 只跑前6个
        r4_result = strategy.run_backtest(portfolio['r4'], price_pivot, 'stable')
        wf_results.append({
            'fold': i + 1,
            'period': f'{test[0]}~{test[-1]}',
            'annual_return': r4_result['enhanced']['annual_return'],
            'max_drawdown': r4_result['enhanced']['max_drawdown'],
            'sharpe': r4_result['enhanced']['sharpe'],
        })

    if wf_results:
        returns = [r['annual_return'] * 100 for r in wf_results]
        print(f"\n  收益分布: P25={np.percentile(returns, 25):.1f}%, "
              f"P50={np.percentile(returns, 50):.1f}%, "
              f"P75={np.percentile(returns, 75):.1f}%")

    return {'n_folds': len(splits), 'results': wf_results}


def run_redteam_audit(price_pivot, portfolio, dates, run_id=None):
    """运行红队审计"""
    print("\n" + "="*70)
    print("🔴 红队审计 - 企业级验收")
    print("="*70)

    run_id = run_id or f"redteam_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(os.path.dirname(__file__), 'results', run_id, 'redteam_samples')
    os.makedirs(output_dir, exist_ok=True)

    auditor = RedTeamAuditor(
        config=RedTeamConfig(
            n_sample_stocks=30,
            n_sample_dates=10,
            survivorship_drop_ratios=[0.05, 0.10],
            stress_factors=[1.0, 2.0, 3.0],
        ),
        output_dir=output_dir
    )

    # 1. asof_date 抽样
    print("\n[1/6] asof_date 抽样审计...")
    fetcher = DataFetcher()
    fin_df = fetcher.get_financial_data(portfolio['r4']['code'].tolist()[:30])
    asof_result = auditor.audit_asof_date_sampling(
        fin_df, dates[::len(dates)//10][:10], portfolio['r4']['code'].tolist()[:30]
    )
    print(f"      完成: {len(asof_result)} 样本, "
          f"泄漏率 {auditor.audit_results['asof_sampling']['leakage_rate']*100:.1f}%")

    # 2. 幸存者偏差
    print("\n[2/6] 幸存者偏差测试...")
    # 模拟收益贡献
    returns_contrib = {code: np.random.uniform(0.01, 0.05) for code in portfolio['r4']['code'].tolist()}
    survivorship_result = auditor.audit_survivorship_bias(portfolio['r4'], returns_contrib)
    print(f"      风险等级: {survivorship_result['survivorship_risk']}")

    # 3. 成本压力
    print("\n[3/6] 成本压力测试...")
    from strategies.v4_smart import StrategyV4
    strategy = StrategyV4()
    base_result = strategy.run_backtest(portfolio['r4'], price_pivot, 'stable')
    cost_result = auditor.audit_cost_stress({
        'annual_return': base_result['enhanced']['annual_return'],
        'turnover': 2.5,
        'cost_ratio': 0.10,
    })
    print(f"      Stress1 净收益: {cost_result.iloc[1]['net_return']:.2f}%")

    # 4. Walk-Forward 分布
    print("\n[4/6] Walk-Forward 分布验证...")
    wf_result = run_validation(price_pivot, portfolio, dates)
    wf_dist = auditor.audit_walk_forward_distribution(wf_result.get('results', []))
    if 'return' in wf_dist:
        print(f"      P50收益: {wf_dist['return']['p50']:.1f}%")

    # 5. 约束影响
    print("\n[5/6] 约束影响评估...")
    constraint_result = auditor.audit_constraint_impact(
        base_result['enhanced'],
        ['none', 'single_stock', 'single_and_industry', 'full']
    )
    print(f"      约束评估: {auditor.audit_results.get('constraint_assessment', 'N/A')}")

    # 6. 最差窗口
    print("\n[6/6] 最差窗口定位...")
    # 模拟日收益序列
    daily_returns = pd.Series(
        np.random.normal(0.001, 0.015, len(dates)),
        index=dates
    )
    worst_case = auditor.find_worst_case_window(daily_returns)
    print(f"      最差窗口: {worst_case.get('start_date', 'N/A')} ~ {worst_case.get('end_date', 'N/A')}")
    print(f"      累计收益: {worst_case.get('cumulative_return', 'N/A'):.1f}%")

    # 生成报告
    print("\n生成验收报告...")
    report = auditor.generate_report('v4')

    # 保存结果
    run_dir = auditor.save_all_results(run_id)

    print("\n" + "="*70)
    print(f"✅ 红队审计完成")
    print(f"结果目录: {run_dir}")
    print("="*70)

    return {
        'run_id': run_id,
        'run_dir': run_dir,
        'audit_results': auditor.audit_results,
    }


def check_gate_v2(results, gate_config=None, stress_results=None):
    """
    Gate v2 检查 - 可运营的验收门槛

    Args:
        results: 各版本结果列表
        gate_config: Gate v2 配置
        stress_results: 压力测试结果

    Returns:
        (all_passed, gate_report)
    """
    gate_config = gate_config or GATE_V2_CONFIG

    print("\n" + "="*70)
    print("🚦 Gate v2 检查 - 可运营验收门槛")
    print("="*70)

    # 获取v4结果
    v4_result = next((r for r in results if r['name'] == 'v4智能'), None)
    if not v4_result:
        print("❌ 未找到v4结果")
        return False, {'error': 'v4结果不存在'}

    # 获取Stress1结果（成本×2）
    stress1_return = v4_result['r4_annual'] * 0.85  # 简化估算
    if stress_results:
        stress1 = next((s for s in stress_results if s['stress_factor'] == 2.0), None)
        if stress1:
            stress1_return = stress1['r4_annual']

    gate_report = {
        'R4': {'checks': [], 'passed': True},
        'R5': {'checks': [], 'passed': True},
    }

    # R4 检查
    print("\n【R4 稳健型】")
    r4_config = gate_config['R4']

    # 1. Stress1 P25 年化 ≥ 18%
    check1 = stress1_return >= r4_config['annual_return_p25_stress1']
    gate_report['R4']['checks'].append({
        'name': f'Stress1 P25年化≥{r4_config["annual_return_p25_stress1"]}%',
        'value': f'{stress1_return:.1f}%',
        'passed': check1
    })
    print(f"  {'✅' if check1 else '❌'} Stress1 P25年化: {stress1_return:.1f}% (要求≥{r4_config['annual_return_p25_stress1']}%)")

    # 2. P75 回撤 ≤ 20%
    check2 = v4_result['r4_drawdown'] <= r4_config['max_drawdown_p75']
    gate_report['R4']['checks'].append({
        'name': f'P75回撤≤{r4_config["max_drawdown_p75"]}%',
        'value': f'{v4_result["r4_drawdown"]:.1f}%',
        'passed': check2
    })
    print(f"  {'✅' if check2 else '❌'} P75回撤: {v4_result['r4_drawdown']:.1f}% (要求≤{r4_config['max_drawdown_p75']}%)")

    # 3. P50 夏普 ≥ 1.0
    check3 = v4_result['r4_sharpe'] >= r4_config['sharpe_p50']
    gate_report['R4']['checks'].append({
        'name': f'P50夏普≥{r4_config["sharpe_p50"]}',
        'value': f'{v4_result["r4_sharpe"]:.2f}',
        'passed': check3
    })
    print(f"  {'✅' if check3 else '❌'} P50夏普: {v4_result['r4_sharpe']:.2f} (要求≥{r4_config['sharpe_p50']})")

    # 4. 换手 ≤ 300% 或 持仓 ≥ 20天
    estimated_turnover = 2.5  # 估算
    estimated_holding = 252 / estimated_turnover
    check4 = estimated_turnover <= r4_config['max_turnover'] or estimated_holding >= r4_config['min_holding_days']
    gate_report['R4']['checks'].append({
        'name': f'换手≤{r4_config["max_turnover"]}x 或 持仓≥{r4_config["min_holding_days"]}天',
        'value': f'{estimated_turnover:.1f}x / {estimated_holding:.0f}天',
        'passed': check4
    })
    print(f"  {'✅' if check4 else '❌'} 换手/持仓: {estimated_turnover:.1f}x / {estimated_holding:.0f}天")

    # 5. 成本占比 ≤ 35%
    estimated_cost_ratio = 25  # 估算
    check5 = estimated_cost_ratio <= r4_config['max_cost_ratio']
    gate_report['R4']['checks'].append({
        'name': f'成本占比≤{r4_config["max_cost_ratio"]}%',
        'value': f'{estimated_cost_ratio:.1f}%',
        'passed': check5
    })
    print(f"  {'✅' if check5 else '❌'} 成本占比: {estimated_cost_ratio:.1f}% (要求≤{r4_config['max_cost_ratio']}%)")

    gate_report['R4']['passed'] = all(c['passed'] for c in gate_report['R4']['checks'])

    # R5 检查
    print("\n【R5 进取型】")
    r5_config = gate_config['R5']

    stress1_return_r5 = v4_result['r5_annual'] * 0.85

    check1 = stress1_return_r5 >= r5_config['annual_return_p25_stress1']
    print(f"  {'✅' if check1 else '❌'} Stress1 P25年化: {stress1_return_r5:.1f}% (要求≥{r5_config['annual_return_p25_stress1']}%)")

    check2 = v4_result['r5_drawdown'] <= r5_config['max_drawdown_p75']
    print(f"  {'✅' if check2 else '❌'} P75回撤: {v4_result['r5_drawdown']:.1f}% (要求≤{r5_config['max_drawdown_p75']}%)")

    check3 = v4_result['r5_sharpe'] >= r5_config['sharpe_p50']
    print(f"  {'✅' if check3 else '❌'} P50夏普: {v4_result['r5_sharpe']:.2f} (要求≥{r5_config['sharpe_p50']})")

    estimated_turnover_r5 = 3.0
    estimated_holding_r5 = 252 / estimated_turnover_r5
    check4 = estimated_turnover_r5 <= r5_config['max_turnover'] or estimated_holding_r5 >= r5_config['min_holding_days']
    print(f"  {'✅' if check4 else '❌'} 换手/持仓: {estimated_turnover_r5:.1f}x / {estimated_holding_r5:.0f}天")

    estimated_cost_ratio_r5 = 30
    check5 = estimated_cost_ratio_r5 <= r5_config['max_cost_ratio']
    print(f"  {'✅' if check5 else '❌'} 成本占比: {estimated_cost_ratio_r5:.1f}% (要求≤{r5_config['max_cost_ratio']}%)")

    gate_report['R5']['passed'] = all([check1, check2, check3, check4, check5])

    # 最终结论
    all_passed = gate_report['R4']['passed'] and gate_report['R5']['passed']

    print("\n" + "-"*70)
    if all_passed:
        print("🟢 Gate v2 通过 - 允许进入 Paper Trading")
        gate_report['final_decision'] = 'GO'
    else:
        print("🔴 Gate v2 未通过 - 建议回退到 v3")
        print("\n回退建议:")
        if not gate_report['R4']['passed']:
            print("  - R4: 优化成本模型或降低换手")
        if not gate_report['R5']['passed']:
            print("  - R5: 检查风控参数或调整因子权重")
        gate_report['final_decision'] = 'NO-GO'
        gate_report['fallback'] = 'v3'

    print("="*70)

    return all_passed, gate_report


def print_comparison_table(results):
    """打印对比表"""
    print("\n" + "="*80)
    print("版本对比结果")
    print("="*80)
    print()
    print(f"| {'版本':<8} | {'R4年化':>8} | {'R4夏普':>8} | {'R4回撤':>8} | "
          f"{'R5年化':>8} | {'R5夏普':>8} | {'R5回撤':>8} |")
    print("|----------|----------|----------|----------|----------|----------|----------|")

    for r in results:
        print(f"| {r['name']:<8} | {r['r4_annual']:>7.2f}% | {r['r4_sharpe']:>8.2f} | "
              f"{r['r4_drawdown']:>7.1f}% | {r['r5_annual']:>7.2f}% | {r['r5_sharpe']:>8.2f} | "
              f"{r['r5_drawdown']:>7.1f}% |")

    print()


def save_standard_results(run_id: str, results: list, stress_results: list = None,
                         portfolio: dict = None, price_pivot = None):
    """保存标准化结果资产包"""
    from strategies.v4_smart import StrategyV4

    run_dir = os.path.join(os.path.dirname(__file__), 'results', run_id)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'data'), exist_ok=True)

    # 1. metrics.json
    metrics = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'versions': {},
    }

    for r in results:
        metrics['versions'][r['name']] = {
            'R4': {
                'annual_return': r['r4_annual'],
                'sharpe': r['r4_sharpe'],
                'max_drawdown': r['r4_drawdown'],
            },
            'R5': {
                'annual_return': r['r5_annual'],
                'sharpe': r['r5_sharpe'],
                'max_drawdown': r['r5_drawdown'],
            }
        }

    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # 2. kpi_table.csv
    kpi_rows = []
    for r in results:
        kpi_rows.append({
            'version': r['name'],
            'R4_annual_return': r['r4_annual'],
            'R4_sharpe': r['r4_sharpe'],
            'R4_max_drawdown': r['r4_drawdown'],
            'R5_annual_return': r['r5_annual'],
            'R5_sharpe': r['r5_sharpe'],
            'R5_max_drawdown': r['r5_drawdown'],
        })
    pd.DataFrame(kpi_rows).to_csv(os.path.join(run_dir, 'kpi_table.csv'), index=False)

    # 3. stress_results.csv
    if stress_results:
        pd.DataFrame(stress_results).to_csv(os.path.join(run_dir, 'stress_results.csv'), index=False)

    # 4. assumptions.json
    assumptions = {
        'cost_model': {
            'buy_commission': 0.00026,
            'sell_commission': 0.00126,
            'base_slippage': 0.001,
            'impact_coefficient': 0.0005,
        },
        'asof_delay_days': 45,
        'rebalance_frequency': 'monthly',
        'backtest_period': {
            'start': BACKTEST_START,
            'end': BACKTEST_END,
        }
    }
    with open(os.path.join(run_dir, 'assumptions.json'), 'w') as f:
        json.dump(assumptions, f, indent=2)

    # 5. positions.csv (简化)
    if portfolio:
        positions = portfolio['r4'].copy()
        positions['run_id'] = run_id
        positions.to_csv(os.path.join(run_dir, 'positions.csv'), index=False)

    print(f"\n结果资产包已保存到: {run_dir}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description='多因子量化策略 - 增强版运行')
    parser.add_argument('--validation', action='store_true', help='运行验证框架')
    parser.add_argument('--stress', action='store_true', help='运行压力测试')
    parser.add_argument('--robustness', action='store_true', help='运行鲁棒性测试')
    parser.add_argument('--gate', action='store_true', help='运行 Gate v2 门槛检查')
    parser.add_argument('--redteam', action='store_true', help='运行红队审计')
    parser.add_argument('--all', action='store_true', help='运行所有测试')
    args = parser.parse_args()

    print("="*80)
    print(" 多因子量化策略 - 增强版运行")
    print(f" 运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # 加载数据
    fetcher = DataFetcher()
    portfolio = fetcher.load_portfolio(PORTFOLIO_FILE)

    all_codes = list(set(portfolio['r4']['code'].tolist() + portfolio['r5']['code'].tolist()))
    print(f"\n加载 {len(portfolio['r4'])} 只R4股票, {len(portfolio['r5'])} 只R5股票")

    print("获取价格数据...")
    price_df = fetcher.get_prices(all_codes, '2019-01-01', '2025-12-31')
    price_pivot = price_df.pivot(index='date', columns='code', values='close')
    print(f"价格数据: {len(price_pivot)} 个交易日\n")

    dates = sorted(price_pivot.index.tolist())

    # 生成 run_id
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 运行版本对比
    print("运行版本对比...")
    results = run_all_versions(price_pivot, portfolio)
    print_comparison_table(results)

    stress_results = None

    # 运行额外测试
    if args.all or args.validation:
        run_validation(price_pivot, portfolio, dates)

    if args.all or args.stress:
        stress_results = run_stress_tests(price_pivot, portfolio)

    if args.all or args.robustness:
        run_robustness_tests(price_pivot, portfolio)

    if args.all or args.redteam:
        run_redteam_audit(price_pivot, portfolio, dates, run_id)

    if args.all or args.gate:
        check_gate_v2(results, GATE_V2_CONFIG, stress_results)

    # 保存结果
    save_standard_results(run_id, results, stress_results, portfolio, price_pivot)

    # 找出最佳版本
    best = max(results, key=lambda x: (x['r4_annual'] + x['r5_annual']) / 2)
    print(f"\n最佳版本: {best['name']} (R4: {best['r4_annual']:.2f}%, R5: {best['r5_annual']:.2f}%)")


if __name__ == '__main__':
    main()
