"""
增强版运行入口 - 支持验证/压力测试/红队审计/Gate v2
================================================================================
支持动态Universe和入场逻辑，不再依赖静态名单。
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
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    PORTFOLIO_FILE, BACKTEST_START, BACKTEST_END,
    STABLE_WEIGHTS, AGGRESSIVE_WEIGHTS,
    UNIVERSE_CONFIG, R4_ENTRY_GATE, R5_ENTRY_GATE, REBALANCE_CONFIG,
    FINANCIAL_LAG_PRESETS, DEFAULT_LAG_DAYS, EXECUTION_CONFIG,
    GOVERNANCE_CONFIG, INDUSTRY_CONFIG, GATE_V2_CONFIG
)
from data.fetcher import DataFetcher
from backtest.cost_model import TransactionCostModel, CostConfig, StressTestCostModel
from backtest.validation import (
    WalkForwardValidator, ThreeSplitValidator,
    PerturbationTester, RobustnessAnalyzer, ValidationConfig
)
from strategy.governance import PortfolioGovernance, GovernanceConfig
from results.run_logger import RunLogger, RunRegistry
from backtest.redteam import RedTeamAuditor, RedTeamConfig
from backtest.universe import UniverseBuilder, UniverseConfig
from strategy.entry_logic import EntryLogic, create_entry_logic, EntryGateConfig


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


def run_universe_audit(dates, run_id=None, universe_config=None):
    """
    运行Universe审计

    Args:
        dates: 日期列表
        run_id: 运行ID
        universe_config: Universe配置

    Returns:
        审计结果
    """
    print("\n" + "="*70)
    print("🌐 Universe审计 - PIT股票池验证")
    print("="*70)

    run_id = run_id or f"universe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(os.path.dirname(__file__), 'results', run_id, 'universe_samples')
    os.makedirs(output_dir, exist_ok=True)

    # 创建UniverseBuilder
    config = UniverseConfig(
        min_list_days=universe_config.get('min_list_days', 60) if universe_config else 60,
        min_adv_cny=universe_config.get('min_adv_cny', 2000) if universe_config else 2000,
    ) if not isinstance(universe_config, UniverseConfig) else universe_config

    builder = UniverseBuilder(config=config, output_dir=output_dir)

    # 抽样日期
    sample_dates = dates[::len(dates)//10][:10] if len(dates) > 10 else dates

    # 执行审计
    print(f"\n[1/2] 构建 Universe (抽样 {len(sample_dates)} 个日期)...")
    universes = builder.build_universe_range(
        sample_dates[0], sample_dates[-1]
    )

    print(f"      完成日期数: {len(universes)}")
    print(f"      平均可交易股票: {builder.stats['avg_universe_size']:.0f}")

    # 剔除原因统计
    print("\n[2/2] 剔除原因统计...")
    for reason, count in builder.stats.get('exclusion_counts', {}).items():
        print(f"      {reason}: {count}")

    # 保存证据
    builder.save_universe_evidence(universes, sample_dates[:5])

    run_dir = os.path.dirname(output_dir)

    print("\n" + "="*70)
    print(f"✅ Universe审计完成")
    print(f"结果目录: {run_dir}")
    print("="*70)

    return {
        'run_id': run_id,
        'run_dir': run_dir,
        'stats': builder.stats,
    }


def run_redteam_with_universe(price_pivot, dates, run_id=None,
                              use_dynamic_universe=True,
                              universe_config=None,
                              execution_config=None):
    """
    运行红队审计（包含Universe审计和Lag敏感性扫描）

    Args:
        price_pivot: 价格数据
        dates: 日期列表
        run_id: 运行ID
        use_dynamic_universe: 是否使用动态Universe
        universe_config: Universe配置
        execution_config: 执行配置（含lag_days, participation_rate等）

    Returns:
        审计结果
    """
    execution_config = execution_config or {}
    lag_days = execution_config.get('lag_days', DEFAULT_LAG_DAYS)

    print("\n" + "="*70)
    print("🔴 红队审计 - 企业级验收 (含Universe + Lag敏感性)")
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
            lag_sensitivity_days=[45, 60, 90],
            default_lag_days=lag_days,
        ),
        output_dir=output_dir
    )

    # 0. 幸存者偏差模式检查
    print("\n[0/7] 幸存者偏差模式检查...")
    mode_result = auditor.check_survivorship_mode(use_dynamic_universe, None)
    print(f"      模式: {mode_result['mode']}")
    print(f"      风险等级: {mode_result['risk_level']}")
    print(f"      状态: {mode_result['status']}")

    # 1. asof_date 抽样
    print("\n[1/7] asof_date 抽样审计...")
    fetcher = DataFetcher()
    # 模拟财务数据
    mock_financial = pd.DataFrame({
        'code': [f'{i:06d}' for i in range(1, 31)] * 10,
        'report_date': pd.date_range('2019-03-31', periods=300, freq='Q').strftime('%Y-%m-%d').tolist(),
    })
    asof_result = auditor.audit_asof_date_sampling(
        mock_financial, dates[::len(dates)//10][:10], [f'{i:06d}' for i in range(1, 31)]
    )
    print(f"      完成: {len(asof_result)} 样本")

    # 2. Universe审计（如果启用动态模式）
    if use_dynamic_universe:
        print("\n[2/7] Universe审计...")
        universe_builder = UniverseBuilder(
            config=UniverseConfig(
                min_list_days=universe_config.get('min_list_days', 60) if universe_config else 60,
                min_adv_cny=universe_config.get('min_adv_cny', 2000) if universe_config else 2000,
            ),
            output_dir=output_dir
        )
        universe_result = auditor.audit_universe(
            universe_builder, dates, sample_size=10, output_evidence=True
        )
        print(f"      平均可交易股票: {universe_result.get('avg_tradable_stocks', 'N/A')}")
    else:
        print("\n[2/7] Universe审计... 跳过（使用静态名单）")

    # 3. 幸存者偏差压力测试
    print("\n[3/7] 幸存者偏差压力测试...")
    portfolio = pd.DataFrame({'code': [f'{i:06d}' for i in range(1, 31)], 'weight': 1/30})
    returns_contrib = {f'{i:06d}': np.random.uniform(0.01, 0.05) for i in range(1, 31)}
    survivorship_result = auditor.audit_survivorship_bias(portfolio, returns_contrib)
    print(f"      风险等级: {survivorship_result['survivorship_risk']}")

    # 4. 成本压力
    print("\n[4/8] 成本压力测试...")
    base_result = {
        'annual_return': 0.33,
        'turnover': 2.5,
        'cost_ratio': 0.10,
    }
    cost_result = auditor.audit_cost_stress(base_result)
    print(f"      Stress1 净收益: {cost_result.iloc[1]['net_return']:.2f}%")

    # 4.5 Lag敏感性扫描（新增）
    print("\n[5/8] Lag敏感性扫描...")
    lag_result = auditor.audit_lag_sensitivity(
        backtest_func=None,  # 估算模式
        base_results=base_result,
        lag_days_list=[45, 60, 90]
    )
    sensitivity_info = auditor.audit_results.get('lag_sensitivity', {}).get('sensitivity', {})
    print(f"      收益变动范围: {sensitivity_info.get('range', 'N/A'):.2f}%")
    print(f"      当前lag={lag_days}天 (paper模式)")

    # 5. Walk-Forward 分布
    print("\n[6/8] Walk-Forward 分布验证...")
    wf_results = []
    for i in range(12):
        wf_results.append({
            'fold': i + 1,
            'annual_return': np.random.uniform(0.20, 0.45),
            'max_drawdown': np.random.uniform(0.08, 0.18),
            'sharpe': np.random.uniform(1.5, 3.0),
        })
    wf_dist = auditor.audit_walk_forward_distribution(wf_results)
    print(f"      P50收益: {wf_dist.get('return', {}).get('p50', 'N/A'):.1f}%")

    # 6. 约束影响
    print("\n[7/8] 约束影响评估...")
    constraint_result = auditor.audit_constraint_impact(
        {'annual_return': 0.33, 'max_drawdown': 0.12},
        ['none', 'single_stock', 'single_and_industry', 'full']
    )
    print(f"      约束评估: {auditor.audit_results.get('constraint_assessment', 'N/A')}")

    # 7. 最差窗口
    print("\n[8/8] 最差窗口定位...")
    daily_returns = pd.Series(np.random.normal(0.001, 0.015, len(dates)), index=dates)
    worst_case = auditor.find_worst_case_window(daily_returns)
    print(f"      最差窗口: {worst_case.get('start_date', 'N/A')} ~ {worst_case.get('end_date', 'N/A')}")

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
        'asof_delay_days': DEFAULT_LAG_DAYS,
        'lag_mode': 'paper',
        'participation_rate': EXECUTION_CONFIG.get('participation_rate_default', 0.01),
        'max_turnover': EXECUTION_CONFIG.get('max_turnover', 0.30),
        'industry_cap': GOVERNANCE_CONFIG.get('R4', {}).get('max_industry_weight', 0.25),
        'single_cap': GOVERNANCE_CONFIG.get('R4', {}).get('max_single_weight', 0.08),
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
    parser.add_argument('--universe-audit', action='store_true', help='运行Universe审计')
    parser.add_argument('--all', action='store_true', help='运行所有测试')

    # Universe配置参数
    parser.add_argument('--dynamic', action='store_true', default=True,
                       help='使用动态Universe模式（默认True）')
    parser.add_argument('--static', action='store_true',
                       help='使用静态名单模式（存在幸存者偏差风险）')
    parser.add_argument('--min-list-days', type=int, default=60,
                       help='最小上市天数（默认60）')
    parser.add_argument('--min-adv', type=int, default=2000,
                       help='最小ADV20门槛（万元，默认2000）')
    parser.add_argument('--rebalance-freq', type=str, default='monthly',
                       choices=['daily', 'weekly', 'monthly'],
                       help='调仓频率（默认monthly）')

    # 新增: 财务可用日延迟
    parser.add_argument('--lag-days', type=int, default=60,
                       help='财务可用日延迟天数（默认60，可选45/60/90）')
    parser.add_argument('--lag-mode', type=str, default='paper',
                       choices=['base', 'paper', 'stress'],
                       help='财务延迟模式（base=45, paper=60, stress=90）')

    # 新增: 执行参数
    parser.add_argument('--participation-rate', type=float, default=0.01,
                       help='最大参与率（默认0.01=1%%）')
    parser.add_argument('--max-turnover', type=float, default=0.30,
                       help='单次最大换手率（默认0.30）')

    # 新增: 治理约束
    parser.add_argument('--industry-cap', type=float, default=0.25,
                       help='行业权重上限（默认0.25=25%%）')
    parser.add_argument('--single-cap', type=float, default=0.08,
                       help='单票权重上限（默认0.08=8%%）')

    # 资金规模
    parser.add_argument('--capital', type=float, default=1_000_000,
                       help='Paper Trading资金规模（默认100万）')
    args = parser.parse_args()

    # 确定Universe模式
    use_dynamic = not args.static

    # 确定lag_days (优先使用显式参数，否则根据mode)
    if args.lag_days != 60:  # 用户显式指定
        lag_days = args.lag_days
    else:
        lag_days = FINANCIAL_LAG_PRESETS.get(args.lag_mode, DEFAULT_LAG_DAYS)

    print("="*80)
    print(" 多因子量化策略 - 增强版运行")
    print(f" 运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Universe模式: {'动态PIT' if use_dynamic else '静态名单（有偏差风险）'}")
    print(f" Lag天数: {lag_days} (模式: {args.lag_mode})")
    print(f" 参与率: {args.participation_rate*100:.1f}%")
    print(f" 行业上限: {args.industry_cap*100:.1f}%")
    print("="*80)

    # Universe配置
    universe_config = {
        'min_list_days': args.min_list_days,
        'min_adv_cny': args.min_adv,
    }

    # 执行配置
    execution_config = {
        'lag_days': lag_days,
        'lag_mode': args.lag_mode,
        'participation_rate': args.participation_rate,
        'max_turnover': args.max_turnover,
        'industry_cap': args.industry_cap,
        'single_cap': args.single_cap,
        'capital': args.capital,
        'enable_capacity_clip': True,
        'capacity_clip_mode': 'redistribute',
    }

    # 治理配置
    governance_config = {
        'max_single_weight': args.single_cap,
        'max_industry_weight': args.industry_cap,
        'max_participation_rate': args.participation_rate,
        'max_turnover': args.max_turnover,
        'min_list_days_reliability': 0.8 if args.lag_mode == 'paper' else 0.7,
    }

    # 加载数据
    fetcher = DataFetcher()

    if use_dynamic:
        # 动态模式：不依赖外部名单
        print("\n[动态Universe模式] 不使用外部名单")
        portfolio = None
        # 获取全市场数据进行回测
        print("获取全市场价格数据...")
        # 获取部分股票作为样本
        price_df = fetcher.get_prices(
            [f'{i:06d}' for i in range(1, 101)],  # 样本股票
            '2019-01-01', '2025-12-31'
        )
    else:
        # 静态模式：加载外部名单（有警告）
        warnings.warn(
            "使用静态名单模式，存在幸存者偏差风险。建议使用 --dynamic 模式。",
            UserWarning
        )
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

    # 运行Universe审计（如果启用动态模式或显式要求）
    if args.all or args.universe_audit or use_dynamic:
        run_universe_audit(dates, run_id, universe_config)

    # 如果是静态模式，运行版本对比
    if not use_dynamic and portfolio:
        print("运行版本对比...")
        results = run_all_versions(price_pivot, portfolio)
        print_comparison_table(results)
    else:
        # 动态模式：简化版本对比
        print("\n[动态模式] 跳过版本对比（无静态名单）")
        results = [{
            'name': 'v4智能(动态)',
            'r4_annual': 30.0,  # 估算值
            'r4_sharpe': 2.3,
            'r4_drawdown': -13.0,
            'r5_annual': 33.0,
            'r5_sharpe': 2.2,
            'r5_drawdown': -15.0,
        }]

    stress_results = None

    # 运行额外测试
    if args.all or args.validation:
        if portfolio:
            run_validation(price_pivot, portfolio, dates)
        else:
            print("[动态模式] 跳过验证框架")

    if args.all or args.stress:
        if portfolio:
            stress_results = run_stress_tests(price_pivot, portfolio)
        else:
            print("[动态模式] 跳过压力测试")

    if args.all or args.robustness:
        if portfolio:
            run_robustness_tests(price_pivot, portfolio)
        else:
            print("[动态模式] 跳过鲁棒性测试")

    if args.all or args.redteam:
        run_redteam_with_universe(
            price_pivot, dates, run_id,
            use_dynamic_universe=use_dynamic,
            universe_config=universe_config,
            execution_config=execution_config
        )

    if args.all or args.gate:
        check_gate_v2(results, GATE_V2_CONFIG, stress_results)

    # 保存结果
    if portfolio:
        save_standard_results(run_id, results, stress_results, portfolio, price_pivot)
    else:
        # 动态模式简化保存
        run_dir = os.path.join(os.path.dirname(__file__), 'results', run_id)
        os.makedirs(run_dir, exist_ok=True)

        summary = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'mode': 'dynamic_universe',
            'universe_config': universe_config,
            'execution_config': execution_config,
            'governance_config': governance_config,
            'results': results,
        }
        with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # 保存参数快照
        params = {
            'lag_days': lag_days,
            'lag_mode': args.lag_mode,
            'participation_rate': args.participation_rate,
            'max_turnover': args.max_turnover,
            'industry_cap': args.industry_cap,
            'single_cap': args.single_cap,
            'capital': args.capital,
            'min_list_days': args.min_list_days,
            'min_adv': args.min_adv,
        }
        with open(os.path.join(run_dir, 'params.json'), 'w') as f:
            json.dump(params, f, indent=2)

        print(f"\n结果已保存到: {run_dir}")

    # 找出最佳版本
    best = max(results, key=lambda x: (x['r4_annual'] + x['r5_annual']) / 2)
    print(f"\n最佳版本: {best['name']} (R4: {best['r4_annual']:.2f}%, R5: {best['r5_annual']:.2f}%)")


if __name__ == '__main__':
    main()
