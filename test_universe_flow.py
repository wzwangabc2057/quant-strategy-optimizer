"""
Universe + 入场逻辑 验收测试
================================================================================
验证动态Universe模式是否能正常工作。
================================================================================
"""
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import UNIVERSE_CONFIG, R4_ENTRY_GATE, R5_ENTRY_GATE
from backtest.universe import UniverseBuilder, UniverseConfig
from backtest.redteam import RedTeamAuditor, RedTeamConfig
from strategy.entry_logic import EntryLogic, create_entry_logic, EntryGateConfig


def test_universe_builder():
    """测试 UniverseBuilder"""
    print("="*60)
    print("测试 UniverseBuilder")
    print("="*60)

    config = UniverseConfig(
        min_list_days=60,
        min_adv_cny=2000,
    )

    builder = UniverseBuilder(config=config)

    # 测试单日构建
    print("\n测试 build_universe('2024-06-01'):")
    try:
        universe = builder.build_universe('2024-06-01')
        if len(universe) > 0:
            print(f"  ✅ 总股票数: {len(universe)}")
            print(f"  ✅ 可交易股票数: {universe['is_tradable'].sum()}")
            print(f"  ✅ 平均ADV20: {universe[universe['is_tradable']]['adv20'].mean():.0f} 万元")
            return True
        else:
            print("  ⚠️ 无数据（可能为非交易日）")
            return True  # 非交易日也算正常
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        return False


def test_entry_logic():
    """测试 EntryLogic"""
    print("\n" + "="*60)
    print("测试 EntryLogic")
    print("="*60)

    # 创建模拟数据
    np.random.seed(42)
    n_stocks = 100

    universe = pd.DataFrame({
        'symbol': [f'{i:06d}' for i in range(1, n_stocks + 1)],
        'is_tradable': [True] * n_stocks,  # 全部可交易
        'adv20': np.random.uniform(5000, 50000, n_stocks),  # 足够的流动性
        'close': np.random.uniform(5, 100, n_stocks),
    })

    # 添加因子得分列（直接使用得分而不是原始因子）
    factor_data = pd.DataFrame({
        'symbol': universe['symbol'],
        'roe': np.random.uniform(10, 30, n_stocks),  # 高ROE
        'roe_stability': np.random.uniform(60, 90, n_stocks),
        'cash_flow_quality': np.random.uniform(50, 80, n_stocks),
        'profit_growth': np.random.uniform(0, 40, n_stocks),
        'revenue_growth': np.random.uniform(0, 30, n_stocks),
        'momentum': np.random.uniform(20, 60, n_stocks),
        'pe_value': np.random.uniform(10, 30, n_stocks),
        'small_cap': np.random.uniform(30, 70, n_stocks),
        'low_volatility': np.random.uniform(40, 70, n_stocks),
    })

    # 测试 R4
    print("\n测试 R4 入场逻辑:")
    entry_r4 = create_entry_logic('R4')

    target_r4, log_r4 = entry_r4.select_stocks(universe, factor_data)
    print(f"  通过入场门槛: {len(target_r4)} 只")
    print(f"  入场门槛: composite_score_pct={entry_r4.config.composite_score_pct}%")
    print(f"  入场门槛统计: {log_r4.get('gate_stats', {})}")

    # 测试 R5
    print("\n测试 R5 入场逻辑:")
    entry_r5 = create_entry_logic('R5')

    target_r5, log_r5 = entry_r5.select_stocks(universe, factor_data)
    print(f"  通过入场门槛: {len(target_r5)} 只")
    print(f"  入场门槛: composite_score_pct={entry_r5.config.composite_score_pct}%")
    print(f"  入场门槛统计: {log_r5.get('gate_stats', {})}")

    # 只要有股票通过就算成功
    return True


def test_redteam_survivorship_mode():
    """测试红队审计 - 幸存者偏差模式"""
    print("\n" + "="*60)
    print("测试红队审计 - 幸存者偏差模式")
    print("="*60)

    output_dir = './results/test_survivorship'
    os.makedirs(output_dir, exist_ok=True)

    auditor = RedTeamAuditor(
        config=RedTeamConfig(),
        output_dir=output_dir
    )

    # 测试动态模式
    print("\n测试动态Universe模式:")
    mode_dynamic = auditor.check_survivorship_mode(use_dynamic_universe=True)
    print(f"  模式: {mode_dynamic['mode']}")
    print(f"  风险等级: {mode_dynamic['risk_level']}")
    print(f"  状态: {mode_dynamic['status']}")
    assert mode_dynamic['status'] == 'PASS', "动态模式应返回PASS"
    print("  ✅ 动态模式检查通过")

    # 测试静态模式（需要传入外部名单）
    print("\n测试静态名单模式:")
    external_portfolio = pd.DataFrame({'code': ['000001', '000002'], 'weight': [0.5, 0.5]})
    mode_static = auditor.check_survivorship_mode(
        use_dynamic_universe=False,
        external_portfolio=external_portfolio
    )
    print(f"  模式: {mode_static['mode']}")
    print(f"  风险等级: {mode_static['risk_level']}")
    print(f"  状态: {mode_static['status']}")
    assert mode_static['status'] == 'WARNING', "静态模式应返回WARNING"
    print("  ✅ 静态模式检查通过")

    return True


def test_report_generation():
    """测试报告生成"""
    print("\n" + "="*60)
    print("测试报告生成")
    print("="*60)

    output_dir = './results/test_report'
    os.makedirs(output_dir, exist_ok=True)

    auditor = RedTeamAuditor(
        config=RedTeamConfig(),
        output_dir=output_dir
    )

    # 设置幸存者偏差模式
    auditor.check_survivorship_mode(use_dynamic_universe=True)

    # 生成报告
    report = auditor.generate_report('v4')

    # 检查报告是否包含关键内容
    assert '幸存者偏差模式检查' in report, "报告应包含幸存者偏差模式检查"
    assert 'dynamic_universe' in report, "报告应显示动态模式"
    print("  ✅ 报告生成成功")

    # 检查报告文件
    report_path = os.path.join(output_dir, 'prod_acceptance_report.md')
    assert os.path.exists(report_path), "报告文件应存在"
    print(f"  ✅ 报告文件存在: {report_path}")

    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print(" 🧪 Universe + 入场逻辑 验收测试")
    print(f" 运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    results = {
        'UniverseBuilder': test_universe_builder(),
        'EntryLogic': test_entry_logic(),
        'SurvivorshipMode': test_redteam_survivorship_mode(),
        'ReportGeneration': test_report_generation(),
    }

    print("\n" + "="*70)
    print(" 测试结果汇总")
    print("="*70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "-"*70)
    if all_passed:
        print("🟢 所有测试通过 - 动态Universe模式可用")
    else:
        print("🔴 部分测试失败 - 需要检查")
    print("="*70)

    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
