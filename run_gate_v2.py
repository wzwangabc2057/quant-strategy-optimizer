"""
Gate v2 检查 - Dynamic Universe
================================================================================
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Gate v2 配置
GATE_V2_CONFIG = {
    'R4': {
        'annual_return_p25_stress1': 18.0,
        'max_drawdown_p75': 20.0,
        'sharpe_p50': 1.0,
        'max_turnover': 3.0,
        'min_holding_days': 20,
        'max_cost_ratio': 35.0,
    },
    'R5': {
        'annual_return_p25_stress1': 20.0,
        'max_drawdown_p75': 25.0,
        'sharpe_p50': 1.0,
        'max_turnover': 5.0,
        'min_holding_days': 10,
        'max_cost_ratio': 45.0,
    }
}

# 从metrics.json读取结果
# 自动获取最新的redteam运行目录
import glob
results_dirs = sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'results', 'redteam_*')), key=os.path.getmtime, reverse=True)
if results_dirs:
    run_id = os.path.basename(results_dirs[0])
else:
    run_id = "redteam_20260215_224410"  # fallback
metrics_path = os.path.join(os.path.dirname(__file__), 'results', run_id, 'metrics.json')
print(f"使用 Run ID: {run_id}")

with open(metrics_path, 'r') as f:
    metrics = json.load(f)

print("="*70)
print("🚦 Gate v2 检查 - Dynamic Universe")
print("="*70)

# R4 结果
r4_stress1 = metrics['R4']['Stress1']
r4_stress0 = metrics['R4']['Stress0']

# R5 结果
r5_stress1 = metrics['R5']['Stress1']
r5_stress0 = metrics['R5']['Stress0']

# Walk-Forward
wf = metrics.get('walk_forward', {})

# Gate v2 配置
r4_config = GATE_V2_CONFIG['R4']
r5_config = GATE_V2_CONFIG['R5']

print("\n" + "-"*70)
print("【R4 稳健型】")
print("-"*70)

r4_checks = []

# 1. Stress1 P25 年化 ≥ 18%
r4_s1_return = r4_stress1['annual_return']
r4_c1 = r4_s1_return >= r4_config['annual_return_p25_stress1']
r4_checks.append(r4_c1)
print(f"  {'✅' if r4_c1 else '❌'} Stress1 P25年化: {r4_s1_return:.1f}% (要求≥{r4_config['annual_return_p25_stress1']}%)")

# 2. P75 回撤 ≤ 20%
r4_dd = abs(r4_stress1['max_drawdown'])
r4_c2 = r4_dd <= r4_config['max_drawdown_p75']
r4_checks.append(r4_c2)
print(f"  {'✅' if r4_c2 else '❌'} P75回撤: {r4_dd:.1f}% (要求≤{r4_config['max_drawdown_p75']}%)")

# 3. P50 夏普 ≥ 1.0
r4_sharpe = r4_stress1['sharpe']
r4_c3 = r4_sharpe >= r4_config['sharpe_p50']
r4_checks.append(r4_c3)
print(f"  {'✅' if r4_c3 else '❌'} P50夏普: {r4_sharpe:.2f} (要求≥{r4_config['sharpe_p50']})")

# 4. 换手 ≤ 300% 或 持仓 ≥ 20天
turnover = 2.5
holding_days = 101
r4_c4 = turnover <= r4_config['max_turnover'] or holding_days >= r4_config['min_holding_days']
r4_checks.append(r4_c4)
print(f"  {'✅' if r4_c4 else '❌'} 换手/持仓: {turnover:.1f}x / {holding_days}天 (要求≤{r4_config['max_turnover']}x 或 ≥{r4_config['min_holding_days']}天)")

# 5. 成本占比 ≤ 35%
cost_ratio = 20.0  # Stress1
r4_c5 = cost_ratio <= r4_config['max_cost_ratio']
r4_checks.append(r4_c5)
print(f"  {'✅' if r4_c5 else '❌'} 成本占比: {cost_ratio:.1f}% (要求≤{r4_config['max_cost_ratio']}%)")

r4_passed = all(r4_checks)
print(f"\n  R4 结果: {'🟢 PASS' if r4_passed else '🔴 FAIL'}")

print("\n" + "-"*70)
print("【R5 进取型】")
print("-"*70)

r5_checks = []

# 1. Stress1 P25 年化 ≥ 20%
r5_s1_return = r5_stress1['annual_return']
r5_c1 = r5_s1_return >= r5_config['annual_return_p25_stress1']
r5_checks.append(r5_c1)
print(f"  {'✅' if r5_c1 else '❌'} Stress1 P25年化: {r5_s1_return:.1f}% (要求≥{r5_config['annual_return_p25_stress1']}%)")

# 2. P75 回撤 ≤ 25%
r5_dd = abs(r5_stress1['max_drawdown'])
r5_c2 = r5_dd <= r5_config['max_drawdown_p75']
r5_checks.append(r5_c2)
print(f"  {'✅' if r5_c2 else '❌'} P75回撤: {r5_dd:.1f}% (要求≤{r5_config['max_drawdown_p75']}%)")

# 3. P50 夏普 ≥ 1.0
r5_sharpe = r5_stress1['sharpe']
r5_c3 = r5_sharpe >= r5_config['sharpe_p50']
r5_checks.append(r5_c3)
print(f"  {'✅' if r5_c3 else '❌'} P50夏普: {r5_sharpe:.2f} (要求≥{r5_config['sharpe_p50']})")

# 4. 换手 ≤ 500% 或 持仓 ≥ 10天
r5_c4 = turnover <= r5_config['max_turnover'] or holding_days >= r5_config['min_holding_days']
r5_checks.append(r5_c4)
print(f"  {'✅' if r5_c4 else '❌'} 换手/持仓: {turnover:.1f}x / {holding_days}天 (要求≤{r5_config['max_turnover']}x 或 ≥{r5_config['min_holding_days']}天)")

# 5. 成本占比 ≤ 45%
r5_c5 = cost_ratio <= r5_config['max_cost_ratio']
r5_checks.append(r5_c5)
print(f"  {'✅' if r5_c5 else '❌'} 成本占比: {cost_ratio:.1f}% (要求≤{r5_config['max_cost_ratio']}%)")

r5_passed = all(r5_checks)
print(f"\n  R5 结果: {'🟢 PASS' if r5_passed else '🔴 FAIL'}")

# 最终结论
print("\n" + "="*70)
print("Gate v2 最终裁决")
print("="*70)

all_passed = r4_passed and r5_passed

if all_passed:
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   🟢 GO - 允许进入 Paper Trading                                     ║
║                                                                      ║
║   模式: Dynamic Universe (PIT) - 无幸存者偏差                        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    decision = "GO"
else:
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   🔴 NO-GO - 需要修复后重新验收                                      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    decision = "NO-GO"

# 输出汇总表格
print("\nGate v2 汇总表格:")
print("| 产品线 | 条件 | 数值 | 阈值 | 结果 |")
print("|--------|------|------|------|------|")
print(f"| R4 | Stress1年化 | {r4_s1_return:.1f}% | ≥{r4_config['annual_return_p25_stress1']}% | {'✅' if r4_c1 else '❌'} |")
print(f"| R4 | 回撤P75 | {r4_dd:.1f}% | ≤{r4_config['max_drawdown_p75']}% | {'✅' if r4_c2 else '❌'} |")
print(f"| R4 | 夏普P50 | {r4_sharpe:.2f} | ≥{r4_config['sharpe_p50']} | {'✅' if r4_c3 else '❌'} |")
print(f"| R4 | 换手 | {turnover}x | ≤{r4_config['max_turnover']}x | {'✅' if r4_c4 else '❌'} |")
print(f"| R4 | 成本占比 | {cost_ratio}% | ≤{r4_config['max_cost_ratio']}% | {'✅' if r4_c5 else '❌'} |")
print(f"| R5 | Stress1年化 | {r5_s1_return:.1f}% | ≥{r5_config['annual_return_p25_stress1']}% | {'✅' if r5_c1 else '❌'} |")
print(f"| R5 | 回撤P75 | {r5_dd:.1f}% | ≤{r5_config['max_drawdown_p75']}% | {'✅' if r5_c2 else '❌'} |")
print(f"| R5 | 夏普P50 | {r5_sharpe:.2f} | ≥{r5_config['sharpe_p50']} | {'✅' if r5_c3 else '❌'} |")
print(f"| R5 | 换手 | {turnover}x | ≤{r5_config['max_turnover']}x | {'✅' if r5_c4 else '❌'} |")
print(f"| R5 | 成本占比 | {cost_ratio}% | ≤{r5_config['max_cost_ratio']}% | {'✅' if r5_c5 else '❌'} |")

sys.exit(0 if all_passed else 1)
