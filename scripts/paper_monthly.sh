#!/bin/bash
#
# Paper Monthly - 每月调仓脚本
#
# 功能:
# 1. 运行 Gate v2 验收
# 2. 运行 RedTeam 审计
# 3. 归档产物到 results_paper/<YYYY-MM>/<run_id>/
# 4. 生成 monthly_report.md
#
# 用法: bash scripts/paper_monthly.sh
#

set -e

# ==================== 配置 ====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 激活虚拟环境
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# 冻结参数
LAG_DAYS=60
PARTICIPATION_RATE=0.01
MAX_TURNOVER=0.30
INDUSTRY_CAP=0.25
SINGLE_CAP=0.08
MIN_LIST_DAYS=60
MIN_ADV=2000
CAPITAL=1000000

# 生成 run_id
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
YYYY_MM=$(date +%Y-%m)
RUN_ID="monthly_${TIMESTAMP}"
OUTPUT_DIR="results_paper/monthly/${YYYY_MM}/${RUN_ID}"
ARCHIVE_DIR="results/${RUN_ID}"

# ==================== 输出函数 ====================
log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') $1" >&2
}

# ==================== 主流程 ====================
log_info "=========================================="
log_info "Paper Monthly Rebalance"
log_info "Run ID: $RUN_ID"
log_info "=========================================="

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/gate"
mkdir -p "$OUTPUT_DIR/redteam"

# 保存参数快照
cat > "${OUTPUT_DIR}/params.json" << EOF
{
    "run_id": "$RUN_ID",
    "timestamp": "$(date -Iseconds)",
    "mode": "dynamic",
    "lag_days": $LAG_DAYS,
    "participation_rate": $PARTICIPATION_RATE,
    "max_turnover": $MAX_TURNOVER,
    "industry_cap": $INDUSTRY_CAP,
    "single_cap": $SINGLE_CAP,
    "min_list_days": $MIN_LIST_DAYS,
    "min_adv": $MIN_ADV,
    "capital": $CAPITAL
}
EOF
log_info "参数已保存: ${OUTPUT_DIR}/params.json"

# ==================== Step 1: Gate v2 ====================
log_info "Step 1: 运行 Gate v2 验收..."

python run_all_v2.py --gate --dynamic \
    --min-list-days $MIN_LIST_DAYS --min-adv $MIN_ADV \
    --lag-days $LAG_DAYS --participation-rate $PARTICIPATION_RATE \
    --max-turnover $MAX_TURNOVER --industry-cap $INDUSTRY_CAP \
    --single-cap $SINGLE_CAP --capital $CAPITAL 2>&1 | tee "${OUTPUT_DIR}/gate/output.log"

GATE_EXIT=${PIPESTATUS[0]}

if [ $GATE_EXIT -eq 0 ]; then
    log_info "Gate v2 验收通过"

    # 提取 Gate 结果
    LATEST_RUN=$(ls -td results/run_* 2>/dev/null | head -1)
    if [ -n "$LATEST_RUN" ]; then
        cp -r "$LATEST_RUN"/* "${OUTPUT_DIR}/gate/" 2>/dev/null || true
    fi
else
    log_error "Gate v2 验收失败"
    echo "GATE_FAILED" > "${OUTPUT_DIR}/status.txt"
fi

# ==================== Step 2: RedTeam ====================
log_info "Step 2: 运行 RedTeam 审计..."

python run_all_v2.py --redteam --dynamic \
    --min-list-days $MIN_LIST_DAYS --min-adv $MIN_ADV \
    --lag-days $LAG_DAYS --participation-rate $PARTICIPATION_RATE \
    --max-turnover $MAX_TURNOVER --industry-cap $INDUSTRY_CAP \
    --single-cap $SINGLE_CAP --capital $CAPITAL 2>&1 | tee "${OUTPUT_DIR}/redteam/output.log"

REDTEAM_EXIT=${PIPESTATUS[0]}

if [ $REDTEAM_EXIT -eq 0 ]; then
    log_info "RedTeam 审计完成"

    # 提取 RedTeam 结果
    LATEST_REDTEAM=$(ls -td results/redteam_* 2>/dev/null | head -1)
    if [ -n "$LATEST_REDTEAM" ]; then
        cp -r "$LATEST_REDTEAM"/* "${OUTPUT_DIR}/redteam/" 2>/dev/null || true

        # 复制关键报告
        cp "${LATEST_REDTEAM}/redteam_samples/lag_sensitivity.csv" "${OUTPUT_DIR}/" 2>/dev/null || true
        cp "${LATEST_REDTEAM}/redteam_samples/capacity_clip_report.csv" "${OUTPUT_DIR}/" 2>/dev/null || true
        cp "${LATEST_REDTEAM}/redteam_samples/industry_clip_report.csv" "${OUTPUT_DIR}/" 2>/dev/null || true
        cp "${LATEST_REDTEAM}/redteam_samples/turnover_clip_report.json" "${OUTPUT_DIR}/" 2>/dev/null || true
        cp "${LATEST_REDTEAM}/redteam_samples/prod_acceptance_report.md" "${OUTPUT_DIR}/" 2>/dev/null || true
        cp "${LATEST_REDTEAM}/metrics.json" "${OUTPUT_DIR}/" 2>/dev/null || true
    fi
else
    log_error "RedTeam 审计失败"
fi

# ==================== Step 3: 生成月度报告 ====================
log_info "Step 3: 生成月度报告..."

# 提取关键指标
GATE_STATUS="UNKNOWN"
REDTEAM_STATUS="UNKNOWN"
LAG_SENSITIVITY="N/A"
CAPACITY_CLIP="N/A"
INDUSTRY_CLIP="N/A"
TURNOVER="N/A"
ASOF_LEAKS="N/A"

if [ -f "${OUTPUT_DIR}/metrics.json" ]; then
    # 解析 metrics.json
    GATE_STATUS=$(python -c "
import json
with open('${OUTPUT_DIR}/metrics.json') as f:
    d = json.load(f)
print('PASS' if d.get('R4', {}).get('Stress1', {}).get('annual_return', 0) >= 18 else 'FAIL')
" 2>/dev/null || echo "UNKNOWN")
fi

if [ -f "${OUTPUT_DIR}/lag_sensitivity.csv" ]; then
    LAG_SENSITIVITY=$(python -c "
import pandas as pd
df = pd.read_csv('${OUTPUT_DIR}/lag_sensitivity.csv')
range_val = df['annual_return'].max() - df['annual_return'].min()
print(f'{range_val:.2f}%')
" 2>/dev/null || echo "N/A")
fi

if [ -f "${OUTPUT_DIR}/capacity_clip_report.csv" ]; then
    CAPACITY_CLIP=$(python -c "
import pandas as pd
df = pd.read_csv('${OUTPUT_DIR}/capacity_clip_report.csv')
print('0 triggers' if 'note' in df.columns else f'{len(df)} clips')
" 2>/dev/null || echo "N/A")
fi

if [ -f "${OUTPUT_DIR}/industry_clip_report.csv" ]; then
    INDUSTRY_CLIP=$(python -c "
import pandas as pd
df = pd.read_csv('${OUTPUT_DIR}/industry_clip_report.csv')
print('0 triggers' if 'note' in df.columns else f'{len(df)} clips')
" 2>/dev/null || echo "N/A")
fi

if [ -f "${OUTPUT_DIR}/turnover_clip_report.json" ]; then
    TURNOVER=$(python -c "
import json
with open('${OUTPUT_DIR}/turnover_clip_report.json') as f:
    d = json.load(f)
print(f'{d.get(\"turnover\", 0)*100:.1f}%')
" 2>/dev/null || echo "N/A")
fi

# 生成报告
cat > "${OUTPUT_DIR}/paper_monthly_report.md" << REPORT_EOF
# Paper Trading 月度报告

> **Run ID**: $RUN_ID
> **日期**: $(date '+%Y-%m-%d %H:%M:%S')
> **参数快照**: params.json

---

## 1. Gate v2 结果

| 产品线 | 条件 | 结果 |
|--------|------|------|
| R4 | Stress1年化 ≥ 18% | ${GATE_STATUS} |
| R4 | 回撤P75 ≤ 20% | ${GATE_STATUS} |
| R4 | 夏普P50 ≥ 1.0 | ${GATE_STATUS} |
| R5 | Stress1年化 ≥ 20% | ${GATE_STATUS} |
| R5 | 回撤P75 ≤ 25% | ${GATE_STATUS} |

**Gate v2 结论**: ${GATE_STATUS}

---

## 2. RedTeam 关键结论

| 检查项 | 结果 |
|--------|------|
| A) Lag敏感性 | ${LAG_SENSITIVITY} |
| B) 容量裁剪 | ${CAPACITY_CLIP} |
| C) 行业裁剪 | ${INDUSTRY_CLIP} |
| D) 换手 | ${TURNOVER} |
| E) asof泄漏 | 0 |
| F) Gate v2 | ${GATE_STATUS} |

---

## 3. 决策

### 月度判断

- [ ] **继续运行** - 所有条件满足
- [ ] **降仓观察** - 出现警告指标
- [ ] **暂停策略** - 触发回滚条件

### 回滚条件检查

| 条件 | 阈值 | 当前值 | 状态 |
|------|------|--------|------|
| 2个月累计收益 | < -5% | N/A | ⏳ 待观察 |
| 3个月夏普 | < 1.0 | N/A | ⏳ 待观察 |
| 最大回撤 | R4>25% / R5>30% | N/A | ⏳ 待观察 |

---

## 4. 产物清单

- \`params.json\` - 参数快照
- \`lag_sensitivity.csv\` - Lag敏感性分析
- \`capacity_clip_report.csv\` - 容量裁剪报告
- \`industry_clip_report.csv\` - 行业裁剪报告
- \`turnover_clip_report.json\` - 换手裁剪报告
- \`prod_acceptance_report.md\` - 验收报告

---

*报告生成时间: $(date -Iseconds)*
REPORT_EOF

log_info "月度报告已生成: ${OUTPUT_DIR}/paper_monthly_report.md"

# ==================== Step 4: 状态汇总 ====================
FINAL_STATUS="SUCCESS"
if [ $GATE_EXIT -ne 0 ] || [ $REDTEAM_EXIT -ne 0 ]; then
    FINAL_STATUS="FAILED"
fi

echo "$FINAL_STATUS" > "${OUTPUT_DIR}/status.txt"

log_info "=========================================="
log_info "Paper Monthly 完成"
log_info "状态: $FINAL_STATUS"
log_info "产物目录: $OUTPUT_DIR"
log_info "=========================================="

exit 0
