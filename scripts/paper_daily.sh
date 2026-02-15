#!/bin/bash
#
# Paper Daily - 每日监控脚本
#
# 功能:
# 1. 更新净值曲线（读取昨日持仓 + 今日行情）
# 2. 计算风险指标（DD、rolling_return、rolling_sharpe）
# 3. 检查运营规则（DE-RISK / EXIT）
# 4. 生成告警报告
#
# 用法: bash scripts/paper_daily.sh [--date YYYY-MM-DD]
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

# 参数
DATE=${1:-$(date +%Y-%m-%d)}
DATE_NO_DASH=$(echo "$DATE" | tr -d '-')

# 目录配置
RESULTS_DIR="results_paper"
DAILY_DIR="${RESULTS_DIR}/daily/${DATE_NO_DASH}"
EQUITY_FILE="${RESULTS_DIR}/equity_curve.csv"

# ==================== 输出函数 ====================
log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') $1" >&2
}

log_warning() {
    echo "[WARNING] $(date '+%Y-%m-%d %H:%M:%S') $1"
}

# ==================== 主流程 ====================
log_info "=========================================="
log_info "Paper Daily Monitor"
log_info "Date: $DATE"
log_info "=========================================="

# 创建输出目录
mkdir -p "$DAILY_DIR"

# 保存参数快照
cat > "${DAILY_DIR}/params.json" << EOF
{
    "date": "$DATE",
    "script": "paper_daily.sh",
    "timestamp": "$(date -Iseconds)",
    "mode": "paper_trading"
}
EOF
log_info "参数已保存: ${DAILY_DIR}/params.json"

# ==================== Step 1: 运行每日监控 ====================
log_info "Step 1: 运行每日监控..."

python -c "
import os
import sys
import json
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目路径
sys.path.insert(0, '$PROJECT_DIR')

from ops.daily_monitor import DailyMonitor
from ops.utils import FROZEN_PARAMS

try:
    # 初始化监控器
    monitor = DailyMonitor(
        results_dir='$RESULTS_DIR',
        capital=FROZEN_PARAMS['capital'],
        profile='R4',  # 默认使用 R4 稳健型
        enable_alerts=True
    )

    # 运行监控
    result = monitor.run(date='$DATE')

    # 输出关键信息
    print('\\n' + '='*50)
    print('每日监控结果')
    print('='*50)
    print(f\"日期: {result['date']}\")
    print(f\"净值: {result['equity']:.2f}\")
    print(f\"日收益: {result['daily_return']:.2%}\")
    print(f\"持仓数: {result['positions_count']}\")
    print(f\"决策: {result['decision']}\")

    if result['alerts']:
        print('\\n告警:')
        for alert in result['alerts']:
            print(f\"  - {alert}\")

    print('='*50)

    # 保存决策到文件
    with open('${DAILY_DIR}/decision.txt', 'w') as f:
        f.write(result['decision'])

    sys.exit(0)

except Exception as e:
    logger.error(f'每日监控失败: {e}')
    import traceback
    traceback.print_exc()

    # 保存错误信息
    with open('${DAILY_DIR}/error.log', 'w') as f:
        f.write(str(e))

    sys.exit(1)
"

MONITOR_EXIT=$?

if [ $MONITOR_EXIT -eq 0 ]; then
    log_info "每日监控完成"
else
    log_error "每日监控失败"
fi

# ==================== Step 2: 汇总报告 ====================
log_info "Step 2: 汇总报告..."

# 读取决策
DECISION="UNKNOWN"
if [ -f "${DAILY_DIR}/decision.txt" ]; then
    DECISION=$(cat "${DAILY_DIR}/decision.txt")
fi

# 读取 KPI
if [ -f "${DAILY_DIR}/daily_kpi.json" ]; then
    log_info "KPI 数据已生成: ${DAILY_DIR}/daily_kpi.json"
fi

# 读取告警
if [ -f "${DAILY_DIR}/alerts.md" ]; then
    log_info "告警报告已生成: ${DAILY_DIR}/alerts.md"
fi

# 读取净值曲线
if [ -f "$EQUITY_FILE" ]; then
    log_info "净值曲线已更新: $EQUITY_FILE"

    # 显示最近的净值记录
    log_info "最近5日净值:"
    tail -5 "$EQUITY_FILE" | while read line; do
        log_info "  $line"
    done
fi

# ==================== Step 3: 检查告警级别 ====================
log_info "Step 3: 检查告警级别..."

ALERT_LEVEL="INFO"
case "$DECISION" in
    "HOLD")
        ALERT_LEVEL="INFO"
        ;;
    "DE-RISK")
        ALERT_LEVEL="WARNING"
        log_warning "触发 DE-RISK 规则，建议降低仓位"
        ;;
    "EXIT")
        ALERT_LEVEL="CRITICAL"
        log_error "触发 EXIT 规则，建议暂停策略"
        ;;
esac

# ==================== Step 4: 生成每日摘要 ====================
log_info "Step 4: 生成每日摘要..."

cat > "${DAILY_DIR}/daily_summary.md" << SUMMARY_EOF
# Paper Trading 每日监控摘要

> **日期**: $DATE
> **决策**: $DECISION
> **告警级别**: $ALERT_LEVEL

---

## 决策说明

| 决策 | 说明 |
|------|------|
| HOLD | 继续运行，所有指标正常 |
| DE-RISK | 降低仓位，出现警告信号 |
| EXIT | 暂停策略，触发回滚条件 |

---

## 运营规则检查

### DE-RISK 触发条件
- [ ] 20日回撤 > 15%
- [ ] 连续负收益天数 >= 10天

### EXIT 触发条件
- [ ] R4 最大回撤 > 25% / R5 最大回撤 > 30%
- [ ] 2个月累计收益 < -5%
- [ ] 3个月滚动夏普 < 1.0

---

## 产物清单

- \`daily_kpi.json\` - 风险指标
- \`alerts.md\` - 告警详情
- \`params.json\` - 参数快照
- \`equity_curve.csv\` - 累计净值曲线

---

*报告生成时间: $(date -Iseconds)*
SUMMARY_EOF

log_info "每日摘要已生成: ${DAILY_DIR}/daily_summary.md"

# ==================== 最终输出 ====================
echo ""
log_info "=========================================="
log_info "Paper Daily 完成"
log_info "决策: $DECISION"
log_info "告警级别: $ALERT_LEVEL"
log_info "产物目录: $DAILY_DIR"
log_info "=========================================="

# 根据决策返回不同退出码
case "$DECISION" in
    "HOLD")
        exit 0
        ;;
    "DE-RISK")
        exit 10  # 警告级别退出码
        ;;
    "EXIT")
        exit 20  # 严重级别退出码
        ;;
    *)
        exit 1
        ;;
esac
