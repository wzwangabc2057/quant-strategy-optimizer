# Paper Trading Operations Guide

> **版本**: v0.4.0-paper-ready
> **冻结日期**: 2026-02-15
> **状态**: Production Ready

---

## 1. 推荐运行环境

| 组件 | 配置 |
|------|------|
| 执行机器 | macstudio@192.168.0.88 |
| ClickHouse | 192.168.0.74:8123 |
| Python | 3.9.6+ |
| 数据表 | stock_data_qfq (19M+ rows), stock_block_em (4,582 stocks) |

---

## 2. 冻结参数

> ⚠️ **警告**: 以下参数已冻结，不得随意修改。如需调整必须重新运行 redteam + gate 验收。

| 参数 | 冻结值 | 说明 |
|------|--------|------|
| `lag_days` | 60 | 财务可用日延迟 (paper模式) |
| `lag_mode` | paper | 保守模式 (base=45, paper=60, stress=90) |
| `participation_rate` | 0.01 (1%) | 最大参与率 |
| `max_turnover` | 0.30 (30%) | 单次最大换手 |
| `industry_cap` | 0.25 (25%) | 行业权重上限 |
| `single_cap` | 0.08 (8%) | 单票权重上限 |
| `min_list_days` | 60 | 最小上市天数 |
| `min_adv` | 2000万元 | 最小ADV20门槛 |
| `rebalance_freq` | monthly | 调仓频率 |
| `capital` | 1,000,000 | Paper Trading资金规模 |

---

## 3. 每月调仓命令

```bash
# SSH 到执行机器
ssh macstudio@192.168.0.88

# 进入项目目录
cd /Users/macstudio/pythontest/quant-strategy-optimizer
source .venv/bin/activate

# 运行 Gate v2 验收
python run_all_v2.py --gate --dynamic \
  --min-list-days 60 --min-adv 2000 \
  --lag-days 60 --participation-rate 0.01 --max-turnover 0.30 \
  --industry-cap 0.25 --single-cap 0.08 --capital 1000000
```

**期望输出**: R4/R5 全部 PASS

---

## 4. 每月红队复验命令

```bash
# 运行完整红队审计
python run_all_v2.py --redteam --dynamic \
  --min-list-days 60 --min-adv 2000 \
  --lag-days 60 --participation-rate 0.01 --max-turnover 0.30 \
  --industry-cap 0.25 --single-cap 0.08 --capital 1000000
```

**期望输出**:
- `lag_sensitivity.csv`: 收益变动 < 3%
- `capacity_clip_report.csv`: 无超限
- `industry_clip_report.csv`: 裁剪后无行业超限
- `turnover_clip_report.json`: 换手 <= 30%
- `prod_acceptance_report.md`: GO 结论

---

## 5. 产物检查清单

每次运行后检查以下文件:

| 文件 | 检查项 |
|------|--------|
| `results/<run_id>/redteam_samples/lag_sensitivity.csv` | 45→90天收益变动 < 3% |
| `results/<run_id>/redteam_samples/capacity_clip_report.csv` | 无超 cap_i 的目标买入 |
| `results/<run_id>/redteam_samples/industry_clip_report.csv` | 裁剪后行业权重 <= 上限 |
| `results/<run_id>/redteam_samples/turnover_clip_report.json` | 换手 <= 30% |
| `results/<run_id>/redteam_samples/asof_samples.csv` | 泄漏行数 = 0 |
| `results/<run_id>/redteam_samples/prod_acceptance_report.md` | 最终裁决 = GO |
| `results/<run_id>/params.json` | 参数与冻结值一致 |

---

## 6. 回滚条件

满足以下任一条件立即回滚到 v3:

| 条件 | 阈值 | 触发动作 |
|------|------|----------|
| 2个月累计收益 | < -5% | 回滚 v3 |
| 3个月夏普比率 | < 1.0 | 回滚 v3 |
| 最大回撤 | R4 > 25% 或 R5 > 30% | 回滚 v3 |

**回滚操作**:
```bash
git checkout v3.x.x
# 重新部署
```

---

## 7. 参数修改流程

如需修改冻结参数:

1. 在 `config.py` 中修改默认值
2. 运行完整 redteam 审计
3. 运行 Gate v2 验收
4. 确认所有条件 PASS
5. 更新本文档的冻结参数表
6. 提交 PR 并附上新 run_id

---

## 8. 自动化运营

### 8.1 目录结构

```
ops/
├── __init__.py          # 模块初始化
├── utils.py             # 工具函数 + 冻结参数
├── alerts.py            # 告警系统 (本地/Telegram/飞书)
├── daily_monitor.py     # 每日监控核心逻辑
└── run_scheduler.py     # APScheduler 调度器

scripts/
├── paper_daily.sh       # 每日监控脚本
└── paper_monthly.sh     # 每月调仓脚本

results_paper/
├── daily/<YYYY-MM-DD>/  # 每日监控产物
│   ├── daily_kpi.json   # 风险指标
│   ├── alerts.md        # 告警报告
│   ├── decision.txt     # 决策 (HOLD/DE-RISK/EXIT)
│   └── params.json      # 参数快照
├── monthly/<YYYY-MM>/   # 月度调仓产物
│   └── <run_id>/
│       ├── params.json  # 参数快照
│       ├── gate/        # Gate v2 结果
│       ├── redteam/     # RedTeam 结果
│       └── paper_monthly_report.md  # 月度报告
├── equity_curve.csv     # 累计净值曲线
└── _runs/               # 调度器运行记录
```

### 8.2 每日监控 (Daily Monitor)

**运行方式**:
```bash
# 手动运行
bash scripts/paper_daily.sh

# 指定日期
bash scripts/paper_daily.sh --date 2026-02-15
```

**决策输出**:
| 决策 | 说明 | 退出码 |
|------|------|--------|
| HOLD | 继续运行 | 0 |
| DE-RISK | 降低仓位 | 10 |
| EXIT | 暂停策略 | 20 |

**DE-RISK 触发条件**:
- 20日回撤 > 15%
- 连续负收益天数 >= 10天

**EXIT 触发条件**:
- R4 最大回撤 > 25% / R5 最大回撤 > 30%
- 2个月累计收益 < -5%
- 3个月滚动夏普 < 1.0

### 8.3 每月调仓 (Monthly Rebalance)

**运行方式**:
```bash
# 手动运行
bash scripts/paper_monthly.sh
```

**产物清单**:
- `params.json` - 参数快照
- `lag_sensitivity.csv` - Lag 敏感性分析
- `capacity_clip_report.csv` - 容量裁剪报告
- `industry_clip_report.csv` - 行业裁剪报告
- `turnover_clip_report.json` - 换手裁剪报告
- `prod_acceptance_report.md` - 验收报告
- `paper_monthly_report.md` - 月度报告

### 8.4 定时调度 (Cron Scheduler)

**运行方式**:
```bash
# 启动 cron 调度器
python ops/run_scheduler.py --mode cron

# 手动执行特定任务
python ops/run_scheduler.py --mode manual --job daily
python ops/run_scheduler.py --mode manual --job monthly
python ops/run_scheduler.py --mode manual --job both
```

**调度时间**:
| 任务 | 时间 | 说明 |
|------|------|------|
| 每日监控 | 18:00 | 每日运行 |
| 每月调仓 | 每月1日 18:30 | 无法获取交易日历，使用固定日期 |

**依赖安装**:
```bash
pip install apscheduler
```

### 8.5 告警配置

**环境变量**:
```bash
# Telegram 告警
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"

# 飞书告警
export FEISHU_WEBHOOK_URL="your_webhook_url"
```

**告警级别**:
| 级别 | 颜色 | 触发条件 |
|------|------|----------|
| INFO | 蓝色 | 决策 = HOLD |
| WARNING | 黄色 | 决策 = DE-RISK |
| CRITICAL | 红色 | 决策 = EXIT |

---

## 9. 历史验收记录

| Run ID | 日期 | 环境 | Gate 结果 |
|--------|------|------|-----------|
| redteam_20260215_224410 | 2026-02-15 | macstudio@192.168.0.88 | R4/R5 10/10 PASS |

---

*文档生成时间: 2026-02-15*
