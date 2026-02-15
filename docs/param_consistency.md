# Parameter Consistency Check

> **检查日期**: 2026-02-15
> **检查范围**: config.py, run_all_v2.py, docs/ops_paper.md

---

## 1. 冻结参数 vs config.py 默认值

| 参数 | ops_paper.md 冻结值 | config.py 默认值 | 一致性 |
|------|---------------------|------------------|--------|
| lag_days | 60 | `DEFAULT_LAG_DAYS = 60` | ✅ 一致 |
| participation_rate | 0.01 | `participation_rate_default = 0.01` | ✅ 一致 |
| max_turnover | 0.30 | `max_turnover = 0.30` | ✅ 一致 |
| industry_cap (R4) | 0.25 | `max_industry_weight = 0.25` | ✅ 一致 |
| single_cap (R4) | 0.08 | `max_single_weight = 0.08` | ✅ 一致 |
| capital | 1,000,000 | `paper_capital = 1_000_000` | ✅ 一致 |

**config.py 一致性结论: 全部一致 ✅**

---

## 2. 冻结参数 vs run_all_v2.py CLI 默认值

| 参数 | ops_paper.md 冻结值 | CLI 默认值 | 一致性 |
|------|---------------------|------------|--------|
| --lag-days | 60 | `default=60` | ✅ 一致 |
| --participation-rate | 0.01 | `default=0.01` | ✅ 一致 |
| --max-turnover | 0.30 | `default=0.30` | ✅ 一致 |
| --industry-cap | 0.25 | `default=0.25` | ✅ 一致 |
| --single-cap | 0.08 | `default=0.08` | ✅ 一致 |
| --capital | 1,000,000 | `default=1_000_000` | ✅ 一致 |
| --min-list-days | 60 | `default=60` | ✅ 一致 |
| --min-adv | 2000 | `default=2000` | ✅ 一致 |

**run_all_v2.py 一致性结论: 全部一致 ✅**

---

## 3. 运行时参数记录

### params.json 输出检查

run_all_v2.py 运行后会生成 `params.json`，包含以下字段:

| 字段 | 是否记录 | 来源 |
|------|----------|------|
| lag_days | ✅ 是 | CLI 参数 |
| lag_mode | ✅ 是 | CLI 参数 |
| participation_rate | ✅ 是 | CLI 参数 |
| max_turnover | ✅ 是 | CLI 参数 |
| industry_cap | ✅ 是 | CLI 参数 |
| single_cap | ✅ 是 | CLI 参数 |
| capital | ✅ 是 | CLI 参数 |
| min_list_days | ✅ 是 | CLI 参数 |
| min_adv | ✅ 是 | CLI 参数 |

**params.json 记录完整性: ✅ 完整**

### assumptions.json 输出检查

| 字段 | 是否记录 | 来源 |
|------|----------|------|
| lag_days | ✅ 是 | EXECUTION_CONFIG |
| lag_mode | ✅ 是 | CLI 参数 |
| participation_rate | ✅ 是 | EXECUTION_CONFIG |
| max_turnover | ✅ 是 | EXECUTION_CONFIG |
| industry_cap | ✅ 是 | GOVERNANCE_CONFIG |
| single_cap | ✅ 是 | GOVERNANCE_CONFIG |

**assumptions.json 记录完整性: ✅ 完整**

---

## 4. 不一致项修复记录

| 项目 | 问题 | 修复动作 | 状态 |
|------|------|----------|------|
| 无 | - | - | ✅ 无需修复 |

---

## 5. 参数一致性结论

```
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   ✅ 参数一致性检查通过                                              ║
║                                                                      ║
║   - config.py 默认值与冻结参数一致                                   ║
║   - run_all_v2.py CLI 默认值与冻结参数一致                           ║
║   - params.json / assumptions.json 完整记录所有参数                  ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

*检查完成时间: 2026-02-15*
