# Universe 构建与入场逻辑设计文档

## 1. 概述

### 1.1 目标
将"静态名单"彻底替换为"历史时点可交易的动态股票池（Point-in-Time Universe）+ 自动筛选入场逻辑"。

### 1.2 核心原则
- **无幸存者偏差**: 使用PIT数据，只包含历史时点实际可交易的股票
- **一致性**: 回测与Paper Trading使用完全相同的股票池逻辑
- **可审计**: 每一步都有日志和证据输出

---

## 2. Universe 定义

### 2.1 PIT Universe (Point-in-Time Universe)

在每个历史时点 `date`，构建一个可交易股票集合：

```python
Universe(date) = {
    symbol: str,
    is_tradable: bool,
    reason_flags: List[str],  # 剔除原因
    adv20: float,             # 20日平均成交额（万元）
    list_days: int,           # 上市天数
    close: float,             # 收盘价
}
```

### 2.2 交易可执行性定义

| 过滤条件 | 参数 | 默认值 | 说明 |
|---------|------|-------|------|
| 存在行情数据 | - | 必须 | 当日有close/volume数据 |
| 成交量 > 0 | - | 必须 | 排除停牌/无成交 |
| 上市天数 >= MIN_LIST_DAYS | MIN_LIST_DAYS | 60 | 排除次新股 |
| 涨跌停检测 | LIMIT_THRESHOLD | 0.095 | 近似检测（见2.3） |
| ADV20 >= MIN_ADV_CNY | MIN_ADV_CNY | 2000万 | 流动性门槛 |

### 2.3 涨跌停近似检测

由于ClickHouse中可能没有limit字段，使用以下近似方法：

```python
def is_limit_up(date, code):
    """检测涨停"""
    ret = (close - prev_close) / prev_close
    volume_spike = volume / adv20_vol > 0.5  # 成交量异常

    # 涨幅接近10%且成交萎缩（封板特征）
    return ret > 0.095 and volume_spike < 0.3

def is_limit_down(date, code):
    """检测跌停"""
    ret = (close - prev_close) / prev_close
    volume_spike = volume / adv20_vol > 0.5

    # 跌幅接近-10%且成交萎缩（封板特征）
    return ret < -0.095 and volume_spike < 0.3
```

### 2.4 剔除原因标志 (reason_flags)

| Flag | 含义 |
|------|------|
| `NO_DATA` | 无行情数据 |
| `SUSPENDED` | 停牌/无成交 |
| `NEW_LISTING` | 上市天数不足 |
| `LIMIT_UP` | 涨停不可买入 |
| `LIMIT_DOWN` | 跌停不可卖出 |
| `LOW_LIQUIDITY` | 流动性不足 |

---

## 3. 入场/调仓流程

### 3.1 完整流程

```
┌─────────────┐
│   日期触发   │  (每月第一个交易日 / 信号事件)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Universe构建 │  → UniverseBuilder.build_universe(date)
│  (PIT数据)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  因子计算    │  → FactorCalculator.calculate(universe)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  综合打分    │  → composite_score = Σ(factor × weight)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  入场门槛    │  → EntryGate.filter(universe, scores)
│ (Entry Gate) │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  权重分配    │  → WeightAssigner.assign(passed_stocks)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  治理约束    │  → Governance.apply_constraints(weights)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  订单生成    │  → OrderGenerator.generate(target_weights)
└─────────────┘
```

### 3.2 入场门槛 (Entry Gate)

**R4 稳健型**:
```python
R4_ENTRY_GATE = {
    'composite_score_pct': 90,      # 综合分位数 >= 90%（Top 10%）
    'quality_min': 60,              # 质量因子得分 >= 60
    'momentum_min': 40,             # 动量因子得分 >= 40（允许反转）
    'excluded_if_momentum_neg': False,
}
```

**R5 进取型**:
```python
R5_ENTRY_GATE = {
    'composite_score_pct': 85,      # 综合分位数 >= 85%（Top 15%）
    'growth_min': 50,               # 成长因子得分 >= 50
    'momentum_min': 50,             # 动量因子得分 >= 50
    'small_cap_boost': True,        # 小市值加分
}
```

---

## 4. 与现有模块的接口

### 4.1 成本模型 (CostModel)

```python
# Universe提供流动性数据用于滑点估算
def estimate_slippage(order_value, adv20):
    participation_rate = order_value / (adv20 * 10000)  # adv20单位：万元
    return base_slippage * (1 + participation_rate * 10)
```

### 4.2 治理模块 (Governance)

```python
# Governance约束应用于EntryGate之后的权重
target_weights = entry_logic.select_stocks(universe, scores)
constrained_weights = governance.apply(
    target_weights,
    constraints=['single_stock', 'industry', 'turnover']
)
```

### 4.3 验证框架 (Validation)

```python
# Walk-Forward验证时，每个fold使用该时点的Universe
for train_dates, test_dates in validator.split(dates):
    universe = builder.build_universe_range(train_dates[0], train_dates[-1])
    # 训练/测试...
```

### 4.4 红队审计 (RedTeam)

```python
# 新增Universe审计项
auditor.audit_universe(
    universe_builder,
    dates=sample_dates,
    output_evidence=True
)
```

---

## 5. 关键参数配置

### 5.1 Universe参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `MIN_LIST_DAYS` | 60 | 最小上市天数 |
| `MIN_ADV_CNY` | 2000万 | 最小20日成交额 |
| `LIMIT_THRESHOLD` | 0.095 | 涨跌停检测阈值 |
| `MAX_POSITIONS` | 50 | 最大持仓数量 |

### 5.2 调仓参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `REBALANCE_FREQ` | 'monthly' | 调仓频率 |
| `PARTICIPATION_RATE` | 0.10 | 最大参与率 |
| `TURNOVER_LIMIT` | 0.30 | 单次换手上限 |

### 5.3 入场门槛参数

| 参数 | R4 | R5 | 说明 |
|------|----|----|------|
| `composite_score_pct` | 90 | 85 | 综合分位数门槛 |
| `quality_min` | 60 | - | 质量因子最低分 |
| `growth_min` | - | 50 | 成长因子最低分 |
| `momentum_min` | 40 | 50 | 动量因子最低分 |

---

## 6. 数据依赖

### 6.1 ClickHouse 表结构

**stock_data_qfq** (复权行情):
```sql
SELECT date, code, close, vol, high, low
FROM stock_data_qfq
WHERE date = '{date}'
```

**stock_basic** (基础信息，可选):
```sql
SELECT code, list_date, industry
FROM stock_basic
```

### 6.2 缺失数据说明

| 数据 | 状态 | 处理方式 |
|------|------|---------|
| `announce_date` | 待确认 | 默认使用report_date + 45天 |
| 退市股票数据 | 待确认 | 使用现有数据中存在的记录 |
| 行业分类 | 待确认 | 暂不约束，或使用外部数据 |

---

## 7. 审计与日志

### 7.1 Universe日志

每次构建输出：
- 日期
- 总股票数
- 可交易股票数
- 剔除原因分布
- ADV分布统计

### 7.2 证据文件

```
results/<run_id>/universe/
├── universe_stats.csv         # 每日Universe统计
├── exclusion_reasons.csv      # 剔除原因汇总
├── sample_20240101.csv        # 抽样日期证据
└── entry_gate_log.csv         # 入场门槛日志
```

---

## 8. 版本兼容

### 8.1 向后兼容

```python
# 如果传入外部名单，发出警告但继续使用
if external_portfolio is not None:
    logger.warning("使用外部名单，存在幸存者偏差风险")
    # 标记为HIGH风险
    use_dynamic_universe = False
else:
    use_dynamic_universe = True
```

### 8.2 版本对比

| 特性 | 旧版(Excel名单) | 新版(PIT Universe) |
|------|----------------|-------------------|
| 幸存者偏差 | HIGH风险 | PASS |
| PIT一致性 | 否 | 是 |
| 自动化程度 | 手动更新 | 全自动 |
| 可审计性 | 低 | 高 |

---

## 9. 待确认事项 (Checklist)

- [ ] ClickHouse 中是否有 `announce_date` 字段？
- [ ] 退市股票数据是否可用？
- [ ] 行业分类数据是否有？
- [ ] 涨跌停字段是否可用？
- [ ] 实际资金规模确认（用于流动性约束）

---

*文档版本: v1.0*
*更新时间: 2026-02-15*
