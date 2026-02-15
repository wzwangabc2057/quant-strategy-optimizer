# 多因子量化策略优化器

独立项目，专注于策略迭代优化和版本对比。

## 项目结构

```
quant-strategy-optimizer/
├── strategies/           # 策略版本
│   ├── v1_benchmark.py   # 基准版本 (原始等权)
│   ├── v2_basic.py       # 基础优化 (因子加权)
│   ├── v3_aggressive.py  # 激进版 (非线性倾斜)
│   └── v4_smart.py       # 智能版 (ATR止损+波动率调整)
├── backtest/            # 回测模块
│   ├── engine.py        # 回测引擎
│   └── metrics.py       # 指标计算
├── factors/             # 因子模块
│   ├── momentum.py      # 动量因子
│   ├── reversal.py      # 反转因子
│   ├── quality.py       # 质量因子
│   └── growth.py        # 成长因子
├── data/               # 数据模块
│   └── fetcher.py      # 数据获取
├── results/            # 回测结果
│   └── comparison.csv  # 版本对比
├── run_all.py          # 运行所有版本对比
└── config.py           # 全局配置
```

## 版本对比

| 版本 | R4年化 | R5年化 | R4夏普 | R5夏普 | 核心优化 |
|------|--------|--------|--------|--------|----------|
| v1基准 | 13.07% | 14.20% | 0.78 | 0.77 | 原始等权 |
| v2基础 | 17.23% | 18.10% | 0.99 | 0.96 | 因子加权 |
| v3激进 | 25.08% | 25.95% | 1.36 | 1.28 | 非线性倾斜 |
| v4智能 | 33.43% | 36.40% | 2.65 | 2.64 | ATR止损+波动率调整 |

## 快速开始

```bash
# 运行所有版本对比
python run_all.py

# 运行单个版本
python strategies/v4_smart.py
```

## 依赖

- pandas
- numpy
- clickhouse-connect
- akshare (可选，用于在线数据)

## 数据源

- ClickHouse (本地): 192.168.0.88:8123
- 股票数据: stock_data_qfq
- 财务数据: stock_financial
