# 多因子量化策略优化器

独立项目，专注于策略迭代优化和版本对比。

## 项目结构

```
quant-strategy-optimizer/
├── strategies/           # 策略版本
│   ├── v1_benchmark.py   # 基准版本 (原始等权)
│   ├── v2_basic.py       # 基础优化 (因子加权)
│   ├── v3_aggressive.py  # 激进版 (非线性倾斜)
│   └── v4_smart.py       # 智能版 (ATR止损+波动率调整) ★
├── strategy/             # 产品化策略模块
│   ├── portfolio_strategy.py  # 组合策略核心类
│   ├── risk_control.py        # 风控模块
│   └── order_generator.py     # 订单生成模块
├── backtest/             # 回测模块
│   ├── engine.py         # 回测引擎
│   └── metrics.py        # 指标计算
├── factors/              # 因子模块
│   ├── momentum.py       # 动量因子
│   ├── reversal.py       # 反转因子
│   ├── quality.py        # 质量因子
│   └── growth.py         # 成长因子
├── data/                 # 数据模块
│   └── fetcher.py        # 数据获取
├── results/              # 回测结果
├── run_all.py            # 运行所有版本对比
├── run_production.py     # 生产环境入口 ★
└── config.py             # 全局配置
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

# 运行最佳版本 (v4智能版)
python strategies/v4_smart.py

# 运行生产环境版本
python run_production.py

# 带参数运行
python run_production.py --start 2020-01-01 --end 2025-12-31
```

## 产品化特性

### 1. 风控模块 (`strategy/risk_control.py`)

- **大盘止损**: 沪深300跌破年线15%触发
- **个股止损**: 单股跌幅超过20%自动减仓
- **移动止损**: 盈利10%后启动，从高点回撤10%触发
- **最大回撤控制**: 回撤超过15%降低仓位
- **持仓跟踪**: 实时跟踪每只股票的持仓成本、天数、止损价

```python
from strategy import RiskController, RiskConfig

config = RiskConfig(
    market_stop_loss=0.85,
    single_stock_stop_loss=0.80,
    max_drawdown_limit=0.15,
)
risk = RiskController(config)
```

### 2. 订单生成模块 (`strategy/order_generator.py`)

- **最小交易金额过滤**: 忽略小额交易
- **整手处理**: 自动处理100股整数倍
- **涨跌停检测**: 涨停不买入，跌停不卖出
- **交易成本计算**: 佣金+滑点

```python
from strategy import OrderGenerator, OrderConfig

gen = OrderGenerator(OrderConfig(
    min_trade_value=1000,
    lot_size=100,
))
sell_orders, buy_orders, summary = gen.generate_rebalance_orders(...)
```

### 3. 组合策略核心类 (`strategy/portfolio_strategy.py`)

整合因子计算、风控、订单生成:

```python
from strategy import PortfolioStrategy, StrategyConfig

config = StrategyConfig(
    name="R4",
    factor_weights={...},
    max_weight=0.08,
    enable_risk_control=True,
)
strategy = PortfolioStrategy(config)

# 执行调仓
sell_orders, buy_orders, summary = strategy.rebalance(
    portfolio=portfolio_df,
    factor_data=factor_df,
    market_prices=market_series,
    current_prices=price_dict,
    total_value=1000000,
)
```

### 4. 因子体系

| 因子类别 | 包含因子 | 数据来源 |
|---------|---------|---------|
| 质量 | ROE、ROE稳定性、现金流质量 | 财务数据 |
| 成长 | 净利润增速、营收增速、PEG | 财务数据 |
| 动量 | 多周期动量(1周/1月/3月/6月) | 价格数据 |
| 反转 | 3个月回撤、RSI反转 | 价格数据 |
| 估值 | PE、股息率 | 估值数据 |

### 5. 市场环境检测

自动检测当前市场环境并调整策略参数:

| 环境 | 触发条件 | 倾斜系数 | 最大权重 |
|------|---------|---------|---------|
| 强牛市 | 60日涨>15%, 20日涨>5%, 波动<25% | 1.4 | 12% |
| 温和牛市 | 60日涨>8%, 均线多头 | 1.2 | 10% |
| 熊市 | 60日跌>15% | 0.4 | 5% |
| 高波动 | 波动率>35% | 0.5 | 6% |
| 中性 | 其他 | 0.9 | 7% |

## 依赖

```
pandas>=1.3.0
numpy>=1.20.0
clickhouse-connect>=0.5.0
```

## 数据源

- ClickHouse (本地): 192.168.0.74:8123
- 股票数据: stock_data_qfq
- 财务数据: stock_financial

## 配置文件

`config.py` 中可配置:

```python
# 回测配置
BACKTEST_START = '2020-01-01'
BACKTEST_END = '2025-12-31'

# 交易成本
BUY_COMMISSION = 0.00026   # 买入0.026%
SELL_COMMISSION = 0.00126  # 卖出0.126% (含印花税)

# 因子权重
STABLE_WEIGHTS = {...}      # 稳健型
AGGRESSIVE_WEIGHTS = {...}  # 进取型
```

## 输出

- 回测结果保存在 `results/` 目录
- 日志输出到控制台
- 订单明细可打印

## License

Private - All rights reserved
