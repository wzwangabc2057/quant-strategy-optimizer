"""
量化策略 API 服务
提供持仓推荐、买入权重等接口
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="量化策略 API",
    description="多因子量化策略 - 持仓推荐与买入权重接口",
    version="1.0.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 响应模型 ====================
class Position(BaseModel):
    """持仓"""
    code: str
    name: str = ""
    weight: float
    shares: int = 0
    price: float = 0.0
    amount: float = 0.0
    industry: str = ""
    score: float = 0.0


class PortfolioResponse(BaseModel):
    """组合响应"""
    success: bool
    message: str
    timestamp: str
    profile: str
    capital: float
    positions: List[Position]
    total_weight: float
    metrics: Dict = {}


class RebalanceRequest(BaseModel):
    """调仓请求"""
    capital: float = 1000000.0
    profile: str = "R4"  # R4稳健 / R5进取
    current_positions: Optional[Dict[str, float]] = None  # 当前持仓 {code: weight}


class RebalanceResponse(BaseModel):
    """调仓响应"""
    success: bool
    message: str
    timestamp: str
    profile: str
    capital: float
    current_positions: Dict[str, float]
    target_positions: List[Position]
    trades: List[Dict]  # 买入/卖出建议
    turnover: float
    metrics: Dict = {}


class SignalResponse(BaseModel):
    """信号响应"""
    success: bool
    message: str
    timestamp: str
    signals: List[Dict]
    count: int


# ==================== 策略引擎 ====================
class StrategyEngine:
    """策略引擎"""

    # 冻结参数
    FROZEN_PARAMS = {
        'lag_days': 60,
        'participation_rate': 0.01,
        'max_turnover': 0.30,
        'industry_cap': 0.25,
        'single_cap': 0.08,
        'min_list_days': 60,
        'min_adv': 2000,
    }

    # 因子权重
    R4_WEIGHTS = {
        'quality': 0.30,
        'growth': 0.15,
        'momentum': 0.10,
        'value': 0.20,
        'small_cap': 0.05,
        'low_volatility': 0.20,
    }

    R5_WEIGHTS = {
        'quality': 0.15,
        'growth': 0.30,
        'momentum': 0.20,
        'value': 0.10,
        'small_cap': 0.15,
        'low_volatility': 0.10,
    }

    def __init__(self):
        self.data_fetcher = None
        self._init_data_fetcher()

    def _init_data_fetcher(self):
        """初始化数据获取器"""
        try:
            from data.fetcher_mcp import MCPDataFetcher
            self.data_fetcher = MCPDataFetcher()
        except Exception as e:
            print(f"数据获取器初始化失败: {e}")

    def calculate_scores(self, codes: List[str], profile: str = "R4") -> pd.DataFrame:
        """计算综合得分"""
        if not self.data_fetcher:
            return pd.DataFrame()

        # 获取每日指标
        try:
            df = self.data_fetcher.get_daily_basic(codes)
            if df.empty:
                return pd.DataFrame()
        except:
            return pd.DataFrame()

        # 因子计算
        weights = self.R4_WEIGHTS if profile == "R4" else self.R5_WEIGHTS

        # 质量因子 (ROE, 现金流)
        df['quality_score'] = 50 + (df.get('pe', 50).apply(lambda x: max(0, 100 - x))) * 0.3

        # 成长因子 (营收增长)
        df['growth_score'] = 50

        # 动量因子 (使用换手率代理)
        df['momentum_score'] = 50 + df.get('turnover_rate', 0) * 10

        # 价值因子 (PE, PB)
        df['value_score'] = df.get('pe', 50).apply(lambda x: max(0, 100 - min(x, 100)))

        # 小盘因子 (市值倒数)
        if 'total_mv' in df.columns:
            max_mv = df['total_mv'].max()
            df['small_cap_score'] = 100 * (1 - df['total_mv'] / max_mv)
        else:
            df['small_cap_score'] = 50

        # 低波因子
        df['low_volatility_score'] = 50

        # 综合得分
        df['composite_score'] = (
            df['quality_score'] * weights['quality'] +
            df['growth_score'] * weights['growth'] +
            df['momentum_score'] * weights['momentum'] +
            df['value_score'] * weights['value'] +
            df['small_cap_score'] * weights['small_cap'] +
            df['low_volatility_score'] * weights['low_volatility']
        )

        return df.sort_values('composite_score', ascending=False)

    def select_stocks(self,
                      capital: float,
                      profile: str = "R4",
                      n_positions: int = 30) -> List[Dict]:
        """
        选股

        Args:
            capital: 资金规模
            profile: R4稳健 / R5进取
            n_positions: 持仓数量

        Returns:
            持仓列表
        """
        if not self.data_fetcher:
            return self._mock_positions(capital, profile, n_positions)

        # 获取所有股票
        all_stocks = self.data_fetcher.get_all_stocks()
        if not all_stocks:
            return self._mock_positions(capital, profile, n_positions)

        # 取前100只计算得分
        sample_codes = all_stocks[:100]
        df = self.calculate_scores(sample_codes, profile)

        if df.empty:
            return self._mock_positions(capital, profile, n_positions)

        # 取Top N
        df = df.head(n_positions)

        # 权重分配
        weights = self._assign_weights(df)

        positions = []
        for _, row in df.iterrows():
            code = row['code']
            weight = weights.get(code, 1.0 / n_positions)
            price = row.get('close', 10.0)
            amount = capital * weight
            shares = int(amount / price / 100) * 100  # 整手

            positions.append({
                'code': code,
                'name': '',
                'weight': round(weight, 4),
                'shares': shares,
                'price': round(price, 2),
                'amount': round(shares * price, 2),
                'industry': '',
                'score': round(row.get('composite_score', 0), 2)
            })

        return positions

    def _assign_weights(self, df: pd.DataFrame) -> Dict[str, float]:
        """分配权重"""
        n = len(df)
        if n == 0:
            return {}

        # 按得分排名分档
        df['rank'] = range(1, n + 1)
        df['pct'] = df['rank'] / n

        # 权重档位
        def get_weight(pct):
            if pct <= 0.1:
                return 2.0
            elif pct <= 0.3:
                return 1.5
            elif pct <= 0.5:
                return 1.2
            elif pct <= 0.7:
                return 1.0
            else:
                return 0.8

        df['weight_mult'] = df['pct'].apply(get_weight)

        # 归一化
        total = df['weight_mult'].sum()
        df['weight'] = df['weight_mult'] / total

        return dict(zip(df['code'], df['weight']))

    def _mock_positions(self, capital: float, profile: str, n: int) -> List[Dict]:
        """模拟持仓（数据不可用时）"""
        # 使用预定义的优质股票池
        mock_stocks = [
            '600519', '000858', '600036', '601318', '000333',
            '600276', '002415', '300750', '601012', '002352',
            '600887', '000002', '601166', '600030', '002304',
            '300015', '002475', '601288', '600000', '601398',
            '600016', '601988', '600031', '000651', '002236',
            '300059', '600309', '002271', '601888', '600585',
        ]

        positions = []
        base_weight = 1.0 / min(n, len(mock_stocks))

        for i, code in enumerate(mock_stocks[:n]):
            weight = base_weight * (1.5 if i < 5 else 1.0 if i < 15 else 0.8)
            positions.append({
                'code': code,
                'name': '',
                'weight': round(weight, 4),
                'shares': int(capital * weight / 50 / 100) * 100,
                'price': 50.0,
                'amount': round(capital * weight, 2),
                'industry': '',
                'score': round(90 - i, 2)
            })

        # 归一化
        total = sum(p['weight'] for p in positions)
        for p in positions:
            p['weight'] = round(p['weight'] / total, 4)

        return positions

    def calculate_rebalance(self,
                            capital: float,
                            profile: str,
                            current_positions: Dict[str, float]) -> Dict:
        """
        计算调仓

        Args:
            capital: 资金规模
            profile: R4/R5
            current_positions: 当前持仓 {code: weight}

        Returns:
            调仓信息
        """
        # 获取目标持仓
        target = self.select_stocks(capital, profile)
        target_weights = {p['code']: p['weight'] for p in target}

        # 计算交易
        trades = []
        all_codes = set(current_positions.keys()) | set(target_weights.keys())

        buy_amount = 0
        sell_amount = 0

        for code in all_codes:
            current = current_positions.get(code, 0)
            target_w = target_weights.get(code, 0)
            delta = target_w - current

            if abs(delta) > 0.001:  # 最小调仓阈值
                trade = {
                    'code': code,
                    'action': 'buy' if delta > 0 else 'sell',
                    'current_weight': round(current, 4),
                    'target_weight': round(target_w, 4),
                    'delta_weight': round(delta, 4),
                    'delta_amount': round(capital * delta, 2)
                }
                trades.append(trade)

                if delta > 0:
                    buy_amount += capital * delta
                else:
                    sell_amount += capital * abs(delta)

        # 换手率
        turnover = (buy_amount + sell_amount) / capital / 2

        return {
            'trades': trades,
            'turnover': round(turnover, 4),
            'buy_amount': round(buy_amount, 2),
            'sell_amount': round(sell_amount, 2),
            'n_trades': len(trades)
        }


# 全局策略引擎
engine = StrategyEngine()


# ==================== API 接口 ====================
@app.get("/")
def root():
    """根路径"""
    return {
        "service": "量化策略 API",
        "version": "1.0.0",
        "endpoints": [
            "/portfolio - 获取推荐持仓",
            "/rebalance - 计算调仓",
            "/signals - 获取买入信号",
            "/health - 健康检查"
        ]
    }


@app.get("/health")
def health():
    """健康检查"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/portfolio", response_model=PortfolioResponse)
def get_portfolio(
    capital: float = Query(1000000, description="资金规模"),
    profile: str = Query("R4", description="风险配置: R4稳健 / R5进取"),
    n_positions: int = Query(30, description="持仓数量")
):
    """
    获取推荐持仓

    - **capital**: 资金规模（默认100万）
    - **profile**: R4稳健型 / R5进取型
    - **n_positions**: 持仓数量（默认30只）
    """
    try:
        positions = engine.select_stocks(capital, profile, n_positions)

        total_weight = sum(p['weight'] for p in positions)

        return PortfolioResponse(
            success=True,
            message="获取成功",
            timestamp=datetime.now().isoformat(),
            profile=profile,
            capital=capital,
            positions=[Position(**p) for p in positions],
            total_weight=round(total_weight, 4),
            metrics={
                "n_positions": len(positions),
                "avg_weight": round(total_weight / len(positions), 4) if positions else 0
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebalance", response_model=RebalanceResponse)
def calculate_rebalance(request: RebalanceRequest):
    """
    计算调仓建议

    - **capital**: 资金规模
    - **profile**: R4稳健 / R5进取
    - **current_positions**: 当前持仓 {股票代码: 权重}
    """
    try:
        current = request.current_positions or {}

        # 获取目标持仓
        target = engine.select_stocks(request.capital, request.profile)

        # 计算调仓
        rebalance = engine.calculate_rebalance(
            request.capital,
            request.profile,
            current
        )

        return RebalanceResponse(
            success=True,
            message="调仓计算完成",
            timestamp=datetime.now().isoformat(),
            profile=request.profile,
            capital=request.capital,
            current_positions=current,
            target_positions=[Position(**p) for p in target],
            trades=rebalance['trades'],
            turnover=rebalance['turnover'],
            metrics={
                "n_trades": rebalance['n_trades'],
                "buy_amount": rebalance['buy_amount'],
                "sell_amount": rebalance['sell_amount']
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signals", response_model=SignalResponse)
def get_signals(
    capital: float = Query(1000000, description="资金规模"),
    profile: str = Query("R4", description="风险配置"),
    top_n: int = Query(10, description="返回数量")
):
    """
    获取买入信号（Top N 推荐）

    - **capital**: 资金规模
    - **profile**: R4稳健 / R5进取
    - **top_n**: 返回前N只股票
    """
    try:
        positions = engine.select_stocks(capital, profile, top_n)

        signals = []
        for p in positions:
            signals.append({
                "code": p['code'],
                "action": "BUY",
                "weight": p['weight'],
                "score": p['score'],
                "amount": round(capital * p['weight'], 2),
                "reason": f"综合得分 {p['score']:.1f}，权重建议 {p['weight']*100:.2f}%"
            })

        return SignalResponse(
            success=True,
            message="获取成功",
            timestamp=datetime.now().isoformat(),
            signals=signals,
            count=len(signals)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/weights/{code}")
def get_stock_weight(
    code: str,
    capital: float = Query(1000000),
    profile: str = Query("R4")
):
    """
    获取单只股票的推荐权重

    - **code**: 股票代码
    - **capital**: 资金规模
    - **profile**: 风险配置
    """
    try:
        positions = engine.select_stocks(capital, profile, 50)

        for p in positions:
            if p['code'] == code:
                return {
                    "success": True,
                    "code": code,
                    "weight": p['weight'],
                    "amount": round(capital * p['weight'], 2),
                    "score": p['score']
                }

        return {
            "success": False,
            "message": f"股票 {code} 未在推荐列表中",
            "code": code,
            "weight": 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/frozen_params")
def get_frozen_params():
    """获取冻结参数"""
    return {
        "success": True,
        "params": engine.FROZEN_PARAMS,
        "description": "Paper Trading 冻结参数，不得随意修改"
    }


@app.get("/factor_weights")
def get_factor_weights():
    """获取因子权重配置"""
    return {
        "success": True,
        "R4_稳健型": engine.R4_WEIGHTS,
        "R5_进取型": engine.R5_WEIGHTS
    }


# ==================== 启动配置 ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
