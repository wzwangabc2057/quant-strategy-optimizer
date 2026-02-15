"""
Ops Utilities - 通用工具函数
"""
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

# 冻结参数
FROZEN_PARAMS = {
    'mode': 'dynamic',
    'lag_days': 60,
    'participation_rate': 0.01,
    'max_turnover': 0.30,
    'industry_cap': 0.25,
    'single_cap': 0.08,
    'min_list_days': 60,
    'min_adv': 2000,
    'rebalance_freq': 'monthly',
    'capital': 1_000_000,
}


def get_trading_date(offset_days: int = 0) -> str:
    """
    获取交易日日期（简化版：使用工作日）

    Args:
        offset_days: 偏移天数

    Returns:
        日期字符串 (YYYY-MM-DD)
    """
    from pandas.tseries.offsets import BDay

    target = datetime.now() + timedelta(days=offset_days)
    # 转换为最近的交易日（工作日）
    target = pd.Timestamp(target) + BDay(0)
    return target.strftime('%Y-%m-%d')


def load_params(params_path: str) -> Dict[str, Any]:
    """加载参数文件"""
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            return json.load(f)
    return {}


def save_params(params_path: str, params: Dict[str, Any]):
    """保存参数文件"""
    os.makedirs(os.path.dirname(params_path), exist_ok=True)
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2, default=str)


def generate_run_id(job_type: str = 'monthly') -> str:
    """生成运行ID"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{job_type}_{timestamp}"


def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def write_json(data: Dict, path: str):
    """写入JSON文件"""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def read_json(path: str) -> Dict:
    """读取JSON文件"""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def append_to_csv(df: pd.DataFrame, path: str):
    """追加数据到CSV（用于累计净值曲线）"""
    ensure_dir(os.path.dirname(path))
    header = not os.path.exists(path)
    df.to_csv(path, mode='a', header=header, index=False)


def calculate_drawdown(equity_series: pd.Series) -> pd.Series:
    """计算回撤序列"""
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    return drawdown


def calculate_rolling_sharpe(returns: pd.Series, window: int = 20) -> float:
    """计算滚动夏普比率（近似）"""
    if len(returns) < window:
        return 0.0
    rolling_ret = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    # 年化因子（假设日频）
    sharpe = (rolling_ret.iloc[-1] * 252) / (rolling_std.iloc[-1] * (252 ** 0.5))
    return float(sharpe) if pd.notna(sharpe) else 0.0
