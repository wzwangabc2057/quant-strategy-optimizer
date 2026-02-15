"""
数据对齐与反泄漏模块
================================================================================
核心功能:
1. asof_date机制 - 财务数据只能使用已披露的数据
2. 反泄漏join - 确保signal_date < trade_date
3. 幸存者偏差处理 - 使用历史时点可得的股票池
4. 支持可配置的财务可用日延迟 (lag_days)
================================================================================
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
import json
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FINANCIAL_LAG_PRESETS, DEFAULT_LAG_DAYS

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        return super().default(obj)


class DataAlignment:
    """数据对齐器 - 防止数据泄漏"""

    # 财务报告披露期限（天数）
    DISCLOSURE_DEADLINES = {
        'Q1': 30,    # 一季报：4月30日前
        'H1': 60,    # 半年报：8月31日前
        'Q3': 30,    # 三季报：10月31日前
        'annual': 120,  # 年报：4月30日前
    }

    def __init__(self, lag_days: int = None, mode: str = 'paper'):
        """
        Args:
            lag_days: 财务可用日延迟天数（覆盖mode）
            mode: 模式 ('base'/'paper'/'stress')，使用预设值
        """
        if lag_days is not None:
            self.lag_days = lag_days
        else:
            self.lag_days = FINANCIAL_LAG_PRESETS.get(mode, DEFAULT_LAG_DAYS)

        self.mode = mode
        self.default_delay_days = self.lag_days

        logger.info(f"DataAlignment 初始化: lag_days={self.lag_days}, mode={mode}")

    def get_config(self) -> Dict:
        """获取当前配置"""
        return {
            'lag_days': self.lag_days,
            'mode': self.mode,
            'presets': FINANCIAL_LAG_PRESETS,
        }

    def save_config(self, filepath: str):
        """保存配置到文件"""
        config = self.get_config()
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2, cls=NumpyEncoder)
        logger.info(f"配置已保存: {filepath}")

    def get_asof_date(self, report_date: str, signal_date: str) -> Optional[str]:
        """
        计算因子可用日期 (asof_date)

        Args:
            report_date: 报告期 (如 '2024-03-31')
            signal_date: 信号日期 (如 '2024-05-15')

        Returns:
            如果报告期数据在信号日已披露，返回asof_date；否则返回None
        """
        report_dt = pd.to_datetime(report_date)
        signal_dt = pd.to_datetime(signal_date)

        # 判断报告类型，使用可配置的延迟
        month = report_dt.month
        if month == 3:
            disclosure_deadline = report_dt + timedelta(days=self.lag_days + 30)
        elif month == 6:
            disclosure_deadline = report_dt + timedelta(days=self.lag_days + 62)
        elif month == 9:
            disclosure_deadline = report_dt + timedelta(days=self.lag_days + 31)
        elif month == 12:
            disclosure_deadline = report_dt + timedelta(days=self.lag_days + 120)
        else:
            disclosure_deadline = report_dt + timedelta(days=self.lag_days)

        # 检查信号日是否在披露截止日之后
        if signal_dt >= disclosure_deadline:
            return disclosure_deadline.strftime('%Y-%m-%d')
        return None

    def align_financial_data(self,
                            financial_df: pd.DataFrame,
                            signal_date: str,
                            date_col: str = 'report_date') -> pd.DataFrame:
        """
        对齐财务数据 - 只保留信号日已可用的数据

        Args:
            financial_df: 财务数据 (需包含 report_date 列)
            signal_date: 信号日期
            date_col: 报告期列名

        Returns:
            对齐后的财务数据
        """
        if date_col not in financial_df.columns:
            logger.warning(f"财务数据缺少 {date_col} 列，使用默认延迟")
            return financial_df

        # 计算每条记录的可用日期
        def check_available(row):
            asof = self.get_asof_date(str(row[date_col]), signal_date)
            return asof is not None

        available_mask = financial_df.apply(check_available, axis=1)

        # 只保留已披露的数据
        aligned_df = financial_df[available_mask].copy()

        # 取最新的已披露数据
        if len(aligned_df) > 0:
            aligned_df = aligned_df.sort_values(date_col, ascending=False)
            aligned_df = aligned_df.drop_duplicates(subset=['code'], keep='first')

        logger.info(f"财务数据对齐 (lag={self.lag_days}天): 原始{len(financial_df)}条 -> 对齐后{len(aligned_df)}条")

        return aligned_df

    def validate_no_leakage(self, signal_date: str, trade_date: str) -> bool:
        """
        验证无数据泄漏

        Args:
            signal_date: 信号日期
            trade_date: 交易日期

        Returns:
            True if valid (no leakage)
        """
        signal_dt = pd.to_datetime(signal_date)
        trade_dt = pd.to_datetime(trade_date)

        # 信号日必须早于交易日
        if signal_dt >= trade_dt:
            logger.error(f"数据泄漏风险: 信号日 {signal_date} >= 交易日 {trade_date}")
            return False

        return True


class AsofJoin:
    """Asof Join - 基于时点对齐的数据合并"""

    def __init__(self, tolerance_days: int = 1):
        """
        Args:
            tolerance_days: 容忍天数差
        """
        self.tolerance_days = tolerance_days

    def merge_asof(self,
                   left_df: pd.DataFrame,
                   right_df: pd.DataFrame,
                   left_on: str,
                   right_on: str,
                   by: str = 'code') -> pd.DataFrame:
        """
        执行 asof merge

        Args:
            left_df: 左表 (通常是需要填充的数据)
            right_df: 右表 (通常是有时间戳的历史数据)
            left_on: 左表时间列
            right_on: 右表时间列
            by: 分组列

        Returns:
            合并后的DataFrame
        """
        # 确保日期格式
        left_df = left_df.copy()
        right_df = right_df.copy()

        left_df[left_on] = pd.to_datetime(left_df[left_on])
        right_df[right_on] = pd.to_datetime(right_df[right_on])

        # 排序
        left_df = left_df.sort_values([by, left_on])
        right_df = right_df.sort_values([by, right_on])

        # 执行 asof merge
        result = pd.merge_asof(
            left_df,
            right_df,
            left_on=left_on,
            right_on=right_on,
            by=by,
            tolerance=pd.Timedelta(days=self.tolerance_days),
            direction='backward'  # 向后查找，只用过去的数据
        )

        return result


class SurvivorshipBiasHandler:
    """幸存者偏差处理器"""

    def __init__(self, delisted_stocks: List[str] = None):
        """
        Args:
            delisted_stocks: 已退市股票列表
        """
        self.delisted_stocks = delisted_stocks or []

    def get_historical_universe(self,
                               all_stocks: List[str],
                               as_of_date: str,
                               listing_dates: Dict[str, str] = None,
                               delisting_dates: Dict[str, str] = None) -> List[str]:
        """
        获取历史时点的有效股票池

        Args:
            all_stocks: 所有股票列表
            as_of_date: 目标日期
            listing_dates: 上市日期字典 {code: date}
            delisting_dates: 退市日期字典 {code: date}

        Returns:
            该时点有效的股票列表
        """
        asof_dt = pd.to_datetime(as_of_date)
        listing_dates = listing_dates or {}
        delisting_dates = delisting_dates or {}

        valid_stocks = []
        excluded_new = []
        excluded_delisted = []

        for code in all_stocks:
            # 检查是否已上市
            listing_date = listing_dates.get(code)
            if listing_date and pd.to_datetime(listing_date) > asof_dt:
                excluded_new.append(code)
                continue

            # 检查是否已退市
            delisting_date = delisting_dates.get(code)
            if delisting_date and pd.to_datetime(delisting_date) <= asof_dt:
                excluded_delisted.append(code)
                continue

            valid_stocks.append(code)

        logger.info(f"股票池过滤 ({as_of_date}): "
                   f"原始{len(all_stocks)} -> 有效{len(valid_stocks)}, "
                   f"剔除新上市{len(excluded_new)}, 剔除已退市{len(excluded_delisted)}")

        return valid_stocks

    def estimate_survivorship_bias(self,
                                  portfolio_returns: pd.DataFrame,
                                  market_returns: pd.Series) -> Dict:
        """
        估算幸存者偏差影响

        Args:
            portfolio_returns: 组合收益
            market_returns: 市场收益

        Returns:
            偏差估算结果
        """
        # 简化估算：假设退市股平均损失50%
        assumed_loss_on_delisted = 0.50

        # 估算退市股占比（基于历史退市率）
        annual_delisting_rate = 0.01  # 约1%年退市率

        # 计算偏差调整
        bias_adjustment = annual_delisting_rate * assumed_loss_on_delisted

        return {
            'estimated_delisting_rate': annual_delisting_rate,
            'assumed_loss_rate': assumed_loss_on_delisted,
            'return_bias_adjustment': bias_adjustment,
            'recommendation': '回测结果应扣除此偏差'
        }


def create_aligned_factor_data(price_df: pd.DataFrame,
                               financial_df: pd.DataFrame,
                               signal_date: str,
                               aligner: DataAlignment = None) -> pd.DataFrame:
    """
    创建对齐后的因子数据

    Args:
        price_df: 价格数据
        financial_df: 财务数据
        signal_date: 信号日期
        aligner: 数据对齐器

    Returns:
        对齐后的因子数据
    """
    aligner = aligner or DataAlignment()

    # 对齐财务数据
    aligned_financial = aligner.align_financial_data(financial_df, signal_date)

    # 合并数据
    factor_df = price_df.merge(aligned_financial, on='code', how='left')

    return factor_df
