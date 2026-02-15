"""
数据获取模块
"""
import pandas as pd
import numpy as np
import clickhouse_connect
from typing import List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLICKHOUSE_HOST, CLICKHOUSE_PORT, PORTFOLIO_FILE


class DataFetcher:
    """数据获取器"""

    def __init__(self, host: str = None, port: int = None):
        self.host = host or CLICKHOUSE_HOST
        self.port = port or CLICKHOUSE_PORT
        # Use compress=False to avoid zstd issues
        self.client = clickhouse_connect.get_client(
            host=self.host,
            port=self.port,
            compress=False,
            query_limit=0
        )

    def get_prices(self, codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """获取复权价格数据"""
        codes_str = "','".join(codes)
        result = self.client.query(f"""
            SELECT toString(date) as date, code, close, vol
            FROM stock_data_qfq
            WHERE code IN ('{codes_str}')
              AND date >= '{start_date}'
              AND date <= '{end_date}'
            ORDER BY date, code
        """)
        df = pd.DataFrame(result.result_rows, columns=['date', 'code', 'close', 'volume'])
        df = df.drop_duplicates(subset=['date', 'code'], keep='last')
        return df

    def get_financial_data(self, codes: List[str]) -> pd.DataFrame:
        """获取财务数据"""
        codes_str = "','".join(codes)
        try:
            result = self.client.query(f"""
                SELECT code, toString(report_date) as report_date, eps, bvps,
                       CASE WHEN bvps > 0 THEN eps / bvps * 100 ELSE NULL END as roe,
                       operating_revenue, net_profit
                FROM stock_financial
                WHERE code IN ('{codes_str}')
                ORDER BY report_date DESC LIMIT 1 BY code
            """)
            return pd.DataFrame(result.result_rows,
                               columns=['code', 'report_date', 'eps', 'bvps', 'roe', 'revenue', 'net_profit'])
        except Exception as e:
            print(f"获取财务数据失败: {e}")
            return pd.DataFrame({'code': codes, 'report_date': None})

    def get_index_data(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取指数数据"""
        try:
            result = self.client.query(f"""
                SELECT toString(date) as date, close
                FROM stock_index
                WHERE code = '{index_code}'
                  AND date >= '{start_date}'
                  AND date <= '{end_date}'
                ORDER BY date
            """)
            if result.result_rows:
                return pd.DataFrame(result.result_rows, columns=['date', 'close'])
        except:
            pass
        return pd.DataFrame()

    def load_portfolio(self, file_path: str = None) -> dict:
        """加载持仓文件"""
        file_path = file_path or PORTFOLIO_FILE

        r4 = pd.read_excel(file_path, sheet_name='稳健型股票组合(风险等级R4)')
        r5 = pd.read_excel(file_path, sheet_name='进取型股票组合(风险等级R5)')

        r4.columns = r5.columns = ['code', 'weight']
        r4['code'] = r4['code'].apply(lambda x: str(x).split('.')[0])
        r5['code'] = r5['code'].apply(lambda x: str(x).split('.')[0])

        # 归一化权重
        r4['weight'] = r4['weight'] / r4['weight'].sum()
        r5['weight'] = r5['weight'] / r5['weight'].sum()

        return {'r4': r4, 'r5': r5}


def get_price_pivot(price_df: pd.DataFrame) -> pd.DataFrame:
    """将价格数据转为宽表"""
    return price_df.pivot(index='date', columns='code', values='close')
