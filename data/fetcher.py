"""
数据获取模块
支持 MCP 服务 (推荐) 和 ClickHouse (备选)
"""
import pandas as pd
import numpy as np
from typing import List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLICKHOUSE_HOST, CLICKHOUSE_PORT, PORTFOLIO_FILE

# 数据源选择
DATA_SOURCE = os.environ.get('DATA_SOURCE', 'mcp')  # 'mcp' or 'clickhouse'


def get_fetcher(source: str = None):
    """
    获取数据获取器

    Args:
        source: 数据源 ('mcp' 或 'clickhouse')

    Returns:
        DataFetcher 实例
    """
    source = source or DATA_SOURCE

    if source == 'mcp':
        from data.fetcher_mcp import MCPDataFetcher
        return MCPDataFetcher()
    else:
        return _ClickHouseDataFetcher()


class _ClickHouseDataFetcher:
    """ClickHouse 数据获取器 (备选)"""

    def __init__(self, host: str = None, port: int = None):
        self.host = host or CLICKHOUSE_HOST
        self.port = port or CLICKHOUSE_PORT
        try:
            import clickhouse_connect
            self.client = clickhouse_connect.get_client(
                host=self.host,
                port=self.port,
                compress=False,
                query_limit=0
            )
        except Exception as e:
            print(f"ClickHouse 连接失败: {e}")
            self.client = None

    def get_prices(self, codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """获取复权价格数据"""
        if self.client is None:
            return pd.DataFrame(columns=['date', 'code', 'close', 'volume'])
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
        if self.client is None:
            return pd.DataFrame({'code': codes, 'report_date': None})
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
        if self.client is None:
            return pd.DataFrame()
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

    def get_industry_classification(self, codes: List[str] = None) -> pd.DataFrame:
        """获取行业分类数据"""
        if self.client is None:
            return pd.DataFrame(columns=['code', 'industry_name'])
        try:
            if codes:
                codes_str = "','".join(codes)
                result = self.client.query(f"""
                    SELECT DISTINCT stock_code as code, block_name as industry_name
                    FROM stock_block_em
                    WHERE block_type = 'industry'
                      AND stock_code IN ('{codes_str}')
                """)
            else:
                result = self.client.query("""
                    SELECT DISTINCT stock_code as code, block_name as industry_name
                    FROM stock_block_em
                    WHERE block_type = 'industry'
                """)

            if result.result_rows:
                return pd.DataFrame(result.result_rows, columns=['code', 'industry_name'])
        except Exception as e:
            print(f"获取行业分类失败: {e}")
        return pd.DataFrame(columns=['code', 'industry_name'])

    def get_stock_listing_date(self, codes: List[str] = None) -> pd.DataFrame:
        """获取股票上市日期"""
        if self.client is None:
            return pd.DataFrame(columns=['code', 'list_date'])
        try:
            if codes:
                codes_str = "','".join(codes)
                result = self.client.query(f"""
                    SELECT code, min(date) as list_date
                    FROM stock_data_qfq
                    WHERE code IN ('{codes_str}')
                    GROUP BY code
                """)
            else:
                result = self.client.query("""
                    SELECT code, min(date) as list_date
                    FROM stock_data_qfq
                    GROUP BY code
                """)

            if result.result_rows:
                df = pd.DataFrame(result.result_rows, columns=['code', 'list_date'])
                df['list_date'] = pd.to_datetime(df['list_date'])
                return df
        except Exception as e:
            print(f"获取上市日期失败: {e}")
        return pd.DataFrame(columns=['code', 'list_date'])

    def load_portfolio(self, file_path: str = None) -> dict:
        """加载持仓文件"""
        file_path = file_path or PORTFOLIO_FILE

        r4 = pd.read_excel(file_path, sheet_name='稳健型股票组合(风险等级R4)')
        r5 = pd.read_excel(file_path, sheet_name='进取型股票组合(风险等级R5)')

        r4.columns = r5.columns = ['code', 'weight']
        r4['code'] = r4['code'].apply(lambda x: str(x).split('.')[0])
        r5['code'] = r5['code'].apply(lambda x: str(x).split('.')[0])

        r4['weight'] = r4['weight'] / r4['weight'].sum()
        r5['weight'] = r5['weight'] / r5['weight'].sum()

        return {'r4': r4, 'r5': r5}


# 默认使用 MCP 数据源
class DataFetcher:
    """数据获取器 - 默认使用 MCP"""

    def __new__(cls):
        return get_fetcher('mcp')


def get_price_pivot(price_df: pd.DataFrame) -> pd.DataFrame:
    """将价格数据转为宽表"""
    return price_df.pivot(index='date', columns='code', values='close')
