"""
数据获取模块 - MCP 版本
通过 MCP 服务获取 A 股数据，替代 ClickHouse
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mcp_client import get_mcp_client


class MCPDataFetcher:
    """基于 MCP 服务的数据获取器"""

    # 交易所代码映射
    EXCHANGE_MAP = {
        'SH': 'SSE',   # 上海
        'SZ': 'SZSE',  # 深圳
        'BJ': 'BSE',   # 北交所
    }

    def __init__(self):
        """初始化数据获取器"""
        self.client = get_mcp_client()
        self._stock_basic_cache = None
        self._industry_cache = None

    def _convert_code(self, code: str) -> str:
        """
        转换股票代码格式
        000001 -> 000001.SZ
        """
        code = str(code).zfill(6)
        if code.startswith(('60', '68')):
            return f"{code}.SH"
        elif code.startswith(('00', '30')):
            return f"{code}.SZ"
        elif code.startswith(('82', '83', '87', '88')):
            return f"{code}.BJ"
        return f"{code}.SZ"  # 默认深圳

    def _strip_code(self, ts_code: str) -> str:
        """
        去掉代码后缀
        000001.SZ -> 000001
        """
        return ts_code.split('.')[0] if '.' in ts_code else ts_code

    def get_prices(self, codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取复权价格数据

        Args:
            codes: 股票代码列表 (如 ['000001', '000002'])
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            DataFrame with columns: [date, code, close, volume]
        """
        start_dt = start_date.replace('-', '')
        end_dt = end_date.replace('-', '')

        all_records = []

        # 批量获取数据
        for code in codes:
            ts_code = self._convert_code(code)
            quotes = self.client.get_daily_quotes(
                ts_code=ts_code,
                start_date=start_dt,
                end_date=end_dt
            )

            for q in quotes:
                if q.get('tradeDate'):
                    all_records.append({
                        'date': q['tradeDate'].replace('-', '') if '-' in q.get('tradeDate', '') else q.get('tradeDate', ''),
                        'code': self._strip_code(q.get('tsCode', '')),
                        'close': float(q.get('close', 0)),
                        'volume': float(q.get('vol', 0))
                    })

        if not all_records:
            return pd.DataFrame(columns=['date', 'code', 'close', 'volume'])

        df = pd.DataFrame(all_records)
        df = df.drop_duplicates(subset=['date', 'code'], keep='last')
        df = df.sort_values(['date', 'code']).reset_index(drop=True)

        return df

    def get_financial_data(self, codes: List[str]) -> pd.DataFrame:
        """
        获取财务数据

        Args:
            codes: 股票代码列表

        Returns:
            DataFrame with columns: [code, report_date, eps, bvps, roe, revenue, net_profit]
        """
        # 使用 daily_basic 获取 PE、PB、市值等指标
        # 这些可以作为财务指标的代理
        all_records = []

        for code in codes:
            ts_code = self._convert_code(code)
            basics = self.client.get_daily_basic(ts_code=ts_code, page_size=1)

            if basics:
                b = basics[0]
                all_records.append({
                    'code': self._strip_code(b.get('tsCode', '')),
                    'report_date': b.get('tradeDate', ''),
                    'eps': float(b.get('eps', 0) or 0),
                    'bvps': float(b.get('bvps', 0) or 0),
                    'roe': float(b.get('roe', 0) or 0) if b.get('roe') else None,
                    'revenue': None,  # MCP 暂无
                    'net_profit': None,  # MCP 暂无
                })

        if not all_records:
            return pd.DataFrame({'code': codes, 'report_date': None})

        return pd.DataFrame(all_records)

    def get_daily_basic(self, codes: List[str], trade_date: str = None) -> pd.DataFrame:
        """
        获取每日指标 (PE, PB, 市值, 换手率等)

        Args:
            codes: 股票代码列表
            trade_date: 交易日期 (YYYY-MM-DD)

        Returns:
            DataFrame with daily basic metrics
        """
        all_records = []

        for code in codes:
            ts_code = self._convert_code(code)
            args = {'ts_code': ts_code, 'page_size': 1}
            if trade_date:
                args['trade_date'] = trade_date.replace('-', '')

            basics = self.client.get_daily_basic(**args)

            for b in basics:
                all_records.append({
                    'code': self._strip_code(b.get('tsCode', '')),
                    'trade_date': b.get('tradeDate', ''),
                    'close': float(b.get('close', 0) or 0),
                    'pe': float(b.get('pe', 0) or 0),
                    'pe_ttm': float(b.get('peTtm', 0) or 0),
                    'pb': float(b.get('pb', 0) or 0),
                    'ps': float(b.get('ps', 0) or 0),
                    'total_mv': float(b.get('totalMv', 0) or 0),
                    'circ_mv': float(b.get('circMv', 0) or 0),
                    'turnover_rate': float(b.get('turnoverRate', 0) or 0),
                    'volume_ratio': float(b.get('volumeRatio', 0) or 0),
                })

        return pd.DataFrame(all_records)

    def get_index_data(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取指数数据

        Args:
            index_code: 指数代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame with columns: [date, close]
        """
        # 指数代码格式化
        ts_code = self._convert_code(index_code)

        quotes = self.client.get_daily_quotes(
            ts_code=ts_code,
            start_date=start_date.replace('-', ''),
            end_date=end_date.replace('-', '')
        )

        records = []
        for q in quotes:
            if q.get('tradeDate'):
                records.append({
                    'date': q['tradeDate'],
                    'close': float(q.get('close', 0))
                })

        return pd.DataFrame(records)

    def get_industry_classification(self, codes: List[str] = None) -> pd.DataFrame:
        """
        获取行业分类数据

        Args:
            codes: 股票代码列表

        Returns:
            DataFrame with columns: [code, industry_name]
        """
        # 使用缓存的行业数据
        if self._industry_cache is not None:
            if codes:
                return self._industry_cache[self._industry_cache['code'].isin(codes)]
            return self._industry_cache

        # 通过股票基本信息获取行业
        all_stocks = self.client.get_stock_basic(page_size=10000)

        records = []
        for stock in all_stocks:
            if stock.get('industry'):
                records.append({
                    'code': self._strip_code(stock.get('tsCode', '')),
                    'industry_name': stock.get('industry', '')
                })

        if records:
            self._industry_cache = pd.DataFrame(records)
            if codes:
                return self._industry_cache[self._industry_cache['code'].isin(codes)]
            return self._industry_cache

        return pd.DataFrame(columns=['code', 'industry_name'])

    def get_stock_listing_date(self, codes: List[str] = None) -> pd.DataFrame:
        """
        获取股票上市日期

        Args:
            codes: 股票代码列表

        Returns:
            DataFrame with columns: [code, list_date]
        """
        # 使用缓存的股票基本信息
        if self._stock_basic_cache is None:
            self._stock_basic_cache = self.client.get_stock_basic(page_size=10000)

        records = []
        for stock in self._stock_basic_cache:
            if stock.get('listDate'):
                code = self._strip_code(stock.get('tsCode', ''))
                if codes is None or code in codes:
                    records.append({
                        'code': code,
                        'list_date': pd.to_datetime(stock['listDate'])
                    })

        return pd.DataFrame(records)

    def get_all_stocks(self, list_status: str = 'L') -> List[str]:
        """
        获取所有股票代码

        Args:
            list_status: 上市状态 (L-上市)

        Returns:
            股票代码列表
        """
        if self._stock_basic_cache is None:
            self._stock_basic_cache = self.client.get_stock_basic(list_status=list_status, page_size=10000)

        return [self._strip_code(s.get('tsCode', '')) for s in self._stock_basic_cache if s.get('tsCode')]

    def get_moneyflow(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        获取大盘资金流向

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            资金流向 DataFrame
        """
        start_dt = start_date.replace('-', '') if start_date else None
        end_dt = end_date.replace('-', '') if end_date else None

        data = self.client.get_moneyflow(start_date=start_dt, end_date=end_dt)

        if not data:
            return pd.DataFrame()

        records = []
        for d in data:
            records.append({
                'date': d.get('tradeDate', ''),
                'buy_elg_vol': float(d.get('buyElgVol', 0) or 0),  # 超大单买入量
                'sell_elg_vol': float(d.get('sellElgVol', 0) or 0),  # 超大单卖出量
                'net_mf_vol': float(d.get('netMfVol', 0) or 0),  # 净流入量
            })

        return pd.DataFrame(records)

    def load_portfolio(self, file_path: str) -> dict:
        """加载持仓文件"""
        r4 = pd.read_excel(file_path, sheet_name='稳健型股票组合(风险等级R4)')
        r5 = pd.read_excel(file_path, sheet_name='进取型股票组合(风险等级R5)')

        r4.columns = r5.columns = ['code', 'weight']
        r4['code'] = r4['code'].apply(lambda x: str(x).split('.')[0])
        r5['code'] = r5['code'].apply(lambda x: str(x).split('.')[0])

        r4['weight'] = r4['weight'] / r4['weight'].sum()
        r5['weight'] = r5['weight'] / r5['weight'].sum()

        return {'r4': r4, 'r5': r5}


# 兼容旧接口
class DataFetcher(MCPDataFetcher):
    """数据获取器 - 兼容旧接口"""
    pass


def get_price_pivot(price_df: pd.DataFrame) -> pd.DataFrame:
    """将价格数据转为宽表"""
    return price_df.pivot(index='date', columns='code', values='close')


if __name__ == '__main__':
    print("测试 MCP 数据获取器...")

    fetcher = MCPDataFetcher()

    # 测试价格获取
    print("\n1. 测试价格获取...")
    prices = fetcher.get_prices(['000001', '000002'], '2026-02-01', '2026-02-10')
    print(f"获取到 {len(prices)} 条价格记录")
    if len(prices) > 0:
        print(prices.head())

    # 测试股票列表
    print("\n2. 测试股票列表...")
    stocks = fetcher.get_all_stocks()
    print(f"获取到 {len(stocks)} 只股票")

    # 测试行业分类
    print("\n3. 测试行业分类...")
    industry = fetcher.get_industry_classification(['000001', '000002'])
    print(industry)
