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

    def get_bulk_prices(self, trade_dates: List[str]) -> pd.DataFrame:
        """
        批量获取多个交易日全市场价格（按日期查询，非逐只股票）

        Args:
            trade_dates: 交易日期列表 (YYYYMMDD 格式)

        Returns:
            DataFrame with columns: [date, code, close, volume]
        """
        all_records = []

        for trade_date in trade_dates:
            # 分页获取（全市场约 5400+ 只，page_size=5000 需要 2 页）
            for page in range(1, 3):
                quotes = self.client.get_daily_quotes_by_date(
                    trade_date=trade_date, page=page, page_size=5000
                )
                if not quotes:
                    break

                for q in quotes:
                    ts_code = q.get('tsCode', '')
                    if not ts_code:
                        continue
                    code = self._strip_code(ts_code)
                    if not code[:1].isdigit():
                        continue
                    trade_dt = q.get('tradeDate', '')
                    if '-' in trade_dt:
                        trade_dt = trade_dt.replace('-', '')

                    all_records.append({
                        'date': trade_dt,
                        'code': code,
                        'close': float(q.get('close', 0) or 0),
                        'volume': float(q.get('vol', 0) or 0),
                    })

                if len(quotes) < 5000:
                    break  # 最后一页不满说明没有下一页了

        if not all_records:
            return pd.DataFrame(columns=['date', 'code', 'close', 'volume'])

        df = pd.DataFrame(all_records)
        df = df.drop_duplicates(subset=['date', 'code'], keep='last')
        df = df.sort_values(['date', 'code']).reset_index(drop=True)
        return df

    def get_bulk_daily_basic(self, trade_date: str) -> pd.DataFrame:
        """
        批量获取某日全市场基本面指标

        Args:
            trade_date: 交易日期 (YYYY-MM-DD 或 YYYYMMDD)

        Returns:
            DataFrame with columns: [code, pe, pb, total_mv, circ_mv, turnover_rate, dv_ratio, ...]
        """
        trade_dt = trade_date.replace('-', '')
        records = []

        # 分页获取（全市场约 5400+ 只）
        for page in range(1, 3):
            basics = self.client.get_daily_basic_by_date(
                trade_date=trade_dt, page=page, page_size=5000
            )
            if not basics:
                break

            for b in basics:
                ts_code = b.get('tsCode', '')
                if not ts_code:
                    continue
                code = self._strip_code(ts_code)
                if not code[:1].isdigit():
                    continue

                records.append({
                    'code': code,
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
                    'dv_ratio': float(b.get('dvRatio', 0) or 0),
                })

            if len(basics) < 5000:
                break

        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)

    def get_recent_trade_dates(self, n_days: int = 120) -> List[str]:
        """
        获取最近 N 个交易日列表

        优先从 ClickHouse (192.168.0.74) stock_index 表查上证指数提取交易日，
        ClickHouse 不可用时 fallback 到 MCP 查个股日线。

        Args:
            n_days: 需要的交易日数量

        Returns:
            交易日列表 (YYYYMMDD 格式，从旧到新)
        """
        start_dt = (datetime.now() - timedelta(days=int(n_days * 1.8))).strftime('%Y-%m-%d')
        end_dt = datetime.now().strftime('%Y-%m-%d')

        # 优先: ClickHouse 查上证指数
        try:
            import clickhouse_connect
            from config import CLICKHOUSE_HOST, CLICKHOUSE_PORT
            ch = clickhouse_connect.get_client(
                host=CLICKHOUSE_HOST, port=CLICKHOUSE_PORT,
                compress=False, query_limit=0
            )
            result = ch.query(f"""
                SELECT DISTINCT toString(date) as trade_date
                FROM stock_index
                WHERE code = '000001'
                  AND date >= '{start_dt}'
                  AND date <= '{end_dt}'
                ORDER BY trade_date
            """)
            if result.result_rows:
                dates = [row[0].replace('-', '') for row in result.result_rows]
                print(f"ClickHouse 获取交易日: {len(dates)} 天")
                return dates[-n_days:] if len(dates) >= n_days else dates
        except Exception as e:
            print(f"ClickHouse 获取交易日失败，fallback 到 MCP: {e}")

        # Fallback: MCP 查个股日线
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=int(n_days * 1.8))).strftime('%Y%m%d')

        quotes = self.client.get_daily_quotes(
            ts_code='000001.SZ',
            start_date=start_date,
            end_date=end_date
        )

        dates = set()
        for q in quotes:
            trade_date = q.get('tradeDate', '')
            if trade_date:
                if '-' in trade_date:
                    trade_date = trade_date.replace('-', '')
                dates.add(trade_date)

        sorted_dates = sorted(dates)
        return sorted_dates[-n_days:] if len(sorted_dates) >= n_days else sorted_dates

    def get_income_growth(self, codes: List[str]) -> pd.DataFrame:
        """
        获取利润增长数据（拉取最新两期利润表计算 YoY）

        Args:
            codes: 股票代码列表

        Returns:
            DataFrame with columns: [code, revenue_growth, profit_growth]
        """
        records = []

        for code in codes:
            ts_code = self._convert_code(code)
            try:
                income_data = self.client.get_income_data(ts_code=ts_code, page_size=4)
                if len(income_data) >= 2:
                    # 按报告期排序，取最新两期
                    sorted_data = sorted(income_data, key=lambda x: x.get('endDate', ''), reverse=True)
                    latest = sorted_data[0]
                    prev = sorted_data[1]

                    # 计算营收增长
                    rev_latest = float(latest.get('revenue', 0) or 0)
                    rev_prev = float(prev.get('revenue', 0) or 0)
                    revenue_growth = ((rev_latest / rev_prev) - 1) * 100 if rev_prev > 0 else 0

                    # 计算利润增长
                    profit_latest = float(latest.get('nIncome', 0) or latest.get('netProfit', 0) or 0)
                    profit_prev = float(prev.get('nIncome', 0) or prev.get('netProfit', 0) or 0)
                    profit_growth = ((profit_latest / profit_prev) - 1) * 100 if profit_prev > 0 else 0

                    records.append({
                        'code': code,
                        'revenue_growth': revenue_growth,
                        'profit_growth': profit_growth,
                    })
                else:
                    records.append({'code': code, 'revenue_growth': 0, 'profit_growth': 0})
            except Exception:
                records.append({'code': code, 'revenue_growth': 0, 'profit_growth': 0})

        if not records:
            return pd.DataFrame(columns=['code', 'revenue_growth', 'profit_growth'])
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
