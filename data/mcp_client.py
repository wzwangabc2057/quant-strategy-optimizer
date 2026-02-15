"""
MCP 服务客户端
通过 HTTP 调用 MCP 服务获取 A 股数据
"""
import requests
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import time


class MCPClient:
    """MCP 服务客户端"""

    def __init__(self,
                 url: str = "http://156.254.5.155:8092/mcp",
                 token: str = "lhjy.653653a5ac6d4f348932d3365abcdeca"):
        """
        初始化 MCP 客户端

        Args:
            url: MCP 服务地址
            token: 认证 Token
        """
        self.url = url
        self.token = token
        self.session_id = None
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {token}"
        }
        self._init_session()

    def _init_session(self):
        """初始化会话"""
        payload = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "quant-strategy", "version": "1.0"}
            },
            "id": 1
        }

        # 先获取 session
        resp = requests.post(self.url, headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }, json=payload)

        # 从响应头获取 session ID
        self.session_id = resp.headers.get('mcp-session-id')

        if not self.session_id:
            # 尝试从 SSE 事件中解析
            for line in resp.text.split('\n'):
                if line.startswith('mcp-session-id:'):
                    self.session_id = line.split(':', 1)[1].strip()
                    break

        if self.session_id:
            self.headers['mcp-session-id'] = self.session_id
            print(f"MCP 会话已建立: {self.session_id[:8]}...")
        else:
            print("警告: 未能获取 MCP 会话 ID")

    def _parse_sse_response(self, text: str) -> Dict:
        """解析 SSE 格式响应"""
        result = {}
        for line in text.split('\n'):
            if line.startswith('data:'):
                try:
                    result = json.loads(line[5:].strip())
                except:
                    pass
        return result

    def call_tool(self, tool_name: str, arguments: Dict) -> Any:
        """
        调用 MCP 工具

        Args:
            tool_name: 工具名称
            arguments: 参数

        Returns:
            工具返回结果
        """
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": int(time.time() * 1000) % 100000
        }

        try:
            resp = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
            data = self._parse_sse_response(resp.text)

            if 'result' in data and 'content' in data['result']:
                # 解析返回的文本内容
                content = data['result']['content']
                if isinstance(content, list):
                    # 合并所有文本
                    texts = []
                    for item in content:
                        if item.get('type') == 'text':
                            texts.append(item.get('text', ''))
                    return texts
                return content
            elif 'error' in data:
                print(f"MCP 错误: {data['error']}")
                return None
        except Exception as e:
            print(f"调用 MCP 工具失败: {e}")
            return None

    def get_daily_quotes(self,
                         ts_code: str = None,
                         start_date: str = None,
                         end_date: str = None,
                         page: int = 1,
                         page_size: int = 5000) -> List[Dict]:
        """
        获取日线行情

        Args:
            ts_code: 股票代码 (如 000001.SZ)
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            page: 页码
            page_size: 每页数量

        Returns:
            行情数据列表
        """
        args = {'page': page, 'page_size': page_size}
        if ts_code:
            args['ts_code'] = ts_code
        if start_date:
            args['start_date'] = start_date
        if end_date:
            args['end_date'] = end_date

        result = self.call_tool('tushare_daily_quote_query', args)
        if result:
            records = []
            for text in result:
                try:
                    records.append(json.loads(text))
                except:
                    pass
            return records
        return []

    def get_daily_basic(self,
                        ts_code: str = None,
                        start_date: str = None,
                        end_date: str = None,
                        page: int = 1,
                        page_size: int = 5000) -> List[Dict]:
        """
        获取每日指标 (PE, PB, 市值等)

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            指标数据列表
        """
        args = {'page': page, 'page_size': page_size}
        if ts_code:
            args['ts_code'] = ts_code
        if start_date:
            args['start_date'] = start_date
        if end_date:
            args['end_date'] = end_date

        result = self.call_tool('tushare_daily_basic_query', args)
        if result:
            records = []
            for text in result:
                try:
                    records.append(json.loads(text))
                except:
                    pass
            return records
        return []

    def get_stock_basic(self,
                        exchange: str = None,
                        list_status: str = 'L',
                        page: int = 1,
                        page_size: int = 5000) -> List[Dict]:
        """
        获取股票基本信息

        Args:
            exchange: 交易所 (SSE/SZSE/BSE)
            list_status: 上市状态 (L-上市 D-退市 P-暂停)
            page: 页码
            page_size: 每页数量

        Returns:
            股票基本信息列表
        """
        args = {'list_status': list_status, 'page': page, 'page_size': page_size}
        if exchange:
            args['exchange'] = exchange

        result = self.call_tool('tushare_stock_basic_query', args)
        if result:
            records = []
            for text in result:
                try:
                    records.append(json.loads(text))
                except:
                    pass
            return records
        return []

    def get_ths_members(self, ts_code: str) -> List[Dict]:
        """
        获取同花顺概念/行业成分

        Args:
            ts_code: 概念/行业指数代码

        Returns:
            成分股列表
        """
        result = self.call_tool('tushare_ths_member_query', {'ts_code': ts_code})
        if result:
            records = []
            for text in result:
                try:
                    records.append(json.loads(text))
                except:
                    pass
            return records
        return []

    def get_moneyflow(self,
                      start_date: str = None,
                      end_date: str = None) -> List[Dict]:
        """
        获取大盘资金流向

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            资金流向数据
        """
        args = {'page': 1, 'page_size': 1000}
        if start_date:
            args['start_date'] = start_date
        if end_date:
            args['end_date'] = end_date

        result = self.call_tool('tushare_moneyflow_hsgt_query', args)
        if result:
            records = []
            for text in result:
                try:
                    records.append(json.loads(text))
                except:
                    pass
            return records
        return []


# 单例客户端
_client = None

def get_mcp_client() -> MCPClient:
    """获取 MCP 客户端单例"""
    global _client
    if _client is None:
        _client = MCPClient()
    return _client


if __name__ == '__main__':
    # 测试
    client = get_mcp_client()

    print("测试日线行情...")
    quotes = client.get_daily_quotes(ts_code='000001.SZ', start_date='20260201', end_date='20260210')
    print(f"获取到 {len(quotes)} 条记录")
    if quotes:
        print(quotes[0])

    print("\n测试股票基本信息...")
    basic = client.get_stock_basic(exchange='SSE', page_size=10)
    print(f"获取到 {len(basic)} 条记录")
    if basic:
        print(basic[0])
