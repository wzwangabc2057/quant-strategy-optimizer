"""
Alerts Module - 告警模块
支持本地日志、Telegram、飞书
"""
import os
import json
import logging
import requests
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class AlertManager:
    """告警管理器"""

    def __init__(self,
                 enable_local: bool = True,
                 enable_telegram: bool = False,
                 enable_feishu: bool = False,
                 logs_dir: str = 'logs'):
        """
        初始化告警管理器

        Args:
            enable_local: 启用本地日志告警
            enable_telegram: 启用 Telegram 告警
            enable_feishu: 启用飞书告警
            logs_dir: 日志目录
        """
        self.enable_local = enable_local
        self.enable_telegram = enable_telegram
        self.enable_feishu = enable_feishu
        self.logs_dir = logs_dir

        # 从环境变量读取配置
        self.telegram_token = os.environ.get('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        self.feishu_webhook = os.environ.get('FEISHU_WEBHOOK_URL')

        # 如果配置了环境变量，自动启用
        if self.telegram_token and self.telegram_chat_id:
            self.enable_telegram = True
        if self.feishu_webhook:
            self.enable_feishu = True

        os.makedirs(logs_dir, exist_ok=True)

    def send(self,
             title: str,
             message: str,
             level: str = 'INFO',
             details: Dict[str, Any] = None) -> bool:
        """
        发送告警

        Args:
            title: 告警标题
            message: 告警消息
            level: 告警级别 (INFO / WARNING / ERROR / CRITICAL)
            details: 详细信息

        Returns:
            是否发送成功
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        full_message = f"[{level}] {title}\n{timestamp}\n\n{message}"

        if details:
            full_message += f"\n\n详情:\n```json\n{json.dumps(details, indent=2, ensure_ascii=False)}\n```"

        success = True

        # 本地告警
        if self.enable_local:
            self._send_local(title, full_message, level)

        # Telegram 告警
        if self.enable_telegram:
            if not self._send_telegram(full_message):
                success = False

        # 飞书告警
        if self.enable_feishu:
            if not self._send_feishu(title, message, level):
                success = False

        return success

    def _send_local(self, title: str, message: str, level: str):
        """本地日志告警"""
        alert_path = os.path.join(self.logs_dir, 'alerts_latest.md')
        with open(alert_path, 'w') as f:
            f.write(f"# {title}\n\n")
            f.write(f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**级别**: {level}\n\n")
            f.write(message)

        logger.info(f"告警已写入: {alert_path}")

    def _send_telegram(self, message: str) -> bool:
        """发送 Telegram 告警"""
        if not self.telegram_token or not self.telegram_chat_id:
            logger.warning("Telegram 配置缺失")
            return False

        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            response = requests.post(url, data=data, timeout=10)
            if response.status_code == 200:
                logger.info("Telegram 告警发送成功")
                return True
            else:
                logger.error(f"Telegram 告警发送失败: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Telegram 告警发送异常: {e}")
            return False

    def _send_feishu(self, title: str, message: str, level: str) -> bool:
        """发送飞书告警"""
        if not self.feishu_webhook:
            logger.warning("飞书 Webhook 配置缺失")
            return False

        try:
            color_map = {
                'INFO': 'blue',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red',
            }

            payload = {
                "msg_type": "interactive",
                "card": {
                    "header": {
                        "title": {
                            "tag": "plain_text",
                            "content": title
                        },
                        "template": color_map.get(level, 'blue')
                    },
                    "elements": [
                        {
                            "tag": "markdown",
                            "content": message
                        }
                    ]
                }
            }

            response = requests.post(
                self.feishu_webhook,
                json=payload,
                timeout=10
            )
            if response.status_code == 200:
                logger.info("飞书告警发送成功")
                return True
            else:
                logger.error(f"飞书告警发送失败: {response.text}")
                return False
        except Exception as e:
            logger.error(f"飞书告警发送异常: {e}")
            return False


def send_alert(title: str, message: str, level: str = 'INFO', **kwargs) -> bool:
    """快捷发送告警函数"""
    manager = AlertManager()
    return manager.send(title, message, level, kwargs)


def format_daily_alert(decision: str,
                        equity_change: float,
                        drawdown: float,
                        rolling_sharpe: float,
                        alerts: list) -> str:
    """格式化每日告警消息"""
    emoji_map = {
        'HOLD': '✅',
        'DE-RISK': '⚠️',
        'EXIT': '🚨',
    }

    emoji = emoji_map.get(decision, '📊')

    msg = f"{emoji} **每日监控报告**\n\n"
    msg += f"**决策**: {decision}\n"
    msg += f"**净值变动**: {equity_change:+.2%}\n"
    msg += f"**当前回撤**: {drawdown:.2%}\n"
    msg += f"**滚动夏普**: {rolling_sharpe:.2f}\n\n"

    if alerts:
        msg += "**触发规则**:\n"
        for alert in alerts:
            msg += f"- {alert}\n"

    return msg
