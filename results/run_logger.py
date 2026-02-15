"""
运行日志与审计模块
================================================================================
包含:
- run_id生成
- 配置快照
- 结果归档
- 审计日志
================================================================================
"""
import os
import json
import hashlib
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class RunLogger:
    """运行日志器"""

    def __init__(self, base_dir: str = None):
        """
        Args:
            base_dir: 结果存储目录
        """
        self.base_dir = base_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'results'
        )
        self.run_id = None
        self.run_dir = None
        self.config_snapshot = {}
        self.data_version = {}
        self.results = {}

    def generate_run_id(self, prefix: str = 'run') -> str:
        """
        生成运行ID

        Format: {prefix}_{YYYYMMDD_HHMMSS}_{short_hash}
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        short_hash = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:6]
        return f"{prefix}_{timestamp}_{short_hash}"

    def start_run(self,
                  run_id: str = None,
                  config: Dict = None,
                  description: str = '') -> str:
        """
        开始新运行

        Args:
            run_id: 指定运行ID（可选）
            config: 配置字典
            description: 运行描述

        Returns:
            run_id
        """
        self.run_id = run_id or self.generate_run_id()
        self.run_dir = os.path.join(self.base_dir, self.run_id)

        # 创建目录
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, 'plots'), exist_ok=True)

        # 保存配置快照
        self.config_snapshot = config or {}
        self.config_snapshot['run_id'] = self.run_id
        self.config_snapshot['start_time'] = datetime.now().isoformat()
        self.config_snapshot['description'] = description

        # 记录代码版本
        self._save_code_version()

        # 保存配置
        self._save_config()

        logger.info(f"开始运行: {self.run_id}")
        return self.run_id

    def _save_code_version(self):
        """保存代码版本信息"""
        version_info = {
            'timestamp': datetime.now().isoformat(),
        }

        # Git信息
        try:
            version_info['git_commit'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            version_info['git_branch'] = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            version_info['git_status'] = subprocess.check_output(
                ['git', 'status', '--short'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except:
            version_info['git_commit'] = 'unknown'
            version_info['git_branch'] = 'unknown'

        # 文件哈希（关键文件）
        key_files = [
            'config.py',
            'strategies/v4_smart.py',
            'backtest/engine.py',
        ]

        file_hashes = {}
        for f in key_files:
            filepath = os.path.join(os.path.dirname(self.base_dir), f)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as file:
                    file_hashes[f] = hashlib.md5(file.read()).hexdigest()[:8]

        version_info['file_hashes'] = file_hashes

        self.config_snapshot['version'] = version_info

        # 保存版本文件
        version_path = os.path.join(self.run_dir, 'version.json')
        with open(version_path, 'w') as f:
            json.dump(version_info, f, indent=2)

    def _save_config(self):
        """保存配置"""
        config_path = os.path.join(self.run_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config_snapshot, f, indent=2, default=str)

    def log_data_version(self,
                        data_source: str,
                        min_date: str,
                        max_date: str,
                        n_records: int = None,
                        extra_info: Dict = None):
        """
        记录数据版本

        Args:
            data_source: 数据源名称
            min_date: 最小日期
            max_date: 最大日期
            n_records: 记录数
            extra_info: 额外信息
        """
        info = {
            'source': data_source,
            'min_date': min_date,
            'max_date': max_date,
            'n_records': n_records,
            'logged_at': datetime.now().isoformat(),
        }
        if extra_info:
            info.update(extra_info)

        self.data_version[data_source] = info

    def log_metrics(self, metrics: Dict, name: str = 'main'):
        """
        记录指标

        Args:
            metrics: 指标字典
            name: 指标集名称
        """
        if name not in self.results:
            self.results[name] = {}

        self.results[name].update(metrics)
        self.results[name]['logged_at'] = datetime.now().isoformat()

    def save_holdings(self, holdings: pd.DataFrame, name: str = 'holdings'):
        """保存持仓明细"""
        if holdings is None or len(holdings) == 0:
            return

        path = os.path.join(self.run_dir, 'data', f'{name}.csv')
        holdings.to_csv(path, index=False)
        logger.debug(f"持仓保存到: {path}")

    def save_trades(self, trades: pd.DataFrame, name: str = 'trades'):
        """保存交易明细"""
        if trades is None or len(trades) == 0:
            return

        path = os.path.join(self.run_dir, 'data', f'{name}.csv')
        trades.to_csv(path, index=False)
        logger.debug(f"交易保存到: {path}")

    def save_returns(self, returns: pd.DataFrame, name: str = 'returns'):
        """保存收益序列"""
        if returns is None or len(returns) == 0:
            return

        path = os.path.join(self.run_dir, 'data', f'{name}.csv')
        returns.to_csv(path, index=False)
        logger.debug(f"收益保存到: {path}")

    def save_exposure(self, exposure: Dict, name: str = 'exposure'):
        """保存风格/行业暴露"""
        path = os.path.join(self.run_dir, 'data', f'{name}.json')
        with open(path, 'w') as f:
            json.dump(exposure, f, indent=2, default=str)

    def end_run(self, summary: Dict = None):
        """
        结束运行

        Args:
            summary: 运行摘要
        """
        end_time = datetime.now()

        # 更新配置
        self.config_snapshot['end_time'] = end_time.isoformat()

        # 保存数据版本
        self.config_snapshot['data_version'] = self.data_version

        # 保存结果
        self.config_snapshot['results'] = self.results

        # 保存摘要
        if summary:
            self.config_snapshot['summary'] = summary

        # 重新保存配置
        self._save_config()

        # 生成摘要报告
        self._generate_summary_report()

        logger.info(f"运行结束: {self.run_id}")

    def _generate_summary_report(self):
        """生成摘要报告"""
        report_path = os.path.join(self.run_dir, 'summary.md')

        with open(report_path, 'w') as f:
            f.write(f"# 回测运行报告\n\n")
            f.write(f"**运行ID**: {self.run_id}\n\n")
            f.write(f"**时间**: {self.config_snapshot.get('start_time', '')} ~ "
                   f"{self.config_snapshot.get('end_time', '')}\n\n")

            f.write("## 配置\n\n")
            f.write("```json\n")
            f.write(json.dumps(self.config_snapshot.get('factor_weights', {}), indent=2))
            f.write("\n```\n\n")

            f.write("## 数据版本\n\n")
            for source, info in self.data_version.items():
                f.write(f"- {source}: {info.get('min_date')} ~ {info.get('max_date')}\n")

            f.write("\n## 结果\n\n")
            for name, metrics in self.results.items():
                f.write(f"### {name}\n\n")
                if isinstance(metrics, dict):
                    for k, v in metrics.items():
                        if isinstance(v, float):
                            f.write(f"- {k}: {v:.4f}\n")
                        else:
                            f.write(f"- {k}: {v}\n")
                f.write("\n")

    def get_run_path(self, filename: str = '') -> str:
        """获取运行目录路径"""
        if filename:
            return os.path.join(self.run_dir, filename)
        return self.run_dir


class RunRegistry:
    """运行注册表 - 管理所有历史运行"""

    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'results'
        )
        self.registry_file = os.path.join(self.base_dir, 'registry.json')
        self._load_registry()

    def _load_registry(self):
        """加载注册表"""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {'runs': []}

    def _save_registry(self):
        """保存注册表"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def register(self, run_id: str, config: Dict, results: Dict):
        """注册运行"""
        entry = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'config_summary': {
                'strategy': config.get('strategy', 'unknown'),
                'start_date': config.get('start_date'),
                'end_date': config.get('end_date'),
            },
            'results_summary': {
                'annual_return': results.get('annual_return'),
                'max_drawdown': results.get('max_drawdown'),
                'sharpe': results.get('sharpe'),
            },
        }

        self.registry['runs'].append(entry)
        self._save_registry()

    def list_runs(self, limit: int = 20) -> List[Dict]:
        """列出最近运行"""
        return self.registry['runs'][-limit:]

    def get_best_run(self, metric: str = 'sharpe') -> Optional[Dict]:
        """获取最佳运行"""
        if not self.registry['runs']:
            return None

        return max(
            self.registry['runs'],
            key=lambda x: x['results_summary'].get(metric, 0) or 0
        )

    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """对比多个运行"""
        comparison = []

        for run_id in run_ids:
            run_dir = os.path.join(self.base_dir, run_id)
            config_path = os.path.join(run_dir, 'config.json')

            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)

                comparison.append({
                    'run_id': run_id,
                    'strategy': config.get('strategy', 'unknown'),
                    'annual_return': config.get('results', {}).get('main', {}).get('annual_return'),
                    'sharpe': config.get('results', {}).get('main', {}).get('sharpe'),
                    'max_drawdown': config.get('results', {}).get('main', {}).get('max_drawdown'),
                })

        return pd.DataFrame(comparison)


def create_run(run_id: str = None,
              config: Dict = None,
              base_dir: str = None) -> RunLogger:
    """
    创建新运行的便捷函数

    Args:
        run_id: 运行ID
        config: 配置
        base_dir: 基础目录

    Returns:
        RunLogger实例
    """
    logger = RunLogger(base_dir)
    logger.start_run(run_id, config)
    return logger
