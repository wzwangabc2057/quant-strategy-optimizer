"""
Run Scheduler - APScheduler 定时调度器
支持 cron 模式和 manual 模式
"""
import os
import sys
import json
import logging
import argparse
import subprocess
from datetime import datetime
from typing import Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/paper_scheduler.log'),
    ]
)
logger = logging.getLogger(__name__)

# 确保日志目录存在
os.makedirs('logs', exist_ok=True)
os.makedirs('results_paper/_runs', exist_ok=True)


def run_job(job_type: str, max_retries: int = 2) -> bool:
    """
    运行任务

    Args:
        job_type: 任务类型 (daily / monthly)
        max_retries: 最大重试次数

    Returns:
        是否成功
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = f"{timestamp}_{job_type}"
    run_dir = f"results_paper/_runs/{run_id}"

    os.makedirs(run_dir, exist_ok=True)

    # 保存参数快照
    params = {
        'run_id': run_id,
        'job_type': job_type,
        'timestamp': timestamp,
        'status': 'started',
    }
    with open(f"{run_dir}/params.json", 'w') as f:
        json.dump(params, f, indent=2)

    # 选择脚本
    script = f"scripts/paper_{job_type}.sh"

    logger.info(f"开始执行任务: {job_type}, run_id: {run_id}")

    # 执行脚本（带重试）
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"执行: {script} (尝试 {attempt + 1}/{max_retries + 1})")

            result = subprocess.run(
                ['bash', script],
                capture_output=True,
                text=True,
                timeout=600,  # 10分钟超时
            )

            # 保存输出
            with open(f"{run_dir}/stdout.log", 'w') as f:
                f.write(result.stdout)
            with open(f"{run_dir}/stderr.log", 'w') as f:
                f.write(result.stderr)
            with open(f"{run_dir}/exit_code.txt", 'w') as f:
                f.write(str(result.returncode))

            if result.returncode == 0:
                logger.info(f"任务成功: {job_type}, run_id: {run_id}")
                params['status'] = 'success'
                with open(f"{run_dir}/params.json", 'w') as f:
                    json.dump(params, f, indent=2)
                return True
            else:
                logger.warning(f"任务失败: {job_type}, exit_code={result.returncode}")
                if attempt < max_retries:
                    logger.info("准备重试...")

        except subprocess.TimeoutExpired:
            logger.error(f"任务超时: {job_type}")
            with open(f"{run_dir}/stderr.log", 'a') as f:
                f.write("\nERROR: Task timeout after 600 seconds")
        except Exception as e:
            logger.error(f"任务异常: {e}")
            with open(f"{run_dir}/stderr.log", 'w') as f:
                f.write(f"ERROR: {str(e)}")

    # 所有重试都失败
    params['status'] = 'failed'
    with open(f"{run_dir}/params.json", 'w') as f:
        json.dump(params, f, indent=2)

    logger.error(f"任务最终失败: {job_type}, run_id: {run_id}")
    return False


def run_scheduler_cron():
    """以 cron 模式运行调度器"""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error("请安装 APScheduler: pip install apscheduler")
        return

    scheduler = BlockingScheduler()

    # 每日任务: 18:00
    scheduler.add_job(
        run_job,
        CronTrigger(hour=18, minute=0),
        args=['daily'],
        id='paper_daily',
        name='Paper Trading Daily Monitor',
        max_instances=1,
    )
    logger.info("已添加每日任务: 18:00")

    # 每月任务: 每月1日 18:30
    # 注意: 无法获取交易日历，使用每月1日替代
    scheduler.add_job(
        run_job,
        CronTrigger(day=1, hour=18, minute=30),
        args=['monthly'],
        id='paper_monthly',
        name='Paper Trading Monthly Rebalance',
        max_instances=1,
    )
    logger.info("已添加每月任务: 每月1日 18:30")
    logger.warning("注意: 无法获取交易日历，使用每月1日作为调仓日")

    logger.info("调度器启动...")
    scheduler.start()


def run_manual(job_type: str):
    """以手动模式运行任务"""
    logger.info(f"手动执行任务: {job_type}")
    success = run_job(job_type)
    if success:
        logger.info("任务执行成功")
    else:
        logger.error("任务执行失败")
    return success


def main():
    parser = argparse.ArgumentParser(description='Paper Trading Scheduler')
    parser.add_argument(
        '--mode',
        choices=['cron', 'manual'],
        default='manual',
        help='运行模式: cron(定时) / manual(手动)'
    )
    parser.add_argument(
        '--job',
        choices=['daily', 'monthly', 'both'],
        default='daily',
        help='手动模式下执行的任务类型'
    )

    args = parser.parse_args()

    if args.mode == 'cron':
        run_scheduler_cron()
    else:
        if args.job == 'both':
            run_manual('daily')
            run_manual('monthly')
        else:
            run_manual(args.job)


if __name__ == '__main__':
    main()
