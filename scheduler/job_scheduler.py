"""
定时任务调度模块 - 基于APScheduler实现每日自动数据采集
"""
import time
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from config.settings import SCHEDULE_HOUR, SCHEDULE_MINUTE, LOG_DIR
from modules.data_collector import collector
from modules.stock_pool import pool_manager


# 配置日志
logger.add(
    LOG_DIR / "scheduler_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO"
)


class StockScheduler:
    """股票数据定时调度器"""

    def __init__(self):
        self.scheduler = BackgroundScheduler(
            timezone="Asia/Shanghai",
            job_defaults={
                "coalesce": True,  # 合并错过的任务
                "max_instances": 1,  # 同一任务最多同时运行1个实例
            }
        )

    def start(self):
        """启动调度器"""
        # 每日收盘后采集数据 (默认16:00)
        self.scheduler.add_job(
            func=self.daily_collection_job,
            trigger=CronTrigger(hour=SCHEDULE_HOUR, minute=SCHEDULE_MINUTE),
            id="daily_collection",
            name="每日数据采集",
            replace_existing=True,
        )

        # 每30分钟同步一次实时行情 (仅交易时段)
        self.scheduler.add_job(
            func=self.realtime_sync_job,
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour="9-11,13-14",
                minute="*/30"
            ),
            id="realtime_sync",
            name="实时行情同步",
            replace_existing=True,
        )

        # 每日凌晨2点做数据清理
        self.scheduler.add_job(
            func=self.cleanup_job,
            trigger=CronTrigger(hour=2, minute=0),
            id="daily_cleanup",
            name="数据清理",
            replace_existing=True,
        )

        self.scheduler.start()
        logger.info(f"定时调度器已启动 - 每日{SCHEDULE_HOUR:02d}:{SCHEDULE_MINUTE:02d}采集数据")
        self._print_jobs()

    def stop(self):
        """停止调度器"""
        self.scheduler.shutdown(wait=False)
        logger.info("定时调度器已停止")

    def _print_jobs(self):
        """打印所有任务"""
        for job in self.scheduler.get_jobs():
            logger.info(f"  任务: {job.name} | 下次执行: {job.next_run_time}")

    def daily_collection_job(self):
        """每日数据采集任务"""
        logger.info("=" * 50)
        logger.info("开始每日数据采集...")
        start = time.time()

        try:
            # 获取股票池代码
            codes = pool_manager.get_codes()

            if not codes:
                logger.warning("股票池为空，跳过数据采集")
                return

            # 同步日线数据 (最近10天，确保不遗漏)
            count = collector.sync_daily_data(codes, days_back=10)
            logger.info(f"日线数据采集完成: {count} 条")

            # 同步实时行情
            rt_count = collector.sync_realtime_data(codes)
            logger.info(f"实时行情同步完成: {rt_count} 条")

            elapsed = time.time() - start
            logger.info(f"每日数据采集完成，耗时 {elapsed:.1f}秒")
            logger.info("=" * 50)

        except Exception as e:
            logger.error(f"每日数据采集失败: {e}", exc_info=True)

    def realtime_sync_job(self):
        """实时行情同步任务"""
        try:
            codes = pool_manager.get_codes()
            if codes:
                collector.sync_realtime_data(codes)
        except Exception as e:
            logger.error(f"实时行情同步失败: {e}")

    def cleanup_job(self):
        """数据清理任务 - 清理超过90天的日志"""
        import os
        from pathlib import Path

        cutoff = datetime.now().timestamp() - 90 * 24 * 3600
        for f in LOG_DIR.glob("*.log"):
            if f.stat().st_mtime < cutoff:
                f.unlink()
                logger.info(f"清理旧日志: {f.name}")

    def run_now(self, job_id: str = "daily_collection"):
        """立即执行指定任务"""
        job = self.scheduler.get_job(job_id)
        if job:
            job.func()
        else:
            logger.warning(f"任务 {job_id} 不存在")

    def get_status(self) -> dict:
        """获取调度器状态"""
        jobs_info = []
        for job in self.scheduler.get_jobs():
            jobs_info.append({
                "id": job.id,
                "name": job.name,
                "next_run": str(job.next_run_time) if job.next_run_time else "N/A",
            })

        return {
            "running": self.scheduler.running,
            "jobs": jobs_info,
        }


# 单例
stock_scheduler = StockScheduler()


if __name__ == "__main__":
    from utils.database import init_database
    init_database()

    # 测试：立即执行一次采集
    stock_scheduler.start()
    stock_scheduler.run_now("daily_collection")

    # 保持运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stock_scheduler.stop()
