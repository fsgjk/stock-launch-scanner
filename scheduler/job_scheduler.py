"""
定时任务调度模块 - 每日15:05自动采集+起涨点扫描
"""
import time
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import LOG_DIR
from modules.data_collector import collector
from modules.stock_pool import pool_manager

logger.add(
    LOG_DIR / "scheduler_{time:YYYY-MM-DD}.log",
    rotation="1 day", retention="30 days", level="INFO"
)


class StockScheduler:
    """股票数据定时调度器"""

    def __init__(self):
        self.scheduler = BackgroundScheduler(
            timezone="Asia/Shanghai",
            job_defaults={"coalesce": True, "max_instances": 1}
        )

    def start(self):
        # === 核心任务：每日15:05 采集+扫描 ===
        self.scheduler.add_job(
            func=self.daily_scan_job,
            trigger=CronTrigger(day_of_week="mon-fri", hour=15, minute=5),
            id="daily_scan",
            name="每日收盘采集+起涨点扫描",
            replace_existing=True,
        )

        # 每30分钟同步实时行情
        self.scheduler.add_job(
            func=self.realtime_sync_job,
            trigger=CronTrigger(day_of_week="mon-fri", hour="9-11,13-14", minute="*/30"),
            id="realtime_sync",
            name="实时行情同步",
            replace_existing=True,
        )

        # 每日凌晨2点清理
        self.scheduler.add_job(
            func=self.cleanup_job,
            trigger=CronTrigger(hour=2, minute=0),
            id="daily_cleanup",
            name="数据清理",
            replace_existing=True,
        )

        self.scheduler.start()
        logger.info("定时调度器已启动 - 每日15:05采集+扫描")
        self._print_jobs()

    def stop(self):
        self.scheduler.shutdown(wait=False)
        logger.info("定时调度器已停止")

    def _print_jobs(self):
        for job in self.scheduler.get_jobs():
            logger.info(f"  任务: {job.name} | 下次: {job.next_run_time}")

    # ========== 核心：每日采集+扫描 ==========
    def daily_scan_job(self):
        """每日15:05: 采集今日收盘数据 + 起涨点扫描"""
        logger.info("=" * 50)
        logger.info("🚀 开始每日收盘采集+起涨点扫描...")
        start = time.time()

        try:
            import akshare as ak
            import pandas as pd
            import numpy as np
            import sqlite3

            today = datetime.now().strftime('%Y-%m-%d')
            logger.info(f"日期: {today}")

            conn = sqlite3.connect(str(Path(__file__).parent.parent / "data" / "stock_system.db"))

            # 检查今日数据
            cnt = conn.execute("SELECT COUNT(*) FROM stock_daily WHERE trade_date=?", (today,)).fetchone()[0]

            if cnt == 0:
                logger.info("今日数据不存在，开始采集...")
                df_spot = ak.stock_zh_a_spot()
                df_spot['code'] = df_spot['代码'].str.replace(r'^(sh|sz|bj)', '', regex=True)
                logger.info(f"获取行情 {len(df_spot)} 只")

                codes = df_spot['code'].tolist()
                batch_size = 500
                inserted = 0

                for batch_idx in range(0, len(codes), batch_size):
                    batch_codes = codes[batch_idx:batch_idx + batch_size]
                    placeholders = ','.join(['?'] * len(batch_codes))

                    df_hist = pd.read_sql_query(f"""
                        SELECT code, trade_date, close, high, low, volume
                        FROM stock_daily WHERE code IN ({placeholders})
                        ORDER BY code, trade_date
                    """, conn, params=batch_codes)

                    spot_subset = df_spot[df_spot['code'].isin(batch_codes)]

                    rows = []
                    for _, spot in spot_subset.iterrows():
                        code = spot['code']
                        hist = df_hist[df_hist['code'] == code].sort_values('trade_date')
                        if len(hist) < 20:
                            continue
                        hist = hist.tail(120).reset_index(drop=True)

                        close = float(spot['最新价']) if pd.notna(spot.get('最新价')) else None
                        if close is None or close == 0:
                            continue

                        open_p = float(spot['今开']) if pd.notna(spot.get('今开')) else None
                        high = float(spot['最高']) if pd.notna(spot.get('最高')) else None
                        low = float(spot['最低']) if pd.notna(spot.get('最低')) else None
                        pre_close = float(spot['昨收']) if pd.notna(spot.get('昨收')) else None
                        volume = float(spot['成交量']) if pd.notna(spot.get('成交量')) else None
                        amount = float(spot['成交额']) if pd.notna(spot.get('成交额')) else None
                        pct = float(spot['涨跌幅']) if pd.notna(spot.get('涨跌幅')) else None
                        amp = float(spot['振幅']) if pd.notna(spot.get('振幅')) else None
                        turnover = float(spot['换手率']) if pd.notna(spot.get('换手率')) else None

                        closes = np.append(hist['close'].values, close)
                        highs_arr = np.append(hist['high'].values, high) if high else closes
                        lows_arr = np.append(hist['low'].values, low) if low else closes

                        def ma(s, p):
                            return float(np.mean(s[-p:])) if len(s) >= p else None

                        # 均线
                        ma5_v = ma(closes, 5)
                        ma10_v = ma(closes, 10)
                        ma20_v = ma(closes, 20)
                        ma60_v = ma(closes, 60) if len(closes) >= 60 else None
                        ma120_v = ma(closes, 120) if len(closes) >= 120 else None
                        ma250_v = ma(closes, 250) if len(closes) >= 250 else None

                        # KDJ
                        kdj_k = kdj_d = kdj_j = None
                        if len(closes) >= 12:
                            rsv_list = []
                            for i in range(8, len(closes)):
                                ln = float(np.min(lows_arr[max(0,i-8):i+1]))
                                hn = float(np.max(highs_arr[max(0,i-8):i+1]))
                                rsv_list.append((closes[i] - ln) / (hn - ln) * 100 if hn > ln else 50)
                            if rsv_list:
                                rsv_arr = np.array(rsv_list)
                                k_arr = pd.Series(rsv_arr).ewm(alpha=1/3, adjust=False).mean().values
                                d_arr = pd.Series(k_arr).ewm(alpha=1/3, adjust=False).mean().values
                                kdj_k = float(k_arr[-1])
                                kdj_d = float(d_arr[-1])
                                kdj_j = float(3*k_arr[-1] - 2*d_arr[-1])

                        # MACD
                        macd_dif = macd_dea = macd_hist = None
                        if len(closes) >= 35:
                            e12 = pd.Series(closes).ewm(span=12, adjust=False).mean().values
                            e26 = pd.Series(closes).ewm(span=26, adjust=False).mean().values
                            dif = e12 - e26
                            dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
                            macd_dif = float(dif[-1])
                            macd_dea = float(dea[-1])
                            macd_hist = float(2*(dif[-1] - dea[-1]))

                        # RSI
                        def calc_rsi(c, period):
                            if len(c) < period + 1:
                                return None
                            d = np.diff(c)
                            g = pd.Series(np.where(d>0, d, 0)).ewm(alpha=1/period, adjust=False).mean().iloc[-1]
                            l = pd.Series(np.where(d<0, -d, 0)).ewm(alpha=1/period, adjust=False).mean().iloc[-1]
                            return float(100 - 100/(1 + g/l)) if l > 0 else 100.0

                        rsi6 = calc_rsi(closes, 6)
                        rsi14 = calc_rsi(closes, 14)
                        rsi24 = calc_rsi(closes, 24)

                        # 布林带
                        boll_upper = boll_mid = boll_lower = None
                        if len(closes) >= 20:
                            boll_mid = float(np.mean(closes[-20:]))
                            boll_std = float(np.std(closes[-20:]))
                            boll_upper = boll_mid + 2*boll_std
                            boll_lower = boll_mid - 2*boll_std

                        # 量能
                        vols = np.append(hist['volume'].values, volume) if volume else hist['volume'].values
                        vol_ma5 = ma(vols, 5)
                        vol_ma10 = ma(vols, 10)
                        vol_ma20 = ma(vols, 20)
                        vol_ratio = float(volume / vol_ma5) if volume and vol_ma5 and vol_ma5 > 0 else None

                        rows.append((
                            code, today, open_p, high, low, close, pre_close, pct, amp,
                            volume, amount, turnover,
                            vol_ma5, vol_ma10, vol_ma20, vol_ratio,
                            kdj_k, kdj_d, kdj_j,
                            macd_dif, macd_dea, macd_hist,
                            ma5_v, ma10_v, ma20_v, ma60_v, ma120_v, ma250_v,
                            rsi6, rsi14, rsi24,
                            boll_upper, boll_mid, boll_lower,
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        ))

                    if rows:
                        conn.executemany("""
                            INSERT OR REPLACE INTO stock_daily (code, trade_date, open, high, low, close, pre_close, pct_change, amplitude,
                                volume, amount, turnover_rate, volume_ma5, volume_ma10, volume_ma20, volume_ratio,
                                kdj_k, kdj_d, kdj_j, macd_dif, macd_dea, macd_hist,
                                ma5, ma10, ma20, ma60, ma120, ma250,
                                rsi6, rsi14, rsi24, boll_upper, boll_mid, boll_lower, update_time)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, rows)
                        inserted += len(rows)

                    if (batch_idx // batch_size + 1) % 5 == 0:
                        conn.commit()

                conn.commit()
                logger.info(f"今日数据采集完成: {inserted} 条")
            else:
                logger.info(f"今日数据已存在: {cnt} 条")

            conn.close()

            # 运行起涨点扫描
            logger.info("开始起涨点扫描...")
            from modules.launch_scanner import LaunchPointScanner
            scanner = LaunchPointScanner()
            result = scanner.run_full_scan(top_n=200)
            logger.info(f"扫描完成: {result['candidates_count']} 只候选, 得分范围 {min(c['score'] for c in result['candidates'])}-{max(c['score'] for c in result['candidates'])}")

            elapsed = time.time() - start
            logger.info(f"✅ 每日任务完成，耗时 {elapsed:.0f}秒")

        except Exception as e:
            logger.error(f"每日任务失败: {e}", exc_info=True)

    def realtime_sync_job(self):
        try:
            codes = pool_manager.get_codes()
            if codes:
                collector.sync_realtime_data(codes)
        except Exception as e:
            logger.error(f"实时行情同步失败: {e}")

    def cleanup_job(self):
        import os
        cutoff = datetime.now().timestamp() - 90 * 24 * 3600
        for f in LOG_DIR.glob("*.log"):
            if f.stat().st_mtime < cutoff:
                f.unlink()
                logger.info(f"清理旧日志: {f.name}")

    def run_now(self, job_id="daily_scan"):
        job = self.scheduler.get_job(job_id)
        if job:
            job.func()
        else:
            logger.warning(f"任务 {job_id} 不存在")

    def get_status(self):
        return {
            "running": self.scheduler.running,
            "jobs": [{"id": j.id, "name": j.name, "next_run": str(j.next_run_time)} for j in self.scheduler.get_jobs()],
        }


stock_scheduler = StockScheduler()

if __name__ == "__main__":
    from utils.database import init_database
    init_database()
    stock_scheduler.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stock_scheduler.stop()
