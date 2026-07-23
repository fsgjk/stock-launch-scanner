"""
数据库模型 - 优化版，支持全量A股数据存储（~3300万行）
- 日线表与技术指标合并，减少JOIN
- 使用覆盖索引优化查询
- 批量写入优化
"""
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

from config.settings import DATA_DIR

DB_PATH = DATA_DIR / "stock_system.db"


def get_connection():
    """获取数据库连接"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db():
    """数据库连接上下文管理器"""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def get_bulk_db():
    """批量写入优化的数据库连接"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=-8000000")  # 8GB cache
    conn.execute("PRAGMA page_size=65536")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database():
    """初始化数据库表结构（全量版）"""
    with get_db() as conn:
        cursor = conn.cursor()

        # ===== 核心表：全量日线+技术指标（合并） =====
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_daily (
                code TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                -- 价格数据
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                pre_close REAL,
                pct_change REAL,
                amplitude REAL,
                -- 量能数据
                volume REAL,
                amount REAL,
                turnover_rate REAL,
                volume_ma5 REAL,
                volume_ma10 REAL,
                volume_ma20 REAL,
                volume_ratio REAL,
                -- KDJ 指标
                kdj_k REAL,
                kdj_d REAL,
                kdj_j REAL,
                -- MACD 指标
                macd_dif REAL,
                macd_dea REAL,
                macd_hist REAL,
                -- 均线
                ma5 REAL,
                ma10 REAL,
                ma20 REAL,
                ma60 REAL,
                ma120 REAL,
                ma250 REAL,
                -- RSI
                rsi6 REAL,
                rsi14 REAL,
                rsi24 REAL,
                -- 布林带
                boll_upper REAL,
                boll_mid REAL,
                boll_lower REAL,
                -- 其他
                update_time TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (code, trade_date)
            ) WITHOUT ROWID
        """)

        # ===== 股票池表 =====
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_pool (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL,
                name TEXT,
                market TEXT DEFAULT 'A',
                group_name TEXT DEFAULT '默认分组',
                added_date TEXT DEFAULT (date('now')),
                is_active INTEGER DEFAULT 1,
                notes TEXT,
                UNIQUE(code, group_name)
            )
        """)

        # ===== 股票基本信息表 =====
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_info (
                code TEXT PRIMARY KEY,
                name TEXT,
                industry TEXT,
                area TEXT,
                market TEXT,
                list_date TEXT,
                total_market_cap REAL,
                circulating_market_cap REAL,
                pe_ratio REAL,
                pb_ratio REAL,
                update_time TEXT
            )
        """)

        # ===== 实时行情缓存 =====
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS realtime_quote (
                code TEXT PRIMARY KEY,
                name TEXT,
                open REAL, high REAL, low REAL, price REAL, pre_close REAL,
                volume REAL, amount REAL, pct_change REAL, amplitude REAL,
                turnover_rate REAL, update_time TEXT
            )
        """)

        # ===== 数据采集进度表 =====
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS collection_progress (
                code TEXT PRIMARY KEY,
                name TEXT,
                status TEXT DEFAULT 'pending',
                last_date TEXT,
                total_days INTEGER DEFAULT 0,
                error_msg TEXT,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # ===== 采集日志 =====
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_collection_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT,
                start_time TEXT,
                end_time TEXT,
                status TEXT,
                records_count INTEGER,
                stock_count INTEGER,
                error_message TEXT
            )
        """)

        # ===== 起涨点扫描结果表 =====
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS launch_scan_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date TEXT NOT NULL,
                scan_time TEXT NOT NULL,
                total_scanned INTEGER,
                total_candidates INTEGER,
                hard_filter_passed INTEGER,
                winner_sample_count INTEGER,
                latest_trade_date TEXT,
                scan_params TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS launch_scan_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id INTEGER NOT NULL,
                code TEXT NOT NULL,
                close REAL, pct_change REAL, score INTEGER,
                kdj_k REAL, kdj_d REAL, kdj_j REAL,
                rsi6 REAL, rsi14 REAL, rsi24 REAL,
                dev_ma20 REAL, dev_ma60 REAL,
                down_days INTEGER, dd_60 REAL,
                volume_ratio REAL, price_pct_20d REAL, boll_pos REAL,
                macd_dif REAL, macd_hist REAL,
                score_breakdown TEXT,
                FOREIGN KEY (scan_id) REFERENCES launch_scan_results(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS winner_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL,
                pct_20d REAL,
                scan_date TEXT,
                kdj_k REAL, kdj_d REAL,
                rsi14 REAL, dev_ma60 REAL,
                down_days INTEGER, dd_60 REAL,
                boll_pos REAL, vol_ratio REAL,
                macd_dif REAL, macd_hist REAL,
                price_pct REAL, lp_date TEXT,
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(code, scan_date)
            )
        """)

        # ===== 索引 =====
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_date ON stock_daily(trade_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pool_group ON stock_pool(group_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_progress_status ON collection_progress(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scan_results_date ON launch_scan_results(scan_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scan_candidates_scan ON launch_scan_candidates(scan_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scan_candidates_score ON launch_scan_candidates(score DESC)")

        conn.commit()

    print(f"数据库初始化完成: {DB_PATH}")


def migrate_old_data():
    """将旧表数据迁移到新表"""
    import os
    old_tables = ["daily_quote", "technical_indicators"]
    with get_db() as conn:
        for t in old_tables:
            try:
                conn.execute(f"SELECT COUNT(*) FROM {t}")
                print(f"旧表 {t} 存在，可手动迁移")
            except:
                pass


if __name__ == "__main__":
    init_database()
