"""
全量A股数据采集引擎
- 采集2000年至今所有A股日线数据
- 实时计算 KDJ、MACD、RSI、MA、布林带、量能指标
- 支持断点续传、进度追踪、并发控制
"""
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from loguru import logger

from utils.database import get_db, get_bulk_db
from config.settings import (
    MAX_RETRIES, REQUEST_INTERVAL, KDJ_N, KDJ_M1, KDJ_M2,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL, RSI_PERIOD,
    BOLL_PERIOD, BOLL_STD, MA_PERIODS, DATA_DIR
)


class FullStockCollector:
    """全量A股数据采集器"""

    def __init__(self):
        self._ak = None
        self._all_codes = None
        self._batch_size = 200  # 每批写入条数
        self._request_delay = 0.8  # 请求间隔（秒）

    @property
    def ak(self):
        if self._ak is None:
            import akshare as ak
            self._ak = ak
        return self._ak

    # ==================== 股票列表 ====================

    def get_all_stock_list(self, refresh: bool = False) -> pd.DataFrame:
        """获取全部A股列表"""
        if self._all_codes is not None and not refresh:
            return self._all_codes

        logger.info("获取A股全量列表...")
        df = self.ak.stock_zh_a_spot()
        df = df[["代码", "名称"]].copy()
        df["代码"] = df["代码"].astype(str).str.replace(r"^(sh|sz|bj)", "", regex=True)
        df = df.rename(columns={"代码": "code", "名称": "name"})
        self._all_codes = df
        logger.info(f"A股总数: {len(df)}")
        return df

    # ==================== 历史K线获取 ====================

    def _code_to_sina_symbol(self, code: str) -> str:
        """将6位代码转为新浪symbol格式"""
        code_str = str(code).zfill(6)
        if code_str.startswith(("6", "5")):
            return f"sh{code_str}"
        elif code_str.startswith(("0", "3", "2")):
            return f"sz{code_str}"
        elif code_str.startswith(("8", "4", "9")):
            return f"bj{code_str}"
        return f"sz{code_str}"

    def fetch_stock_history(self, code: str, start_date: str = "20000101",
                            end_date: str = None) -> Optional[pd.DataFrame]:
        """获取单只股票全量历史K线"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        symbol = self._code_to_sina_symbol(code)

        for attempt in range(MAX_RETRIES):
            try:
                df = self.ak.stock_zh_a_daily(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )

                if df.empty:
                    return None

                # 标准化列名
                df = df.rename(columns={
                    "date": "trade_date",
                    "open": "open", "high": "high",
                    "low": "low", "close": "close",
                    "volume": "volume", "amount": "amount",
                    "turnover": "turnover_rate",
                })
                df["trade_date"] = df["trade_date"].astype(str)
                return df

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(REQUEST_INTERVAL * (attempt + 1))
                else:
                    logger.error(f"{code} 获取失败: {e}")
                    return None

    # ==================== 技术指标计算 ====================

    @staticmethod
    def calc_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标（向量化）"""
        if df.empty:
            return df

        df = df.sort_values("trade_date").reset_index(drop=True)
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float)

        n = len(df)

        # === 价格衍生指标 ===
        df["pre_close"] = np.concatenate([[close[0]], close[:-1]])
        df["pct_change"] = np.where(
            df["pre_close"] != 0,
            (close - df["pre_close"].values) / df["pre_close"].values * 100,
            0
        )
        df["amplitude"] = np.where(
            df["pre_close"] != 0,
            (high - low) / df["pre_close"].values * 100,
            0
        )

        # === 均线 (向量化) ===
        for p in MA_PERIODS:
            if n >= p:
                ma = np.full(n, np.nan)
                cumsum = np.cumsum(np.insert(close, 0, 0))
                ma[p-1:] = (cumsum[p:] - cumsum[:-p]) / p
                df[f"ma{p}"] = ma
            else:
                df[f"ma{p}"] = np.nan

        # === KDJ ===
        if n >= KDJ_N:
            # 使用pandas rolling加速
            low_min = pd.Series(low).rolling(KDJ_N).min().values
            high_max = pd.Series(high).rolling(KDJ_N).max().values
            rsv = np.where(
                (high_max - low_min) != 0,
                (close - low_min) / (high_max - low_min) * 100,
                50
            )
            rsv = np.nan_to_num(rsv, nan=50)

            # EMA for K and D
            alpha_k = 1.0 / KDJ_M1
            alpha_d = 1.0 / KDJ_M2
            k = np.full(n, 50.0)
            d = np.full(n, 50.0)
            for i in range(1, n):
                k[i] = alpha_k * rsv[i] + (1 - alpha_k) * k[i-1]
                d[i] = alpha_d * k[i] + (1 - alpha_d) * d[i-1]
            j = 3 * k - 2 * d

            df["kdj_k"] = k
            df["kdj_d"] = d
            df["kdj_j"] = j
        else:
            df["kdj_k"] = np.nan
            df["kdj_d"] = np.nan
            df["kdj_j"] = np.nan

        # === MACD ===
        ema_fast = pd.Series(close).ewm(span=MACD_FAST, adjust=False).mean().values
        ema_slow = pd.Series(close).ewm(span=MACD_SLOW, adjust=False).mean().values
        dif = ema_fast - ema_slow
        dea = pd.Series(dif).ewm(span=MACD_SIGNAL, adjust=False).mean().values
        hist = 2 * (dif - dea)

        df["macd_dif"] = dif
        df["macd_dea"] = dea
        df["macd_hist"] = hist

        # === RSI ===
        for period in [6, 14, 24]:
            delta = np.diff(close, prepend=close[0])
            gain = np.maximum(delta, 0)
            loss = np.abs(np.minimum(delta, 0))
            avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().values
            avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().values
            rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
            rsi = np.where(avg_loss != 0, 100 - 100 / (1 + rs), 50)
            df[f"rsi{period}"] = rsi

        # === 布林带 ===
        if n >= BOLL_PERIOD:
            mid = pd.Series(close).rolling(BOLL_PERIOD).mean().values
            std = pd.Series(close).rolling(BOLL_PERIOD).std().values
            df["boll_mid"] = mid
            df["boll_upper"] = mid + BOLL_STD * std
            df["boll_lower"] = mid - BOLL_STD * std
        else:
            df["boll_mid"] = np.nan
            df["boll_upper"] = np.nan
            df["boll_lower"] = np.nan

        # === 量能指标 ===
        if n >= 5:
            df["volume_ma5"] = pd.Series(volume).rolling(5).mean().values
        if n >= 10:
            df["volume_ma10"] = pd.Series(volume).rolling(10).mean().values
        if n >= 20:
            df["volume_ma20"] = pd.Series(volume).rolling(20).mean().values

        # 量比 = 当日成交量 / 5日均量
        df["volume_ratio"] = np.where(
            df["volume_ma5"].values > 0,
            volume / df["volume_ma5"].values,
            1.0
        )

        return df

    # ==================== 批量采集核心 ====================

    def init_progress_table(self, codes: list, names: dict):
        """初始化采集进度表"""
        with get_bulk_db() as conn:
            for code in codes:
                conn.execute("""
                    INSERT OR IGNORE INTO collection_progress (code, name, status)
                    VALUES (?, ?, 'pending')
                """, (code, names.get(code, "")))

    def update_progress(self, code: str, status: str, last_date: str = None,
                        total_days: int = 0, error_msg: str = None):
        """更新采集进度"""
        with get_db() as conn:
            conn.execute("""
                UPDATE collection_progress
                SET status=?, last_date=?, total_days=?, error_msg=?, updated_at=datetime('now')
                WHERE code=?
            """, (status, last_date, total_days, error_msg, code))

    def get_pending_codes(self) -> List[str]:
        """获取待采集的股票代码"""
        with get_db() as conn:
            rows = conn.execute(
                "SELECT code FROM collection_progress WHERE status IN ('pending', 'failed')"
            ).fetchall()
        return [r["code"] for r in rows]

    def get_progress_summary(self) -> Dict:
        """获取采集进度摘要"""
        with get_db() as conn:
            total = conn.execute("SELECT COUNT(*) as c FROM collection_progress").fetchone()["c"]
            done = conn.execute(
                "SELECT COUNT(*) as c FROM collection_progress WHERE status='done'"
            ).fetchone()["c"]
            pending = conn.execute(
                "SELECT COUNT(*) as c FROM collection_progress WHERE status='pending'"
            ).fetchone()["c"]
            failed = conn.execute(
                "SELECT COUNT(*) as c FROM collection_progress WHERE status='failed'"
            ).fetchone()["c"]
            total_records = conn.execute(
                "SELECT COUNT(*) as c FROM stock_daily"
            ).fetchone()["c"]

        return {
            "total_stocks": total,
            "done": done,
            "pending": pending,
            "failed": failed,
            "total_records": total_records,
            "progress_pct": f"{done/total*100:.1f}%" if total > 0 else "0%",
        }

    def save_batch(self, batch: List[Dict]):
        """批量写入数据"""
        if not batch:
            return 0

        sql = """INSERT OR REPLACE INTO stock_daily
            (code, trade_date, open, high, low, close, pre_close, pct_change, amplitude,
             volume, amount, turnover_rate, volume_ma5, volume_ma10, volume_ma20, volume_ratio,
             kdj_k, kdj_d, kdj_j, macd_dif, macd_dea, macd_hist,
             ma5, ma10, ma20, ma60, ma120, ma250,
             rsi6, rsi14, rsi24, boll_upper, boll_mid, boll_lower)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

        with get_bulk_db() as conn:
            conn.executemany(sql, batch)
        return len(batch)

    def collect_single_stock(self, code: str, name: str = "",
                             start_date: str = "20000101") -> int:
        """采集单只股票的完整数据，返回写入条数"""
        df = self.fetch_stock_history(code, start_date)
        if df is None or df.empty:
            self.update_progress(code, "failed", error_msg="无数据")
            return 0

        # 计算技术指标
        df = self.calc_all_indicators(df)

        # 准备批量写入数据
        cols = [
            "code", "trade_date", "open", "high", "low", "close",
            "pre_close", "pct_change", "amplitude",
            "volume", "amount", "turnover_rate",
            "volume_ma5", "volume_ma10", "volume_ma20", "volume_ratio",
            "kdj_k", "kdj_d", "kdj_j",
            "macd_dif", "macd_dea", "macd_hist",
            "ma5", "ma10", "ma20", "ma60", "ma120", "ma250",
            "rsi6", "rsi14", "rsi24",
            "boll_upper", "boll_mid", "boll_lower",
        ]

        batch = []
        total = 0
        for _, row in df.iterrows():
            record = [code]
            for col in cols[1:]:
                val = row.get(col, np.nan)
                if isinstance(val, float) and np.isnan(val):
                    record.append(None)
                elif isinstance(val, (np.integer,)):
                    record.append(int(val))
                elif isinstance(val, (np.floating,)):
                    record.append(float(val))
                else:
                    record.append(str(val) if val is not None else None)
            batch.append(tuple(record))

            if len(batch) >= self._batch_size:
                self.save_batch(batch)
                total += len(batch)
                batch = []

        if batch:
            self.save_batch(batch)
            total += len(batch)

        last_date = str(df["trade_date"].max())
        self.update_progress(code, "done", last_date=last_date, total_days=total)

        return total

    def run_full_collection(self, start_date: str = "20000101",
                            resume: bool = True, max_stocks: int = None):
        """执行全量采集

        Args:
            start_date: 起始日期
            resume: 是否断点续传
            max_stocks: 最大采集数量（None=全部）
        """
        # 获取股票列表
        stock_df = self.get_all_stock_list()
        all_codes = stock_df["code"].tolist()
        names = dict(zip(stock_df["code"], stock_df["name"]))

        if max_stocks:
            all_codes = all_codes[:max_stocks]

        logger.info(f"准备采集 {len(all_codes)} 只股票")

        # 初始化进度表
        if not resume:
            with get_db() as conn:
                conn.execute("DELETE FROM collection_progress")
        self.init_progress_table(all_codes, names)

        # 获取待采集列表
        if resume:
            pending = self.get_pending_codes()
            logger.info(f"断点续传: {len(pending)} 只待采集")
        else:
            pending = all_codes

        if not pending:
            logger.info("所有股票已采集完成!")
            return

        # 开始批量采集
        start_time = time.time()
        success = 0
        total_records = 0

        for i, code in enumerate(pending):
            name = names.get(code, "")
            try:
                records = self.collect_single_stock(code, name, start_date)
                total_records += records
                success += 1

                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                eta = avg_time * (len(pending) - i - 1)

                if (i + 1) % 50 == 0 or i == 0:
                    summary = self.get_progress_summary()
                    logger.info(
                        f"[{summary['progress_pct']}] {i+1}/{len(pending)} "
                        f"✓{code} {name} {records}条 | "
                        f"累计{total_records}条 | "
                        f"速度{avg_time:.1f}s/只 | "
                        f"预计剩余{eta/60:.0f}分钟"
                    )

                time.sleep(self._request_delay)

            except Exception as e:
                logger.error(f"✗ {code} {name} 采集失败: {e}")
                self.update_progress(code, "failed", error_msg=str(e))

        # 完成
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"全量采集完成!")
        logger.info(f"成功: {success}/{len(pending)} 只")
        logger.info(f"总记录: {total_records} 条")
        logger.info(f"耗时: {elapsed/60:.1f} 分钟")
        logger.info(f"速度: {elapsed/len(pending):.1f}s/只")

        # 记录日志
        with get_db() as conn:
            conn.execute("""
                INSERT INTO data_collection_log
                (task_type, start_time, end_time, status, records_count, stock_count)
                VALUES ('full_collection', ?, datetime('now'), 'success', ?, ?)
            """, (datetime.fromtimestamp(start_time).isoformat(), total_records, success))

        return {"success": success, "total_records": total_records, "elapsed_min": elapsed/60}

    def verify_data(self):
        """验证数据完整性"""
        with get_db() as conn:
            # 总记录数
            total = conn.execute("SELECT COUNT(*) as c FROM stock_daily").fetchone()["c"]
            # 股票数
            stocks = conn.execute(
                "SELECT COUNT(DISTINCT code) as c FROM stock_daily"
            ).fetchone()["c"]
            # 日期范围
            dates = conn.execute(
                "SELECT MIN(trade_date) as min_d, MAX(trade_date) as max_d FROM stock_daily"
            ).fetchone()
            # 每只股票的数据量分布
            top = conn.execute("""
                SELECT code, COUNT(*) as cnt FROM stock_daily
                GROUP BY code ORDER BY cnt DESC LIMIT 10
            """).fetchall()

        logger.info(f"数据验证:")
        logger.info(f"  总记录: {total:,} 条")
        logger.info(f"  股票数: {stocks} 只")
        logger.info(f"  日期: {dates['min_d']} ~ {dates['max_d']}")
        logger.info(f"  数据量TOP10:")
        for r in top:
            logger.info(f"    {r['code']}: {r['cnt']} 条")

        return {"total": total, "stocks": stocks, "dates": (dates["min_d"], dates["max_d"])}


# 单例
collector = FullStockCollector()


if __name__ == "__main__":
    from utils.database import init_database
    init_database()
    collector.run_full_collection(max_stocks=5)
