"""
数据采集模块 - 基于AKShare获取A股数据
主要使用新浪接口（sandbox兼容性好），东方财富接口作为备选
"""
import time
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

from utils.database import get_db
from config.settings import MAX_RETRIES, REQUEST_INTERVAL


class StockDataCollector:
    """A股股票数据采集器"""

    def __init__(self):
        self._ak = None

    @property
    def ak(self):
        """懒加载akshare"""
        if self._ak is None:
            try:
                import akshare as ak
                self._ak = ak
            except ImportError:
                raise ImportError("请先安装akshare: pip install akshare")
        return self._ak

    def _retry(max_retries=MAX_RETRIES):
        """重试装饰器"""
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                last_error = None
                retries = max_retries
                for attempt in range(retries):
                    try:
                        return func(self, *args, **kwargs)
                    except Exception as e:
                        last_error = e
                        logger.warning(f"{func.__name__} 第{attempt+1}次尝试失败: {e}")
                        if attempt < retries - 1:
                            time.sleep(REQUEST_INTERVAL * (attempt + 1))
                raise last_error
            return wrapper
        return decorator

    # ==================== 股票列表 ====================

    @_retry(max_retries=2)
    def get_all_stocks(self) -> pd.DataFrame:
        """获取全部A股列表（新浪接口）"""
        df = self.ak.stock_zh_a_spot()
        df = df[["代码", "名称"]].copy()
        df["代码"] = df["代码"].astype(str).str.replace(r"^(sh|sz|bj)", "", regex=True)
        return df.rename(columns={"代码": "code", "名称": "name"})

    # ==================== 实时行情 ====================

    @_retry(max_retries=2)
    def get_realtime_quote(self, codes: list = None) -> pd.DataFrame:
        """获取实时行情（新浪接口）"""
        df = self.ak.stock_zh_a_spot()

        # 统一列名
        df = df.rename(columns={
            "代码": "code", "名称": "name",
            "最新价": "price", "涨跌额": "change",
            "涨跌幅": "pct_change", "昨收": "pre_close",
            "今开": "open", "最高": "high", "最低": "low",
            "成交量": "volume", "成交额": "amount",
        })

        # 清理代码前缀 (sh600519 → 600519)
        df["code"] = df["code"].str.replace(r"^(sh|sz|bj)", "", regex=True)

        if codes:
            target_codes = set(str(c).zfill(6) for c in codes)
            df = df[df["code"].isin(target_codes)]

        return df

    # ==================== 历史K线 ====================

    @_retry(max_retries=3)
    def get_daily_kline(self, code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取日K线数据（新浪接口）

        Args:
            code: 股票代码，如 '600519'
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")

        # 新浪接口需要前缀: sh=上海, sz=深圳, bj=北交所
        code_str = str(code).zfill(6)
        if code_str.startswith(("6", "5")):
            symbol = f"sh{code_str}"
        elif code_str.startswith(("0", "3", "2")):
            symbol = f"sz{code_str}"
        elif code_str.startswith(("8", "4", "9")):
            symbol = f"bj{code_str}"
        else:
            symbol = f"sz{code_str}"

        df = self.ak.stock_zh_a_daily(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"  # 前复权
        )

        # 统一列名
        df = df.rename(columns={
            "date": "trade_date",
            "open": "open", "high": "high",
            "low": "low", "close": "close",
            "volume": "volume", "amount": "amount",
            "outstanding_share": "outstanding_share",
            "turnover": "turnover_rate",
        })

        return df

    # ==================== 数据同步 ====================

    def sync_daily_data(self, codes: list = None, days_back: int = 10):
        """同步日线数据到本地数据库"""
        if codes is None:
            from modules.stock_pool import pool_manager
            codes = pool_manager.get_codes()

        if not codes:
            logger.warning("股票池为空，请先添加股票")
            return 0

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")

        total = 0
        success_count = 0
        for code in codes:
            try:
                df = self.get_daily_kline(code, start_date, end_date)
                if df.empty:
                    logger.warning(f"{code} 无数据")
                    continue

                with get_db() as conn:
                    for _, row in df.iterrows():
                        conn.execute("""
                            INSERT OR REPLACE INTO daily_quote
                            (code, trade_date, open, high, low, close, pre_close,
                             volume, amount, amplitude, pct_change, turnover_rate)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            code,
                            str(row.get("trade_date", "")),
                            float(row.get("open", 0)),
                            float(row.get("high", 0)),
                            float(row.get("low", 0)),
                            float(row.get("close", 0)),
                            float(row.get("close", 0)) - float(row.get("open", 0)),  # pre_close approximate
                            float(row.get("volume", 0)),
                            float(row.get("amount", 0)),
                            float(row.get("high", 0) - row.get("low", 0)) / float(row.get("pre_close", row.get("close", 1)) or 1) * 100 if row.get("close", 0) else 0,
                            0.0,  # pct_change 由后续计算
                            float(row.get("turnover_rate", 0)),
                        ))
                total += len(df)
                success_count += 1
                logger.info(f"✓ {code} 同步 {len(df)} 条日线数据")
                time.sleep(REQUEST_INTERVAL)

            except Exception as e:
                logger.error(f"✗ {code} 同步失败: {e}")

        # 记录日志
        with get_db() as conn:
            conn.execute("""
                INSERT INTO data_collection_log (task_type, start_time, end_time, status, records_count)
                VALUES ('daily_sync', ?, ?, 'success', ?)
            """, (datetime.now().isoformat(), datetime.now().isoformat(), total))

        logger.info(f"日线数据同步完成: {success_count}/{len(codes)} 只股票, 共 {total} 条记录")
        return total

    def sync_realtime_data(self, codes: list = None):
        """同步实时行情"""
        try:
            df = self.get_realtime_quote(codes)

            with get_db() as conn:
                conn.execute("DELETE FROM realtime_quote")  # 清空旧数据
                for _, row in df.iterrows():
                    conn.execute("""
                        INSERT OR REPLACE INTO realtime_quote
                        (code, name, open, high, low, price, pre_close,
                         volume, amount, pct_change, amplitude, turnover_rate, update_time)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        str(row.get("code", "")),
                        str(row.get("name", "")),
                        float(row.get("open", 0)),
                        float(row.get("high", 0)),
                        float(row.get("low", 0)),
                        float(row.get("price", 0)),
                        float(row.get("pre_close", 0)),
                        float(row.get("volume", 0)),
                        float(row.get("amount", 0)),
                        float(row.get("pct_change", 0)),
                        float(row.get("high", 0) - row.get("low", 0)) / float(row.get("pre_close", 1) or 1) * 100,
                        0.0,  # turnover_rate
                        datetime.now().isoformat(),
                    ))

            count = len(df)
            logger.info(f"实时行情同步完成，共 {count} 条记录")
            return count

        except Exception as e:
            logger.error(f"实时行情同步失败: {e}")
            return 0

    # ==================== 市场指数 ====================

    @_retry(max_retries=2)
    def get_market_indices(self) -> pd.DataFrame:
        """获取主要市场指数"""
        try:
            df = self.ak.stock_zh_index_daily_em(symbol="sh000001")
            return df
        except Exception:
            logger.warning("获取指数数据失败")
            return pd.DataFrame()


# 单例
collector = StockDataCollector()


if __name__ == "__main__":
    from utils.database import init_database
    init_database()

    # 测试
    print("测试获取股票列表...")
    df = collector.get_all_stocks()
    print(f"A股总数: {len(df)}")
    print(df.head(5))

    print("\n测试获取日K线...")
    df = collector.get_daily_kline("600519", start_date="20250701", end_date="20250722")
    print(f"获取到 {len(df)} 条")
    print(df.tail(3))
