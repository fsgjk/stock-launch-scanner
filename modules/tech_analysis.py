"""
技术分析模块 - 计算常用技术指标
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from loguru import logger

from utils.database import get_db
from config.settings import (
    MA_PERIODS, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    RSI_PERIOD, KDJ_N, KDJ_M1, KDJ_M2,
    BOLL_PERIOD, BOLL_STD
)


class TechnicalAnalyzer:
    """技术分析器"""

    @staticmethod
    def calc_ma(df: pd.DataFrame, periods: list = None) -> pd.DataFrame:
        """计算移动平均线"""
        if periods is None:
            periods = MA_PERIODS
        if "close" not in df.columns and "收盘" in df.columns:
            df = df.rename(columns={"收盘": "close"})
        for p in periods:
            df[f"ma{p}"] = df["close"].rolling(window=p).mean()
        return df

    @staticmethod
    def calc_macd(df: pd.DataFrame) -> pd.DataFrame:
        """计算MACD指标"""
        close = df["close"] if "close" in df.columns else df["收盘"]

        ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()

        df["macd_dif"] = ema_fast - ema_slow
        df["macd_dea"] = df["macd_dif"].ewm(span=MACD_SIGNAL, adjust=False).mean()
        df["macd_hist"] = 2 * (df["macd_dif"] - df["macd_dea"])
        return df

    @staticmethod
    def calc_rsi(df: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """计算RSI指标"""
        if period is None:
            period = RSI_PERIOD
        close = df["close"] if "close" in df.columns else df["收盘"]

        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def calc_kdj(df: pd.DataFrame) -> pd.DataFrame:
        """计算KDJ指标"""
        close = df["close"] if "close" in df.columns else df["收盘"]
        high = df["high"] if "high" in df.columns else df["最高"]
        low = df["low"] if "low" in df.columns else df["最低"]

        lowest_low = low.rolling(window=KDJ_N).min()
        highest_high = high.rolling(window=KDJ_N).max()

        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
        rsv = rsv.fillna(50)

        df["kdj_k"] = rsv.ewm(alpha=1/KDJ_M1, adjust=False).mean()
        df["kdj_d"] = df["kdj_k"].ewm(alpha=1/KDJ_M2, adjust=False).mean()
        df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]
        return df

    @staticmethod
    def calc_boll(df: pd.DataFrame) -> pd.DataFrame:
        """计算布林带"""
        close = df["close"] if "close" in df.columns else df["收盘"]

        df["boll_mid"] = close.rolling(window=BOLL_PERIOD).mean()
        std = close.rolling(window=BOLL_PERIOD).std()
        df["boll_upper"] = df["boll_mid"] + BOLL_STD * std
        df["boll_lower"] = df["boll_mid"] - BOLL_STD * std
        return df

    @staticmethod
    def calc_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标"""
        # 标准化列名
        col_map = {
            "开盘": "open", "最高": "high", "最低": "low",
            "收盘": "close", "成交量": "volume", "成交额": "amount",
            "日期": "trade_date"
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        df = TechnicalAnalyzer.calc_ma(df)
        df = TechnicalAnalyzer.calc_macd(df)
        df = TechnicalAnalyzer.calc_rsi(df)
        df = TechnicalAnalyzer.calc_kdj(df)
        df = TechnicalAnalyzer.calc_boll(df)
        return df

    def analyze_stock(self, code: str, days: int = 250) -> Dict:
        """对单只股票进行全面技术分析"""
        with get_db() as conn:
            rows = conn.execute("""
                SELECT * FROM daily_quote WHERE code=?
                ORDER BY trade_date DESC LIMIT ?
            """, (code, days)).fetchall()

        if not rows:
            return {"error": f"{code} 无历史数据"}

        df = pd.DataFrame([dict(r) for r in rows])
        df = df.sort_values("trade_date")

        # 计算指标
        df = self.calc_all_indicators(df)
        latest = df.iloc[-1] if len(df) > 0 else None

        if latest is None:
            return {"error": "数据不足"}

        # 信号判断
        signals = self._generate_signals(df, latest)

        return {
            "code": code,
            "latest_date": str(latest.get("trade_date", "")),
            "latest_price": float(latest.get("close", 0)),
            "indicators": {
                "ma5": float(latest.get("ma5", 0)),
                "ma10": float(latest.get("ma10", 0)),
                "ma20": float(latest.get("ma20", 0)),
                "ma60": float(latest.get("ma60", 0)),
                "ma120": float(latest.get("ma120", 0)),
                "ma250": float(latest.get("ma250", 0)),
                "macd_dif": float(latest.get("macd_dif", 0)),
                "macd_dea": float(latest.get("macd_dea", 0)),
                "macd_hist": float(latest.get("macd_hist", 0)),
                "rsi": float(latest.get("rsi", 0)),
                "kdj_k": float(latest.get("kdj_k", 0)),
                "kdj_d": float(latest.get("kdj_d", 0)),
                "kdj_j": float(latest.get("kdj_j", 0)),
                "boll_upper": float(latest.get("boll_upper", 0)),
                "boll_mid": float(latest.get("boll_mid", 0)),
                "boll_lower": float(latest.get("boll_lower", 0)),
            },
            "signals": signals,
            "raw_df": df.tail(60),  # 最近60天数据用于绘图
        }

    def _generate_signals(self, df: pd.DataFrame, latest: pd.Series) -> list:
        """生成技术信号"""
        signals = []

        # MA信号
        if latest.get("close", 0) > latest.get("ma5", 0):
            signals.append({"type": "MA", "signal": "bullish", "desc": "收盘价站上5日均线"})
        else:
            signals.append({"type": "MA", "signal": "bearish", "desc": "收盘价跌破5日均线"})

        # MACD信号
        if latest.get("macd_dif", 0) > latest.get("macd_dea", 0):
            signals.append({"type": "MACD", "signal": "bullish", "desc": "DIF在DEA上方"})
        else:
            signals.append({"type": "MACD", "signal": "bearish", "desc": "DIF在DEA下方"})

        # MACD金叉死叉
        if len(df) >= 2:
            prev = df.iloc[-2]
            if (prev.get("macd_dif", 0) <= prev.get("macd_dea", 0) and
                    latest.get("macd_dif", 0) > latest.get("macd_dea", 0)):
                signals.append({"type": "MACD", "signal": "golden_cross", "desc": "⚠️ MACD金叉"})
            elif (prev.get("macd_dif", 0) >= prev.get("macd_dea", 0) and
                  latest.get("macd_dif", 0) < latest.get("macd_dea", 0)):
                signals.append({"type": "MACD", "signal": "dead_cross", "desc": "⚠️ MACD死叉"})

        # RSI信号
        rsi = latest.get("rsi", 50)
        if rsi > 80:
            signals.append({"type": "RSI", "signal": "overbought", "desc": f"RSI超买({rsi:.1f})"})
        elif rsi < 20:
            signals.append({"type": "RSI", "signal": "oversold", "desc": f"RSI超卖({rsi:.1f})"})
        elif rsi > 50:
            signals.append({"type": "RSI", "signal": "bullish", "desc": f"RSI偏强({rsi:.1f})"})
        else:
            signals.append({"type": "RSI", "signal": "bearish", "desc": f"RSI偏弱({rsi:.1f})"})

        # KDJ信号
        kdj_j = latest.get("kdj_j", 50)
        if kdj_j > 100:
            signals.append({"type": "KDJ", "signal": "overbought", "desc": f"KDJ超买(J={kdj_j:.1f})"})
        elif kdj_j < 0:
            signals.append({"type": "KDJ", "signal": "oversold", "desc": f"KDJ超卖(J={kdj_j:.1f})"})

        # 布林带信号
        close = latest.get("close", 0)
        upper = latest.get("boll_upper", 0)
        lower = latest.get("boll_lower", 0)
        if close > upper:
            signals.append({"type": "BOLL", "signal": "overbought", "desc": "突破布林上轨"})
        elif close < lower:
            signals.append({"type": "BOLL", "signal": "oversold", "desc": "跌破布林下轨"})

        return signals

    def batch_analyze(self, codes: list) -> pd.DataFrame:
        """批量技术分析"""
        results = []
        for code in codes:
            try:
                result = self.analyze_stock(code)
                if "error" not in result:
                    row = {
                        "code": code,
                        "date": result["latest_date"],
                        "price": result["latest_price"],
                        "ma20": result["indicators"]["ma20"],
                        "rsi": result["indicators"]["rsi"],
                        "macd_signal": "bullish" if result["indicators"]["macd_dif"] > result["indicators"]["macd_dea"] else "bearish",
                    }
                    # 汇总信号
                    bull = sum(1 for s in result["signals"] if s["signal"] in ("bullish", "golden_cross", "oversold"))
                    bear = sum(1 for s in result["signals"] if s["signal"] in ("bearish", "dead_cross", "overbought"))
                    row["bullish_count"] = bull
                    row["bearish_count"] = bear
                    row["signal_summary"] = "🟢偏多" if bull > bear else ("🔴偏空" if bear > bull else "⚪中性")
                    results.append(row)
            except Exception as e:
                logger.error(f"分析{code}失败: {e}")

        return pd.DataFrame(results)


# 单例
tech_analyzer = TechnicalAnalyzer()


if __name__ == "__main__":
    from utils.database import init_database
    init_database()
    result = tech_analyzer.analyze_stock("600519")
    print(result.get("signals"))
