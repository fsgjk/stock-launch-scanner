"""
系统配置模块
"""
import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 数据存储目录
DATA_DIR = ROOT_DIR / "data"
STOCK_POOL_DIR = DATA_DIR / "stock_pool"
HISTORY_DIR = DATA_DIR / "history"
REALTIME_DIR = DATA_DIR / "realtime"

# 日志目录
LOG_DIR = ROOT_DIR / "logs"

# 确保目录存在
for d in [DATA_DIR, STOCK_POOL_DIR, HISTORY_DIR, REALTIME_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ========== 定时任务配置 ==========
# 每日数据采集时间 (北京时间, 24小时制)
SCHEDULE_HOUR = 16  # 收盘后 16:00 执行
SCHEDULE_MINUTE = 0

# 是否启用定时任务
SCHEDULER_ENABLED = True

# ========== 数据源配置 ==========
# 数据采集重试次数
MAX_RETRIES = 3
# 请求间隔(秒)
REQUEST_INTERVAL = 0.5

# ========== 股票池默认配置 ==========
# 默认关注指数
DEFAULT_INDEXES = ["000001", "399001", "399006"]  # 上证, 深证, 创业板

# ========== 技术分析默认参数 ==========
# 常用均线周期
MA_PERIODS = [5, 10, 20, 60, 120, 250]
# MACD参数
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
# RSI参数
RSI_PERIOD = 14
# KDJ参数
KDJ_N = 9
KDJ_M1 = 3
KDJ_M2 = 3
# 布林带参数
BOLL_PERIOD = 20
BOLL_STD = 2

# ========== Web Dashboard配置 ==========
STREAMLIT_HOST = "0.0.0.0"
STREAMLIT_PORT = 8501
PAGE_TITLE = "A股股票池分析系统"
PAGE_ICON = "📈"

# ========== 起涨点扫描配置 ==========
SCAN_TOP_N = 200                     # 候选股票数量上限
SCAN_WINNER_THRESHOLD = 20           # 大涨股涨幅阈值(%)
SCAN_LOOKBACK_DAYS = 120             # 历史回溯天数

# 硬条件默认阈值
HARD_FILTER_KDJ_MAX = 35
HARD_FILTER_RSI_MAX = 45
HARD_FILTER_MA60_DEV_MAX = -3
HARD_FILTER_DD60_MAX = -15
HARD_FILTER_VOL_RATIO_MAX = 1.2
HARD_FILTER_DOWN_DAYS_MIN = 2
HARD_FILTER_PCT_CHANGE_MAX = 0.5
HARD_FILTER_BOLL_POS_MAX = 0.4
