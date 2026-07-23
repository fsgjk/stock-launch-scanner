#!/usr/bin/env python3
"""
系统初始化脚本 - 首次运行前执行
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.database import init_database
from modules.stock_pool import pool_manager
from modules.data_collector import collector
from loguru import logger


def init_system():
    """初始化系统"""
    print("=" * 50)
    print("A股股票池分析系统 - 初始化")
    print("=" * 50)

    # 1. 初始化数据库
    print("\n[1/4] 初始化数据库...")
    init_database()
    print("✓ 数据库初始化完成")

    # 2. 导入默认股票
    print("\n[2/4] 导入示例股票...")
    default_stocks = [
        {"code": "600519", "name": "贵州茅台", "group": "重点关注"},
        {"code": "000858", "name": "五粮液", "group": "重点关注"},
        {"code": "300750", "name": "宁德时代", "group": "短线操作"},
        {"code": "600036", "name": "招商银行", "group": "中长线"},
        {"code": "000333", "name": "美的集团", "group": "中长线"},
        {"code": "002415", "name": "海康威视", "group": "观察列表"},
        {"code": "601318", "name": "中国平安", "group": "观察列表"},
        {"code": "000001", "name": "平安银行", "group": "观察列表"},
        {"code": "600900", "name": "长江电力", "group": "中长线"},
        {"code": "002594", "name": "比亚迪", "group": "短线操作"},
        {"code": "601166", "name": "兴业银行", "group": "观察列表"},
        {"code": "600276", "name": "恒瑞医药", "group": "重点关注"},
    ]
    count = pool_manager.import_from_list(default_stocks)
    print(f"✓ 导入 {count} 只示例股票")

    # 3. 首次数据采集
    print("\n[3/4] 首次数据采集（这可能需要几分钟）...")
    codes = pool_manager.get_codes()
    print(f"  股票池共 {len(codes)} 只股票")

    try:
        daily_count = collector.sync_daily_data(codes, days_back=60)
        print(f"✓ 日线数据采集完成: {daily_count} 条")
    except Exception as e:
        print(f"⚠ 日线数据采集出现错误: {e}")

    try:
        rt_count = collector.sync_realtime_data(codes)
        print(f"✓ 实时行情同步完成: {rt_count} 条")
    except Exception as e:
        print(f"⚠ 实时行情同步出现错误: {e}")

    # 4. 完成
    print("\n[4/4] 初始化完成!")
    print("=" * 50)
    print("\n启动命令:")
    print("  Web Dashboard: streamlit run app/dashboard.py")
    print("  定时任务服务:  python scheduler/job_scheduler.py")
    print("=" * 50)


if __name__ == "__main__":
    init_system()
