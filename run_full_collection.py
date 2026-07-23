#!/usr/bin/env python3
"""
全量A股数据采集启动脚本
采集所有5529只A股自2000年至今的日线数据 + KDJ + MACD + 量能指标
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.database import init_database
from modules.full_collector import collector
from loguru import logger


def main():
    print("=" * 60)
    print("全量A股数据采集")
    print("目标: 5529只A股 × 2000年至今的日线 + KDJ + MACD + 量能")
    print("预计耗时: ~2小时")
    print("=" * 60)

    init_database()

    start = time.time()
    result = collector.run_full_collection(
        start_date="20000101",
        resume=True,  # 支持断点续传
    )

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"采集完成! 耗时: {elapsed/3600:.1f} 小时")
    print(f"成功: {result['success']} 只")
    print(f"总记录: {result['total_records']:,} 条")
    print("=" * 60)

    collector.verify_data()


if __name__ == "__main__":
    main()
