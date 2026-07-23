#!/usr/bin/env python3
"""
启动脚本 - 同时启动Web Dashboard和定时任务调度器
"""
import sys
import subprocess
import os
import signal
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_DIR = Path(__file__).parent


def main():
    print("=" * 50)
    print("A股股票池分析系统 - 启动中...")
    print("=" * 50)

    # 启动定时任务调度器
    print("\n[1] 启动定时任务调度器...")
    scheduler_proc = subprocess.Popen(
        [sys.executable, str(PROJECT_DIR / "scheduler" / "job_scheduler.py")],
        cwd=str(PROJECT_DIR),
    )
    print(f"  调度器 PID: {scheduler_proc.pid}")

    # 启动Streamlit Dashboard
    print("\n[2] 启动Web Dashboard...")
    dashboard_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run",
         str(PROJECT_DIR / "app" / "dashboard.py"),
         "--server.port=8501",
         "--server.address=0.0.0.0"],
        cwd=str(PROJECT_DIR),
    )
    print(f"  Dashboard PID: {dashboard_proc.pid}")

    print("\n" + "=" * 50)
    print("系统已启动!")
    print(f"  Web Dashboard: http://localhost:8501")
    print(f"  定时任务: 每日 16:00 自动采集数据")
    print("  按 Ctrl+C 停止所有服务")
    print("=" * 50)

    def cleanup(sig, frame):
        print("\n正在停止服务...")
        scheduler_proc.terminate()
        dashboard_proc.terminate()
        scheduler_proc.wait()
        dashboard_proc.wait()
        print("服务已停止")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # 保持运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup(None, None)


if __name__ == "__main__":
    main()
