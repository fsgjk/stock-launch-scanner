"""
批量回测扫描脚本 - 从指定日期起每个交易日执行扫描
用法: python3.11 batch_backtest.py [start_date] [end_date]
默认: 从 2026-07-01 到最新交易日
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from modules.launch_scanner import LaunchPointScanner
from utils.database import get_db
from datetime import datetime


def main():
    start_date = sys.argv[1] if len(sys.argv) > 1 else '2026-07-01'
    end_date = sys.argv[2] if len(sys.argv) > 2 else None

    scanner = LaunchPointScanner()

    # 获取所有交易日
    with get_db() as conn:
        if end_date:
            rows = conn.execute(
                "SELECT DISTINCT trade_date FROM stock_daily WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
                (start_date, end_date)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT DISTINCT trade_date FROM stock_daily WHERE trade_date >= ? ORDER BY trade_date",
                (start_date,)
            ).fetchall()

    trade_dates = [r[0] for r in rows]
    print(f"📅 从 {start_date} 到 {trade_dates[-1] if trade_dates else 'N/A'}，共 {len(trade_dates)} 个交易日")

    success = 0
    skipped = 0
    failed = 0

    for i, td in enumerate(trade_dates):
        # 检查是否已有该日期的扫描
        with get_db() as conn:
            existing = conn.execute(
                "SELECT COUNT(*) FROM launch_scan_results WHERE scan_date=?", (td,)
            ).fetchone()[0]

        if existing > 0:
            print(f"[{i+1}/{len(trade_dates)}] {td} ⏭️ 已有记录，跳过")
            skipped += 1
            continue

        try:
            result = scanner.run_scan_for_date(td, top_n=200)
            if 'error' in result:
                print(f"[{i+1}/{len(trade_dates)}] {td} ❌ {result['error']}")
                failed += 1
            else:
                print(f"[{i+1}/{len(trade_dates)}] {td} ✅ {result['candidates_count']}只候选 "
                      f"(扫描{result['total_scanned']}只, 大涨股{result['winner_count']}只)")
                success += 1
        except Exception as e:
            print(f"[{i+1}/{len(trade_dates)}] {td} ❌ 异常: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"✅ 成功: {success} | ⏭️ 跳过: {skipped} | ❌ 失败: {failed}")
    print(f"📊 总计: {len(trade_dates)} 个交易日")


if __name__ == '__main__':
    main()
