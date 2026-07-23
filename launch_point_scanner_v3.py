"""
起涨点扫描系统 V3 - 终极版
核心思路：
1. 不是所有低位股都会涨，需要找到「跌透+企稳」的特征
2. 加入严格的硬性条件过滤
3. 目标：50-200只高确定性候选
"""
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DB_PATH = Path("/workspace/stock_analyzer/data/stock_system.db")
OUTPUT_DIR = Path("/workspace/stock_analyzer/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def main():
    print("=" * 70)
    print("起涨点扫描 V3 - 精准版")
    print("=" * 70)

    conn = get_db()

    # 获取日期
    cur = conn.execute("SELECT MAX(trade_date) FROM stock_daily")
    latest_date = cur.fetchone()[0]
    cur = conn.execute("""
        SELECT DISTINCT trade_date FROM stock_daily
        WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT 120
    """, (latest_date,))
    all_dates = [r[0] for r in cur.fetchall()]
    all_dates.sort()
    print(f"扫描日期: {latest_date}, 回溯到: {all_dates[0]}")

    # =====================================================
    # 步骤1: 先从大涨股学习起涨点特征
    # =====================================================
    print("\n" + "-" * 50)
    print("步骤1: 学习大涨股起涨点")
    print("-" * 50)

    start_20d = all_dates[-20] if len(all_dates) >= 20 else all_dates[0]
    df_start = pd.read_sql_query("SELECT code, close FROM stock_daily WHERE trade_date = ?", conn, params=(start_20d,))
    df_end = pd.read_sql_query("SELECT code, close FROM stock_daily WHERE trade_date = ?", conn, params=(latest_date,))
    df = df_start.merge(df_end, on='code', suffixes=('_start', '_end'))
    df['pct_20d'] = (df['close_end'] - df['close_start']) / df['close_start'] * 100
    winners = df[(df['pct_20d'] >= 20) & ~df['code'].str.contains('900|200', na=False)]

    # 学习每只大涨股
    win_features = []
    for _, row in winners.iterrows():
        code = row['code']
        df_s = pd.read_sql_query("""
            SELECT * FROM stock_daily
            WHERE code = ? AND trade_date BETWEEN ? AND ?
            ORDER BY trade_date
        """, conn, params=(code, all_dates[0], latest_date))
        if len(df_s) < 40:
            continue

        df_s = df_s.reset_index(drop=True)
        # 在近20日找最低收盘价
        recent = df_s.iloc[-20:]
        min_idx = recent['close'].idxmin()
        if min_idx >= len(df_s) - 5:
            older = df_s.iloc[:-5]
            if len(older) > 5:
                min_idx = older['close'].idxmin()

        lp_row = df_s.loc[min_idx]

        # 起涨前特征
        before = df_s.loc[:min_idx]
        after = df_s.loc[min_idx:]

        # 连跌天数
        down_days = 0
        for j in range(len(before) - 1, max(0, len(before) - 20), -1):
            if before.iloc[j]['pct_change'] is not None and before.iloc[j]['pct_change'] < 0:
                down_days += 1
            else:
                break

        # 近20日价格分位数
        close_recent = recent['close']
        price_pct = (lp_row['close'] - close_recent.min()) / (close_recent.max() - close_recent.min()) * 100 if close_recent.max() > close_recent.min() else 50

        # 60日回撤
        lookback = df_s.iloc[-min(60, len(df_s)):]
        dd_60 = (lp_row['close'] - lookback['high'].max()) / lookback['high'].max() * 100

        # 是否触及布林下轨
        boll_pos = None
        if lp_row['boll_lower'] and lp_row['boll_upper'] and lp_row['boll_upper'] > lp_row['boll_lower']:
            boll_pos = (lp_row['close'] - lp_row['boll_lower']) / (lp_row['boll_upper'] - lp_row['boll_lower'])

        # MA偏离
        dev_ma60 = (lp_row['close'] - lp_row['ma60']) / lp_row['ma60'] * 100 if lp_row['ma60'] and lp_row['ma60'] > 0 else None
        dev_ma20 = (lp_row['close'] - lp_row['ma20']) / lp_row['ma20'] * 100 if lp_row['ma20'] and lp_row['ma20'] > 0 else None

        # KDJ金叉（起涨前是否出现过）
        kdj_golden = 0
        for j in range(max(0, len(before) - 10), len(before)):
            if before.iloc[j]['kdj_k'] is not None and before.iloc[j]['kdj_d'] is not None:
                if before.iloc[j]['kdj_k'] > before.iloc[j]['kdj_d']:
                    kdj_golden = 1
                    break

        win_features.append({
            'code': code,
            'pct_20d': row['pct_20d'],
            'kdj_k': lp_row['kdj_k'],
            'rsi14': lp_row['rsi14'],
            'dev_ma60': dev_ma60,
            'down_days': down_days,
            'price_pct': price_pct,
            'dd_60': dd_60,
            'boll_pos': boll_pos,
            'vol_ratio': lp_row['volume_ratio'],
            'macd_dif': lp_row['macd_dif'],
            'macd_hist': lp_row['macd_hist'],
            'pct_change': lp_row['pct_change'],
        })

    df_w = pd.DataFrame(win_features).dropna(subset=['kdj_k', 'rsi14', 'dev_ma60', 'dd_60', 'boll_pos'])
    print(f"有效大涨股样本: {len(df_w)}")

    # 打印特征总结
    print(f"\n起涨点核心特征:")
    print(f"  KDJ_K: 均值{df_w['kdj_k'].mean():.1f}, P25={df_w['kdj_k'].quantile(0.25):.1f}, P75={df_w['kdj_k'].quantile(0.75):.1f}")
    print(f"  RSI14: 均值{df_w['rsi14'].mean():.1f}, P25={df_w['rsi14'].quantile(0.25):.1f}, P75={df_w['rsi14'].quantile(0.75):.1f}")
    print(f"  MA60偏离: 均值{df_w['dev_ma60'].mean():.1f}%, P25={df_w['dev_ma60'].quantile(0.25):.1f}%, P75={df_w['dev_ma60'].quantile(0.75):.1f}%")
    print(f"  连跌天数: 均值{df_w['down_days'].mean():.1f}, P75={df_w['down_days'].quantile(0.75):.1f}")
    print(f"  60日回撤: 均值{df_w['dd_60'].mean():.1f}%, P25={df_w['dd_60'].quantile(0.25):.1f}%")
    print(f"  布林位置: 均值{df_w['boll_pos'].mean():.2f}")
    print(f"  量比: 均值{df_w['vol_ratio'].mean():.2f}")
    print(f"  当日下跌: {(df_w['pct_change'] < 0).mean()*100:.0f}%")
    print(f"  KDJ<30: {(df_w['kdj_k'] < 30).mean()*100:.0f}%")
    print(f"  RSI<35: {(df_w['rsi14'] < 35).mean()*100:.0f}%")
    print(f"  破MA60: {(df_w['dev_ma60'] < 0).mean()*100:.0f}%")
    print(f"  连跌>=3: {(df_w['down_days'] >= 3).mean()*100:.0f}%")

    # =====================================================
    # 步骤2: 构建精准筛选条件
    # =====================================================
    print("\n" + "-" * 50)
    print("步骤2: 全市场扫描")
    print("-" * 50)

    # 加载全市场最新数据 + 近60日数据
    df_today = pd.read_sql_query("""
        SELECT code, trade_date, close, open, high, low, pct_change, amplitude,
               kdj_k, kdj_d, kdj_j,
               macd_dif, macd_dea, macd_hist,
               ma5, ma10, ma20, ma60, ma120, ma250,
               rsi6, rsi14, rsi24,
               volume_ratio, turnover_rate, volume,
               boll_upper, boll_mid, boll_lower
        FROM stock_daily WHERE trade_date = ?
    """, conn, params=(latest_date,))

    # 过滤沪深市场
    df_today = df_today[df_today['code'].str.match(r'^(00|30|60|68)')].copy()
    print(f"沪深市场: {len(df_today)}只")

    # 加载近60日数据
    df_hist = pd.read_sql_query("""
        SELECT code, trade_date, close, high, low, pct_change, volume
        FROM stock_daily
        WHERE trade_date BETWEEN ? AND ?
        ORDER BY code, trade_date
    """, conn, params=(all_dates[0], latest_date))

    conn.close()

    # 分组计算衍生指标
    print("计算衍生指标...")
    grouped = df_hist.groupby('code')

    def calc_derived(grp):
        grp = grp.sort_values('trade_date').reset_index(drop=True)
        n = len(grp)
        if n < 10:
            return pd.Series({
                'down_days': np.nan, 'price_pct_20d': np.nan, 'dd_60': np.nan,
                'vol_5d_avg': np.nan, 'prev_volume': np.nan,
                'kdj_golden': np.nan, 'lowest_in_5d': np.nan,
                'vol_shrink_3d': np.nan, 'near_low_10d': np.nan,
            })

        last_close = grp.iloc[-1]['close']

        # 连跌天数
        down_days = 0
        for j in range(n - 1, max(0, n - 20), -1):
            if grp.iloc[j]['pct_change'] is not None and grp.iloc[j]['pct_change'] < 0:
                down_days += 1
            else:
                break

        # 20日价格分位数
        recent_n = min(20, n)
        recent = grp.iloc[-recent_n:]
        c_min, c_max = recent['close'].min(), recent['close'].max()
        price_pct = (last_close - c_min) / (c_max - c_min) * 100 if c_max > c_min else 50

        # 60日最大回撤
        lb_n = min(60, n)
        lb = grp.iloc[-lb_n:]
        dd_60 = (last_close - lb['high'].max()) / lb['high'].max() * 100

        # 近5日均量
        vol_5d = grp.iloc[-5:]['volume'].mean() if n >= 5 else grp['volume'].mean()

        # 前一日成交量
        prev_vol = grp.iloc[-2]['volume'] if n >= 2 else grp.iloc[-1]['volume']

        # 5日内最低价是否在今天附近
        last5_low = grp.iloc[-5:]['low'].min() if n >= 5 else grp.iloc[-1]['low']
        lowest_in_5d = 1 if last_close <= last5_low * 1.01 else 0

        # 近3日缩量（每日量递减）
        vol_shrink = 0
        if n >= 4:
            v3, v2, v1 = grp.iloc[-3]['volume'], grp.iloc[-2]['volume'], grp.iloc[-1]['volume']
            if v1 < v2 < v3:
                vol_shrink = 1

        # 近10日最低价附近（5%以内）
        low10 = grp.iloc[-10:]['low'].min() if n >= 10 else last5_low
        near_low_10d = 1 if last_close <= low10 * 1.05 else 0

        return pd.Series({
            'down_days': down_days,
            'price_pct_20d': price_pct,
            'dd_60': dd_60,
            'vol_5d_avg': vol_5d,
            'prev_volume': prev_vol,
            'kdj_golden': 0,  # 后续单独计算
            'lowest_in_5d': lowest_in_5d,
            'vol_shrink_3d': vol_shrink,
            'near_low_10d': near_low_10d,
        })

    features = grouped.apply(calc_derived).reset_index()

    # 合并
    df = df_today.merge(features, on='code', how='inner')

    # 计算均线偏离和布林位置
    df['dev_ma60'] = np.where((df['ma60'].notna()) & (df['ma60'] > 0),
                               (df['close'] - df['ma60']) / df['ma60'] * 100, np.nan)
    df['dev_ma20'] = np.where((df['ma20'].notna()) & (df['ma20'] > 0),
                               (df['close'] - df['ma20']) / df['ma20'] * 100, np.nan)
    df['boll_pos'] = np.where(
        (df['boll_upper'].notna()) & (df['boll_lower'].notna()) & (df['boll_upper'] > df['boll_lower']),
        (df['close'] - df['boll_lower']) / (df['boll_upper'] - df['boll_lower']), np.nan
    )

    # 清理
    df = df.dropna(subset=['kdj_k', 'rsi14', 'dev_ma60', 'dd_60', 'boll_pos', 'down_days'])
    print(f"有完整数据: {len(df)}只")

    # =====================================================
    # 步骤3: 精准筛选 - 硬条件 + 软评分
    # =====================================================
    print("\n" + "-" * 50)
    print("步骤3: 精准筛选")
    print("-" * 50)

    # --- 硬条件过滤 ---
    # 基于大涨股特征：KDJ<30占60%，RSI<35占59%，破MA60占86%，连跌>=3占45%
    # 但为了缩小范围，采用更严格的条件：

    conditions = {
        'A_KDJ超卖': df['kdj_k'] < 35,           # KDJ偏低
        'B_RSI弱势': df['rsi14'] < 45,            # RSI弱势
        'C_破MA60': df['dev_ma60'] < -3,          # 至少略破MA60
        'D_60日回撤': df['dd_60'] < -15,          # 回撤超15%
        'E_缩量': df['volume_ratio'] < 1.2,       # 未明显放量
        'F_连跌': df['down_days'] >= 2,           # 至少连跌2天
        'G_当日跌或平': df['pct_change'] < 0.5,    # 当日未大涨
        'H_布林下半区': df['boll_pos'] < 0.4,      # 布林下半区
    }

    # 所有条件都满足
    mask = np.ones(len(df), dtype=bool)
    for name, cond in conditions.items():
        mask = mask & cond
        print(f"  {name}: {cond.sum()}只 -> 累计{mask.sum()}只")

    df_filtered = df[mask].copy()
    print(f"\n硬条件过滤后: {len(df_filtered)}只")

    if len(df_filtered) == 0:
        print("无候选！放宽条件...")
        # 放宽
        mask = np.ones(len(df), dtype=bool)
        relaxed = {
            'A_KDJ': df['kdj_k'] < 50,
            'B_RSI': df['rsi14'] < 55,
            'C_MA60': df['dev_ma60'] < 0,
            'D_回撤': df['dd_60'] < -10,
            'E_量': df['volume_ratio'] < 1.5,
        }
        for name, cond in relaxed.items():
            mask = mask & cond
        df_filtered = df[mask].copy()
        print(f"放宽后: {len(df_filtered)}只")

    # --- 综合评分 ---
    def calc_score(row):
        score = 0

        # KDJ (0-5分)
        k = row['kdj_k']
        if k < 15: score += 5
        elif k < 22: score += 4
        elif k < 30: score += 3
        elif k < 40: score += 1

        # RSI (0-4分)
        r = row['rsi14']
        if r < 25: score += 4
        elif r < 30: score += 3
        elif r < 35: score += 2
        elif r < 42: score += 1

        # MA60偏离 (0-5分)
        d = row['dev_ma60']
        if d < -25: score += 5
        elif d < -18: score += 4
        elif d < -12: score += 3
        elif d < -6: score += 2
        elif d < -3: score += 1

        # 连跌 (0-4分)
        dd = row['down_days']
        if dd >= 6: score += 4
        elif dd >= 4: score += 3
        elif dd >= 3: score += 2
        elif dd >= 2: score += 1

        # 60日回撤 (0-3分)
        d60 = row['dd_60']
        if d60 < -40: score += 3
        elif d60 < -30: score += 2
        elif d60 < -20: score += 1

        # 价格分位数 (0-3分)
        pp = row['price_pct_20d']
        if pp < 5: score += 3
        elif pp < 15: score += 2
        elif pp < 25: score += 1

        # MACD (0-2分)
        if row['macd_dif'] < 0 and row['macd_hist'] < 0:
            score += 2
        elif row['macd_dif'] < 0:
            score += 1

        # 成交量 (0-2分)
        vr = row['volume_ratio']
        if vr < 0.6: score += 2
        elif vr < 0.8: score += 1

        # 布林 (0-1分)
        if row['boll_pos'] < 0.1: score += 1

        # 额外加分：近10日最低价附近
        if row.get('near_low_10d', 0): score += 1

        return score

    df_filtered['score'] = df_filtered.apply(calc_score, axis=1)
    df_filtered = df_filtered.sort_values('score', ascending=False)

    # 得分分布
    print(f"\n候选得分分布:")
    for t in [28, 26, 24, 22, 20, 18, 16, 14]:
        cnt = (df_filtered['score'] >= t).sum()
        print(f"  >= {t}分: {cnt}只")

    # 取Top候选
    top_n = 200
    df_top = df_filtered.head(top_n)

    # =====================================================
    # 步骤4: 生成报告
    # =====================================================
    print("\n" + "-" * 50)
    print("步骤4: 生成报告")
    print("-" * 50)

    # 按得分分组
    df_top = df_top.copy()
    df_top['ma60_label'] = df_top['dev_ma60'].apply(
        lambda x: f"{x:+.1f}%")
    df_top['dd60_label'] = df_top['dd_60'].apply(
        lambda x: f"{x:+.1f}%")

    # 生成详细报告
    md_path = OUTPUT_DIR / "launch_point_candidates_v3.md"
    lines = [
        f"# 🎯 起涨点精准扫描报告 V3",
        f"",
        f"**扫描日期**: {latest_date}",
        f"**市场范围**: 沪深主板 + 创业板 + 科创板 ({len(df)}只)",
        f"**筛选逻辑**: 硬条件初筛 → 综合评分排序 → Top {top_n}",
        f"**硬条件**: KDJ<35, RSI<45, 破MA60(>3%), 60日回撤>15%, 缩量, 连跌>=2天, 当日未大涨",
        f"",
        f"## 📊 大涨股起涨点特征（{len(df_w)}只样本）",
        f"",
        f"| 指标 | 均值 | P25 | P75 | 占比 |",
        f"|------|------|-----|-----|------|",
        f"| KDJ_K | {df_w['kdj_k'].mean():.1f} | {df_w['kdj_k'].quantile(0.25):.1f} | {df_w['kdj_k'].quantile(0.75):.1f} | K<30: {(df_w['kdj_k'] < 30).mean()*100:.0f}% |",
        f"| RSI14 | {df_w['rsi14'].mean():.1f} | {df_w['rsi14'].quantile(0.25):.1f} | {df_w['rsi14'].quantile(0.75):.1f} | RSI<35: {(df_w['rsi14'] < 35).mean()*100:.0f}% |",
        f"| MA60偏离 | {df_w['dev_ma60'].mean():.1f}% | {df_w['dev_ma60'].quantile(0.25):.1f}% | {df_w['dev_ma60'].quantile(0.75):.1f}% | 破MA60: {(df_w['dev_ma60'] < 0).mean()*100:.0f}% |",
        f"| 连跌天数 | {df_w['down_days'].mean():.1f} | {df_w['down_days'].quantile(0.25):.1f} | {df_w['down_days'].quantile(0.75):.1f} | >=3天: {(df_w['down_days'] >= 3).mean()*100:.0f}% |",
        f"| 60日回撤 | {df_w['dd_60'].mean():.1f}% | {df_w['dd_60'].quantile(0.25):.1f}% | - | >20%: {(df_w['dd_60'] < -20).mean()*100:.0f}% |",
        f"| 量比 | {df_w['vol_ratio'].mean():.2f} | - | - | <1: {(df_w['vol_ratio'] < 1).mean()*100:.0f}% |",
        f"| 当日下跌 | - | - | - | {(df_w['pct_change'] < 0).mean()*100:.0f}% |",
        f"",
        f"## 🏆 起涨点候选 Top {top_n}",
        f"",
        f"*按综合评分排序，得分越高越符合大涨股的起涨点特征*",
        f"",
    ]

    # 按得分分组输出
    for score in sorted(df_top['score'].unique(), reverse=True):
        group = df_top[df_top['score'] == score]
        lines.append(f"### ⭐ {int(score)}分段 ({len(group)}只)")
        lines.append("")
        lines.append("| # | 代码 | 收盘 | 涨跌% | K | RSI | MA60 | 连跌 | 回撤 | 量比 | 分位% | MACD |")
        lines.append("|---|------|------|-------|----|-----|------|------|------|------|-------|------|")
        for i, (_, r) in enumerate(group.iterrows()):
            macd_sign = "🔴零下绿" if (r['macd_dif'] < 0 and r['macd_hist'] < 0) else \
                        ("🟡零下红" if r['macd_dif'] < 0 else "🟢零上")
            lines.append(
                f"| {i+1} | {r['code']} | {r['close']:.2f} | {r['pct_change']:+.2f}% | "
                f"{r['kdj_k']:.0f} | {r['rsi14']:.0f} | {r['dev_ma60']:+.0f}% | "
                f"{r['down_days']:.0f}天 | {r['dd_60']:+.0f}% | {r['volume_ratio']:.2f} | "
                f"{r['price_pct_20d']:.0f}% | {macd_sign} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## 💡 使用说明",
        "",
        "1. **起涨点**不是精确的买入点，而是一个**高概率区域**",
        "2. 高分候选意味着当前状态与历史大涨股起涨前状态高度相似",
        "3. 建议结合以下因素进一步筛选：",
        "   - 行业/板块热度",
        "   - 近期是否有催化剂（业绩预告、政策利好等）",
        "   - KDJ是否出现金叉",
        "   - 成交量是否极度萎缩后开始温和放量",
        "4. **风险提示**: 历史规律不代表未来，仅供参考",
        "",
        "---",
        f"*报告由起涨点扫描系统V3自动生成 | {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
    ]

    md_path.write_text('\n'.join(lines), encoding='utf-8')

    # CSV
    csv_path = OUTPUT_DIR / "launch_point_candidates_v3.csv"
    df_top[['code', 'close', 'pct_change', 'score', 'kdj_k', 'rsi14',
            'dev_ma60', 'down_days', 'dd_60', 'volume_ratio', 'price_pct_20d',
            'macd_dif', 'macd_hist', 'boll_pos']].to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"\n报告: {md_path}")
    print(f"CSV: {csv_path}")

    # 打印TOP 30
    print("\n" + "=" * 70)
    print(f"TOP 30 起涨点候选")
    print("=" * 70)
    print(f"{'#':<3} {'代码':<8} {'收盘':>8} {'涨跌':>8} {'得分':>4} {'K':>5} {'RSI':>5} {'MA60':>7} {'连跌':>4} {'回撤':>7} {'量比':>5} {'分位':>5}")
    print("-" * 85)
    for i, (_, r) in enumerate(df_top.head(30).iterrows()):
        print(f"{i+1:<3} {r['code']:<8} {r['close']:>8.2f} {r['pct_change']:>+7.2f}% {r['score']:>4.0f} "
              f"{r['kdj_k']:>5.0f} {r['rsi14']:>5.0f} {r['dev_ma60']:>+6.0f}% "
              f"{r['down_days']:>4.0f}天 {r['dd_60']:>+6.0f}% {r['volume_ratio']:>5.2f} {r['price_pct_20d']:>4.0f}%")

    print(f"\n共 {len(df_top)} 只候选")
    print(f"完整报告: {md_path}")

    return md_path


if __name__ == "__main__":
    main()
