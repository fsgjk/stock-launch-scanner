"""
起涨点扫描系统 V2 - 增强版
- 多维度评分（满分30分）
- 加入连续下跌天数、价格分位数、累计跌幅等区分维度
- 目标：筛选出50-200只高确定性候选
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


def step1_extract_launch_points():
    """提取大涨股起涨点特征"""
    print("=" * 70)
    print("步骤1: 提取起涨点特征")
    print("=" * 70)

    conn = get_db()

    cur = conn.execute("SELECT MAX(trade_date) FROM stock_daily")
    latest_date = cur.fetchone()[0]

    cur = conn.execute("""
        SELECT DISTINCT trade_date FROM stock_daily
        WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT 20
    """, (latest_date,))
    dates = [r[0] for r in cur.fetchall()]
    start_date, end_date = dates[-1], dates[0]
    print(f"区间: {start_date} ~ {end_date}")

    df_start = pd.read_sql_query("SELECT code, close FROM stock_daily WHERE trade_date = ?", conn, params=(start_date,))
    df_end = pd.read_sql_query("SELECT code, close FROM stock_daily WHERE trade_date = ?", conn, params=(end_date,))
    df = df_start.merge(df_end, on='code', suffixes=('_start', '_end'))
    df['pct_20d'] = (df['close_end'] - df['close_start']) / df['close_start'] * 100
    winners = df[(df['pct_20d'] >= 20) & ~df['code'].str.contains('900|200', na=False)].copy()
    print(f"大涨股: {len(winners)}只")

    # 获取20日区间内前10个和后10个交易日
    cur = conn.execute("""
        SELECT DISTINCT trade_date FROM stock_daily
        WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT 60
    """, (latest_date,))
    all_dates = [r[0] for r in cur.fetchall()]
    all_dates.sort()
    lookback_start = all_dates[0]  # 约60个交易日前

    launch_points = []
    for i, (_, row) in enumerate(winners.iterrows()):
        code = row['code']
        if i % 40 == 0:
            print(f"  处理 {i}/{len(winners)}: {code}")

        # 获取更长时间的数据用于精确定位
        df_stock = pd.read_sql_query("""
            SELECT trade_date, open, high, low, close, volume, pct_change,
                   kdj_k, kdj_d, kdj_j,
                   macd_dif, macd_dea, macd_hist,
                   ma5, ma10, ma20, ma60, ma120, ma250,
                   rsi6, rsi14, rsi24,
                   boll_upper, boll_mid, boll_lower,
                   volume_ratio, turnover_rate, amplitude
            FROM stock_daily
            WHERE code = ? AND trade_date BETWEEN ? AND ?
            ORDER BY trade_date
        """, conn, params=(code, lookback_start, end_date))

        if len(df_stock) < 30:
            continue

        df_stock = df_stock.reset_index(drop=True)

        # 方法：从最新日向前找局部最低点
        # 在最近20个交易日内找最低收盘价
        recent_n = min(20, len(df_stock))
        recent = df_stock.iloc[-recent_n:]
        min_idx = recent['close'].idxmin()

        # 如果最低点在最后5天，说明可能还在跌，往前找
        if min_idx >= len(df_stock) - 5:
            older = df_stock.iloc[:-5]
            if len(older) > 5:
                min_idx = older['close'].idxmin()

        launch_row = df_stock.loc[min_idx]
        launch_date = launch_row['trade_date']

        # 确认这个点之后确实涨了
        after = df_stock.loc[min_idx:]
        if len(after) < 5:
            continue

        max_after_close = after['close'].max()
        launch_close = launch_row['close']
        gain_after = (max_after_close - launch_close) / launch_close * 100
        if gain_after < 10:  # 起涨后至少涨10%
            continue

        # 计算起涨前连续下跌天数
        before = df_stock.loc[:min_idx]
        down_days = 0
        for j in range(len(before) - 1, max(0, len(before) - 15), -1):
            if before.iloc[j]['pct_change'] is not None and before.iloc[j]['pct_change'] < 0:
                down_days += 1
            else:
                break

        # 起涨前5日累计跌幅
        if len(before) >= 5:
            pre5_start = before.iloc[-5]['close']
            pre5_cum = (launch_close - pre5_start) / pre5_start * 100
        else:
            pre5_cum = 0

        # 近20日价格分位数
        close_series = recent['close']
        price_percentile = (launch_close - close_series.min()) / (close_series.max() - close_series.min()) * 100 if close_series.max() > close_series.min() else 50

        # 距离60日最高点的回撤
        if len(df_stock) >= 60:
            max60 = df_stock.iloc[-60:]['high'].max()
            drawdown_60d = (launch_close - max60) / max60 * 100
        else:
            drawdown_60d = (launch_close - df_stock['high'].max()) / df_stock['high'].max() * 100

        lp = {
            'code': code,
            'launch_date': launch_date,
            'launch_close': launch_close,
            'gain_after': gain_after,
            'pct_20d': row['pct_20d'],
            # 当日特征
            'pct_change': launch_row['pct_change'],
            'amplitude': launch_row['amplitude'],
            'kdj_k': launch_row['kdj_k'],
            'kdj_d': launch_row['kdj_d'],
            'kdj_j': launch_row['kdj_j'],
            'macd_dif': launch_row['macd_dif'],
            'macd_dea': launch_row['macd_dea'],
            'macd_hist': launch_row['macd_hist'],
            'rsi6': launch_row['rsi6'],
            'rsi14': launch_row['rsi14'],
            'rsi24': launch_row['rsi24'],
            'volume_ratio': launch_row['volume_ratio'],
            'turnover_rate': launch_row['turnover_rate'],
            # 新增维度
            'down_days': down_days,
            'pre5_cum_pct': pre5_cum,
            'price_percentile': price_percentile,
            'drawdown_60d': drawdown_60d,
        }

        # 均线偏离
        for ma_name, ma_col in [('ma5', 'ma5'), ('ma10', 'ma10'), ('ma20', 'ma20'), ('ma60', 'ma60'), ('ma120', 'ma120'), ('ma250', 'ma250')]:
            ma_val = launch_row[ma_col]
            if ma_val and ma_val > 0:
                lp[f'dev_{ma_name}'] = (launch_close - ma_val) / ma_val * 100
            else:
                lp[f'dev_{ma_name}'] = None

        # 布林带
        if launch_row['boll_lower'] and launch_row['boll_upper'] and (launch_row['boll_upper'] - launch_row['boll_lower']) > 0:
            lp['boll_position'] = (launch_close - launch_row['boll_lower']) / (launch_row['boll_upper'] - launch_row['boll_lower'])
        else:
            lp['boll_position'] = None

        launch_points.append(lp)

    conn.close()
    df_lp = pd.DataFrame(launch_points)
    print(f"\n提取到 {len(df_lp)} 个有效起涨点")
    return df_lp


def step2_analyze(df_lp):
    """深度统计分析"""
    print("\n" + "=" * 70)
    print("步骤2: 深度统计分析")
    print("=" * 70)

    df = df_lp.dropna(subset=['kdj_k', 'rsi14', 'dev_ma60', 'down_days', 'price_percentile']).copy()
    print(f"有效样本: {len(df)}")

    stats = {}

    metrics = {
        'kdj_k': 'KDJ_K',
        'rsi14': 'RSI14',
        'dev_ma60': '偏离MA60(%)',
        'dev_ma20': '偏离MA20(%)',
        'down_days': '连跌天数',
        'pre5_cum_pct': '前5日累计跌幅(%)',
        'price_percentile': '20日价格分位数(%)',
        'drawdown_60d': '60日最大回撤(%)',
        'volume_ratio': '量比',
        'pct_change': '当日涨跌幅(%)',
        'boll_position': '布林带位置',
    }

    for col, name in metrics.items():
        if col in df.columns:
            data = df[col].dropna()
            stats[name] = {
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'p10': data.quantile(0.1),
                'p25': data.quantile(0.25),
                'p75': data.quantile(0.75),
                'p90': data.quantile(0.9),
            }
            print(f"\n{name}: 均值={data.mean():.2f}, 中位数={data.median():.2f}, "
                  f"P25={data.quantile(0.25):.2f}, P75={data.quantile(0.75):.2f}")

    # 关键比例
    print("\n--- 关键比例 ---")
    print(f"KDJ_K < 30: {(df['kdj_k'] < 30).mean()*100:.0f}%")
    print(f"RSI14 < 35: {(df['rsi14'] < 35).mean()*100:.0f}%")
    print(f"跌破MA60: {(df['dev_ma60'] < 0).mean()*100:.0f}%")
    print(f"跌破MA60超10%: {(df['dev_ma60'] < -10).mean()*100:.0f}%")
    print(f"连跌>=3天: {(df['down_days'] >= 3).mean()*100:.0f}%")
    print(f"价格分位数<20%: {(df['price_percentile'] < 20).mean()*100:.0f}%")
    print(f"60日回撤>15%: {(df['drawdown_60d'] < -15).mean()*100:.0f}%")
    print(f"量比<1: {(df['volume_ratio'] < 1).mean()*100:.0f}%")
    print(f"当日下跌: {(df['pct_change'] < 0).mean()*100:.0f}%")

    return df, stats


def step3_build_model(df_lp):
    """构建增强评分模型（满分30分）"""
    print("\n" + "=" * 70)
    print("步骤3: 构建增强评分模型")
    print("=" * 70)

    scoring_rules = {
        # KDJ位置 (5分) - 核心指标
        'kdj': {
            'weight': 5,
            'rules': [
                ('kdj_k < 15', 5, 'KDJ极度超卖(K<15)'),
                ('kdj_k >= 15 and kdj_k < 25', 4, 'KDJ深度超卖(K 15-25)'),
                ('kdj_k >= 25 and kdj_k < 35', 3, 'KDJ超卖(K 25-35)'),
                ('kdj_k >= 35 and kdj_k < 50', 1, 'KDJ偏低(K 35-50)'),
                ('kdj_k >= 50', 0, 'KDJ中性'),
            ]
        },

        # RSI位置 (4分)
        'rsi': {
            'weight': 4,
            'rules': [
                ('rsi14 < 25', 4, 'RSI极度超卖(<25)'),
                ('rsi14 >= 25 and rsi14 < 32', 3, 'RSI深度超卖(25-32)'),
                ('rsi14 >= 32 and rsi14 < 38', 2, 'RSI超卖(32-38)'),
                ('rsi14 >= 38 and rsi14 < 45', 1, 'RSI偏弱(38-45)'),
                ('rsi14 >= 45', 0, 'RSI中性'),
            ]
        },

        # 均线偏离 (5分) - 核心指标
        'ma_dev': {
            'weight': 5,
            'rules': [
                ('dev_ma60 < -20', 5, '深度破MA60(<-20%)'),
                ('dev_ma60 >= -20 and dev_ma60 < -12', 4, '跌破MA60(-20~-12%)'),
                ('dev_ma60 >= -12 and dev_ma60 < -5', 3, '跌破MA60(-12~-5%)'),
                ('dev_ma60 >= -5 and dev_ma60 < 0', 2, '略破MA60(-5~0%)'),
                ('dev_ma60 >= 0 and dev_ma60 < 8', 1, '略高于MA60'),
                ('dev_ma60 >= 8', 0, '远离MA60上方'),
            ]
        },

        # 连跌天数 (4分) - 新增区分维度
        'down_days': {
            'weight': 4,
            'rules': [
                ('down_days >= 5', 4, '连跌5天+'),
                ('down_days >= 3 and down_days < 5', 3, '连跌3-4天'),
                ('down_days >= 2 and down_days < 3', 2, '连跌2天'),
                ('down_days >= 1 and down_days < 2', 1, '跌1天'),
                ('down_days < 1', 0, '未跌'),
            ]
        },

        # 价格分位数 (4分) - 新增区分维度
        'price_percentile': {
            'weight': 4,
            'rules': [
                ('price_percentile < 10', 4, '价格在20日最低10%'),
                ('price_percentile >= 10 and price_percentile < 25', 3, '价格在20日低10-25%'),
                ('price_percentile >= 25 and price_percentile < 40', 2, '价格在20日低25-40%'),
                ('price_percentile >= 40 and price_percentile < 60', 1, '价格在20日中位'),
                ('price_percentile >= 60', 0, '价格偏高'),
            ]
        },

        # 60日回撤 (3分)
        'drawdown': {
            'weight': 3,
            'rules': [
                ('drawdown_60d < -30', 3, '60日回撤>30%'),
                ('drawdown_60d >= -30 and drawdown_60d < -20', 2, '60日回撤20-30%'),
                ('drawdown_60d >= -20 and drawdown_60d < -10', 1, '60日回撤10-20%'),
                ('drawdown_60d >= -10', 0, '回撤较小'),
            ]
        },

        # MACD状态 (2分)
        'macd': {
            'weight': 2,
            'rules': [
                ('macd_dif < 0 and macd_hist < 0', 2, 'MACD零轴下绿柱'),
                ('macd_dif < 0 and macd_hist >= 0', 1, 'MACD零轴下红柱'),
                ('macd_dif >= 0', 0, 'MACD零轴上'),
            ]
        },

        # 成交量 (2分)
        'volume': {
            'weight': 2,
            'rules': [
                ('volume_ratio < 0.6', 2, '极度缩量(<0.6)'),
                ('volume_ratio >= 0.6 and volume_ratio < 0.85', 1, '缩量(0.6-0.85)'),
                ('volume_ratio >= 0.85', 0, '量正常/放量'),
            ]
        },

        # 布林带 (1分)
        'boll': {
            'weight': 1,
            'rules': [
                ('boll_position < 0.1', 1, '触及布林下轨'),
                ('boll_position >= 0.1', 0, '非下轨'),
            ]
        },
    }

    total_weight = sum(c['weight'] for c in scoring_rules.values())
    print(f"总分: {total_weight}")

    # 在样本上验证
    df = df_lp.dropna(subset=['kdj_k', 'rsi14', 'dev_ma60', 'down_days', 'price_percentile',
                               'drawdown_60d', 'macd_dif', 'macd_hist', 'volume_ratio', 'boll_position']).copy()

    scores = []
    for _, row in df.iterrows():
        score = 0
        local_vars = row.to_dict()
        for name, cfg in scoring_rules.items():
            for condition, points, _ in cfg['rules']:
                try:
                    if eval(condition, {"__builtins__": {}}, local_vars):
                        score += points
                        break
                except:
                    pass
        scores.append(score)

    df['score'] = scores
    scores_arr = np.array(scores)

    print(f"\n样本评分分布 (满分{total_weight}):")
    print(f"  均值: {scores_arr.mean():.1f}, 中位数: {np.median(scores_arr):.1f}")
    for t in [28, 26, 24, 22, 20, 18, 16, 14, 12]:
        pct = (scores_arr >= t).mean() * 100
        print(f"  >= {t}分: {pct:.0f}%")

    # 选阈值：覆盖60-70%大涨股
    for t in range(total_weight, 0, -1):
        if (scores_arr >= t).mean() >= 0.65:
            best_threshold = t
            break

    print(f"\n推荐阈值: >= {best_threshold}分 (覆盖65%大涨股)")

    return scoring_rules, best_threshold


def step4_scan_market(scoring_rules, threshold):
    """全市场扫描"""
    print("\n" + "=" * 70)
    print("步骤4: 全市场扫描")
    print("=" * 70)

    conn = get_db()

    cur = conn.execute("SELECT MAX(trade_date) FROM stock_daily")
    latest_date = cur.fetchone()[0]

    # 获取最近60个交易日日期列表
    cur = conn.execute("""
        SELECT DISTINCT trade_date FROM stock_daily
        WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT 60
    """, (latest_date,))
    all_dates = [r[0] for r in cur.fetchall()]
    all_dates.sort()
    lookback_start = all_dates[0]

    print(f"扫描日期: {latest_date}, 回溯: {lookback_start}")

    # 加载所有股票的最新日数据
    print("加载最新日数据...")
    df_today = pd.read_sql_query("""
        SELECT code, trade_date, close, open, pct_change, amplitude,
               kdj_k, kdj_d, kdj_j,
               macd_dif, macd_dea, macd_hist,
               ma5, ma10, ma20, ma60, ma120, ma250,
               rsi6, rsi14, rsi24,
               volume_ratio, turnover_rate,
               boll_upper, boll_mid, boll_lower
        FROM stock_daily WHERE trade_date = ?
    """, conn, params=(latest_date,))

    # 过滤：沪深主板+创业板+科创板
    df_today = df_today[df_today['code'].str.match(r'^(00|30|60|68)')].copy()
    print(f"沪深市场股票: {len(df_today)}")

    # 批量加载近60日数据
    print("加载近60日数据用于计算衍生指标...")
    df_hist = pd.read_sql_query("""
        SELECT code, trade_date, close, high, pct_change
        FROM stock_daily
        WHERE trade_date BETWEEN ? AND ?
        ORDER BY code, trade_date
    """, conn, params=(lookback_start, latest_date))

    conn.close()

    # 为每只股票计算衍生指标
    print("计算衍生指标...")

    # 分组计算
    grouped = df_hist.groupby('code')

    def calc_features(grp):
        grp = grp.sort_values('trade_date').reset_index(drop=True)
        if len(grp) < 10:
            return pd.Series({
                'down_days': np.nan, 'pre5_cum_pct': np.nan,
                'price_percentile': np.nan, 'drawdown_60d': np.nan
            })

        last_close = grp.iloc[-1]['close']

        # 连跌天数
        down_days = 0
        for j in range(len(grp) - 1, max(0, len(grp) - 20), -1):
            if grp.iloc[j]['pct_change'] is not None and grp.iloc[j]['pct_change'] < 0:
                down_days += 1
            else:
                break

        # 前5日累计跌幅
        if len(grp) >= 5:
            pre5_start = grp.iloc[-5]['close']
            pre5_cum = (last_close - pre5_start) / pre5_start * 100
        else:
            pre5_cum = 0

        # 20日价格分位数
        recent_n = min(20, len(grp))
        recent = grp.iloc[-recent_n:]
        c_min, c_max = recent['close'].min(), recent['close'].max()
        price_percentile = (last_close - c_min) / (c_max - c_min) * 100 if c_max > c_min else 50

        # 60日最大回撤
        lookback_n = min(60, len(grp))
        lookback = grp.iloc[-lookback_n:]
        max_high = lookback['high'].max()
        drawdown_60d = (last_close - max_high) / max_high * 100

        return pd.Series({
            'down_days': down_days,
            'pre5_cum_pct': pre5_cum,
            'price_percentile': price_percentile,
            'drawdown_60d': drawdown_60d
        })

    features = grouped.apply(calc_features).reset_index()
    print(f"  完成 {len(features)} 只股票的衍生指标计算")

    # 合并
    df = df_today.merge(features, on='code', how='inner')

    # 计算均线偏离和布林带位置
    df['dev_ma60'] = np.where(
        (df['ma60'].notna()) & (df['ma60'] > 0),
        (df['close'] - df['ma60']) / df['ma60'] * 100, np.nan
    )
    df['boll_position'] = np.where(
        (df['boll_upper'].notna()) & (df['boll_lower'].notna()) &
        (df['boll_upper'] - df['boll_lower'] > 0),
        (df['close'] - df['boll_lower']) / (df['boll_upper'] - df['boll_lower']), np.nan
    )

    # 过滤无关键数据的
    required_cols = ['kdj_k', 'rsi14', 'dev_ma60', 'down_days', 'price_percentile',
                     'drawdown_60d', 'macd_dif', 'macd_hist', 'volume_ratio']
    df = df.dropna(subset=required_cols)
    print(f"有完整数据: {len(df)}只")

    # 排除ST（简单过滤：排除特定代码段，实际上需要更精确的过滤）
    # 这里先不过滤ST

    # 评分
    print("计算评分...")
    scores = np.zeros(len(df), dtype=int)
    details_list = []

    for idx, (_, row) in enumerate(df.iterrows()):
        score = 0
        details = []
        local_vars = row.to_dict()
        for name, cfg in scoring_rules.items():
            for condition, points, label in cfg['rules']:
                try:
                    if eval(condition, {"__builtins__": {}}, local_vars):
                        if points > 0:
                            details.append(label)
                        score += points
                        break
                except:
                    pass
        scores[idx] = score
        if score >= threshold:
            details_list.append({
                'code': row['code'],
                'close': row['close'],
                'pct_change': row['pct_change'],
                'score': score,
                'kdj_k': row['kdj_k'],
                'rsi14': row['rsi14'],
                'dev_ma60': row['dev_ma60'],
                'down_days': row['down_days'],
                'price_percentile': row['price_percentile'],
                'drawdown_60d': row['drawdown_60d'],
                'volume_ratio': row['volume_ratio'],
                'macd_hist': row['macd_hist'],
                'macd_dif': row['macd_dif'],
                'details': '; '.join(details),
            })

    df['score'] = scores

    # 得分分布
    print("\n全市场得分分布:")
    total_weight = sum(c['weight'] for c in scoring_rules.values())
    for t in range(total_weight, total_weight - 16, -2):
        cnt = (scores >= t).sum()
        print(f"  >= {t}分: {cnt}只 ({cnt/len(df)*100:.1f}%)")

    # 筛选
    candidates = sorted(details_list, key=lambda x: x['score'], reverse=True)
    print(f"\n阈值 >= {threshold}分: {len(candidates)}只候选")

    return candidates, df, latest_date


def step5_generate_report(candidates, latest_date, scoring_rules, threshold):
    """生成详细报告"""
    print("\n" + "=" * 70)
    print("步骤5: 生成报告")
    print("=" * 70)

    # Markdown报告
    md_path = OUTPUT_DIR / "launch_point_candidates_v2.md"
    lines = [
        f"# 起涨点扫描报告 V2",
        f"",
        f"**扫描日期**: {latest_date}",
        f"**扫描范围**: 沪深主板 + 创业板 + 科创板",
        f"**评分模型**: 多维度增强版（满分30分）",
        f"**筛选阈值**: >= {threshold}分",
        f"**候选数量**: {len(candidates)}只",
        f"",
        f"## 评分模型",
        f"",
        f"| 维度 | 满分 | 说明 |",
        f"|------|------|------|",
    ]
    for name, cfg in scoring_rules.items():
        lines.append(f"| {name} | {cfg['weight']} | {cfg['rules'][0][2]} |")

    lines += [
        f"",
        f"## 起涨点特征回顾（来自大涨股统计）",
        f"",
        f"- KDJ超卖(K<30): 约60%",
        f"- 跌破MA60: 约88%",
        f"- RSI14<40: 约71%",
        f"- 当日下跌: 约94%",
        f"- 量比<1: 约65%",
        f"",
        f"## 候选列表 (得分 >= {threshold}分，共{len(candidates)}只)",
        f"",
    ]

    # 按得分分组
    score_groups = {}
    for c in candidates:
        s = c['score']
        score_groups.setdefault(s, []).append(c)

    for score in sorted(score_groups.keys(), reverse=True):
        group = score_groups[score]
        lines.append(f"### 得分 {score} 分 ({len(group)}只)")
        lines.append("")
        lines.append("| 代码 | 收盘价 | 涨跌% | KDJ_K | RSI14 | MA60偏离 | 连跌 | 分位数 | 回撤 | 量比 | MACD | 特征 |")
        lines.append("|------|--------|-------|-------|-------|----------|------|--------|------|------|------|------|")
        for c in group:
            macd_sign = "🔴" if c['macd_hist'] > 0 else "🟢"
            macd_pos = "零轴下" if c['macd_dif'] < 0 else "零轴上"
            lines.append(
                f"| {c['code']} | {c['close']:.2f} | {c['pct_change']:+.2f} | "
                f"{c['kdj_k']:.1f} | {c['rsi14']:.1f} | {c['dev_ma60']:+.1f}% | "
                f"{c['down_days']:.0f} | {c['price_percentile']:.1f}% | {c['drawdown_60d']:+.1f}% | "
                f"{c['volume_ratio']:.2f} | {macd_sign}{macd_pos} | {c['details'][:60]} |"
            )
        lines.append("")

    lines += [
        "---",
        "*报告由起涨点扫描系统V2自动生成*"
    ]

    md_path.write_text('\n'.join(lines), encoding='utf-8')

    # 同时输出CSV
    csv_path = OUTPUT_DIR / "launch_point_candidates_v2.csv"
    df_candidates = pd.DataFrame(candidates)
    df_candidates.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"Markdown报告: {md_path}")
    print(f"CSV数据: {csv_path}")

    return md_path, csv_path


def main():
    # 步骤1-3: 从大涨股学习
    df_lp = step1_extract_launch_points()
    df_lp, stats = step2_analyze(df_lp)
    scoring_rules, threshold = step3_build_model(df_lp)

    # 步骤4-5: 全市场扫描
    candidates, df_all, latest_date = step4_scan_market(scoring_rules, threshold)
    md_path, csv_path = step5_generate_report(candidates, latest_date, scoring_rules, threshold)

    # 打印TOP 50
    print("\n" + "=" * 70)
    print(f"TOP 50 起涨点候选")
    print("=" * 70)
    print(f"{'代码':<8} {'收盘':>8} {'涨跌':>8} {'得分':>4} {'K':>6} {'RSI':>6} {'MA60':>8} {'连跌':>4} {'分位':>6} {'回撤':>8} {'量比':>6}")
    print("-" * 90)
    for c in candidates[:50]:
        print(f"{c['code']:<8} {c['close']:>8.2f} {c['pct_change']:>+7.2f}% {c['score']:>4} "
              f"{c['kdj_k']:>6.1f} {c['rsi14']:>6.1f} {c['dev_ma60']:>+7.1f}% "
              f"{c['down_days']:>4.0f} {c['price_percentile']:>5.1f}% {c['drawdown_60d']:>+7.1f}% {c['volume_ratio']:>6.2f}")

    print(f"\n... 共 {len(candidates)} 只候选")
    print(f"\n报告文件: {md_path}")
    print(f"CSV文件: {csv_path}")


if __name__ == "__main__":
    main()
