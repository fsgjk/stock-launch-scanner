"""
起涨点扫描系统
- 步骤1: 从大涨股中提取起涨点特征
- 步骤2: 统计分析起涨点特征
- 步骤3: 构建评分模型
- 步骤4: 全市场扫描
"""
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

DB_PATH = Path("/workspace/stock_analyzer/data/stock_system.db")


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def step1_find_winning_stocks():
    """找出最近20个交易日涨幅>20%的股票，并精确定位每只股票的起涨点"""
    print("=" * 70)
    print("步骤1: 找出大涨股并定位起涨点")
    print("=" * 70)

    conn = get_db()

    # 获取最新交易日
    cur = conn.execute("SELECT MAX(trade_date) FROM stock_daily")
    latest_date = cur.fetchone()[0]
    print(f"最新交易日: {latest_date}")

    # 获取20个交易日前（约一个月）
    cur = conn.execute("""
        SELECT DISTINCT trade_date FROM stock_daily
        WHERE trade_date <= ?
        ORDER BY trade_date DESC LIMIT 20
    """, (latest_date,))
    dates = [r[0] for r in cur.fetchall()]
    start_date = dates[-1]
    end_date = dates[0]
    print(f"20日区间: {start_date} ~ {end_date}")

    # 获取每只股票在起始日和结束日的收盘价
    df_start = pd.read_sql_query("""
        SELECT code, close FROM stock_daily WHERE trade_date = ?
    """, conn, params=(start_date,))
    df_end = pd.read_sql_query("""
        SELECT code, close FROM stock_daily WHERE trade_date = ?
    """, conn, params=(end_date,))

    df = df_start.merge(df_end, on='code', suffixes=('_start', '_end'))
    df['pct_20d'] = (df['close_end'] - df['close_start']) / df['close_start'] * 100
    winners = df[df['pct_20d'] >= 20].copy()
    print(f"20日涨幅>=20%的股票: {len(winners)}只")

    # 过滤掉ST、退市等
    winners = winners[~winners['code'].str.contains('900|200', na=False)]
    print(f"过滤B股后: {len(winners)}只")

    # 对每只大涨股，精确定位起涨点
    # 起涨点定义：在20日区间内，找到涨幅开始加速的那一天
    # 方法：从结束日向前追溯，找最低点（起涨点）
    launch_points = []

    for i, (_, row) in enumerate(winners.iterrows()):
        code = row['code']
        if i % 50 == 0:
            print(f"  处理 {i}/{len(winners)}: {code}")

        # 获取该股票20日区间内的所有数据
        df_stock = pd.read_sql_query("""
            SELECT trade_date, open, high, low, close, volume, pct_change,
                   kdj_k, kdj_d, kdj_j,
                   macd_dif, macd_dea, macd_hist,
                   ma5, ma10, ma20, ma60, ma120, ma250,
                   rsi6, rsi14, rsi24,
                   boll_upper, boll_mid, boll_lower,
                   volume_ma5, volume_ma10, volume_ma20, volume_ratio,
                   turnover_rate
            FROM stock_daily
            WHERE code = ? AND trade_date BETWEEN ? AND ?
            ORDER BY trade_date
        """, conn, params=(code, start_date, end_date))

        if len(df_stock) < 15:
            continue

        # 找最低收盘价的日期作为起涨点
        min_idx = df_stock['close'].idxmin()
        launch_row = df_stock.loc[min_idx]

        # 如果最低点在最后3天，说明可能还在跌，跳过
        if min_idx >= len(df_stock) - 3:
            # 尝试找前半段的最低点
            half = len(df_stock) // 2
            first_half = df_stock.iloc[:half]
            if len(first_half) > 0:
                min_idx = first_half['close'].idxmin()
                launch_row = df_stock.loc[min_idx]

        launch_date = launch_row['trade_date']
        launch_close = launch_row['close']
        end_close = df_stock.iloc[-1]['close']
        total_gain = (end_close - launch_close) / launch_close * 100

        # 提取起涨点特征
        lp = {
            'code': code,
            'launch_date': launch_date,
            'launch_close': launch_close,
            'end_close': end_close,
            'total_gain': total_gain,
            'pct_20d': row['pct_20d'],
            # 起涨点当日特征
            'pct_change': launch_row['pct_change'],
            'kdj_k': launch_row['kdj_k'],
            'kdj_d': launch_row['kdj_d'],
            'kdj_j': launch_row['kdj_j'],
            'macd_dif': launch_row['macd_dif'],
            'macd_dea': launch_row['macd_dea'],
            'macd_hist': launch_row['macd_hist'],
            'rsi6': launch_row['rsi6'],
            'rsi14': launch_row['rsi14'],
            'rsi24': launch_row['rsi24'],
            'ma5': launch_row['ma5'],
            'ma10': launch_row['ma10'],
            'ma20': launch_row['ma20'],
            'ma60': launch_row['ma60'],
            'ma120': launch_row['ma120'],
            'ma250': launch_row['ma250'],
            'volume': launch_row['volume'],
            'volume_ratio': launch_row['volume_ratio'],
            'turnover_rate': launch_row['turnover_rate'],
            'boll_upper': launch_row['boll_upper'],
            'boll_mid': launch_row['boll_mid'],
            'boll_lower': launch_row['boll_lower'],
        }

        # 计算偏离均线的百分比
        if launch_row['ma60'] and launch_row['ma60'] > 0:
            lp['dev_ma60'] = (launch_close - launch_row['ma60']) / launch_row['ma60'] * 100
        else:
            lp['dev_ma60'] = None
        if launch_row['ma20'] and launch_row['ma20'] > 0:
            lp['dev_ma20'] = (launch_close - launch_row['ma20']) / launch_row['ma20'] * 100
        else:
            lp['dev_ma20'] = None

        # MACD状态
        lp['macd_below_zero'] = 1 if (lp['macd_dif'] and lp['macd_dif'] < 0) else 0
        lp['macd_golden_cross'] = 1 if (lp['macd_dif'] and lp['macd_dea'] and lp['macd_dif'] > lp['macd_dea']) else 0

        # KDJ超卖
        lp['kdj_oversold'] = 1 if (lp['kdj_k'] and lp['kdj_k'] < 30) else 0
        lp['kdj_deep_oversold'] = 1 if (lp['kdj_k'] and lp['kdj_k'] < 20) else 0

        # RSI超卖
        lp['rsi_oversold'] = 1 if (lp['rsi14'] and lp['rsi14'] < 40) else 0

        # 布林带位置
        if lp['boll_lower'] and lp['boll_upper'] and (lp['boll_upper'] - lp['boll_lower']) > 0:
            lp['boll_position'] = (launch_close - lp['boll_lower']) / (lp['boll_upper'] - lp['boll_lower'])
        else:
            lp['boll_position'] = None

        launch_points.append(lp)

    conn.close()

    df_lp = pd.DataFrame(launch_points)
    print(f"\n提取到 {len(df_lp)} 个起涨点样本")

    return df_lp


def step2_analyze_patterns(df_lp):
    """分析起涨点特征分布"""
    print("\n" + "=" * 70)
    print("步骤2: 起涨点特征统计分析")
    print("=" * 70)

    # 清理数据
    df = df_lp.dropna(subset=['kdj_k', 'rsi14', 'dev_ma60']).copy()

    print(f"\n有效样本数: {len(df)}")
    print(f"平均总涨幅: {df['total_gain'].mean():.1f}%")
    print(f"��位数总涨幅: {df['total_gain'].median():.1f}%")

    print("\n--- KDJ 特征 ---")
    print(f"KDJ_K 均值: {df['kdj_k'].mean():.1f}, 中位数: {df['kdj_k'].median():.1f}")
    print(f"KDJ_K 分布: <20: {(df['kdj_k'] < 20).mean()*100:.0f}%, 20-30: {((df['kdj_k'] >= 20) & (df['kdj_k'] < 30)).mean()*100:.0f}%, 30-50: {((df['kdj_k'] >= 30) & (df['kdj_k'] < 50)).mean()*100:.0f}%, >=50: {(df['kdj_k'] >= 50).mean()*100:.0f}%")
    print(f"KDJ_D 均值: {df['kdj_d'].mean():.1f}, 中位数: {df['kdj_d'].median():.1f}")
    print(f"KDJ_J 均值: {df['kdj_j'].mean():.1f}, 中位数: {df['kdj_j'].median():.1f}")
    print(f"KDJ超卖(K<30): {(df['kdj_k'] < 30).mean()*100:.0f}%")
    print(f"KDJ深度超卖(K<20): {(df['kdj_k'] < 20).mean()*100:.0f}%")

    print("\n--- RSI 特征 ---")
    print(f"RSI6 均值: {df['rsi6'].mean():.1f}")
    print(f"RSI14 均值: {df['rsi14'].mean():.1f}, 中位数: {df['rsi14'].median():.1f}")
    print(f"RSI24 均值: {df['rsi24'].mean():.1f}")
    print(f"RSI14分布: <30: {(df['rsi14'] < 30).mean()*100:.0f}%, 30-40: {((df['rsi14'] >= 30) & (df['rsi14'] < 40)).mean()*100:.0f}%, 40-50: {((df['rsi14'] >= 40) & (df['rsi14'] < 50)).mean()*100:.0f}%, >=50: {(df['rsi14'] >= 50).mean()*100:.0f}%")

    print("\n--- 均线偏离 特征 ---")
    print(f"偏离MA20 均值: {df['dev_ma20'].mean():.1f}%")
    print(f"偏离MA60 均值: {df['dev_ma60'].mean():.1f}%, 中位数: {df['dev_ma60'].median():.1f}%")
    print(f"跌破MA20: {(df['dev_ma20'] < 0).mean()*100:.0f}%")
    print(f"跌破MA60: {(df['dev_ma60'] < 0).mean()*100:.0f}%")
    print(f"跌破MA60超过-10%: {(df['dev_ma60'] < -10).mean()*100:.0f}%")
    print(f"跌破MA60超过-20%: {(df['dev_ma60'] < -20).mean()*100:.0f}%")

    print("\n--- MACD 特征 ---")
    print(f"MACD_DIF<0: {(df['macd_dif'] < 0).mean()*100:.0f}%")
    print(f"MACD金叉(DIF>DEA): {(df['macd_dif'] > df['macd_dea']).mean()*100:.0f}%")
    print(f"MACD绿柱缩短(hist<0): {(df['macd_hist'] < 0).mean()*100:.0f}%")

    print("\n--- 成交量 特征 ---")
    print(f"量比均值: {df['volume_ratio'].mean():.2f}")
    print(f"量比<1(缩量): {(df['volume_ratio'] < 1).mean()*100:.0f}%")
    print(f"量比<0.8(明显缩量): {(df['volume_ratio'] < 0.8).mean()*100:.0f}%")

    print("\n--- 起涨点当日涨跌幅 ---")
    print(f"均值: {df['pct_change'].mean():.2f}%")
    print(f"下跌: {(df['pct_change'] < 0).mean()*100:.0f}%")
    print(f"跌幅>2%: {(df['pct_change'] < -2).mean()*100:.0f}%")

    print("\n--- 布林带位置 ---")
    print(f"布林带位置均值: {df['boll_position'].mean():.2f}")
    print(f"触及下轨(<0.1): {(df['boll_position'] < 0.1).mean()*100:.0f}%")
    print(f"下轨附近(<0.2): {(df['boll_position'] < 0.2).mean()*100:.0f}%")

    return df


def build_scoring_model(df_lp):
    """基于统计特征构建评分模型"""
    print("\n" + "=" * 70)
    print("步骤3: 构建评分模型")
    print("=" * 70)

    # 评分维度（满分20分）
    scoring_rules = {
        # KDJ (5分) - 70%样本KDJ_K<30
        'kdj_score': {
            'weight': 5,
            'rules': [
                ('kdj_k < 20', 5, 'KDJ深度超卖(K<20)'),
                ('kdj_k >= 20 and kdj_k < 30', 4, 'KDJ超卖(K 20-30)'),
                ('kdj_k >= 30 and kdj_k < 40', 2, 'KDJ偏低(K 30-40)'),
                ('kdj_k >= 40 and kdj_k < 50', 1, 'KDJ中性偏低'),
                ('kdj_k >= 50', 0, 'KDJ中性'),
            ]
        },

        # RSI (4分) - 93%样本RSI14<40
        'rsi_score': {
            'weight': 4,
            'rules': [
                ('rsi14 < 30', 4, 'RSI深度超卖(<30)'),
                ('rsi14 >= 30 and rsi14 < 35', 3, 'RSI超卖(30-35)'),
                ('rsi14 >= 35 and rsi14 < 40', 2, 'RSI偏弱(35-40)'),
                ('rsi14 >= 40 and rsi14 < 50', 1, 'RSI中性偏弱'),
                ('rsi14 >= 50', 0, 'RSI中性'),
            ]
        },

        # 均线位置 (5分) - 91%样本跌破MA60
        'ma_score': {
            'weight': 5,
            'rules': [
                ('dev_ma60 < -15', 5, '深度跌破MA60(<-15%)'),
                ('dev_ma60 >= -15 and dev_ma60 < -5', 4, '跌破MA60(-15% ~ -5%)'),
                ('dev_ma60 >= -5 and dev_ma60 < 0', 3, '略跌破MA60(-5% ~ 0)'),
                ('dev_ma60 >= 0 and dev_ma60 < 5', 1, '略高于MA60'),
                ('dev_ma60 >= 5', 0, '高于MA60较多'),
            ]
        },

        # MACD (3分)
        'macd_score': {
            'weight': 3,
            'rules': [
                ('macd_hist < 0 and macd_dif < 0', 3, 'MACD零轴下绿柱'),
                ('macd_hist < 0 and macd_dif >= 0', 2, 'MACD零轴上绿柱'),
                ('macd_hist >= 0 and macd_dif < 0', 1, 'MACD零轴下红柱'),
                ('macd_hist >= 0 and macd_dif >= 0', 0, 'MACD零轴上红柱'),
            ]
        },

        # 成交量 (2分)
        'volume_score': {
            'weight': 2,
            'rules': [
                ('volume_ratio < 0.7', 2, '明显缩量(<0.7)'),
                ('volume_ratio >= 0.7 and volume_ratio < 1.0', 1, '缩量(0.7-1.0)'),
                ('volume_ratio >= 1.0', 0, '放量'),
            ]
        },

        # 布林带 (1分)
        'boll_score': {
            'weight': 1,
            'rules': [
                ('boll_position < 0.15', 1, '触及布林下轨'),
                ('boll_position >= 0.15', 0, '布林中轨以上'),
            ]
        },
    }

    print("\n评分维度:")
    for name, cfg in scoring_rules.items():
        print(f"  {name}: 满分{cfg['weight']}分")

    # 在样本上测试评分分布
    df = df_lp.dropna(subset=['kdj_k', 'rsi14', 'dev_ma60', 'macd_hist', 'macd_dif', 'volume_ratio', 'boll_position']).copy()

    total_scores = []
    for _, row in df.iterrows():
        score = 0
        for name, cfg in scoring_rules.items():
            for condition, points, _ in cfg['rules']:
                # 用eval执行条件判断
                try:
                    local_vars = row.to_dict()
                    if eval(condition, {"__builtins__": {}}, local_vars):
                        score += points
                        break
                except:
                    pass
        total_scores.append(score)

    df['score'] = total_scores
    print(f"\n样本评分分布:")
    print(f"  均值: {np.mean(total_scores):.1f}")
    print(f"  中位数: {np.median(total_scores):.1f}")
    for threshold in [18, 16, 14, 12, 10, 8]:
        print(f"  >= {threshold}分: {(np.array(total_scores) >= threshold).mean()*100:.0f}%")

    # 确定最优阈值：让70%以上的大涨股能通过
    target_pct = 0.7
    best_threshold = 0
    for t in range(20, 0, -1):
        if (np.array(total_scores) >= t).mean() >= target_pct:
            best_threshold = t
            break

    print(f"\n推荐筛选阈值: >= {best_threshold}分 (覆盖{target_pct*100:.0f}%大涨股)")

    return scoring_rules, best_threshold


def step4_scan_market(scoring_rules, threshold):
    """全市场扫描起涨点"""
    print("\n" + "=" * 70)
    print("步骤4: 全市场扫描起涨点")
    print("=" * 70)

    conn = get_db()

    # 获取最新交易日
    cur = conn.execute("SELECT MAX(trade_date) FROM stock_daily")
    latest_date = cur.fetchone()[0]
    print(f"扫描日期: {latest_date}")

    # 获取所有股票的最新数据
    print("加载全市场最新数据...")
    df_all = pd.read_sql_query("""
        SELECT code, trade_date, close, open, pct_change,
               kdj_k, kdj_d, kdj_j,
               macd_dif, macd_dea, macd_hist,
               ma5, ma10, ma20, ma60, ma120, ma250,
               rsi6, rsi14, rsi24,
               volume, volume_ratio, turnover_rate,
               boll_upper, boll_mid, boll_lower
        FROM stock_daily
        WHERE trade_date = ?
    """, conn, params=(latest_date,))

    print(f"全市场股票数: {len(df_all)}")

    # 计算衍生指标
    df_all['dev_ma60'] = np.where(
        (df_all['ma60'].notna()) & (df_all['ma60'] > 0),
        (df_all['close'] - df_all['ma60']) / df_all['ma60'] * 100,
        np.nan
    )
    df_all['dev_ma20'] = np.where(
        (df_all['ma20'].notna()) & (df_all['ma20'] > 0),
        (df_all['close'] - df_all['ma20']) / df_all['ma20'] * 100,
        np.nan
    )
    df_all['boll_position'] = np.where(
        (df_all['boll_upper'].notna()) & (df_all['boll_lower'].notna()) &
        (df_all['boll_upper'] - df_all['boll_lower'] > 0),
        (df_all['close'] - df_all['boll_lower']) / (df_all['boll_upper'] - df_all['boll_lower']),
        np.nan
    )

    # 过滤掉无关键数据的
    df = df_all.dropna(subset=['kdj_k', 'rsi14', 'dev_ma60', 'macd_hist', 'macd_dif']).copy()
    print(f"有效股票数(有完整数据): {len(df)}")

    # 排除ST、*ST、退市等
    # 排除北交所(8开头)、B股(9开头)
    df = df[df['code'].str.match(r'^(00|30|60|68)')].copy()
    print(f"过滤后(沪深主板+创业板+科创板): {len(df)}")

    # 评分
    print("计算评分...")
    scores = np.zeros(len(df), dtype=int)
    detail_lines = []

    for idx, (_, row) in enumerate(df.iterrows()):
        score = 0
        details = []
        for name, cfg in scoring_rules.items():
            local_vars = row.to_dict()
            for condition, points, label in cfg['rules']:
                try:
                    if eval(condition, {"__builtins__": {}}, local_vars):
                        if points > 0:
                            details.append(f"{label}(+{points})")
                        score += points
                        break
                except Exception as e:
                    pass
        scores[idx] = score
        if score >= threshold:
            detail_lines.append({
                'code': row['code'],
                'close': row['close'],
                'pct_change': row['pct_change'],
                'score': score,
                'kdj_k': row['kdj_k'],
                'rsi14': row['rsi14'],
                'dev_ma60': row['dev_ma60'],
                'volume_ratio': row['volume_ratio'],
                'macd_hist': row['macd_hist'],
                'details': '; '.join(details) if details else '',
            })

    df['score'] = scores

    # 统计得分分布
    print("\n全市场得分分布:")
    for t in [18, 16, 14, 12, 10, 8, 6]:
        cnt = (scores >= t).sum()
        print(f"  >= {t}分: {cnt}只 ({cnt/len(df)*100:.1f}%)")

    # 筛选
    candidates = [d for d in detail_lines if d['score'] >= threshold]
    candidates.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n推荐阈值 >= {threshold}分: {len(candidates)}只候选")

    conn.close()
    return candidates, df, latest_date


def generate_report(candidates, latest_date, scoring_rules, threshold):
    """生成候选报告"""
    print("\n" + "=" * 70)
    print("步骤5: 生成报告")
    print("=" * 70)

    report_path = Path("/workspace/stock_analyzer/output/launch_point_candidates.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# 起涨点扫描报告")
    lines.append(f"")
    lines.append(f"**扫描日期**: {latest_date}")
    lines.append(f"**扫描范围**: 沪深主板 + 创业板 + 科创板")
    lines.append(f"**筛选阈值**: >= {threshold}分")
    lines.append(f"**候选数量**: {len(candidates)}只")
    lines.append(f"")
    lines.append(f"## 评分模型")
    lines.append(f"")
    lines.append(f"| 维度 | 满分 | 说明 |")
    lines.append(f"|------|------|------|")
    for name, cfg in scoring_rules.items():
        desc = cfg['rules'][0][2].split('(')[0].strip()
        lines.append(f"| {name} | {cfg['weight']} | {desc}等 |")
    lines.append(f"")
    lines.append(f"## 候选列表 (得分 >= {threshold})")
    lines.append(f"")

    # 按得分分组
    score_groups = {}
    for c in candidates:
        s = c['score']
        if s not in score_groups:
            score_groups[s] = []
        score_groups[s].append(c)

    for score in sorted(score_groups.keys(), reverse=True):
        group = score_groups[score]
        lines.append(f"### 得分 {score} 分 ({len(group)}只)")
        lines.append(f"")
        lines.append(f"| 代码 | 收盘价 | 涨跌幅 | KDJ_K | RSI14 | 偏离MA60 | 量比 | MACD | 得分明细 |")
        lines.append(f"|------|--------|--------|-------|-------|----------|------|------|----------|")
        for c in group[:30]:  # 每组最多30只
            macd_sign = "红" if c['macd_hist'] > 0 else "绿"
            lines.append(
                f"| {c['code']} | {c['close']:.2f} | {c['pct_change']:+.2f}% | "
                f"{c['kdj_k']:.1f} | {c['rsi14']:.1f} | {c['dev_ma60']:+.1f}% | "
                f"{c['volume_ratio']:.2f} | {macd_sign} | {c['details']} |"
            )
        lines.append(f"")

    lines.append(f"---")
    lines.append(f"*报告由起涨点扫描系统自动生成*")

    report_content = '\n'.join(lines)
    report_path.write_text(report_content, encoding='utf-8')
    print(f"报告已保存: {report_path}")

    return report_path


def main():
    # 步骤1: 找大涨股并提取起涨点
    df_lp = step1_find_winning_stocks()

    # 步骤2: 统计分析
    df_lp = step2_analyze_patterns(df_lp)

    # 步骤3: 构建评分模型
    scoring_rules, threshold = build_scoring_model(df_lp)

    # 步骤4: 全市场扫描
    candidates, df_all, latest_date = step4_scan_market(scoring_rules, threshold)

    # 步骤5: 生成报告
    report_path = generate_report(candidates, latest_date, scoring_rules, threshold)

    # 打印TOP 30
    print("\n" + "=" * 70)
    print(f"TOP 30 起涨点候选 (共{len(candidates)}只)")
    print("=" * 70)
    print(f"{'代码':<8} {'收盘':>8} {'涨跌':>8} {'得分':>4} {'KDJ_K':>6} {'RSI14':>6} {'MA60偏离':>8} {'量比':>6}")
    print("-" * 70)
    for c in candidates[:30]:
        print(f"{c['code']:<8} {c['close']:>8.2f} {c['pct_change']:>+7.2f}% {c['score']:>4} "
              f"{c['kdj_k']:>6.1f} {c['rsi14']:>6.1f} {c['dev_ma60']:>+7.1f}% {c['volume_ratio']:>6.2f}")

    print(f"\n... 共 {len(candidates)} 只候选，详见报告文件")


if __name__ == "__main__":
    main()
