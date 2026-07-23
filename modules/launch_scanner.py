"""
起涨点扫描核心模块
从 launch_point_scanner_v3.py 提取并模块化，供 Dashboard 和命令行共用
"""
import sqlite3
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DB_PATH = Path("/workspace/stock_analyzer/data/stock_system.db")


class LaunchPointScanner:
    """起涨点扫描器 — 学习大涨股特征，全市场扫描起涨点候选"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ==================== 步骤1: 学习大涨股 ====================

    def get_latest_date(self, conn):
        cur = conn.execute("SELECT MAX(trade_date) FROM stock_daily")
        return cur.fetchone()[0]

    def get_date_range(self, conn, latest_date, lookback_days=120):
        cur = conn.execute("""
            SELECT DISTINCT trade_date FROM stock_daily
            WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT ?
        """, (latest_date, lookback_days))
        dates = [r[0] for r in cur.fetchall()]
        dates.sort()
        return dates

    def find_winner_samples(self, conn, latest_date, all_dates):
        """找出近20日涨幅>=20%的大涨股"""
        start_20d = all_dates[-20] if len(all_dates) >= 20 else all_dates[0]
        df_start = pd.read_sql_query(
            "SELECT code, close FROM stock_daily WHERE trade_date = ?", conn, params=(start_20d,))
        df_end = pd.read_sql_query(
            "SELECT code, close FROM stock_daily WHERE trade_date = ?", conn, params=(latest_date,))
        df = df_start.merge(df_end, on='code', suffixes=('_start', '_end'))
        df['pct_20d'] = (df['close_end'] - df['close_start']) / df['close_start'] * 100
        winners = df[(df['pct_20d'] >= 20) & ~df['code'].str.contains('900|200', na=False)]
        return winners, start_20d

    def extract_winner_features(self, conn, winner_codes, all_dates, latest_date):
        """提取每只大涨股的起涨点特征"""
        features = []
        for code in winner_codes:
            df_s = pd.read_sql_query("""
                SELECT * FROM stock_daily
                WHERE code = ? AND trade_date BETWEEN ? AND ?
                ORDER BY trade_date
            """, conn, params=(code, all_dates[0], latest_date))
            if len(df_s) < 40:
                continue

            df_s = df_s.reset_index(drop=True)
            recent = df_s.iloc[-20:]
            min_idx = recent['close'].idxmin()
            if min_idx >= len(df_s) - 5:
                older = df_s.iloc[:-5]
                if len(older) > 5:
                    min_idx = older['close'].idxmin()

            lp_row = df_s.loc[min_idx]
            before = df_s.loc[:min_idx]

            # 连跌天数
            down_days = 0
            for j in range(len(before) - 1, max(0, len(before) - 20), -1):
                if before.iloc[j]['pct_change'] is not None and before.iloc[j]['pct_change'] < 0:
                    down_days += 1
                else:
                    break

            # 价格分位数
            close_recent = recent['close']
            price_pct = (lp_row['close'] - close_recent.min()) / (close_recent.max() - close_recent.min()) * 100 \
                if close_recent.max() > close_recent.min() else 50

            # 60日回撤
            lookback_n = min(60, len(df_s))
            lookback = df_s.iloc[-lookback_n:]
            dd_60 = (lp_row['close'] - lookback['high'].max()) / lookback['high'].max() * 100

            # 布林位置
            boll_pos = None
            if lp_row['boll_lower'] and lp_row['boll_upper'] and lp_row['boll_upper'] > lp_row['boll_lower']:
                boll_pos = (lp_row['close'] - lp_row['boll_lower']) / (lp_row['boll_upper'] - lp_row['boll_lower'])

            dev_ma60 = (lp_row['close'] - lp_row['ma60']) / lp_row['ma60'] * 100 \
                if lp_row['ma60'] and lp_row['ma60'] > 0 else None

            features.append({
                'code': code, 'pct_20d': lp_row.get('pct_change', 0),
                'kdj_k': lp_row['kdj_k'], 'kdj_d': lp_row['kdj_d'],
                'rsi14': lp_row['rsi14'],
                'dev_ma60': dev_ma60, 'down_days': down_days,
                'price_pct': price_pct, 'dd_60': dd_60,
                'boll_pos': boll_pos, 'vol_ratio': lp_row['volume_ratio'],
                'macd_dif': lp_row['macd_dif'], 'macd_hist': lp_row['macd_hist'],
                'pct_change': lp_row['pct_change'], 'lp_date': lp_row['trade_date'],
            })

        return pd.DataFrame(features)

    def compute_winner_statistics(self, df_w):
        """计算大涨股起涨点统计摘要"""
        df = df_w.dropna(subset=['kdj_k', 'rsi14', 'dev_ma60', 'dd_60', 'boll_pos'])
        if len(df) == 0:
            return {}
        return {
            'sample_count': len(df),
            'kdj_mean': float(df['kdj_k'].mean()),
            'kdj_median': float(df['kdj_k'].median()),
            'kdj_below_30_pct': float((df['kdj_k'] < 30).mean() * 100),
            'rsi_mean': float(df['rsi14'].mean()),
            'rsi_median': float(df['rsi14'].median()),
            'rsi_below_35_pct': float((df['rsi14'] < 35).mean() * 100),
            'dev_ma60_mean': float(df['dev_ma60'].mean()),
            'dev_ma60_below_0_pct': float((df['dev_ma60'] < 0).mean() * 100),
            'down_days_mean': float(df['down_days'].mean()),
            'down_days_ge3_pct': float((df['down_days'] >= 3).mean() * 100),
            'dd_60_mean': float(df['dd_60'].mean()),
            'dd_60_below_m20_pct': float((df['dd_60'] < -20).mean() * 100),
            'vol_ratio_mean': float(df['vol_ratio'].mean()),
            'vol_ratio_below_1_pct': float((df['vol_ratio'] < 1).mean() * 100),
            'pct_change_below_0_pct': float((df['pct_change'] < 0).mean() * 100),
        }

    # ==================== 步骤2: 全市场扫描 ====================

    def load_today_data(self, conn, latest_date):
        """加载最新交易日全市场数据"""
        df = pd.read_sql_query("""
            SELECT code, trade_date, close, open, high, low, pct_change, amplitude,
                   kdj_k, kdj_d, kdj_j,
                   macd_dif, macd_dea, macd_hist,
                   ma5, ma10, ma20, ma60, ma120, ma250,
                   rsi6, rsi14, rsi24,
                   volume_ratio, turnover_rate, volume,
                   boll_upper, boll_mid, boll_lower
            FROM stock_daily WHERE trade_date = ?
        """, conn, params=(latest_date,))
        df = df[df['code'].str.match(r'^(00|30|60|68)')].copy()
        return df

    def load_historical_data(self, conn, date_range):
        """加载近60日历史数据"""
        df = pd.read_sql_query("""
            SELECT code, trade_date, close, high, low, pct_change, volume
            FROM stock_daily
            WHERE trade_date BETWEEN ? AND ?
            ORDER BY code, trade_date
        """, conn, params=(date_range[0], date_range[-1]))
        return df

    def calc_derived_features(self, df_hist):
        """计算衍生指标"""
        grouped = df_hist.groupby('code')

        def calc(grp):
            grp = grp.sort_values('trade_date').reset_index(drop=True)
            n = len(grp)
            if n < 10:
                return pd.Series({
                    'down_days': np.nan, 'price_pct_20d': np.nan, 'dd_60': np.nan,
                    'vol_5d_avg': np.nan, 'near_low_10d': np.nan,
                })
            last_close = grp.iloc[-1]['close']
            down_days = 0
            for j in range(n - 1, max(0, n - 20), -1):
                if grp.iloc[j]['pct_change'] is not None and grp.iloc[j]['pct_change'] < 0:
                    down_days += 1
                else:
                    break
            recent_n = min(20, n)
            recent = grp.iloc[-recent_n:]
            c_min, c_max = recent['close'].min(), recent['close'].max()
            price_pct = (last_close - c_min) / (c_max - c_min) * 100 if c_max > c_min else 50
            lb_n = min(60, n)
            lb = grp.iloc[-lb_n:]
            dd_60 = (last_close - lb['high'].max()) / lb['high'].max() * 100
            vol_5d = grp.iloc[-5:]['volume'].mean() if n >= 5 else grp['volume'].mean()
            low10 = grp.iloc[-10:]['low'].min() if n >= 10 else grp.iloc[-1]['low']
            near_low = 1 if last_close <= low10 * 1.05 else 0
            return pd.Series({
                'down_days': down_days, 'price_pct_20d': price_pct,
                'dd_60': dd_60, 'vol_5d_avg': vol_5d, 'near_low_10d': near_low,
            })

        features = grouped.apply(calc).reset_index()
        return features

    # ==================== 步骤3: 精准筛选 ====================

    def apply_hard_filters(self, df):
        """硬条件过滤，返回过滤后数据和统计"""
        conditions = [
            ('KDJ超卖', df['kdj_k'] < 35),
            ('RSI弱势', df['rsi14'] < 45),
            ('破MA60', df['dev_ma60'] < -3),
            ('60日回撤', df['dd_60'] < -15),
            ('缩量', df['volume_ratio'] < 1.2),
            ('连跌', df['down_days'] >= 2),
            ('当日未大涨', df['pct_change'] < 0.5),
            ('布林下半区', df['boll_pos'] < 0.4),
        ]
        filter_stats = {'total': len(df)}
        mask = np.ones(len(df), dtype=bool)
        for name, cond in conditions:
            mask = mask & cond
            filter_stats[name] = int(cond.sum())
        filter_stats['passed'] = int(mask.sum())
        return df[mask].copy(), filter_stats

    def calc_score_with_breakdown(self, row):
        """计算综合评分并返回明细"""
        score = 0
        breakdown = {}

        # KDJ (0-5)
        k = row['kdj_k']
        if k < 15: pts, breakdown['kdj'] = 5, 5
        elif k < 22: pts, breakdown['kdj'] = 4, 4
        elif k < 30: pts, breakdown['kdj'] = 3, 3
        elif k < 40: pts, breakdown['kdj'] = 1, 1
        else: pts, breakdown['kdj'] = 0, 0
        score += pts

        # RSI (0-4)
        r = row['rsi14']
        if r < 25: pts, breakdown['rsi'] = 4, 4
        elif r < 30: pts, breakdown['rsi'] = 3, 3
        elif r < 35: pts, breakdown['rsi'] = 2, 2
        elif r < 42: pts, breakdown['rsi'] = 1, 1
        else: pts, breakdown['rsi'] = 0, 0
        score += pts

        # MA60偏离 (0-5)
        d = row['dev_ma60']
        if d < -25: pts, breakdown['ma60_dev'] = 5, 5
        elif d < -18: pts, breakdown['ma60_dev'] = 4, 4
        elif d < -12: pts, breakdown['ma60_dev'] = 3, 3
        elif d < -6: pts, breakdown['ma60_dev'] = 2, 2
        elif d < -3: pts, breakdown['ma60_dev'] = 1, 1
        else: pts, breakdown['ma60_dev'] = 0, 0
        score += pts

        # 连跌 (0-4)
        dd = row['down_days']
        if dd >= 6: pts, breakdown['down_days'] = 4, 4
        elif dd >= 4: pts, breakdown['down_days'] = 3, 3
        elif dd >= 3: pts, breakdown['down_days'] = 2, 2
        elif dd >= 2: pts, breakdown['down_days'] = 1, 1
        else: pts, breakdown['down_days'] = 0, 0
        score += pts

        # 60日回撤 (0-3)
        d60 = row['dd_60']
        if d60 < -40: pts, breakdown['dd_60'] = 3, 3
        elif d60 < -30: pts, breakdown['dd_60'] = 2, 2
        elif d60 < -20: pts, breakdown['dd_60'] = 1, 1
        else: pts, breakdown['dd_60'] = 0, 0
        score += pts

        # 价格分位数 (0-3)
        pp = row['price_pct_20d']
        if pp < 5: pts, breakdown['price_pct'] = 3, 3
        elif pp < 15: pts, breakdown['price_pct'] = 2, 2
        elif pp < 25: pts, breakdown['price_pct'] = 1, 1
        else: pts, breakdown['price_pct'] = 0, 0
        score += pts

        # MACD (0-2)
        if row['macd_dif'] < 0 and row['macd_hist'] < 0:
            pts, breakdown['macd'] = 2, 2
        elif row['macd_dif'] < 0:
            pts, breakdown['macd'] = 1, 1
        else:
            pts, breakdown['macd'] = 0, 0
        score += pts

        # 成交量 (0-2)
        vr = row['volume_ratio']
        if vr < 0.6: pts, breakdown['volume'] = 2, 2
        elif vr < 0.8: pts, breakdown['volume'] = 1, 1
        else: pts, breakdown['volume'] = 0, 0
        score += pts

        # 布林 (0-1)
        if row['boll_pos'] < 0.1: pts, breakdown['boll'] = 1, 1
        else: pts, breakdown['boll'] = 0, 0
        score += pts

        # 额外 (0-1)
        if row.get('near_low_10d', 0): pts, breakdown['near_low'] = 1, 1
        else: pts, breakdown['near_low'] = 0, 0
        score += pts

        return score, breakdown

    # ==================== 步骤4: 持久化 ====================

    def save_scan_results(self, conn, scan_date, df_top, stats, filter_stats, winner_count, latest_trade_date):
        """保存扫描结果到数据库"""
        scan_time = datetime.now().strftime('%H:%M:%S')
        scan_params = json.dumps({
            'top_n': len(df_top), 'winner_threshold': 20,
            'hard_filters': {
                'kdj_max': 35, 'rsi_max': 45, 'ma60_dev_max': -3,
                'dd60_max': -15, 'vol_ratio_max': 1.2, 'down_days_min': 2,
                'pct_change_max': 0.5, 'boll_pos_max': 0.4,
            }
        })

        cur = conn.execute("""
            INSERT INTO launch_scan_results (scan_date, scan_time, total_scanned, total_candidates,
                hard_filter_passed, winner_sample_count, latest_trade_date, scan_params)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (scan_date, scan_time, filter_stats['total'], len(df_top),
              filter_stats['passed'], winner_count, latest_trade_date, scan_params))
        scan_id = cur.lastrowid

        # 批量插入候选
        rows = []
        for _, r in df_top.iterrows():
            rows.append((
                scan_id, r['code'], r['close'], r['pct_change'], int(r['score']),
                r['kdj_k'], r['kdj_d'], r['kdj_j'],
                r.get('rsi6'), r['rsi14'], r.get('rsi24'),
                r.get('dev_ma20'), r['dev_ma60'],
                int(r['down_days']), r['dd_60'],
                r['volume_ratio'], r['price_pct_20d'], r['boll_pos'],
                r['macd_dif'], r['macd_hist'],
                json.dumps(r['score_breakdown'], ensure_ascii=False) if 'score_breakdown' in r else '{}'
            ))
        conn.executemany("""
            INSERT INTO launch_scan_candidates (scan_id, code, close, pct_change, score,
                kdj_k, kdj_d, kdj_j, rsi6, rsi14, rsi24,
                dev_ma20, dev_ma60, down_days, dd_60,
                volume_ratio, price_pct_20d, boll_pos, macd_dif, macd_hist, score_breakdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)

        conn.commit()
        return scan_id

    def save_winner_templates(self, conn, df_w, scan_date):
        """保存大涨股模板"""
        rows = []
        for _, r in df_w.iterrows():
            rows.append((
                r['code'], r.get('pct_20d', 0), scan_date,
                r['kdj_k'], r['kdj_d'], r['rsi14'], r['dev_ma60'],
                int(r['down_days']), r['dd_60'],
                r['boll_pos'], r['vol_ratio'],
                r['macd_dif'], r['macd_hist'],
                r.get('price_pct', 0), r.get('lp_date', ''),
            ))
        conn.executemany("""
            INSERT OR REPLACE INTO winner_templates (code, pct_20d, scan_date,
                kdj_k, kdj_d, rsi14, dev_ma60, down_days, dd_60,
                boll_pos, vol_ratio, macd_dif, macd_hist, price_pct, lp_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        conn.commit()

    def load_scan_results(self, conn, scan_id=None, scan_date=None):
        """加载扫描结果"""
        if scan_id:
            cur = conn.execute("SELECT * FROM launch_scan_results WHERE id = ?", (scan_id,))
        elif scan_date:
            cur = conn.execute(
                "SELECT * FROM launch_scan_results WHERE scan_date = ? ORDER BY id DESC LIMIT 1",
                (scan_date,))
        else:
            cur = conn.execute("SELECT * FROM launch_scan_results ORDER BY id DESC LIMIT 1")
        scan_info = cur.fetchone()
        if not scan_info:
            return None, pd.DataFrame()
        scan_info = dict(scan_info)

        candidates = pd.read_sql_query(
            "SELECT * FROM launch_scan_candidates WHERE scan_id = ? ORDER BY score DESC",
            conn, params=(scan_info['id'],))
        return scan_info, candidates

    def get_available_scans(self, conn):
        """获取所有可用扫描日期"""
        cur = conn.execute("""
            SELECT id, scan_date, scan_time, total_candidates, total_scanned, winner_sample_count
            FROM launch_scan_results ORDER BY id DESC LIMIT 50
        """)
        return [dict(r) for r in cur.fetchall()]

    def get_winner_templates(self, conn):
        """加载大涨股模板"""
        return pd.read_sql_query("SELECT * FROM winner_templates", conn)

    def get_winner_stats(self, conn):
        """获取大涨股统计"""
        df = pd.read_sql_query(
            "SELECT kdj_k, kdj_d, rsi14, dev_ma60, down_days, dd_60, boll_pos, vol_ratio, pct_20d, macd_dif, macd_hist FROM winner_templates",
            conn)
        if df.empty:
            return {}
        df = df.dropna()
        return {
            'count': len(df),
            'kdj_mean': float(df['kdj_k'].mean()),
            'kdj_below_30_pct': float((df['kdj_k'] < 30).mean() * 100),
            'rsi_mean': float(df['rsi14'].mean()),
            'rsi_below_35_pct': float((df['rsi14'] < 35).mean() * 100),
            'dev_ma60_mean': float(df['dev_ma60'].mean()),
            'dev_ma60_below_0_pct': float((df['dev_ma60'] < 0).mean() * 100),
            'down_days_ge3_pct': float((df['down_days'] >= 3).mean() * 100),
            'dd_60_mean': float(df['dd_60'].mean()),
            'vol_below_1_pct': float((df['vol_ratio'] < 1).mean() * 100),
            'pct_20d_mean': float(df['pct_20d'].mean()),
        }

    # ==================== 完整扫描流程 ====================

    def run_full_scan(self, progress_callback=None, top_n=200):
        """执行完整扫描流程"""
        conn = self._get_conn()
        try:
            # 阶段1: 准备
            if progress_callback:
                progress_callback('init', 5, '正在获取最新交易日...')

            latest_date = self.get_latest_date(conn)
            all_dates = self.get_date_range(conn, latest_date, 120)
            scan_date = datetime.now().strftime('%Y-%m-%d')

            # 阶段2: 学习大涨股
            if progress_callback:
                progress_callback('learn', 15, '正在识别大涨股并提取起涨点特征...')

            winners, _ = self.find_winner_samples(conn, latest_date, all_dates)
            winner_count = len(winners)
            df_w = self.extract_winner_features(conn, winners['code'].tolist(), all_dates, latest_date)
            df_w = df_w.dropna(subset=['kdj_k', 'rsi14', 'dev_ma60', 'dd_60', 'boll_pos'])

            if not df_w.empty:
                self.save_winner_templates(conn, df_w, scan_date)

            # 阶段3: 全市场数据
            if progress_callback:
                progress_callback('load', 35, '正在加载全市场最新数据...')

            df_today = self.load_today_data(conn, latest_date)
            df_hist = self.load_historical_data(conn, (all_dates[0], latest_date))
            features = self.calc_derived_features(df_hist)

            if progress_callback:
                progress_callback('merge', 50, '正在合并数据并计算指标...')

            df = df_today.merge(features, on='code', how='inner')
            df['dev_ma60'] = np.where(
                (df['ma60'].notna()) & (df['ma60'] > 0),
                (df['close'] - df['ma60']) / df['ma60'] * 100, np.nan)
            df['dev_ma20'] = np.where(
                (df['ma20'].notna()) & (df['ma20'] > 0),
                (df['close'] - df['ma20']) / df['ma20'] * 100, np.nan)
            df['boll_pos'] = np.where(
                (df['boll_upper'].notna()) & (df['boll_lower'].notna()) & (df['boll_upper'] > df['boll_lower']),
                (df['close'] - df['boll_lower']) / (df['boll_upper'] - df['boll_lower']), np.nan)
            df = df.dropna(subset=['kdj_k', 'rsi14', 'dev_ma60', 'dd_60', 'boll_pos', 'down_days'])

            # 阶段4: 筛选评分
            if progress_callback:
                progress_callback('filter', 65, f'正在硬条件筛选 ({len(df)}只)...')

            df_filtered, filter_stats = self.apply_hard_filters(df)

            if progress_callback:
                progress_callback('score', 80, f'正在综合评分 ({len(df_filtered)}只)...')

            scores = []
            breakdowns = []
            for _, row in df_filtered.iterrows():
                s, bd = self.calc_score_with_breakdown(row)
                scores.append(s)
                breakdowns.append(bd)

            df_filtered['score'] = scores
            df_filtered['score_breakdown'] = breakdowns
            df_top = df_filtered.nlargest(top_n, 'score')

            # 阶段5: 保存
            if progress_callback:
                progress_callback('save', 95, '正在保存扫描结果...')

            scan_id = self.save_scan_results(
                conn, scan_date, df_top,
                self.compute_winner_statistics(df_w) if not df_w.empty else {},
                filter_stats, winner_count, latest_date)

            if progress_callback:
                progress_callback('done', 100, f'扫描完成! 候选{len(df_top)}只')

            # 构建返回结果
            candidates_data = []
            for _, r in df_top.iterrows():
                candidates_data.append({
                    'code': r['code'], 'close': float(r['close']),
                    'pct_change': float(r['pct_change']), 'score': int(r['score']),
                    'kdj_k': float(r['kdj_k']), 'kdj_d': float(r['kdj_d']), 'kdj_j': float(r['kdj_j']),
                    'rsi14': float(r['rsi14']),
                    'dev_ma60': float(r['dev_ma60']), 'down_days': int(r['down_days']),
                    'dd_60': float(r['dd_60']), 'volume_ratio': float(r['volume_ratio']),
                    'price_pct_20d': float(r['price_pct_20d']), 'boll_pos': float(r['boll_pos']),
                    'macd_dif': float(r['macd_dif']), 'macd_hist': float(r['macd_hist']),
                    'score_breakdown': r['score_breakdown'],
                })

            return {
                'scan_id': scan_id,
                'scan_date': scan_date,
                'candidates': candidates_data,
                'candidates_count': len(candidates_data),
                'filter_stats': filter_stats,
                'winner_count': winner_count,
                'winner_stats': self.compute_winner_statistics(df_w) if not df_w.empty else {},
                'latest_trade_date': latest_date,
            }
        finally:
            conn.close()
