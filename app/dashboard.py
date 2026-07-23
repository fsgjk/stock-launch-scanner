"""
A股起涨点扫描系统 V6 - 跟踪验证版
左侧日期 | 右侧：代码/名称/收盘价/持仓天数/累计涨跌幅跟踪
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import PAGE_TITLE, PAGE_ICON
from utils.database import init_database, get_db
from modules.launch_scanner import LaunchPointScanner
from scheduler.job_scheduler import stock_scheduler

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide",
                   initial_sidebar_state="expanded")
init_database()
scanner = LaunchPointScanner()

if 'scheduler_started' not in st.session_state:
    try:
        stock_scheduler.start()
        st.session_state.scheduler_started = True
    except:
        pass

if 'selected_scan_id' not in st.session_state:
    st.session_state.selected_scan_id = None
if 'scan_running' not in st.session_state:
    st.session_state.scan_running = False

# ========== 缓存 ==========
@st.cache_data(ttl=300)
def get_daily_data(code, days=250):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM stock_daily WHERE code=? ORDER BY trade_date DESC LIMIT ?",
            (code, days)).fetchall()
    return pd.DataFrame([dict(r) for r in rows]).sort_values("trade_date") if rows else pd.DataFrame()

@st.cache_data(ttl=300)
def get_scan_dates():
    with get_db() as conn:
        cur = conn.execute("""
            SELECT id, scan_date, scan_time, total_candidates, latest_trade_date
            FROM launch_scan_results ORDER BY id DESC LIMIT 60
        """)
        return [dict(r) for r in cur.fetchall()]

@st.cache_data(ttl=300)
def get_scan_candidates(scan_id):
    with get_db() as conn:
        return pd.read_sql_query(
            "SELECT * FROM launch_scan_candidates WHERE scan_id=? ORDER BY score DESC",
            conn, params=(scan_id,))

@st.cache_data(ttl=3600)
def get_stock_names(codes):
    if not codes:
        return {}
    with get_db() as conn:
        placeholders = ','.join(['?'] * len(codes))
        cur = conn.execute(f"SELECT code, name FROM stock_info WHERE code IN ({placeholders})", codes)
        return {r['code']: r['name'] for r in cur.fetchall()}

@st.cache_data(ttl=300)
def get_tracking_pivoted(codes, entry_date):
    """获取一批股票从entry_date起每个交易日的收盘价，计算累计涨跌幅"""
    with get_db() as conn:
        placeholders = ','.join(['?'] * len(codes))
        df = pd.read_sql_query(f"""
            SELECT code, trade_date, close
            FROM stock_daily
            WHERE code IN ({placeholders}) AND trade_date >= ?
            ORDER BY code, trade_date
        """, conn, params=codes + [entry_date])
    if df.empty:
        return {}, []

    # 获取所有交易日（排除非交易日）
    all_dates = sorted(df['trade_date'].unique())

    # 为每只股票计算累计涨跌幅
    result = {}
    for code in codes:
        sdf = df[df['code'] == code].set_index('trade_date')['close']
        if sdf.empty or entry_date not in sdf.index:
            continue

        entry_price = sdf.loc[entry_date]
        cum_pct = {}
        for d in all_dates:
            if d >= entry_date and d in sdf.index:
                cum_pct[d] = (sdf.loc[d] - entry_price) / entry_price * 100
            elif d > entry_date:
                cum_pct[d] = None  # 未开市或停牌
        result[code] = {
            'entry_price': entry_price,
            'cum_pct': cum_pct,
            'days': len([d for d in all_dates if d >= entry_date and d in sdf.index]),
        }
    return result, all_dates

# ========== 侧边栏 ==========
with st.sidebar:
    st.title("🎯 起涨点扫描")

    st.button("🔍 今日扫描", type="primary", use_container_width=True,
              key="scan_btn", disabled=st.session_state.scan_running,
              on_click=lambda: setattr(st.session_state, 'scan_running', True))

    try:
        sch = stock_scheduler.get_status()
        for j in sch.get('jobs', []):
            if j['id'] == 'daily_scan':
                st.caption(f"⏰ 下次自动: {j['next_run']}")
    except:
        pass

    st.divider()
    st.caption("📅 扫描日期")

    scan_dates = get_scan_dates()
    if scan_dates:
        for s in scan_dates:
            sid = s['id']
            label = f"{s['scan_date']} ({s['total_candidates']}只)"
            is_sel = st.session_state.selected_scan_id == sid
            if st.button(label, key=f"d_{sid}", use_container_width=True,
                         type="primary" if is_sel else "secondary"):
                st.session_state.selected_scan_id = sid
                st.rerun()
    else:
        st.info("暂无扫描")

    st.divider()
    st.caption("数据: AKShare/新浪")


# ========== 运行扫描 ==========
if st.session_state.scan_running:
    progress_bar = st.progress(0, "扫描中...")
    def progress_cb(stage, pct, msg):
        progress_bar.progress(pct, msg)
        if stage == 'done':
            st.session_state.scan_running = False
    with st.spinner("正在执行..."):
        result = scanner.run_full_scan(progress_callback=progress_cb, top_n=200)
        get_scan_dates.clear()
        get_scan_candidates.clear()
        st.session_state.selected_scan_id = result['scan_id']
    progress_bar.progress(100, "完成!")
    st.success(f"✅ {result['candidates_count']}只候选")
    st.rerun()

# ========== 主区域 ==========
selected_id = st.session_state.selected_scan_id
if not selected_id and scan_dates:
    selected_id = scan_dates[0]['id']
    st.session_state.selected_scan_id = selected_id

if selected_id:
    df_c = get_scan_candidates(selected_id)
    scan_info = next((s for s in scan_dates if s['id'] == selected_id), {})

    if df_c.empty:
        st.warning("无候选数据")
    else:
        scan_date = scan_info.get('latest_trade_date', '')
        candidates = df_c.to_dict('records')
        codes = [c['code'] for c in candidates]
        names = get_stock_names(codes)

        # 获取跟踪数据
        tracking, all_track_dates = get_tracking_pivoted(codes, scan_date)
        # 只保留 entry_date 之后的日期
        track_dates = [d for d in all_track_dates if d > scan_date]

        st.title(f"📅 {scan_info.get('scan_date', '')} 起涨点候选")

        # 概览
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("候选数", f"{len(candidates)}只")
        with col2: st.metric("数据日期", scan_date)

        # 后续统计
        if track_dates and tracking:
            up_count = 0
            down_count = 0
            for c in candidates:
                code = c['code']
                if code in tracking and track_dates:
                    last_d = track_dates[-1]
                    cum = tracking[code]['cum_pct'].get(last_d)
                    if cum is not None:
                        if cum > 0: up_count += 1
                        else: down_count += 1
            with col3:
                st.metric("最新累计", f"📈{up_count}涨 📉{down_count}跌" if up_count + down_count > 0 else "暂无后续数据")

        st.divider()

        # ===== 核心表格 =====
        st.subheader("📋 股票列表 & 累计涨跌幅跟踪")

        # 构建表头：基本信息 + 技术指标 + 评分明细 + 累计涨跌幅
        base_columns = ['代码', '名称', '当日收盘', '当日涨跌', '得分', '评分明细', '持仓天数']
        metric_columns = ['KDJ_K', 'RSI14', '连跌天', '60日回撤', 'MA60偏离', '量比']
        all_static_cols = base_columns + metric_columns

        table_data = []
        for c in candidates:
            code = c['code']
            name = names.get(code, '')
            tk = tracking.get(code, {})

            # 解析评分明细
            try:
                bd = json.loads(c.get('score_breakdown', '{}'))
                bd_str = '|'.join([f"{k}:{v}" for k, v in bd.items()])
            except:
                bd_str = ''

            row = {
                '代码': code,
                '名称': name,
                '当日收盘': f"{c['close']:.2f}",
                '当日涨跌': f"{c['pct_change']:+.2f}%" if c.get('pct_change') is not None else '-',
                '得分': c['score'],
                '评分明细': bd_str,
                '持仓天数': tk.get('days', 0) if tk else 0,
                'KDJ_K': f"{c['kdj_k']:.1f}" if c.get('kdj_k') is not None else '-',
                'RSI14': f"{c['rsi14']:.1f}" if c.get('rsi14') is not None else '-',
                '连跌天': int(c['down_days']) if c.get('down_days') is not None else 0,
                '60日回撤': f"{c['dd_60']:+.1f}%" if c.get('dd_60') is not None else '-',
                'MA60偏离': f"{c['dev_ma60']:+.1f}%" if c.get('dev_ma60') is not None else '-',
                '量比': f"{c['volume_ratio']:.2f}" if c.get('volume_ratio') is not None else '-',
            }

            for td in track_dates:
                if tk and td in tk['cum_pct']:
                    v = tk['cum_pct'][td]
                    row[td] = v if v is not None else None
                else:
                    row[td] = None

            table_data.append(row)

        df_table = pd.DataFrame(table_data)

        # 样式函数
        def color_score(val):
            try:
                v = int(val)
                if v >= 24: return 'color: #D32F2F; font-weight: bold; font-size: 14px'
                elif v >= 20: return 'color: #FF5722; font-weight: bold'
                elif v >= 16: return 'color: #FF9800; font-weight: bold'
                return 'color: #FFC107'
            except: return ''

        def color_pct_change(val):
            """当日涨跌颜色：涨红跌绿"""
            if isinstance(val, str) and val != '-':
                try:
                    v = float(val.replace('%', '').replace('+', ''))
                    if v > 0: return 'color: #ef5350'
                    elif v < 0: return 'color: #26a69a'
                except: pass
            return ''

        def color_cum(val):
            if val is None or pd.isna(val):
                return ''
            if val > 0:
                return 'color: #ef5350; font-weight: bold'
            else:
                return 'color: #26a69a; font-weight: bold'

        def color_metric(val, low_good=True):
            """技术指标颜色：根据超卖程度着色"""
            if val == '-' or val is None:
                return ''
            try:
                v = float(str(val).replace('%', '').replace('+', ''))
            except:
                return ''
            if low_good:
                if v < 10: return 'color: #D32F2F; font-weight: bold'  # 极度超卖
                elif v < 20: return 'color: #FF5722'
                elif v < 30: return 'color: #FF9800'
                return ''
            else:
                # 量比越接近1越好（缩量），越低越好
                if v < 0.5: return 'color: #D32F2F; font-weight: bold'
                elif v < 0.7: return 'color: #FF5722'
                elif v < 0.9: return 'color: #FF9800'
                return ''

        def color_down_days(val):
            """连跌天数：越多越极端"""
            try:
                v = int(val)
                if v >= 6: return 'color: #D32F2F; font-weight: bold'
                elif v >= 4: return 'color: #FF5722; font-weight: bold'
                elif v >= 2: return 'color: #FF9800'
                return ''
            except: return ''

        def color_dd(val):
            """60日回撤：越大越好（跌得越多越有反弹空间）"""
            if val == '-' or val is None: return ''
            try:
                v = float(str(val).replace('%', '').replace('+', ''))
                if v <= -50: return 'color: #D32F2F; font-weight: bold'
                elif v <= -35: return 'color: #FF5722; font-weight: bold'
                elif v <= -25: return 'color: #FF9800'
                return ''
            except: return ''

        def color_dev_ma60(val):
            """MA60偏离：越负越超卖"""
            if val == '-' or val is None: return ''
            try:
                v = float(str(val).replace('%', '').replace('+', ''))
                if v <= -30: return 'color: #D32F2F; font-weight: bold'
                elif v <= -20: return 'color: #FF5722; font-weight: bold'
                elif v <= -10: return 'color: #FF9800'
                return ''
            except: return ''

        # 应用样式
        style_subsets = {
            '得分': color_score,
            '当日涨跌': color_pct_change,
            'KDJ_K': color_metric,
            'RSI14': color_metric,
            '连跌天': color_down_days,
            '60日回撤': color_dd,
            'MA60偏离': color_dev_ma60,
            '量比': lambda v: color_metric(v, low_good=False),
        }

        styled = df_table.style
        for col, fn in style_subsets.items():
            if col in df_table.columns:
                styled = styled.map(fn, subset=[col])

        for td in track_dates:
            styled = styled.map(color_cum, subset=[td])
            styled = styled.format({td: lambda v: f"{v:+.2f}%" if v is not None and not (isinstance(v, float) and pd.isna(v)) else '-'})

        st.dataframe(styled, use_container_width=True, hide_index=True,
                     height=min(800, 35 * len(table_data) + 40),
                     column_config={
                         '评分明细': st.column_config.TextColumn('评分明细', width='small'),
                     })

        st.caption(f"📊 跟踪日期: {', '.join(track_dates) if track_dates else '暂无后续交易日'} | 🔴红=累计涨 🟢绿=累计跌 | KDJ_K/RSI14越红=超卖越严重 | 连跌天/回撤/偏离越红=越极端 | 评分明细: kdj/rsi/ma60_dev/down_days/dd_60/price_pct/macd/volume/boll/near_low")

        # 总体胜率曲线
        if track_dates and tracking:
            st.divider()
            st.subheader("📈 整体跟踪表现")

            # 每天计算平均累计涨幅和涨跌比
            daily_stats = []
            for td in track_dates:
                vals = []
                for c in candidates:
                    code = c['code']
                    if code in tracking and td in tracking[code]['cum_pct']:
                        v = tracking[code]['cum_pct'][td]
                        if v is not None:
                            vals.append(v)
                if vals:
                    up = sum(1 for v in vals if v > 0)
                    down = sum(1 for v in vals if v < 0)
                    daily_stats.append({
                        'date': td,
                        'avg_cum': np.mean(vals),
                        'median_cum': np.median(vals),
                        'up_pct': up / len(vals) * 100,
                        'count': len(vals),
                    })

            if daily_stats:
                ds_df = pd.DataFrame(daily_stats)

                col_ch1, col_ch2 = st.columns(2)
                with col_ch1:
                    # 平均累计涨跌幅曲线
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=ds_df['date'], y=ds_df['avg_cum'], mode='lines+markers',
                        name='平均累计涨跌幅', line=dict(color='#FF5722', width=2),
                        fill='tozeroy', fillcolor='rgba(255,87,34,0.1)',
                    ))
                    fig.add_hline(y=0, line_dash='dash', line_color='gray')
                    fig.update_layout(title='平均累计涨跌幅', height=300,
                                      template='plotly_white', yaxis_ticksuffix='%')
                    st.plotly_chart(fig, use_container_width=True)

                with col_ch2:
                    # 胜率曲线
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=ds_df['date'], y=ds_df['up_pct'], mode='lines+markers',
                        name='上涨比例', line=dict(color='#ef5350', width=2),
                        fill='tozeroy', fillcolor='rgba(239,83,80,0.1)',
                    ))
                    fig2.add_hline(y=50, line_dash='dash', line_color='gray')
                    fig2.update_layout(title='累计上涨比例（胜率）', height=300,
                                       template='plotly_white', yaxis_ticksuffix='%',
                                       yaxis_range=[0, 100])
                    st.plotly_chart(fig2, use_container_width=True)

                st.caption(f"📊 跟踪{daily_stats[-1]['count']}只有效数据 | 最新平均累计: {daily_stats[-1]['avg_cum']:+.2f}% | 上涨比例: {daily_stats[-1]['up_pct']:.1f}%")

        # K线图查看
        st.divider()
        st.subheader("📈 个股K线图")
        view_opts = [f"{c['code']} {names.get(c['code'], '')} ⭐{c['score']}" for c in candidates]
        view_sel = st.selectbox("选择股票", view_opts, key="kline")
        if view_sel:
            code = view_sel.split()[0]
            df_k = get_daily_data(code, 250)
            if not df_k.empty:
                fig = make_subplots(
                    rows=4, cols=1, shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.35, 0.2, 0.2, 0.25],
                    subplot_titles=('K线+均线', 'MACD', 'KDJ', '成交量+RSI'),
                )
                fig.add_trace(go.Candlestick(
                    x=df_k['trade_date'], open=df_k['open'], high=df_k['high'],
                    low=df_k['low'], close=df_k['close'], name='K线',
                    increasing_line_color='#ef5350', decreasing_line_color='#26a69a',
                ), row=1, col=1)
                for p, c in [(20, '#FF9800'), (60, '#2196F3')]:
                    col = f'ma{p}'
                    if col in df_k.columns:
                        fig.add_trace(go.Scatter(
                            x=df_k['trade_date'], y=df_k[col], mode='lines',
                            name=f'MA{p}', line=dict(width=1, color=c),
                        ), row=1, col=1)
                # 标注入选日
                for r in [1, 2, 3, 4]:
                    fig.add_vline(x=scan_date, line_width=1.5, line_dash='dash',
                                  line_color='#FF5722', opacity=0.6, row=r, col=1)

                # MACD
                mc = np.where(df_k['macd_hist'].values > 0, '#ef5350', '#26a69a')
                fig.add_trace(go.Bar(x=df_k['trade_date'], y=df_k['macd_hist'],
                                      name='MACD', marker_color=mc, marker_line_width=0), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_k['trade_date'], y=df_k['macd_dif'],
                                          mode='lines', name='DIF', line=dict(width=1, color='#2196F3')), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_k['trade_date'], y=df_k['macd_dea'],
                                          mode='lines', name='DEA', line=dict(width=1, color='#FF9800')), row=2, col=1)

                # KDJ
                fig.add_trace(go.Scatter(x=df_k['trade_date'], y=df_k['kdj_k'],
                                          mode='lines', name='K', line=dict(width=1, color='#2196F3')), row=3, col=1)
                fig.add_trace(go.Scatter(x=df_k['trade_date'], y=df_k['kdj_d'],
                                          mode='lines', name='D', line=dict(width=1, color='#FF9800')), row=3, col=1)
                fig.add_hline(y=20, line_dash='dash', line_color='#26a69a', row=3, col=1)
                fig.add_hline(y=80, line_dash='dash', line_color='#ef5350', row=3, col=1)

                # Vol + RSI
                vc = np.where(df_k['close'].values > df_k['open'].values, '#ef5350', '#26a69a')
                fig.add_trace(go.Bar(x=df_k['trade_date'], y=df_k['volume'],
                                      name='量', marker_color=vc, marker_line_width=0, opacity=0.4), row=4, col=1)
                fig.add_trace(go.Scatter(x=df_k['trade_date'], y=df_k['rsi14'],
                                          mode='lines', name='RSI14', line=dict(width=1.5, color='#9C27B0')), row=4, col=1)

                fig.update_layout(height=650, hovermode='x unified', template='plotly_white',
                                  xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 从左侧选择日期，或点击「今日扫描」")

st.divider()
st.caption(f"© A股起涨点扫描系统 V6 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
