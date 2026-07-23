"""
A股起涨点扫描系统 V5 - 日期列表 + 股票跟踪版
布局：��侧日期列表 | 右侧候选股票列表+后续涨跌跟踪
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
from modules.stock_pool import pool_manager
from modules.data_collector import collector
from modules.launch_scanner import LaunchPointScanner
from scheduler.job_scheduler import stock_scheduler

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide",
                   initial_sidebar_state="expanded")

init_database()
scanner = LaunchPointScanner()

# 启动调度器
if 'scheduler_started' not in st.session_state:
    try:
        stock_scheduler.start()
        st.session_state.scheduler_started = True
    except:
        pass

# ========== Session State ==========
if 'selected_scan_id' not in st.session_state:
    st.session_state.selected_scan_id = None
if 'expanded_code' not in st.session_state:
    st.session_state.expanded_code = None
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
            SELECT id, scan_date, scan_time, total_candidates, total_scanned,
                   hard_filter_passed, winner_sample_count, latest_trade_date
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
def get_tracking_data(codes, scan_date, max_days=20):
    """获取一批股票在扫描日期之后的每日涨跌幅"""
    with get_db() as conn:
        placeholders = ','.join(['?'] * len(codes))
        df = pd.read_sql_query(f"""
            SELECT code, trade_date, close, pct_change
            FROM stock_daily
            WHERE code IN ({placeholders}) AND trade_date >= ?
            ORDER BY code, trade_date
        """, conn, params=codes + [scan_date])
    return df

# ========== 图表函数 ==========
def build_kline_chart(df, code, scan_date=None):
    """K线+指标图"""
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.35, 0.2, 0.2, 0.25],
        subplot_titles=('K线+均线', 'MACD', 'KDJ', '成交量+RSI'),
    )
    # K线
    fig.add_trace(go.Candlestick(
        x=df['trade_date'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='K线',
        increasing_line_color='#ef5350', decreasing_line_color='#26a69a',
    ), row=1, col=1)
    for p, color in [(20, '#FF9800'), (60, '#2196F3')]:
        col = f'ma{p}'
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df['trade_date'], y=df[col], mode='lines',
                name=f'MA{p}', line=dict(width=1, color=color),
            ), row=1, col=1)
    if scan_date:
        for r in [1, 2, 3, 4]:
            fig.add_vline(x=scan_date, line_width=1.5, line_dash='dash',
                          line_color='#FF5722', opacity=0.6, row=r, col=1)

    # MACD
    macd_colors = np.where(df['macd_hist'].values > 0, '#ef5350', '#26a69a')
    fig.add_trace(go.Bar(x=df['trade_date'], y=df['macd_hist'], name='MACD柱',
                          marker_color=macd_colors, marker_line_width=0), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['trade_date'], y=df['macd_dif'], mode='lines',
                              name='DIF', line=dict(width=1, color='#2196F3')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['trade_date'], y=df['macd_dea'], mode='lines',
                              name='DEA', line=dict(width=1, color='#FF9800')), row=2, col=1)

    # KDJ
    fig.add_trace(go.Scatter(x=df['trade_date'], y=df['kdj_k'], mode='lines',
                              name='K', line=dict(width=1, color='#2196F3')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['trade_date'], y=df['kdj_d'], mode='lines',
                              name='D', line=dict(width=1, color='#FF9800')), row=3, col=1)
    fig.add_hline(y=20, line_dash='dash', line_color='#26a69a', line_width=0.8, row=3, col=1)
    fig.add_hline(y=80, line_dash='dash', line_color='#ef5350', line_width=0.8, row=3, col=1)

    # Volume + RSI
    vol_colors = np.where(df['close'].values > df['open'].values, '#ef5350', '#26a69a')
    fig.add_trace(go.Bar(x=df['trade_date'], y=df['volume'], name='成交量',
                          marker_color=vol_colors, marker_line_width=0, opacity=0.4), row=4, col=1)
    fig.add_trace(go.Scatter(x=df['trade_date'], y=df['rsi14'], mode='lines',
                              name='RSI(14)', line=dict(width=1.5, color='#9C27B0')), row=4, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='#26a69a', line_width=0.8, row=4, col=1)
    fig.add_hline(y=70, line_dash='dash', line_color='#ef5350', line_width=0.8, row=4, col=1)

    fig.update_layout(height=650, hovermode='x unified', template='plotly_white',
                      xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def get_score_color(score):
    if score >= 24: return '#D32F2F'
    elif score >= 20: return '#FF5722'
    elif score >= 16: return '#FF9800'
    else: return '#FFC107'


# ========== 左侧边栏：日期列表 ==========
with st.sidebar:
    st.title("🎯 起涨点扫描")

    # 扫描按钮
    st.button("🔍 今日扫描", type="primary", use_container_width=True,
              key="scan_btn", disabled=st.session_state.scan_running,
              on_click=lambda: setattr(st.session_state, 'scan_running', True))

    # 调度器状态
    try:
        sch = stock_scheduler.get_status()
        for j in sch.get('jobs', []):
            if j['id'] == 'daily_scan':
                st.caption(f"⏰ 下次: {j['next_run']}")
    except:
        pass

    st.divider()

    # 日期列表
    scan_dates = get_scan_dates()
    if scan_dates:
        st.caption(f"共 {len(scan_dates)} 次扫描记录")

        for s in scan_dates:
            sid = s['id']
            label = f"{s['scan_date']} ({s['total_candidates']}只)"
            is_selected = st.session_state.selected_scan_id == sid

            btn_style = "primary" if is_selected else "secondary"
            if st.button(label, key=f"date_{sid}", use_container_width=True,
                         type=btn_style if is_selected else "secondary"):
                st.session_state.selected_scan_id = sid
                st.session_state.expanded_code = None
                st.rerun()
    else:
        st.info("暂无扫描记录")

    st.divider()
    st.caption("数据来源: AKShare/新浪")


# ========== 运行扫描 ==========
if st.session_state.scan_running:
    progress_bar = st.progress(0, "准备扫描...")
    status_text = st.empty()

    def progress_cb(stage, pct, msg):
        progress_bar.progress(pct, msg)
        if stage == 'done':
            st.session_state.scan_running = False

    with st.spinner("正在执行起涨点扫描..."):
        result = scanner.run_full_scan(progress_callback=progress_cb, top_n=200)
        get_scan_dates.clear()
        get_scan_candidates.clear()
        st.session_state.selected_scan_id = result['scan_id']
        st.session_state.expanded_code = None

    progress_bar.progress(100, "扫描完成!")
    st.success(f"✅ 扫描完成! {result['candidates_count']} 只候选")
    st.rerun()


# ========== 右侧主区域 ==========
selected_id = st.session_state.selected_scan_id
if not selected_id and scan_dates:
    selected_id = scan_dates[0]['id']
    st.session_state.selected_scan_id = selected_id

if selected_id:
    df_candidates = get_scan_candidates(selected_id)
    scan_info = next((s for s in scan_dates if s['id'] == selected_id), {})

    if df_candidates.empty:
        st.warning("该扫描无候选数据")
    else:
        scan_date = scan_info.get('latest_trade_date', '')
        candidates = df_candidates.to_dict('records')
        codes = [c['code'] for c in candidates]
        names = get_stock_names(codes)

        # 标题行
        st.title(f"📅 {scan_info.get('scan_date', '')} 起涨点候选")
        col_h1, col_h2, col_h3, col_h4 = st.columns(4)
        with col_h1:
            st.metric("候选股票", f"{len(candidates)}只")
        with col_h2:
            st.metric("硬过滤通过", f"{scan_info.get('hard_filter_passed', '?')}只")
        with col_h3:
            st.metric("扫描总数", f"{scan_info.get('total_scanned', '?')}只")
        with col_h4:
            avg_score = np.mean([c['score'] for c in candidates])
            st.metric("平均得分", f"{avg_score:.1f}")

        # 后续跟踪数据
        tracking_df = get_tracking_data(codes, scan_date)
        tracking_available = not tracking_df.empty

        # 获取后续交易日列表
        if tracking_available:
            track_dates = sorted(tracking_df['trade_date'].unique())
            # 只取扫描日期之后的
            track_dates = [d for d in track_dates if d > scan_date]
        else:
            track_dates = []

        st.divider()

        # 股票列表表格
        st.subheader(f"📋 候选股票列表")

        # 构建表格数据
        table_rows = []
        for c in candidates:
            code = c['code']
            name = names.get(code, '')
            row = {
                '代码': code,
                '名称': name,
                '收盘价': f"{c['close']:.2f}",
                '当日涨跌': f"{c['pct_change']:+.2f}%",
                '得分': c['score'],
                'KDJ_K': f"{c['kdj_k']:.0f}",
                'RSI14': f"{c['rsi14']:.0f}",
                'MA60偏离': f"{c['dev_ma60']:+.0f}%",
                '连跌': f"{c['down_days']}天",
                '60日回撤': f"{c['dd_60']:+.0f}%",
                '量比': f"{c['volume_ratio']:.2f}",
            }

            # 后续跟踪：每天涨跌幅
            if tracking_available:
                code_track = tracking_df[tracking_df['code'] == code]
                for td in track_dates[:10]:  # 最多显示10天
                    td_row = code_track[code_track['trade_date'] == td]
                    if not td_row.empty:
                        pct = td_row.iloc[0]['pct_change']
                        row[td] = f"{pct:+.2f}%" if pct is not None else '-'
                    else:
                        row[td] = '-'

            table_rows.append(row)

        df_table = pd.DataFrame(table_rows)

        # 得分列颜色
        def color_score(val):
            try:
                v = int(val)
                return f'color: {get_score_color(v)}; font-weight: bold'
            except:
                return ''

        # 跟踪列颜色
        def color_track(val):
            if isinstance(val, str) and val.startswith('+'):
                return 'color: #ef5350; font-weight: bold'
            elif isinstance(val, str) and val.startswith('-'):
                return 'color: #26a69a; font-weight: bold'
            return ''

        styled = df_table.style.applymap(color_score, subset=['得分'])
        for td in track_dates[:10]:
            styled = styled.applymap(color_track, subset=[td])

        st.dataframe(styled, use_container_width=True, hide_index=True,
                     height=min(800, 35 * len(table_rows) + 40))

        # 跟踪说明
        if track_dates:
            st.caption(f"📊 后续跟踪列: {', '.join(track_dates[:10])} | 🔴红=涨 🟢绿=跌")

        # 展开查看K线图
        st.divider()
        st.subheader("📈 K线图查看")

        col_sel1, col_sel2 = st.columns([2, 2])
        with col_sel1:
            view_code = st.selectbox(
                "选择股票查看K线图",
                [f"{c['code']} {names.get(c['code'], '')} ⭐{c['score']}" for c in candidates],
                key="kline_select"
            )
        if view_code:
            view_code_pure = view_code.split()[0]
            df_kline = get_daily_data(view_code_pure, 250)
            if not df_kline.empty:
                fig = build_kline_chart(df_kline, view_code_pure, scan_date)
                st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 从左侧选择日期查看候选股票，或点击「今日扫描」")

st.divider()
st.caption(f"© A股起涨点扫描系统 V5 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
