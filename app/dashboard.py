"""
A股起涨点扫描系统 V4 - 图形画廊版
核心：每天收盘后扫描，每只候选股展示K线图形卡片
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

# ========== 页面配置 ==========
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide",
                   initial_sidebar_state="collapsed")

init_database()
scanner = LaunchPointScanner()

# ========== Session State ==========
def init_session():
    defaults = {
        'scan_result': None,
        'scan_running': False,
        'selected_date': None,
        'expanded_code': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_session()

# ========== 缓存 ==========
@st.cache_data(ttl=300)
def get_daily_data(code, days=120):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM stock_daily WHERE code=? ORDER BY trade_date DESC LIMIT ?",
            (code, days)
        ).fetchall()
    if rows:
        return pd.DataFrame([dict(r) for r in rows]).sort_values("trade_date")
    return pd.DataFrame()

@st.cache_data(ttl=300)
def get_scan_dates():
    with get_db() as conn:
        cur = conn.execute("""
            SELECT id, scan_date, scan_time, total_candidates, total_scanned,
                   winner_sample_count, latest_trade_date
            FROM launch_scan_results ORDER BY id DESC LIMIT 30
        """)
        return [dict(r) for r in cur.fetchall()]

@st.cache_data(ttl=300)
def get_scan_candidates(scan_id):
    with get_db() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM launch_scan_candidates WHERE scan_id=? ORDER BY score DESC",
            conn, params=(scan_id,))
    return df

def get_winner_stats():
    with get_db() as conn:
        return scanner.get_winner_stats(conn)

# ========== 图表构建函数 ==========
def build_thumbnail_chart(df, code, highlight_date=None):
    """构建缩略K线图（小尺寸，用于卡片）"""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.5, 0.25, 0.25],
    )

    # K线
    fig.add_trace(go.Candlestick(
        x=df['trade_date'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='',
        increasing_line_color='#ef5350', decreasing_line_color='#26a69a',
        showlegend=False,
    ), row=1, col=1)

    # MA20, MA60
    for p, color in [(20, '#FF9800'), (60, '#2196F3')]:
        col = f'ma{p}'
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df['trade_date'], y=df[col], mode='lines',
                line=dict(width=0.8, color=color), showlegend=False,
            ), row=1, col=1)

    # 起涨点竖线
    if highlight_date:
        for r in [1, 2, 3]:
            fig.add_vline(x=highlight_date, line_width=1, line_dash='dash',
                          line_color='#FF5722', opacity=0.5, row=r, col=1)

    # MACD 柱
    macd_colors = np.where(df['macd_hist'].values > 0, '#ef5350', '#26a69a')
    fig.add_trace(go.Bar(
        x=df['trade_date'], y=df['macd_hist'], marker_color=macd_colors,
        marker_line_width=0, showlegend=False,
    ), row=2, col=1)

    # KDJ K线
    fig.add_trace(go.Scatter(
        x=df['trade_date'], y=df['kdj_k'], mode='lines',
        line=dict(width=1, color='#2196F3'), showlegend=False,
    ), row=3, col=1)
    fig.add_hline(y=20, line_dash='dash', line_color='#26a69a', line_width=0.5, row=3, col=1)
    fig.add_hline(y=80, line_dash='dash', line_color='#ef5350', line_width=0.5, row=3, col=1)

    fig.update_layout(
        height=220, margin=dict(l=0, r=0, t=5, b=0),
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        xaxis_visible=False,
    )
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)

    return fig


def build_full_chart(df, code, highlight_date=None):
    """构建完整大图"""
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.35, 0.2, 0.2, 0.25],
        subplot_titles=('K线+均线', 'MACD', 'KDJ', '成交量+RSI'),
    )

    # Row 1: K线 + 均线
    fig.add_trace(go.Candlestick(
        x=df['trade_date'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='K线',
        increasing_line_color='#ef5350', decreasing_line_color='#26a69a',
    ), row=1, col=1)

    for p, color in [(20, '#FF9800'), (60, '#2196F3'), (120, '#9C27B0')]:
        col = f'ma{p}'
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df['trade_date'], y=df[col], mode='lines',
                name=f'MA{p}', line=dict(width=1.2, color=color),
            ), row=1, col=1)

    if 'boll_upper' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['trade_date'], y=df['boll_upper'], mode='lines',
            name='布林上轨', line=dict(dash='dash', width=0.5, color='gray'),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df['trade_date'], y=df['boll_lower'], mode='lines',
            name='布林下轨', line=dict(dash='dash', width=0.5, color='gray'),
            fill='tonexty', fillcolor='rgba(128,128,128,0.06)',
        ), row=1, col=1)

    if highlight_date:
        for r in [1, 2, 3, 4]:
            fig.add_vline(x=highlight_date, line_width=1.5, line_dash='dash',
                          line_color='#FF5722', opacity=0.6, row=r, col=1)

    # Row 2: MACD
    macd_colors = np.where(df['macd_hist'].values > 0, '#ef5350', '#26a69a')
    fig.add_trace(go.Bar(
        x=df['trade_date'], y=df['macd_hist'], name='MACD柱',
        marker_color=macd_colors, marker_line_width=0,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df['trade_date'], y=df['macd_dif'], mode='lines',
        name='DIF', line=dict(width=1.2, color='#2196F3'),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df['trade_date'], y=df['macd_dea'], mode='lines',
        name='DEA', line=dict(width=1.2, color='#FF9800'),
    ), row=2, col=1)
    fig.add_hline(y=0, line_width=0.5, line_color='gray', row=2, col=1)

    # Row 3: KDJ
    fig.add_trace(go.Scatter(
        x=df['trade_date'], y=df['kdj_k'], mode='lines',
        name='K', line=dict(width=1.2, color='#2196F3'),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df['trade_date'], y=df['kdj_d'], mode='lines',
        name='D', line=dict(width=1.2, color='#FF9800'),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df['trade_date'], y=df['kdj_j'], mode='lines',
        name='J', line=dict(width=1, color='#9C27B0', dash='dot'),
    ), row=3, col=1)
    fig.add_hline(y=80, line_dash='dash', line_color='#ef5350', line_width=0.8, row=3, col=1)
    fig.add_hline(y=20, line_dash='dash', line_color='#26a69a', line_width=0.8, row=3, col=1)

    # Row 4: 成交量 + RSI
    vol_colors = np.where(df['close'].values > df['open'].values, '#ef5350', '#26a69a')
    fig.add_trace(go.Bar(
        x=df['trade_date'], y=df['volume'], name='成交量',
        marker_color=vol_colors, marker_line_width=0, opacity=0.4,
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df['trade_date'], y=df['rsi14'], mode='lines',
        name='RSI(14)', line=dict(width=1.5, color='#9C27B0'),
    ), row=4, col=1)
    fig.add_hline(y=70, line_dash='dash', line_color='#ef5350', line_width=0.8, row=4, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='#26a69a', line_width=0.8, row=4, col=1)

    fig.update_layout(
        height=800, hovermode='x unified',
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def get_score_color(score):
    if score >= 24: return '#D32F2F'
    elif score >= 20: return '#FF5722'
    elif score >= 16: return '#FF9800'
    else: return '#FFC107'


# ========== 主界面 ==========
st.title("🎯 A股起涨点扫描")

# --- 顶栏：操作区 ---
col_top1, col_top2, col_top3, col_top4 = st.columns([1.5, 1, 2, 1.5])

with col_top1:
    top_n = st.selectbox("候选数量", [50, 100, 200, 300], index=2)

with col_top2:
    scan_btn = st.button("🔍 今日扫描", type="primary", use_container_width=True,
                         disabled=st.session_state.scan_running)

with col_top3:
    scan_dates = get_scan_dates()
    if scan_dates:
        date_options = [f"{s['scan_date']} {s['scan_time']} ({s['total_candidates']}只)" for s in scan_dates]
        selected_label = st.selectbox("📅 历史扫描", date_options, key="history_scan")
        selected_idx = date_options.index(selected_label)
        st.session_state.selected_date = scan_dates[selected_idx]
    else:
        st.info("暂无扫描记录")

with col_top4:
    if st.session_state.selected_date:
        st.metric("选中扫描", f"{st.session_state.selected_date['total_candidates']}只候选")
        st.caption(f"数据日期: {st.session_state.selected_date.get('latest_trade_date', '?')}")

st.divider()

# --- 运行扫描 ---
if scan_btn and not st.session_state.scan_running:
    st.session_state.scan_running = True
    st.session_state.scan_result = None

if st.session_state.scan_running:
    progress_bar = st.progress(0, "准备扫描...")

    def progress_cb(stage, pct, msg):
        progress_bar.progress(pct, msg)
        if stage == 'done':
            st.session_state.scan_running = False

    with st.spinner("正在执行起涨点扫描..."):
        result = scanner.run_full_scan(progress_callback=progress_cb, top_n=top_n)
        st.session_state.scan_result = result
        get_scan_dates.clear()
        get_scan_candidates.clear()

    progress_bar.progress(100, "扫描完成!")
    st.success(f"✅ 扫描完成! 发现 {result['candidates_count']} 只起涨点候选")
    st.rerun()

# --- 加载候选数据 ---
candidates = None
scan_info = None

if st.session_state.scan_result and st.session_state.scan_result.get('candidates'):
    candidates = st.session_state.scan_result['candidates']
    scan_info = st.session_state.scan_result
elif st.session_state.selected_date:
    df_c = get_scan_candidates(st.session_state.selected_date['id'])
    if not df_c.empty:
        candidates = df_c.to_dict('records')
        scan_info = st.session_state.selected_date

# --- 展示候选图形画廊 ---
if candidates:
    # 得分筛选
    scores = [c['score'] for c in candidates]
    min_score = min(scores)
    max_score = max(scores)

    col_f1, col_f2, col_f3 = st.columns([1, 1, 3])
    with col_f1:
        score_min = st.slider("最低得分", min_score, max_score, max_score - 8 if max_score > 8 else min_score)
    with col_f2:
        sort_by = st.selectbox("排序", ["得分↓", "得分↑", "KDJ最低", "RSI最低", "回撤最大"])

    # 排序
    if sort_by == "得分↓":
        candidates.sort(key=lambda x: x['score'], reverse=True)
    elif sort_by == "得分↑":
        candidates.sort(key=lambda x: x['score'])
    elif sort_by == "KDJ最低":
        candidates.sort(key=lambda x: x.get('kdj_k', 99))
    elif sort_by == "RSI最低":
        candidates.sort(key=lambda x: x.get('rsi14', 99))
    elif sort_by == "回撤最大":
        candidates.sort(key=lambda x: x.get('dd_60', 0))

    filtered = [c for c in candidates if c['score'] >= score_min]

    with col_f3:
        st.caption(f"显示 {len(filtered)}/{len(candidates)} 只候选")

    st.divider()

    # --- 图形画廊：每行3个 ---
    cols_per_row = 3
    total_filtered = len(filtered)

    # 分页
    per_page = 30
    total_pages = max(1, (total_filtered - 1) // per_page + 1)
    page = st.number_input(f"第", 1, total_pages, 1, label_visibility="collapsed")
    st.caption(f"第 {page}/{total_pages} 页，共 {total_filtered} 只")

    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, total_filtered)
    page_candidates = filtered[start_idx:end_idx]

    # 渲染卡片
    for row_start in range(0, len(page_candidates), cols_per_row):
        row_candidates = page_candidates[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for i, c in enumerate(row_candidates):
            with cols[i]:
                code = c['code']
                score = c['score']
                score_color = get_score_color(score)

                # 加载K线数据
                df = get_daily_data(code, 120)
                if df.empty:
                    st.warning(f"{code} 无数据")
                    continue

                highlight_date = scan_info.get('latest_trade_date', df['trade_date'].iloc[-1]) if scan_info else df['trade_date'].iloc[-1]

                # 缩略图
                fig_thumb = build_thumbnail_chart(df, code, highlight_date)
                st.plotly_chart(fig_thumb, use_container_width=True, config={'displayModeBar': False})

                # 指标行
                macd_sign = "🔴零下" if (c.get('macd_dif', 0) < 0 and c.get('macd_hist', 0) < 0) else ("🟡零下" if c.get('macd_dif', 0) < 0 else "🟢")

                st.markdown(f"""
                <div style='border-left:3px solid {score_color}; padding-left:8px; margin-bottom:4px'>
                <b style='font-size:1.1em'>{code}</b>
                <span style='color:{score_color};font-weight:bold;float:right'>⭐{score}</span>
                </div>
                """, unsafe_allow_html=True)

                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.caption(f"K:{c.get('kdj_k', 0):.0f} RSI:{c.get('rsi14', 0):.0f}")
                with col_m2:
                    st.caption(f"MA60:{c.get('dev_ma60', 0):+.0f}%")
                with col_m3:
                    st.caption(f"量比:{c.get('volume_ratio', 0):.2f}")

                col_m4, col_m5, col_m6 = st.columns(3)
                with col_m4:
                    st.caption(f"连跌{c.get('down_days', 0)}天")
                with col_m5:
                    st.caption(f"回撤{c.get('dd_60', 0):+.0f}%")
                with col_m6:
                    st.caption(f"{macd_sign}")

                # 展开按钮
                expand_key = f"expand_{code}"
                if st.button("🔍 查看大图", key=expand_key, use_container_width=True):
                    st.session_state.expanded_code = code if st.session_state.expanded_code != code else None
                    st.rerun()

                # 展开大图
                if st.session_state.expanded_code == code:
                    with st.container():
                        st.markdown(f"### {code} — 起涨点分析")
                        fig_full = build_full_chart(df, code, highlight_date)
                        st.plotly_chart(fig_full, use_container_width=True)

                        # 评分明细
                        bd = c.get('score_breakdown', {})
                        if isinstance(bd, str):
                            try: bd = json.loads(bd)
                            except: bd = {}
                        if bd:
                            st.caption(f"评分明细: {bd}")

                        if st.button("收起", key=f"collapse_{code}"):
                            st.session_state.expanded_code = None
                            st.rerun()

                st.divider()

else:
    # 没有扫描结果时
    st.info("""
    ### 👆 点击「今日扫描」开始

    系统将自动：
    1. 分析最近大涨股的起涨点特征
    2. 全市场扫描符合起涨点条件的股票
    3. 以图形画廊展示每只候选股的K线图

    或者从历史扫描中选择已有结果查看。
    """)

    # 显示大涨股统计
    winner_stats = get_winner_stats()
    if winner_stats:
        st.divider()
        st.subheader("📊 大涨股起涨点特征（参考）")
        cols = st.columns(6)
        with cols[0]: st.metric("样本", f"{winner_stats['count']}只")
        with cols[1]: st.metric("KDJ<30", f"{winner_stats['kdj_below_30_pct']:.0f}%")
        with cols[2]: st.metric("RSI<35", f"{winner_stats['rsi_below_35_pct']:.0f}%")
        with cols[3]: st.metric("破MA60", f"{winner_stats['dev_ma60_below_0_pct']:.0f}%")
        with cols[4]: st.metric("缩量", f"{winner_stats['vol_below_1_pct']:.0f}%")
        with cols[5]: st.metric("连跌≥3天", f"{winner_stats['down_days_ge3_pct']:.0f}%")

st.divider()
st.caption(f"© A股起涨点扫描系统 V4 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
