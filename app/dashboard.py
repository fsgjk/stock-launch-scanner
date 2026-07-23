"""
A股股票池分析系统 - Streamlit Web Dashboard V3
新增：起涨点扫描、起涨点图形、大涨股模板
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
from modules.chart_builder import ChartBuilder

# ========== 页面配置 ==========
st.set_page_config(
    page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide",
    initial_sidebar_state="expanded",
)

init_database()
scanner = LaunchPointScanner()
chart_builder = ChartBuilder()

# ========== Session State ==========
def init_session():
    defaults = {
        'scan_result': None,
        'selected_scan_id': None,
        'selected_candidate_code': None,
        'show_template_overlay': False,
        'scan_running': False,
        'menu_override': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ========== 缓存函数 ==========
@st.cache_data(ttl=300)
def get_pool_df(group=None):
    df = pool_manager.get_pool()
    if group and group != "全部":
        df = df[df["group_name"] == group]
    return df

@st.cache_data(ttl=60)
def get_realtime_data(codes):
    with get_db() as conn:
        if not codes:
            return pd.DataFrame()
        placeholders = ",".join(["?"] * len(codes))
        rows = conn.execute(
            f"SELECT code, name, price, pct_change, volume, amount, turnover_rate, amplitude FROM realtime_quote WHERE code IN ({placeholders}) ORDER BY pct_change DESC",
            codes
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

@st.cache_data(ttl=300)
def get_daily_data(code, days=250):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM stock_daily WHERE code=? ORDER BY trade_date DESC LIMIT ?",
            (code, days)
        ).fetchall()
    if rows:
        df = pd.DataFrame([dict(r) for r in rows]).sort_values("trade_date")
        return df
    return pd.DataFrame()

@st.cache_data(ttl=300)
def get_collection_summary():
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) as c FROM collection_progress").fetchone()
        done = conn.execute("SELECT COUNT(*) as c FROM collection_progress WHERE status='done'").fetchone()
        records = conn.execute("SELECT COUNT(*) as c FROM stock_daily").fetchone()
        date_range = conn.execute("SELECT MIN(trade_date) as min_d, MAX(trade_date) as max_d FROM stock_daily").fetchone()
    return {
        "total_stocks": total["c"] if total else 0,
        "done": done["c"] if done else 0,
        "total_records": records["c"] if records else 0,
        "date_min": date_range["min_d"] if date_range else "N/A",
        "date_max": date_range["max_d"] if date_range else "N/A",
        "progress": f"{done['c']/total['c']*100:.1f}%" if total and total["c"] > 0 else "0%",
    }

@st.cache_data(ttl=3600)
def get_winner_stats_cached():
    with get_db() as conn:
        return scanner.get_winner_stats(conn)

@st.cache_data(ttl=300)
def get_available_scans_cached():
    with get_db() as conn:
        return scanner.get_available_scans(conn)

# ========== 辅助函数 ==========
def navigate_to_chart(code):
    st.session_state.selected_candidate_code = code
    st.session_state.menu_override = "📊 起涨点图形"

def get_score_color(score):
    if score >= 24: return '#D32F2F'  # 深红
    elif score >= 20: return '#FF5722'  # 橙红
    elif score >= 16: return '#FF9800'  # 橙
    else: return '#FFC107'  # 黄

def get_score_bg(score):
    if score >= 24: return 'background-color: #FFCDD2'
    elif score >= 20: return 'background-color: #FFE0B2'
    elif score >= 16: return 'background-color: #FFF9C4'
    else: return ''

# ========== 侧边栏 ==========
with st.sidebar:
    st.title("📈 A股股票池分析 V3")

    # 处理菜单跳转
    menu_options = ["🚀 起涨点扫描", "📊 起涨点图形", "🏆 大涨股模板",
                    "📋 股票池管理", "📊 实时行情", "🔬 技术分析",
                    "📈 K线图表", "📦 数据管理", "🔍 股票搜索"]
    if st.session_state.menu_override:
        default_idx = menu_options.index(st.session_state.menu_override) if st.session_state.menu_override in menu_options else 0
        st.session_state.menu_override = None
    else:
        default_idx = 0

    menu = st.radio("导航菜单", menu_options, label_visibility="collapsed", index=default_idx)

    st.divider()
    groups = pool_manager.get_groups()
    selected_group = st.selectbox("分组筛选", ["全部"] + groups, key="sidebar_group")

    cs = get_collection_summary()
    st.divider()
    st.caption(f"📦 数据库: {cs['total_records']:,} 条记录")
    st.caption(f"📊 采集进度: {cs['progress']}")
    st.caption(f"📅 日期: {cs['date_min']} ~ {cs['date_max']}")
    st.caption(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ================================================================
# 页面1: 起涨点扫描
# ================================================================
if menu == "🚀 起涨点扫描":
    st.title("🚀 起涨点扫描")
    st.markdown("基于大涨股起涨点特征，全市场扫描潜在起涨点候选")

    # --- 控制面板 ---
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1.5, 2])
    with col_ctrl1:
        top_n = st.slider("候选数量上限", 50, 500, 200, 50)
    with col_ctrl2:
        scan_btn = st.button("🔍 运行扫描", type="primary", use_container_width=True,
                             disabled=st.session_state.scan_running)

    # --- 历史扫描选择 ---
    scans = get_available_scans_cached()
    with col_ctrl3:
        if scans:
            scan_options = {f"{s['scan_date']} {s['scan_time']} ({s['total_candidates']}只)": s['id'] for s in scans}
            selected_label = st.selectbox("📜 历史扫描", ["最新"] + list(scan_options.keys()))
            if selected_label != "最新":
                st.session_state.selected_scan_id = scan_options[selected_label]
            else:
                st.session_state.selected_scan_id = None

    # --- 运行扫描 ---
    if scan_btn and not st.session_state.scan_running:
        st.session_state.scan_running = True
        st.session_state.scan_result = None

    if st.session_state.scan_running:
        progress_bar = st.progress(0, "准备扫描...")
        status_area = st.empty()

        def progress_cb(stage, pct, msg):
            progress_bar.progress(pct, msg)
            if stage == 'done':
                st.session_state.scan_running = False

        with st.spinner("正在执行起涨点扫描..."):
            result = scanner.run_full_scan(progress_callback=progress_cb, top_n=top_n)
            st.session_state.scan_result = result
            get_available_scans_cached.clear()

        progress_bar.progress(100, "扫描完成!")
        st.success(f"✅ 扫描完成! 发现 {result['candidates_count']} 只起涨点候选")

    # --- 显示结果 ---
    result = st.session_state.scan_result
    if not result and st.session_state.selected_scan_id:
        with get_db() as conn:
            scan_info, candidates_df = scanner.load_scan_results(conn, scan_id=st.session_state.selected_scan_id)
        if scan_info and not candidates_df.empty:
            result = {
                'scan_id': scan_info['id'],
                'scan_date': scan_info['scan_date'],
                'candidates': candidates_df.to_dict('records'),
                'candidates_count': len(candidates_df),
                'filter_stats': json.loads(scan_info.get('scan_params', '{}')).get('hard_filters', {}),
                'latest_trade_date': scan_info['latest_trade_date'],
            }
            result['filter_stats']['total'] = scan_info.get('total_scanned', 0)
            result['filter_stats']['passed'] = scan_info.get('hard_filter_passed', 0)

    if result and result.get('candidates'):
        candidates = result['candidates']

        # --- 概览指标 ---
        st.divider()
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: st.metric("📊 总扫描", f"{result.get('filter_stats', {}).get('total', '?')}只")
        with col2: st.metric("🎯 候选数", f"{len(candidates)}只")
        with col3: st.metric("🏆 大涨样本", f"{result.get('winner_count', '?')}只")
        with col4: st.metric("🔍 硬过滤通过", f"{result.get('filter_stats', {}).get('passed', '?')}只")
        with col5: st.metric("📅 数据日期", result.get('latest_trade_date', '?'))

        # --- 得分分布 ---
        scores = [c['score'] for c in candidates]
        col_dist1, col_dist2 = st.columns([1, 3])
        with col_dist1:
            fig_score = go.Figure(go.Histogram(x=scores, nbinsx=15, marker_color='#FF5722'))
            fig_score.update_layout(title='得分分布', height=200, margin=dict(l=10, r=10, t=30, b=10),
                                     template='plotly_white', xaxis_title='评分', yaxis_title='数量')
            st.plotly_chart(fig_score, use_container_width=True)

        # --- 候选表格 ---
        with col_dist2:
            st.subheader(f"候选列表 ({len(candidates)}只)")

            score_filter = st.slider("最低得分", min(scores), max(scores),
                                     max(scores) - 6 if max(scores) > 6 else min(scores))

            filtered = [c for c in candidates if c['score'] >= score_filter]

            # 构建表格数据
            table_data = []
            for c in filtered:
                table_data.append({
                    '代码': c['code'],
                    '收盘价': f"{c['close']:.2f}",
                    '涨跌幅': f"{c['pct_change']:+.2f}%",
                    '得分': c['score'],
                    'KDJ_K': f"{c['kdj_k']:.0f}",
                    'RSI14': f"{c['rsi14']:.0f}",
                    'MA60偏离': f"{c['dev_ma60']:+.0f}%",
                    '连跌': f"{c['down_days']}天",
                    '回撤': f"{c['dd_60']:+.0f}%",
                    '量比': f"{c['volume_ratio']:.2f}",
                    '操作': '🔍',
                })

            df_display = pd.DataFrame(table_data)

            # 得分列颜色
            def color_score(val):
                v = int(val)
                return f'color: {get_score_color(v)}; font-weight: bold; font-size: 1.1em'

            styled = df_display.style.applymap(color_score, subset=['得分'])

            st.dataframe(styled, use_container_width=True, hide_index=True,
                         column_config={'操作': st.column_config.Column(width='small')})

        # --- 点击查看图表 ---
        st.divider()
        st.markdown("### 点击查看起涨点图形")
        cols = st.columns(8)
        for i, c in enumerate(filtered[:32]):
            with cols[i % 8]:
                score_color = get_score_color(c['score'])
                btn_label = f"{c['code']}\n⭐{c['score']}"
                if st.button(btn_label, key=f"view_{c['code']}", use_container_width=True):
                    navigate_to_chart(c['code'])
                    st.rerun()

    elif not st.session_state.scan_running:
        st.info("👆 点击「运行扫描」按钮开始全市场起涨点扫描，或从历史扫描中选择查看")


# ================================================================
# 页面2: 起涨点图形
# ================================================================
elif menu == "📊 起涨点图形":
    st.title("📊 起涨点图形")

    # 获取候选列表
    candidate_codes = []
    candidate_map = {}
    if st.session_state.scan_result and st.session_state.scan_result.get('candidates'):
        for c in st.session_state.scan_result['candidates']:
            candidate_codes.append(c['code'])
            candidate_map[c['code']] = c
    elif st.session_state.selected_scan_id:
        with get_db() as conn:
            _, cdf = scanner.load_scan_results(conn, scan_id=st.session_state.selected_scan_id)
        if not cdf.empty:
            for _, r in cdf.iterrows():
                candidate_codes.append(r['code'])
                candidate_map[r['code']] = dict(r)

    if not candidate_codes:
        st.warning("请先在「起涨点扫描」页面运行扫描或选择历史扫描")
    else:
        col_sel1, col_sel2, col_sel3 = st.columns([2, 1, 1])
        with col_sel1:
            default_code = st.session_state.selected_candidate_code if st.session_state.selected_candidate_code in candidate_codes else candidate_codes[0]
            selected_code = st.selectbox("选择候选股票", candidate_codes,
                                         index=candidate_codes.index(default_code))
        with col_sel2:
            days = st.slider("显示天数", 30, 250, 120)
        with col_sel3:
            show_template = st.checkbox("对比大涨股模板")

        if selected_code:
            st.session_state.selected_candidate_code = selected_code
            info = candidate_map.get(selected_code, {})

            # --- 评分指标面板 ---
            if info:
                score = info.get('score', 0)
                bd = info.get('score_breakdown', {})
                if isinstance(bd, str):
                    try: bd = json.loads(bd)
                    except: bd = {}

                st.markdown(f"### ⭐ 综合评分: **{score}分**")
                cols_m = st.columns(10)
                metric_labels = [
                    ('KDJ', bd.get('kdj', 0), 5, f"K={info.get('kdj_k', 0):.0f}"),
                    ('RSI', bd.get('rsi', 0), 4, f"{info.get('rsi14', 0):.0f}"),
                    ('MA60', bd.get('ma60_dev', 0), 5, f"{info.get('dev_ma60', 0):+.0f}%"),
                    ('连跌', bd.get('down_days', 0), 4, f"{info.get('down_days', 0)}天"),
                    ('回撤', bd.get('dd_60', 0), 3, f"{info.get('dd_60', 0):+.0f}%"),
                    ('分位', bd.get('price_pct', 0), 3, f"{info.get('price_pct_20d', 0):.0f}%"),
                    ('MACD', bd.get('macd', 0), 2, ''),
                    ('量比', bd.get('volume', 0), 2, f"{info.get('volume_ratio', 0):.2f}"),
                    ('布林', bd.get('boll', 0), 1, ''),
                    ('低价', bd.get('near_low', 0), 1, ''),
                ]
                for i, (name, val, max_val, extra) in enumerate(metric_labels):
                    with cols_m[i]:
                        color = '#4CAF50' if val >= max_val * 0.7 else ('#FF9800' if val >= max_val * 0.4 else '#9E9E9E')
                        st.markdown(
                            f"<div style='text-align:center;padding:4px;border-radius:4px;background:{color}20'>"
                            f"<small>{name}</small><br><b style='color:{color}'>{val}/{max_val}</b>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

            # --- K线图表 ---
            df = get_daily_data(selected_code, days)
            if not df.empty:
                highlight = info.get('latest_trade_date') if 'latest_trade_date' in info else df['trade_date'].iloc[-1]

                st.divider()
                st.markdown("### 📈 K线分析图")

                fig = chart_builder.build_full_analysis_chart(df, selected_code, highlight_date=highlight)
                st.plotly_chart(fig, use_container_width=True)

                # --- 模板对比 ---
                if show_template:
                    st.divider()
                    st.markdown("### 📊 与大涨股模板对比")

                    with get_db() as conn:
                        df_w = scanner.get_winner_templates(conn)

                    if not df_w.empty:
                        col_t1, col_t2 = st.columns(2)
                        with col_t1:
                            fig_radar = chart_builder.build_score_radar(bd)
                            st.plotly_chart(fig_radar, use_container_width=True)
                        with col_t2:
                            winner_stats = get_winner_stats_cached()
                            if winner_stats:
                                st.markdown("#### 大涨股起涨点平均特征")
                                st.markdown(f"""
                                - KDJ<30: **{winner_stats.get('kdj_below_30_pct', 0):.0f}%** (当前: {info.get('kdj_k', 0):.0f})
                                - RSI<35: **{winner_stats.get('rsi_below_35_pct', 0):.0f}%** (当前: {info.get('rsi14', 0):.0f})
                                - 破MA60: **{winner_stats.get('dev_ma60_below_0_pct', 0):.0f}%** (当前偏离: {info.get('dev_ma60', 0):+.0f}%)
                                - 连跌≥3天: **{winner_stats.get('down_days_ge3_pct', 0):.0f}%** (当前: {info.get('down_days', 0)}天)
                                - 平均回撤: **{winner_stats.get('dd_60_mean', 0):.0f}%** (当前: {info.get('dd_60', 0):+.0f}%)
                                - 缩量: **{winner_stats.get('vol_below_1_pct', 0):.0f}%** (当前量比: {info.get('volume_ratio', 0):.2f})
                                """)

                        # 特征分布对比
                        st.markdown("#### 当前股票 vs 大涨股特征分布")
                        col_h1, col_h2, col_h3 = st.columns(3)
                        with col_h1:
                            fig_kdj = chart_builder.build_distribution_histogram(
                                df_w['kdj_k'].dropna(), 'KDJ_K分布', 'KDJ_K',
                                highlight_value=info.get('kdj_k'), color='#2196F3')
                            st.plotly_chart(fig_kdj, use_container_width=True)
                        with col_h2:
                            fig_rsi = chart_builder.build_distribution_histogram(
                                df_w['rsi14'].dropna(), 'RSI14分布', 'RSI14',
                                highlight_value=info.get('rsi14'), color='#4CAF50')
                            st.plotly_chart(fig_rsi, use_container_width=True)
                        with col_h3:
                            fig_ma = chart_builder.build_distribution_histogram(
                                df_w['dev_ma60'].dropna(), 'MA60偏离分布', '偏离%',
                                highlight_value=info.get('dev_ma60'), color='#FF9800')
                            st.plotly_chart(fig_ma, use_container_width=True)
            else:
                st.warning(f"{selected_code} 无数据")


# ================================================================
# 页面3: 大涨股模板
# ================================================================
elif menu == "🏆 大涨股模板":
    st.title("🏆 大涨股模板")
    st.markdown("基于大涨股（20日涨幅>20%）的起涨点特征统计与可视化")

    winner_stats = get_winner_stats_cached()

    if not winner_stats:
        st.warning("暂无大涨股模板数据，请先运行起涨点扫描")
    else:
        # --- 统计摘要 ---
        st.markdown("### 📊 大涨股起涨点特征（样本统计）")
        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
        with col_s1:
            st.metric("📦 样本数", f"{winner_stats['count']}只")
        with col_s2:
            st.metric("📉 KDJ<30", f"{winner_stats['kdj_below_30_pct']:.0f}%",
                      f"均值{winner_stats['kdj_mean']:.1f}")
        with col_s3:
            st.metric("📉 RSI<35", f"{winner_stats['rsi_below_35_pct']:.0f}%",
                      f"均值{winner_stats['rsi_mean']:.1f}")
        with col_s4:
            st.metric("📉 破MA60", f"{winner_stats['dev_ma60_below_0_pct']:.0f}%",
                      f"均值{winner_stats['dev_ma60_mean']:.1f}%")
        with col_s5:
            st.metric("📉 缩量", f"{winner_stats['vol_below_1_pct']:.0f}%",
                      f"均值涨幅{winner_stats['pct_20d_mean']:.1f}%")

        st.divider()

        # --- 关键发现 ---
        st.markdown("""
        ### 💡 关键发现
        > **起涨点当天94%的股票在下跌！** 大涨股的起点几乎都是「跌出来的机会」。
        > 典型特征：KDJ超卖 + 跌破MA60 + 大幅回撤 + 缩量连跌。
        """)

        # --- 特征分布图 ---
        with get_db() as conn:
            df_w = scanner.get_winner_templates(conn)

        if not df_w.empty:
            st.markdown("### 📈 特征分布直方图")
            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                fig1 = chart_builder.build_distribution_histogram(
                    df_w['kdj_k'].dropna(), 'KDJ_K 分布', 'KDJ_K', color='#2196F3')
                st.plotly_chart(fig1, use_container_width=True)
            with col_d2:
                fig2 = chart_builder.build_distribution_histogram(
                    df_w['rsi14'].dropna(), 'RSI14 分布', 'RSI14', color='#4CAF50')
                st.plotly_chart(fig2, use_container_width=True)
            with col_d3:
                fig3 = chart_builder.build_distribution_histogram(
                    df_w['dev_ma60'].dropna(), 'MA60偏离 分布', '偏离%', color='#FF9800')
                st.plotly_chart(fig3, use_container_width=True)

            col_d4, col_d5, col_d6 = st.columns(3)
            with col_d4:
                fig4 = chart_builder.build_distribution_histogram(
                    df_w['down_days'].dropna(), '连跌天数 分布', '天数', color='#9C27B0')
                st.plotly_chart(fig4, use_container_width=True)
            with col_d5:
                fig5 = chart_builder.build_distribution_histogram(
                    df_w['dd_60'].dropna(), '60日回撤 分布', '回撤%', color='#F44336')
                st.plotly_chart(fig5, use_container_width=True)
            with col_d6:
                fig6 = chart_builder.build_distribution_histogram(
                    df_w['vol_ratio'].dropna(), '量比 分布', '量比', color='#00BCD4')
                st.plotly_chart(fig6, use_container_width=True)

            # --- 大涨股列表 ---
            st.divider()
            st.markdown("### 🖼️ 大涨股列表")

            page_size = 12
            page = st.number_input("页码", 1, max(1, (len(df_w) - 1) // page_size + 1), 1)
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, len(df_w))

            page_df = df_w.iloc[start_idx:end_idx]
            cols = st.columns(4)
            for i, (_, r) in enumerate(page_df.iterrows()):
                with cols[i % 4]:
                    st.markdown(f"**{r['code']}** | 涨幅: {r['pct_20d']:.1f}%")
                    st.markdown(f"K:{r['kdj_k']:.0f} RSI:{r['rsi14']:.0f} MA60:{r['dev_ma60']:+.0f}%")
                    st.markdown(f"连跌{r['down_days']:.0f}天 | 回撤{r['dd_60']:+.0f}% | 量比{r['vol_ratio']:.2f}")
                    st.divider()


# ================================================================
# 页面4: 股票池管理（保留并增强）
# ================================================================
elif menu == "📋 股票池管理":
    st.title("📋 股票池管理")

    tab1, tab2, tab3 = st.tabs(["我的股票池", "添加股票", "批量导入"])

    with tab1:
        df = get_pool_df(selected_group)
        if not df.empty:
            codes = df["code"].tolist()
            rt = get_realtime_data(codes)
            rt_map = dict(zip(rt["code"], rt["pct_change"])) if not rt.empty else {}
            price_map = dict(zip(rt["code"], rt["price"])) if not rt.empty else {}

            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("股票总数", len(df))
            with col2: st.metric("分组数", df["group_name"].nunique())
            with col3: st.metric("📈 上涨", f"{sum(1 for v in rt_map.values() if v and v > 0)}只")
            with col4: st.metric("📉 下跌", f"{sum(1 for v in rt_map.values() if v and v < 0)}只")

            # 关联起涨点评分
            scan_scores = {}
            if st.session_state.scan_result and st.session_state.scan_result.get('candidates'):
                for c in st.session_state.scan_result['candidates']:
                    scan_scores[c['code']] = c['score']

            display_data = []
            for _, row in df.iterrows():
                code = row["code"]
                entry = {
                    "代码": code, "名称": row["name"], "分组": row["group_name"],
                    "最新价": f"{price_map.get(code, 0):.2f}" if code in price_map else "-",
                    "涨跌幅(%)": f"{rt_map.get(code, 0):.2f}" if code in rt_map else "-",
                    "起涨点评分": scan_scores.get(code, '-'),
                    "加入日期": row["added_date"], "备注": row["notes"] or "",
                }
                display_data.append(entry)

            st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)

            with st.expander("移除股票"):
                rm_code = st.text_input("股票代码", key="rm_code")
                if st.button("确认移除") and rm_code:
                    pool_manager.remove_stock(rm_code)
                    st.success(f"已移除 {rm_code}")
                    st.rerun()
        else:
            st.info("股票池为空，请先添加股票")

    with tab2:
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1: new_code = st.text_input("股票代码", placeholder="如: 600519")
        with col_a2: new_name = st.text_input("股票名称", placeholder="如: 贵州茅台")
        with col_a3: new_group = st.selectbox("分组", groups, key="new_group")
        new_notes = st.text_input("备注", placeholder="可选")
        if st.button("➕ 添加到股票池", type="primary") and new_code:
            if pool_manager.add_stock(new_code, new_name, new_group, new_notes):
                st.success(f"成功添加 {new_code} {new_name}")
                st.rerun()

    with tab3:
        batch_text = st.text_area("输入股票列表", placeholder="600519,贵州茅台\n000858,五粮液", height=200)
        batch_group = st.selectbox("导入到分组", groups, key="batch_group")
        if st.button("📥 批量导入") and batch_text.strip():
            stocks = []
            for line in batch_text.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2: stocks.append({"code": parts[0], "name": parts[1]})
                elif parts[0]: stocks.append({"code": parts[0], "name": ""})
            count = pool_manager.import_from_list(stocks, batch_group)
            st.success(f"成功导入 {count}/{len(stocks)} 只股票")
            st.rerun()


# ================================================================
# 页面5: 实时行情（保留并增强）
# ================================================================
elif menu == "📊 实时行情":
    st.title("📊 实时行情")
    codes = pool_manager.get_codes()
    if codes:
        rt = get_realtime_data(codes)
        if not rt.empty:
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1: st.metric("总数", len(rt))
            with col2: st.metric("📈 上涨", f"{len(rt[rt['pct_change']>0])}只")
            with col3: st.metric("📉 下跌", f"{len(rt[rt['pct_change']<0])}只")
            with col4: st.metric("平均涨跌", f"{rt['pct_change'].mean():.2f}%")
            with col5: st.metric("最大涨幅", f"{rt['pct_change'].max():.2f}%")

            colors = ["#ef5350" if x < 0 else "#26a69a" for x in rt["pct_change"]]
            fig = go.Figure(go.Bar(
                x=rt["code"] + " " + rt["name"], y=rt["pct_change"],
                marker_color=colors, text=rt["pct_change"].round(2), textposition="outside",
            ))
            fig.update_layout(title="涨跌幅分布", height=400, showlegend=False, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

            # 标注扫描候选
            scan_codes = set()
            if st.session_state.scan_result and st.session_state.scan_result.get('candidates'):
                scan_codes = {c['code'] for c in st.session_state.scan_result['candidates']}

            rt_display = rt.copy()
            rt_display['起涨点候选'] = rt_display['code'].apply(lambda x: '⭐' if x in scan_codes else '')

            st.dataframe(rt_display.rename(columns={
                "code": "代码", "name": "名称", "price": "最新价",
                "pct_change": "涨跌幅(%)", "volume": "成交量", "amount": "成交额",
                "turnover_rate": "换手率(%)", "amplitude": "振幅(%)",
            }), use_container_width=True, hide_index=True)
        else:
            st.warning("暂无实时行情，请先执行数据同步")
    else:
        st.warning("股票池为空")


# ================================================================
# 页面6: 技术分析（保留）
# ================================================================
elif menu == "🔬 技术分析":
    st.title("🔬 技术分析")
    codes = pool_manager.get_codes()
    if codes:
        selected_code = st.selectbox(
            "选择股票", codes,
            format_func=lambda x: f"{x} - {dict(zip(get_pool_df()['code'], get_pool_df()['name'])).get(x, '')}"
        )
        if selected_code:
            df = get_daily_data(selected_code, 250)
            if df.empty:
                st.warning(f"{selected_code} 暂无数据")
            else:
                latest = df.iloc[-1]
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1: st.metric("最新价", f"{latest['close']:.2f}")
                with col2: st.metric("RSI(14)", f"{latest.get('rsi14', 0):.1f}")
                with col3:
                    macd_sig = "多头" if latest.get('macd_dif', 0) > latest.get('macd_dea', 0) else "空头"
                    st.metric("MACD", macd_sig)
                with col4: st.metric("KDJ-K", f"{latest.get('kdj_k', 0):.1f}")
                with col5: st.metric("量比", f"{latest.get('volume_ratio', 0):.2f}")

                st.subheader("📡 技术信号")
                sig_cols = st.columns(4)
                signals = []
                close = latest["close"]
                if close > latest.get("ma20", close): signals.append(("🟢", "MA20", "站上20日均线"))
                else: signals.append(("🔴", "MA20", "跌破20日均线"))
                if latest.get("macd_dif", 0) > latest.get("macd_dea", 0): signals.append(("🟢", "MACD", "DIF在DEA上方"))
                else: signals.append(("🔴", "MACD", "DIF在DEA下方"))
                rsi = latest.get("rsi14", 50)
                if rsi > 80: signals.append(("🔴", "RSI", f"超买({rsi:.1f})"))
                elif rsi < 20: signals.append(("🟢", "RSI", f"超卖({rsi:.1f})"))
                else: signals.append(("⚪", "RSI", f"{rsi:.1f}"))
                kdj_j = latest.get("kdj_j", 50)
                if kdj_j > 100: signals.append(("🔴", "KDJ", f"J值超买({kdj_j:.1f})"))
                elif kdj_j < 0: signals.append(("🟢", "KDJ", f"J值超卖({kdj_j:.1f})"))
                else: signals.append(("⚪", "KDJ", f"J={kdj_j:.1f}"))
                for i, (emoji, typ, desc) in enumerate(signals):
                    with sig_cols[i]: st.info(f"{emoji} **{typ}**: {desc}")

                fig = chart_builder.build_full_analysis_chart(df, selected_code)
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("📋 原始数据"):
                    st.dataframe(df.tail(20)[["trade_date", "open", "high", "low", "close",
                        "pct_change", "volume", "kdj_k", "kdj_d", "kdj_j",
                        "macd_dif", "macd_dea", "rsi14"]].rename(columns={
                        "trade_date": "日期", "open": "开盘", "high": "最高", "low": "最低",
                        "close": "收盘", "pct_change": "涨跌幅", "volume": "成交量",
                    }), use_container_width=True, hide_index=True)


# ================================================================
# 页面7: K线图表（保留）
# ================================================================
elif menu == "📈 K线图表":
    st.title("📈 K线图表")
    codes = pool_manager.get_codes()
    if codes:
        selected_codes = st.multiselect(
            "选择股票（可多选对比）", codes, default=codes[:1] if codes else [],
            format_func=lambda x: f"{x} - {dict(zip(get_pool_df()['code'], get_pool_df()['name'])).get(x, '')}"
        )
        if selected_codes:
            days = st.slider("显示天数", 30, 500, 120)
            chart_type = st.selectbox("图表类型", ["K线图", "收盘价对比", "涨跌幅对比", "成交量对比"])

            if chart_type == "K线图" and len(selected_codes) == 1:
                code = selected_codes[0]
                df = get_daily_data(code, days)
                if not df.empty:
                    fig = chart_builder.build_kline_chart(df, code, height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"{code} 无数据")

            elif chart_type == "收盘价对比":
                fig = go.Figure()
                for code in selected_codes:
                    df = get_daily_data(code, days)
                    if not df.empty:
                        base = df["close"].iloc[0]
                        df["norm"] = (df["close"] / base - 1) * 100
                        name = dict(zip(get_pool_df()['code'], get_pool_df()['name'])).get(code, code)
                        fig.add_trace(go.Scatter(x=df["trade_date"], y=df["norm"], mode="lines", name=f"{code} {name}"))
                fig.update_layout(title="收盘价走势对比（归一化%）", yaxis_title="涨跌幅(%)", height=500, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "涨跌幅对比":
                fig = go.Figure()
                for code in selected_codes:
                    df = get_daily_data(code, days)
                    if not df.empty:
                        name = dict(zip(get_pool_df()['code'], get_pool_df()['name'])).get(code, code)
                        fig.add_trace(go.Scatter(x=df["trade_date"], y=df["pct_change"], mode="lines", name=f"{code} {name}"))
                fig.add_hline(y=0, line_dash="solid", line_color="gray")
                fig.update_layout(title="每日涨跌幅对比", yaxis_title="涨跌幅(%)", height=500, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

            else:
                fig = go.Figure()
                for code in selected_codes:
                    df = get_daily_data(code, days)
                    if not df.empty:
                        name = dict(zip(get_pool_df()['code'], get_pool_df()['name'])).get(code, code)
                        fig.add_trace(go.Scatter(x=df["trade_date"], y=df["volume_ma5"], mode="lines", name=f"{code} {name}"))
                fig.update_layout(title="成交量对比（5日均量）", yaxis_title="成交量", height=500, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)


# ================================================================
# 页面8: 数据管理（保留并增强）
# ================================================================
elif menu == "📦 数据管理":
    st.title("📦 数据管理")

    cs = get_collection_summary()
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("数据库总记录", f"{cs['total_records']:,}")
    with col2: st.metric("已采集股票", f"{cs['done']}/{cs['total_stocks']}")
    with col3: st.metric("采集进度", cs['progress'])
    with col4: st.metric("日期范围", f"{cs['date_min']}~{cs['date_max']}")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 同步股票池实时行情", use_container_width=True):
            codes = pool_manager.get_codes()
            if codes:
                count = collector.sync_realtime_data(codes)
                st.success(f"实时行情同步完成: {count}条")
                st.rerun()
            else:
                st.warning("股票池为空")

    # --- 扫描历史 ---
    st.divider()
    st.subheader("📜 扫描历史")
    scans = get_available_scans_cached()
    if scans:
        scan_df = pd.DataFrame(scans).rename(columns={
            'scan_date': '扫描日期', 'scan_time': '扫描时间',
            'total_candidates': '候选数', 'total_scanned': '扫描总数',
            'winner_sample_count': '大涨样本'
        })
        st.dataframe(scan_df[['扫描日期', '扫描时间', '候选数', '扫描总数', '大涨样本']],
                     use_container_width=True, hide_index=True)
    else:
        st.info("暂无扫描历史")

    # --- 采集进度 ---
    with st.expander("📊 采集进度详情"):
        with get_db() as conn:
            done_list = conn.execute(
                "SELECT code, name, last_date, total_days FROM collection_progress WHERE status='done' ORDER BY last_date DESC LIMIT 20"
            ).fetchall()
            failed_list = conn.execute(
                "SELECT code, name, error_msg FROM collection_progress WHERE status='failed'"
            ).fetchall()
        if done_list:
            st.subheader(f"最近完成 ({len(done_list)}只)")
            st.dataframe(pd.DataFrame([dict(r) for r in done_list]).rename(columns={
                "code": "代码", "name": "名称", "last_date": "最新日期", "total_days": "天数",
            }), use_container_width=True, hide_index=True)
        if failed_list:
            st.subheader(f"失败 ({len(failed_list)}只)")
            st.dataframe(pd.DataFrame([dict(r) for r in failed_list]), use_container_width=True, hide_index=True)

    with st.expander("📈 数据分布"):
        with get_db() as conn:
            year_dist = conn.execute("""
                SELECT SUBSTR(trade_date,1,4) as year, COUNT(*) as cnt
                FROM stock_daily GROUP BY year ORDER BY year
            """).fetchall()
        if year_dist:
            years = [r["year"] for r in year_dist]
            counts = [r["cnt"] for r in year_dist]
            fig = go.Figure(go.Bar(x=years, y=counts, text=counts, textposition="outside"))
            fig.update_layout(title="每年数据量分布", height=300, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)


# ================================================================
# 页面9: 股票搜索（保留）
# ================================================================
elif menu == "🔍 股票搜索":
    st.title("🔍 股票搜索")
    search_query = st.text_input("搜索A股股票", placeholder="输入代码或名称关键词...")

    if search_query:
        with get_db() as conn:
            local = conn.execute(
                "SELECT DISTINCT code, name FROM stock_daily WHERE code LIKE ? OR name LIKE ? LIMIT 20",
                (f"%{search_query}%", f"%{search_query}%")
            ).fetchall()

        if local:
            st.subheader("📌 数据库中匹配")
            for row in local:
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1: st.write(f"**{row['code']}** - {row['name']}")
                with col2:
                    sg = st.selectbox("分组", groups, key=f"grp_{row['code']}", label_visibility="collapsed")
                with col3:
                    if st.button("➕ 添加", key=f"add_{row['code']}"):
                        pool_manager.add_stock(row["code"], row["name"], sg)
                        st.success(f"已添加 {row['code']} {row['name']}")
                        st.rerun()

        if len(search_query) >= 2:
            with st.spinner("在线搜索中..."):
                try:
                    all_stocks = collector.get_all_stocks()
                    mask = all_stocks["code"].str.contains(search_query) | all_stocks["name"].str.contains(search_query)
                    results = all_stocks[mask].head(20)
                    if not results.empty:
                        st.subheader("🔎 全网搜索结果")
                        for _, row in results.iterrows():
                            col1, col2, col3 = st.columns([1.5, 1, 1])
                            with col1: st.write(f"**{row['code']}** - {row['name']}")
                            with col2:
                                sg2 = st.selectbox("分组", groups, key=f"grp2_{row['code']}", label_visibility="collapsed")
                            with col3:
                                if st.button("➕ 添加", key=f"add2_{row['code']}"):
                                    pool_manager.add_stock(row["code"], row["name"], sg2)
                                    st.success(f"已添加 {row['code']} {row['name']}")
                                    st.rerun()
                except Exception as e:
                    st.error(f"搜索失败: {e}")

st.divider()
st.caption(f"© A股股票池分析系统 V3 | 数据来源: AKShare/新浪 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
