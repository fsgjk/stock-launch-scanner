"""
A股起涨点扫描系统 V7 - 优化UI + 点击查看K线
布局：左侧日期面板 | 右侧概览卡片 + 可点击列表 + K线展开
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

# 定时调度器
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
if 'expanded_stock' not in st.session_state:
    st.session_state.expanded_stock = None

# ==================== CSS ====================
st.markdown("""
<style>
    /* 全局 */
    .main .block-container { padding-top: 1rem; }
    
    /* 概览卡片 */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .label { font-size: 0.8rem; color: #8892b0; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; margin: 0.3rem 0; }
    .metric-card .sub { font-size: 0.75rem; color: #6c757d; }
    .metric-red { color: #ef5350; }
    .metric-green { color: #26a69a; }
    .metric-gold { color: #FF9800; }
    .metric-blue { color: #42a5f5; }
    
    /* 表格行悬停 */
    .stock-row { cursor: pointer; }
    .stock-row:hover { background: rgba(66, 165, 245, 0.08) !important; }
    
    /* 侧边栏日期项 */
    .date-item { 
        padding: 0.5rem 0.8rem; 
        border-radius: 6px; 
        margin-bottom: 4px;
        cursor: pointer;
        transition: all 0.15s;
    }
    .date-item:hover { background: rgba(66, 165, 245, 0.1); }
    .date-item.active { background: rgba(66, 165, 245, 0.2); border-left: 3px solid #42a5f5; }
    
    /* 分数徽章 */
    .score-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .score-high { background: #c62828; color: white; }
    .score-mid { background: #e65100; color: white; }
    .score-low { background: #f57f17; color: white; }
    
    /* 涨跌颜色 */
    .up { color: #ef5350; font-weight: 600; }
    .down { color: #26a69a; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ==================== 缓存函数 ====================
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
            FROM launch_scan_results ORDER BY scan_date DESC LIMIT 120
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

    all_dates = sorted(df['trade_date'].unique())
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
                cum_pct[d] = None
        result[code] = {
            'entry_price': entry_price,
            'cum_pct': cum_pct,
            'days': len([d for d in all_dates if d >= entry_date and d in sdf.index]),
        }
    return result, all_dates

@st.cache_data(ttl=3600)
def get_tracking_summary_for_dates():
    """获取所有扫描日期的跟踪汇总（用于侧边栏显示）"""
    with get_db() as conn:
        cur = conn.execute("""
            SELECT scan_date, latest_trade_date, total_candidates 
            FROM launch_scan_results ORDER BY scan_date DESC LIMIT 120
        """)
        return [dict(r) for r in cur.fetchall()]


# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("### 🎯 起涨点扫描")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("🔍 手动扫描", use_container_width=True,
                      disabled=st.session_state.scan_running):
            st.session_state.scan_running = True
            st.rerun()
    with col_b:
        if st.button("🔄 刷新", use_container_width=True):
            get_scan_dates.clear()
            get_scan_candidates.clear()
            get_tracking_pivoted.clear()
            st.rerun()

    try:
        sch = stock_scheduler.get_status()
        for j in sch.get('jobs', []):
            if j['id'] == 'daily_scan':
                st.caption(f"⏰ 下次自动扫描: {j['next_run']}")
    except:
        pass

    st.divider()
    st.caption("📅 扫描日期（点击查看）")

    scan_dates = get_scan_dates()
    if not scan_dates:
        st.info("暂无扫描记录，请点击「手动扫描」")
    else:
        for s in scan_dates:
            sid = s['id']
            is_sel = st.session_state.selected_scan_id == sid
            label = f"📅 {s['scan_date']}"
            cap = f"{s['total_candidates']}只候选"

            if st.button(f"{'🔵 ' if is_sel else ''}{label}  — {cap}",
                         key=f"d_{sid}", use_container_width=True,
                         type="primary" if is_sel else "secondary"):
                st.session_state.selected_scan_id = sid
                st.session_state.expanded_stock = None
                st.rerun()

    st.divider()
    st.caption(f"数据源: AKShare/新浪 | V7")

# ==================== 扫描执行 ====================
if st.session_state.scan_running:
    progress_bar = st.progress(0, "扫描中...")
    def progress_cb(stage, pct, msg):
        progress_bar.progress(pct, msg)
        if stage == 'done':
            st.session_state.scan_running = False
    with st.spinner("正在全市场扫描..."):
        result = scanner.run_full_scan(progress_callback=progress_cb, top_n=200)
        get_scan_dates.clear()
        get_scan_candidates.clear()
        st.session_state.selected_scan_id = result['scan_id']
    progress_bar.progress(100, "完成!")
    st.success(f"✅ 扫描完成！{result['candidates_count']}只候选股票")
    st.rerun()

# ==================== 主区域 ====================
selected_id = st.session_state.selected_scan_id
if not selected_id and scan_dates:
    selected_id = scan_dates[0]['id']
    st.session_state.selected_scan_id = selected_id

if not selected_id:
    st.info("👈 从左侧选择扫描日期，或点击「手动扫描」开始")
    st.stop()

# 加载选中日期的数据
df_c = get_scan_candidates(selected_id)
scan_info = next((s for s in scan_dates if s['id'] == selected_id), {})

if df_c.empty:
    st.warning("该日期无候选数据")
    st.stop()

scan_date = scan_info.get('latest_trade_date', '')
candidates = df_c.to_dict('records')
codes = [c['code'] for c in candidates]
names = get_stock_names(codes)

# 获取跟踪数据
tracking, all_track_dates = get_tracking_pivoted(codes, scan_date)
track_dates = [d for d in all_track_dates if d > scan_date]

# ==================== 顶部概览 ====================
st.markdown(f"## 📅 {scan_info.get('scan_date', '')} 起涨点候选池")

# 概览指标
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("候选股票", f"{len(candidates)}只", delta=None)

with col2:
    avg_score = np.mean([c['score'] for c in candidates])
    st.metric("平均得分", f"{avg_score:.1f}/30")

with col3:
    if track_dates and tracking:
        # 最新跟踪日期的平均累计涨跌幅
        last_date = track_dates[-1]
        valid_vals = []
        for c in candidates:
            code = c['code']
            if code in tracking and last_date in tracking[code]['cum_pct']:
                v = tracking[code]['cum_pct'][last_date]
                if v is not None:
                    valid_vals.append(v)
        if valid_vals:
            avg_ret = np.mean(valid_vals)
            st.metric("平均累计收益", f"{avg_ret:+.2f}%",
                      delta=f"{last_date}",
                      delta_color="normal" if avg_ret > 0 else "inverse")
        else:
            st.metric("平均累计收益", "—")
    else:
        st.metric("平均累计收益", "—")

with col4:
    if track_dates and tracking:
        last_date = track_dates[-1]
        up_count = 0
        total_valid = 0
        for c in candidates:
            code = c['code']
            if code in tracking and last_date in tracking[code]['cum_pct']:
                v = tracking[code]['cum_pct'][last_date]
                if v is not None:
                    total_valid += 1
                    if v > 0:
                        up_count += 1
        if total_valid > 0:
            win_rate = up_count / total_valid * 100
            st.metric("累计胜率", f"{win_rate:.1f}%",
                      delta=f"{up_count}/{total_valid}")
        else:
            st.metric("累计胜率", "—")
    else:
        st.metric("累计胜率", "��")

with col5:
    if track_dates:
        st.metric("跟踪天数", f"{len(track_dates)}天")
    else:
        st.metric("跟踪天数", "0天")

st.divider()

# ==================== 主表格 + K线联动 ====================
st.subheader("📋 股票列表（点击行查看K线图）")

# 构建表格数据
table_data = []
for c in candidates:
    code = c['code']
    name = names.get(code, '')
    tk = tracking.get(code, {})

    # 最新累计涨跌幅
    latest_cum = None
    if track_dates and tk:
        for td in reversed(track_dates):
            v = tk['cum_pct'].get(td)
            if v is not None:
                latest_cum = v
                break

    row = {
        '代码': code,
        '名称': name,
        '收盘': round(c['close'], 2),
        '涨跌': round(c['pct_change'], 2) if c.get('pct_change') is not None else None,
        '得分': int(c['score']),
        '持仓天数': tk.get('days', 0) if tk else 0,
        '累计涨跌': round(latest_cum, 2) if latest_cum is not None else None,
        '连跌': int(c['down_days']) if c.get('down_days') is not None else 0,
        '60日回撤': round(c['dd_60'], 1) if c.get('dd_60') is not None else None,
        'MA60偏离': round(c['dev_ma60'], 1) if c.get('dev_ma60') is not None else None,
        '量比': round(c['volume_ratio'], 2) if c.get('volume_ratio') is not None else None,
    }

    # 保存原始数据
    row['_code'] = code
    row['_score_breakdown'] = c.get('score_breakdown', '{}')

    table_data.append(row)

df_table = pd.DataFrame(table_data)

# --- 构建 display DataFrame（去掉内部列） ---
display_cols = ['代码', '名称', '收盘', '涨跌', '得分', '持仓天数',
                '累计涨跌', '连跌', '60日回撤', 'MA60偏离', '量比']

df_display = df_table[display_cols].copy()

# --- 颜色函数 ---
def style_score(val):
    try:
        v = int(val)
        if v >= 24: return 'background: #c62828; color: white; font-weight: bold'
        elif v >= 20: return 'background: #e65100; color: white; font-weight: bold'
        elif v >= 16: return 'background: #f57f17; color: white; font-weight: bold'
        return 'background: #f9a825; color: white'
    except: return ''

def style_pct(val, is_cum=False):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ''
    try:
        v = float(val)
        if v > 0:
            return 'color: #ef5350; font-weight: bold'
        elif v < 0:
            return 'color: #26a69a; font-weight: bold'
    except: pass
    return ''

def style_kdj(val):
    if val is None or (isinstance(val, float) and pd.isna(val)): return ''
    try:
        v = float(val)
        if v < 10: return 'color: #c62828; font-weight: bold'
        elif v < 20: return 'color: #e65100; font-weight: bold'
        elif v < 30: return 'color: #f57f17'
    except: pass
    return ''

def style_days(val):
    try:
        v = int(val)
        if v >= 6: return 'color: #c62828; font-weight: bold'
        elif v >= 4: return 'color: #e65100; font-weight: bold'
        elif v >= 2: return 'color: #f57f17'
    except: pass
    return ''

def style_dd(val):
    if val is None or (isinstance(val, float) and pd.isna(val)): return ''
    try:
        v = float(val)
        if v <= -50: return 'color: #c62828; font-weight: bold'
        elif v <= -35: return 'color: #e65100; font-weight: bold'
        elif v <= -25: return 'color: #f57f17'
    except: pass
    return ''

def style_dev(val):
    if val is None or (isinstance(val, float) and pd.isna(val)): return ''
    try:
        v = float(val)
        if v <= -30: return 'color: #c62828; font-weight: bold'
        elif v <= -20: return 'color: #e65100; font-weight: bold'
        elif v <= -10: return 'color: #f57f17'
    except: pass
    return ''

def style_vol(val):
    if val is None or (isinstance(val, float) and pd.isna(val)): return ''
    try:
        v = float(val)
        if v < 0.5: return 'color: #c62828; font-weight: bold'
        elif v < 0.7: return 'color: #e65100; font-weight: bold'
        elif v < 0.9: return 'color: #f57f17'
    except: pass
    return ''

# 应用样式
styled = df_display.style
styled = styled.map(style_score, subset=['得分'])
styled = styled.map(style_pct, subset=['涨跌'])
styled = styled.map(style_days, subset=['连跌'])
styled = styled.map(style_dd, subset=['60日回撤'])
styled = styled.map(style_dev, subset=['MA60偏离'])
styled = styled.map(style_vol, subset=['量比'])
styled = styled.map(style_pct, subset=['累计涨跌'])

# 格式化
fmt = {'收盘': '{:.2f}', '涨跌': '{:+.2f}%',
       '累计涨跌': lambda v: f"{v:+.2f}%" if v is not None and not (isinstance(v, float) and pd.isna(v)) else '-',
       '60日回撤': '{:+.1f}%', 'MA60偏离': '{:+.1f}%', '量比': '{:.2f}'}

styled = styled.format(fmt)

# 使用 data_editor 实现可点击选择
st.caption(f"📊 跟踪至最新交易日 | 🔴红涨 🟢绿跌 | 累计涨跌 = 从入选日至今的累计涨跌幅")

# 使用 selectbox 做股票选择 + 下方展示K线
# 按得分排序的选项
view_opts = [f"⭐{c['score']:2d} | {c['code']} {names.get(c['code'], '')} | ¥{c['close']:.2f} | "
             f"连跌{c['down_days']}天 回撤{c['dd_60']:.1f}%" 
             for c in candidates]
view_codes = [c['code'] for c in candidates]

# 表格区域
st.dataframe(styled, use_container_width=True, hide_index=True,
             height=min(600, 35 * len(table_data) + 40),
             column_config={'得分': st.column_config.NumberColumn('得分', width='small')})

# ==================== K线图查看器 ====================
st.divider()
st.subheader("📈 个股K线分析")

# 两列：左侧选择器，右侧评分明细
col_sel, col_detail = st.columns([3, 2])

with col_sel:
    view_sel = st.selectbox(
        "选择股票查看K线图（或直接在上方表格搜索）",
        view_opts, key="kline_sel",
        placeholder="输入代码或名称搜索...")

with col_detail:
    if view_sel:
        sel_code = view_sel.split('|')[1].strip().split()[0]
        sel_candidate = next((c for c in candidates if c['code'] == sel_code), None)
        if sel_candidate:
            try:
                bd = json.loads(sel_candidate.get('score_breakdown', '{}'))
                bd_items = [(k, v) for k, v in bd.items()]
                cols = st.columns(len(bd_items))
                label_map = {
                    'kdj': 'KDJ', 'rsi': 'RSI', 'ma60_dev': 'MA60偏离',
                    'down_days': '连跌', 'dd_60': '60日回撤',
                    'price_pct': '价格分位', 'macd': 'MACD',
                    'volume': '量能', 'boll': '布林', 'near_low': '近低'
                }
                for i, (k, v) in enumerate(bd_items):
                    pct = v / 5 * 100 if k != 'near_low' else (100 if v else 0)
                    with cols[i]:
                        st.metric(label_map.get(k, k), f"{v}/5",
                                  delta=f"{pct:.0f}%")
            except:
                pass

if view_sel:
    sel_code = view_sel.split('|')[1].strip().split()[0]
    sel_name = names.get(sel_code, '')
    sel_candidate = next((c for c in candidates if c['code'] == sel_code), None)

    df_k = get_daily_data(sel_code, 250)

    if not df_k.empty:
        # 计算入选后的表现
        post_perf = ""
        if sel_candidate and track_dates and sel_code in tracking:
            tk = tracking[sel_code]
            if track_dates:
                last_d = track_dates[-1]
                last_cum = tk['cum_pct'].get(last_d)
                if last_cum is not None:
                    color = "🔴" if last_cum > 0 else "🟢"
                    post_perf = f" | 入选后累计: {color} {last_cum:+.2f}%"

        st.markdown(f"#### {sel_code} {sel_name} | 入选价: ¥{sel_candidate['close']:.2f} | 得分: {sel_candidate['score']}{post_perf}")

        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.35, 0.2, 0.2, 0.25],
            subplot_titles=('K线 + MA20/MA60', 'MACD', 'KDJ', '成交量 + RSI14'),
        )

        # K线
        fig.add_trace(go.Candlestick(
            x=df_k['trade_date'], open=df_k['open'], high=df_k['high'],
            low=df_k['low'], close=df_k['close'], name='K线',
            increasing_line_color='#ef5350', decreasing_line_color='#26a69a',
            showlegend=False,
        ), row=1, col=1)

        # 均线
        for p, color in [(20, '#FF9800'), (60, '#42a5f5')]:
            col_ma = f'ma{p}'
            if col_ma in df_k.columns:
                fig.add_trace(go.Scatter(
                    x=df_k['trade_date'], y=df_k[col_ma], mode='lines',
                    name=f'MA{p}', line=dict(width=1.2, color=color),
                ), row=1, col=1)

        # 入选日标注（用 add_shape 避免 add_vline annotation bug）
        fig.add_vline(x=scan_date, line_width=2, line_dash='dash',
                      line_color='#FF5722', opacity=0.8,
                      row=1, col=1)

        # MACD
        mc = np.where(df_k['macd_hist'].values > 0, '#ef5350', '#26a69a')
        fig.add_trace(go.Bar(x=df_k['trade_date'], y=df_k['macd_hist'],
                              name='MACD柱', marker_color=mc, marker_line_width=0,
                              showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_k['trade_date'], y=df_k['macd_dif'],
                                  mode='lines', name='DIF', line=dict(width=1.2, color='#42a5f5')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_k['trade_date'], y=df_k['macd_dea'],
                                  mode='lines', name='DEA', line=dict(width=1.2, color='#FF9800')), row=2, col=1)
        fig.add_hline(y=0, line_dash='solid', line_color='gray', line_width=0.5, row=2, col=1)

        # KDJ
        fig.add_trace(go.Scatter(x=df_k['trade_date'], y=df_k['kdj_k'],
                                  mode='lines', name='K', line=dict(width=1.2, color='#42a5f5')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_k['trade_date'], y=df_k['kdj_d'],
                                  mode='lines', name='D', line=dict(width=1.2, color='#FF9800')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_k['trade_date'], y=df_k['kdj_j'],
                                  mode='lines', name='J', line=dict(width=1, color='#ab47bc', dash='dot')), row=3, col=1)
        fig.add_hline(y=20, line_dash='dash', line_color='#26a69a', line_width=0.8, row=3, col=1)
        fig.add_hline(y=80, line_dash='dash', line_color='#ef5350', line_width=0.8, row=3, col=1)

        # 成交量 + RSI
        vc = np.where(df_k['close'].values > df_k['open'].values, '#ef5350', '#26a69a')
        fig.add_trace(go.Bar(x=df_k['trade_date'], y=df_k['volume'],
                              name='成交量', marker_color=vc, marker_line_width=0,
                              opacity=0.35, showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=df_k['trade_date'], y=df_k['rsi14'],
                                  mode='lines', name='RSI14', line=dict(width=1.5, color='#ab47bc')), row=4, col=1)
        fig.add_hline(y=30, line_dash='dash', line_color='#26a69a', line_width=0.8, row=4, col=1)
        fig.add_hline(y=70, line_dash='dash', line_color='#ef5350', line_width=0.8, row=4, col=1)

        fig.update_layout(
            height=680,
            hovermode='x unified',
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation='h', yanchor='top', y=1.02, xanchor='left', x=0,
                        font=dict(size=10)),
            font=dict(size=11),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.15)')

        st.plotly_chart(fig, use_container_width=True)

# ==================== 整体跟踪表现 ====================
if track_dates and tracking:
    st.divider()
    st.subheader("📈 整体跟踪表现")

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
            daily_stats.append({
                'date': td, 'avg_cum': np.mean(vals),
                'median_cum': np.median(vals),
                'up_pct': up / len(vals) * 100,
                'count': len(vals),
            })

    if daily_stats:
        ds_df = pd.DataFrame(daily_stats)

        col_ch1, col_ch2 = st.columns(2)
        with col_ch1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ds_df['date'], y=ds_df['avg_cum'], mode='lines+markers',
                name='平均��计收益', line=dict(color='#FF5722', width=2.5),
                fill='tozeroy', fillcolor='rgba(255,87,34,0.08)',
                marker=dict(size=6),
            ))
            fig.add_hline(y=0, line_dash='dash', line_color='gray')
            fig.update_layout(
                title='📊 平均累计涨跌幅', height=320,
                template='plotly_dark', yaxis_ticksuffix='%',
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_ch2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=ds_df['date'], y=ds_df['up_pct'], mode='lines+markers',
                name='上涨比例', line=dict(color='#66bb6a', width=2.5),
                fill='tozeroy', fillcolor='rgba(102,187,106,0.08)',
                marker=dict(size=6),
            ))
            fig2.add_hline(y=50, line_dash='dash', line_color='gray',
                           annotation_text='50%基线', annotation_position='bottom right')
            fig2.update_layout(
                title='🎯 累计上���比例（胜率）', height=320,
                template='plotly_dark', yaxis_ticksuffix='%',
                yaxis_range=[0, 100],
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig2, use_container_width=True)

        latest_stat = daily_stats[-1]
        st.caption(f"📊 跟踪{latest_stat['count']}只有效数据 | "
                   f"最新平均累计: {latest_stat['avg_cum']:+.2f}% | "
                   f"中位数: {latest_stat['median_cum']:+.2f}% | "
                   f"上涨比例: {latest_stat['up_pct']:.1f}%")

st.divider()
st.caption(f"© A股起涨点扫描系统 V7 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
           f"数据来源: AKShare/新浪 | 仅供参考，不构成投资建议")
