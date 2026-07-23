"""
Plotly图表构建工具模块
统一管理K线图、技术指标图、起涨点标注、模板叠加等图表
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ChartBuilder:
    """图表构建器"""

    @staticmethod
    def build_kline_chart(df, code, highlight_date=None, show_boll=True,
                          show_ma=True, height=500):
        """构建K线+均线+布林带图表"""
        fig = go.Figure()

        # K线
        fig.add_trace(go.Candlestick(
            x=df['trade_date'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name=code,
            increasing_line_color='#ef5350', decreasing_line_color='#26a69a',
        ))

        # 均线
        if show_ma:
            for p, color in [(20, '#FF9800'), (60, '#2196F3'), (120, '#9C27B0')]:
                col = f'ma{p}'
                if col in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df['trade_date'], y=df[col], mode='lines',
                        name=f'MA{p}', line=dict(width=1.5, color=color),
                    ))

        # 布林带
        if show_boll and 'boll_upper' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['trade_date'], y=df['boll_upper'], mode='lines',
                name='布林上轨', line=dict(dash='dash', width=0.8, color='rgba(128,128,128,0.5)'),
            ))
            fig.add_trace(go.Scatter(
                x=df['trade_date'], y=df['boll_lower'], mode='lines',
                name='布林下轨', line=dict(dash='dash', width=0.8, color='rgba(128,128,128,0.5)'),
                fill='tonexty', fillcolor='rgba(128,128,128,0.08)',
            ))

        # 起涨点标注
        if highlight_date:
            fig.add_vline(
                x=highlight_date, line_width=2, line_dash='dash',
                line_color='#FF5722', opacity=0.8,
                annotation_text='起涨点', annotation_position='top',
                annotation_font=dict(color='#FF5722', size=12),
            )

        fig.update_layout(
            title=f'{code} K线图',
            height=height, hovermode='x unified',
            xaxis_rangeslider_visible=True,
            template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20),
        )
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(200,200,200,0.3)')
        fig.update_yaxes(title_text='价格', showgrid=True, gridwidth=0.5, gridcolor='rgba(200,200,200,0.3)')

        return fig

    @staticmethod
    def build_full_analysis_chart(df, code, highlight_date=None, height=950):
        """构建完整分析图表：K线 + MACD + KDJ + Volume/RSI"""
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.35, 0.2, 0.2, 0.25],
            subplot_titles=('K线 + 均线 + 布林带', 'MACD', 'KDJ', '成交量 + RSI'),
        )

        # === Row 1: K线 + 均线 + 布林带 ===
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
                fill='tonexty', fillcolor='rgba(128,128,128,0.08)',
            ), row=1, col=1)

        if highlight_date:
            for r in [1, 2, 3, 4]:
                fig.add_vline(
                    x=highlight_date, line_width=1.5, line_dash='dash',
                    line_color='#FF5722', opacity=0.6, row=r, col=1,
                )

        # === Row 2: MACD ===
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

        # === Row 3: KDJ ===
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

        # === Row 4: 成交量 + RSI ===
        vol_colors = np.where(df['close'].values > df['open'].values, '#ef5350', '#26a69a')
        fig.add_trace(go.Bar(
            x=df['trade_date'], y=df['volume'], name='成交量',
            marker_color=vol_colors, marker_line_width=0, opacity=0.5,
        ), row=4, col=1)

        # RSI 叠加（双Y轴）
        fig.add_trace(go.Scatter(
            x=df['trade_date'], y=df['rsi14'], mode='lines',
            name='RSI(14)', line=dict(width=1.5, color='#9C27B0'),
            yaxis='y5',
        ), row=4, col=1)

        fig.update_layout(
            height=height, hovermode='x unified',
            template='plotly_white',
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=20, r=60, t=40, b=20),
        )

        # RSI 参考线
        fig.add_hline(y=70, line_dash='dash', line_color='#ef5350', line_width=0.8, row=4, col=1)
        fig.add_hline(y=30, line_dash='dash', line_color='#26a69a', line_width=0.8, row=4, col=1)

        return fig

    @staticmethod
    def build_score_radar(score_breakdown):
        """构建评分雷达图"""
        if not score_breakdown:
            return go.Figure()

        categories = list(score_breakdown.keys())
        values = list(score_breakdown.values())
        max_vals = {'kdj': 5, 'rsi': 4, 'ma60_dev': 5, 'down_days': 4,
                     'dd_60': 3, 'price_pct': 3, 'macd': 2, 'volume': 2,
                     'boll': 1, 'near_low': 1}

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values, theta=categories, fill='toself',
            name='评分', fillcolor='rgba(255,87,34,0.3)',
            line=dict(color='#FF5722', width=2),
        ))
        fig.add_trace(go.Scatterpolar(
            r=[max_vals.get(c, 5) for c in categories],
            theta=categories, fill=None,
            name='满分', line=dict(color='rgba(128,128,128,0.5)', dash='dash'),
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            height=300, margin=dict(l=40, r=40, t=20, b=20),
            template='plotly_white',
        )
        return fig

    @staticmethod
    def build_filter_funnel(filter_stats):
        """构建过滤漏斗图"""
        stages = ['总股票数', 'KDJ超卖', 'RSI弱势', '破MA60', '60日回撤',
                   '缩量', '连跌', '当日未大涨', '布林下半区']
        values = [filter_stats.get('total', 0)]

        for key in ['KDJ超卖', 'RSI弱势', '破MA60', '60日回撤', '缩量', '连跌', '当日未大涨', '布林下半区']:
            values.append(filter_stats.get(key, 0))
        values.append(filter_stats.get('passed', 0))
        stages.append('最终通过')

        fig = go.Figure(go.Funnel(
            y=stages, x=values,
            textinfo='value+percent initial',
            marker=dict(
                color=['#2196F3', '#42A5F5', '#64B5F6', '#90CAF9',
                       '#BBDEFB', '#E3F2FD', '#BBDEFB', '#90CAF9',
                       '#64B5F6', '#FF5722'],
            ),
        ))
        fig.update_layout(height=350, margin=dict(l=100, r=20, t=20, b=20),
                          template='plotly_white')
        return fig

    @staticmethod
    def build_winner_gallery(df_w, code, launch_date=None):
        """构建大涨股起涨点前后的K线图"""
        fig = go.Figure()

        # 简化K线
        fig.add_trace(go.Candlestick(
            x=df_w['trade_date'], open=df_w['open'], high=df_w['high'],
            low=df_w['low'], close=df_w['close'], name=code,
            increasing_line_color='#ef5350', decreasing_line_color='#26a69a',
        ))

        if launch_date and launch_date in df_w['trade_date'].values:
            fig.add_vline(
                x=launch_date, line_width=2, line_dash='dash',
                line_color='#FF5722', opacity=0.8,
                annotation_text='起涨点',
            )

        fig.update_layout(
            title=f'{code} 大涨股模板',
            height=280, hovermode='x',
            template='plotly_white',
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False,
        )
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)

        return fig

    @staticmethod
    def build_distribution_histogram(data, title, x_label, highlight_value=None, color='#2196F3'):
        """构建特征分布直方图"""
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=data, nbinsx=20, marker_color=color,
            marker_line_color='white', marker_line_width=0.5,
            opacity=0.8, name='分布',
        ))
        if highlight_value is not None:
            fig.add_vline(
                x=highlight_value, line_width=2, line_dash='dash',
                line_color='#FF5722',
                annotation_text=f'当前: {highlight_value:.1f}',
            )
        fig.update_layout(
            title=title, height=250, template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title=x_label, yaxis_title='数量',
            showlegend=False,
        )
        return fig
