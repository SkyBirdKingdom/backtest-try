import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine
import re

# --- 配置 ---
DB_URL = "postgresql://postgres:123456@127.0.0.1:5432/nordpool_db?client_encoding=utf8"

st.set_page_config(page_title="NordPool 策略分析中心", layout="wide")

@st.cache_resource
def get_engine():
    return create_engine(DB_URL)

# --- 核心数据获取函数 ---
def get_all_run_ids():
    query = "SELECT DISTINCT run_id FROM backtest_trades ORDER BY run_id DESC"
    try:
        return pd.read_sql(query, get_engine())['run_id'].tolist()
    except: return []

def get_contracts_by_run(run_ids):
    if not run_ids: return []
    ids_str = "', '".join(run_ids)
    query = f"SELECT DISTINCT contract_name FROM backtest_contract_stats WHERE run_id IN ('{ids_str}')"
    return pd.read_sql(query, get_engine())['contract_name'].tolist()

def get_market_data(contract_name):
    query = f"SELECT trade_time, price, volume, trade_id FROM trades WHERE contract_name = '{contract_name}' ORDER BY trade_time ASC"
    df = pd.read_sql(query, get_engine())
    return df.drop_duplicates(subset=['trade_id'])

def get_backtest_trades(contract_name, run_id):
    query = f"SELECT * FROM backtest_trades WHERE contract_name = '{contract_name}' AND run_id = '{run_id}' ORDER BY timestamp ASC"
    return pd.read_sql(query, get_engine())

# --- 页面一：单合约深度分析 ---
def render_single_contract_analysis():
    st.sidebar.header("🔍 过滤条件")
    all_runs = get_all_run_ids()
    selected_run = st.sidebar.selectbox("1. 选择回测批次 (Run ID)", all_runs)
    
    relevant_contracts = get_contracts_by_run([selected_run]) if selected_run else []
    selected_contract = st.sidebar.selectbox("2. 选择合约", relevant_contracts)
    
    show_market_dots = st.sidebar.checkbox("显示市场成交散点", value=False)

    if selected_run and selected_contract:
        market_df = get_market_data(selected_contract)
        trades_df = get_backtest_trades(selected_contract, selected_run)

        st.title(f"📈 交易分析: {selected_contract}")
        
        # 指标栏
        if not trades_df.empty:
            total_pnl = trades_df['pnl'].sum()
            c1, c2, c3 = st.columns(3)
            c1.metric("总盈亏", f"{total_pnl:.2f} EUR")
            c2.metric("交易笔数", len(trades_df))
            c3.metric("平均每笔盈亏", f"{(total_pnl/len(trades_df)):.2f} EUR")

        # 图表展示
        if not market_df.empty:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            # 市场价格背景线 (设置 hoverinfo='skip' 解决干扰)
            fig.add_trace(go.Scattergl(
                x=market_df['trade_time'], y=market_df['price'],
                name='市场价格', line=dict(color='rgba(100, 100, 100, 0.2)', width=1),
                hoverinfo='skip'
            ), row=1, col=1)

            if show_market_dots:
                fig.add_trace(go.Scattergl(
                    x=market_df['trade_time'], y=market_df['price'], mode='markers',
                    name='市场成交', marker=dict(size=3, color='rgba(100, 150, 250, 0.2)'),
                    hoverinfo='skip'
                ), row=1, col=1)

            # 回测买卖点
            if not trades_df.empty:
                for action, color, symbol in [('BUY', '#00CC96', 'triangle-up'), ('SELL', '#EF553B', 'triangle-down')]:
                    mask = trades_df['action'] == action
                    sub = trades_df[mask]
                    fig.add_trace(go.Scattergl(
                        x=sub['timestamp'], y=sub['price'], mode='markers',
                        name=f'策略-{action}', marker=dict(symbol=symbol, size=12, color=color, line=dict(width=1, color='white')),
                        customdata=sub[['strategy', 'size', 'pnl', 'open_strategy']],
                        hovertemplate="<b>%{name}</b><br>价格: %{y}<br>数量: %{customdata[1]}<br>策略: %{customdata[0]}<br>盈亏: %{customdata[2]}<extra></extra>"
                    ), row=1, col=1)

                trades_df['cum_pnl'] = trades_df['pnl'].cumsum()
                fig.add_trace(go.Scattergl(
                    x=trades_df['timestamp'], y=trades_df['cum_pnl'],
                    name='累计盈亏', fill='tozeroy', line=dict(color='gold')
                ), row=2, col=1)

            fig.update_layout(height=700, hovermode='x unified', xaxis2_rangeslider_visible=True)
            st.plotly_chart(fig, use_container_width=True)

        # 恢复数据表格
        tab1, tab2 = st.tabs(["📝 回测交易明细", "📋 真实市场成交数据"])
        with tab1:
            st.dataframe(trades_df.style.highlight_max(axis=0, subset=['pnl'], color='#90ee90'), use_container_width=True)
        with tab2:
            st.dataframe(market_df, use_container_width=True)

# --- 页面二：日度多合约对比 ---
def render_daily_comparison():
    st.title("📅 日度多合约收益对比分析")
    
    col1, col2 = st.columns(2)
    with col1:
        target_date = st.date_input("选择目标日期", value=pd.to_datetime("2026-01-08"))
    with col2:
        all_runs = get_all_run_ids()
        selected_runs = st.multiselect("选择参与对比的 Run ID", all_runs, default=all_runs[:1] if all_runs else [])

    if not selected_runs:
        st.warning("请至少选择一个 Run ID")
        return

    date_str = target_date.strftime("%Y%m%d")
    run_ids_sql = "', '".join(selected_runs)
    
    # 匹配 PH-YYYYMMDD-xx 或 QH-YYYYMMDD-xx 的逻辑
    query = f"""
        SELECT run_id, contract_name, SUM(pnl) as total_pnl, COUNT(*) as trade_count
        FROM backtest_trades
        WHERE run_id IN ('{run_ids_sql}')
          AND (contract_name LIKE 'PH-{date_str}-%%' OR contract_name LIKE 'QH-{date_str}-%%')
        GROUP BY run_id, contract_name
    """
    
    df_daily = pd.read_sql(query, get_engine())

    if df_daily.empty:
        st.info(f"未找到日期 {target_date} 相关的合约回测数据。")
    else:
        # 数据透视以便对比
        comparison_df = df_daily.pivot(index='contract_name', columns='run_id', values='total_pnl').fillna(0)
        
        st.subheader("📊 各合约收益对比 (EUR)")
        st.bar_chart(comparison_df)

        # 汇总统计
        st.subheader("📋 统计摘要")
        summary = df_daily.groupby('run_id').agg({
            'total_pnl': 'sum',
            'contract_name': 'nunique',
            'trade_count': 'sum'
        }).rename(columns={'contract_name': '涉及合约数', 'total_pnl': '当日总盈亏', 'trade_count': '总交易笔数'})
        
        st.table(summary.style.background_gradient(cmap='RdYlGn', subset=['当日总盈亏']))

        with st.expander("查看原始统计数据"):
            st.dataframe(df_daily, use_container_width=True)

# --- 页面三：月度盈亏统计 (修改版) ---
def render_monthly_statistics():
    st.title("🗓️ 月度盈亏统计分析")
    
    all_runs = get_all_run_ids()
    selected_run = st.selectbox("选择回测批次 (Run ID) 以进行月度统计", all_runs)
    
    if not selected_run:
        return

    # 从 backtest_trades 中获取该 run_id 的所有数据
    query = f"""
        SELECT contract_name, pnl, action, size 
        FROM backtest_trades 
        WHERE run_id = '{selected_run}'
    """
    df = pd.read_sql(query, get_engine())

    if df.empty:
        st.warning("该 Run ID 没有相关的交易数据。")
        return

    # 1. 使用正则从 contract_name 中提取 YYYYMMDD 日期 (支持 PH-YYYYMMDD-xx 和 QH-YYYYMMDD-xx)
    df['date_str'] = df['contract_name'].str.extract(r'-(\d{8})-')
    
    # 2. 转换为时间格式，并提取出 YYYY-MM 月份标识
    df['month'] = pd.to_datetime(df['date_str'], format='%Y%m%d', errors='coerce').dt.strftime('%Y-%m')
    df = df.dropna(subset=['month'])
    
    if df.empty:
        st.error("未能从合约名称中成功解析出日期，请检查 contract_name 的命名格式。")
        return

    # 3. 基础按月聚合统计（包含所有订单记录）
    monthly_stats = df.groupby('month').agg(
        total_pnl=('pnl', 'sum'),
        total_executions=('pnl', 'count'), # 总成交动作（含开仓+平仓）
        unique_contracts=('contract_name', 'nunique')
    ).reset_index()

    # 4. 提取有效平仓单（pnl 绝对值大于0的才算作平仓结算）
    # 这样可以过滤掉所有建仓时 pnl=0 的占位记录，分母才准确
    close_trades_df = df[df['pnl'].abs() > 0.0001]
    close_stats = close_trades_df.groupby('month').agg(
        close_trades=('pnl', 'count')
    ).reset_index()

    # 5. 提取盈利单（pnl > 0）
    win_trades_df = df[df['pnl'] > 0]
    win_stats = win_trades_df.groupby('month').agg(
        win_trades=('pnl', 'count')
    ).reset_index()

    # 6. 数据合并
    monthly_stats = pd.merge(monthly_stats, close_stats, on='month', how='left').fillna(0)
    monthly_stats = pd.merge(monthly_stats, win_stats, on='month', how='left').fillna(0)

    # 7. 修正胜率计算逻辑：用 盈利单 / 有效平仓单
    monthly_stats['win_rate'] = (monthly_stats['win_trades'] / monthly_stats['close_trades'].replace(0, 1) * 100).round(2)

    # 按照月份排序
    monthly_stats = monthly_stats.sort_values('month')

    # --- 界面展示 ---
    st.subheader("📊 月度总盈亏走势")
    # 绘制柱状图，盈利为绿色，亏损为红色
    colors = ['#00CC96' if pnl >= 0 else '#EF553B' for pnl in monthly_stats['total_pnl']]
    
    fig = go.Figure(data=[
        go.Bar(
            name='月度盈亏', 
            x=monthly_stats['month'], 
            y=monthly_stats['total_pnl'],
            marker_color=colors,
            text=monthly_stats['total_pnl'].round(2),
            textposition='auto'
        )
    ])
    fig.update_layout(
        xaxis_title="月份",
        yaxis_title="净盈亏 (EUR)",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 月度统计数据明细")
    # 重命名列并格式化展示 (特意把 总成交动作 和 实际平仓笔数 都显示出来方便你核对)
    display_df = monthly_stats[['month', 'total_pnl', 'total_executions', 'close_trades', 'unique_contracts', 'win_rate']].copy()
    display_df.columns = ['月份', '总盈亏 (EUR)', '总成交动作(含开平仓)', '有效平仓笔数', '参与合约数量', '平仓胜率 (%)']
    
    # 设置背景色渐变方便查看
    st.dataframe(
        display_df.style.background_gradient(cmap='RdYlGn', subset=['总盈亏 (EUR)'])
                        .format({
                            '总盈亏 (EUR)': "{:.2f}", 
                            '平仓胜率 (%)': "{:.2f}%",
                            '总成交动作(含开平仓)': "{:.0f}",
                            '有效平仓笔数': "{:.0f}",
                            '参与合约数量': "{:.0f}"
                        }), 
        use_container_width=True
    )

# --- 主导航 ---
def main():
    st.sidebar.title("🧭 导航")
    page = st.sidebar.radio("跳转至", ["单合约深度分析", "日度多合约对比", "月度盈亏统计"])
    
    if page == "单合约深度分析":
        render_single_contract_analysis()
    elif page == "日度多合约对比":
        render_daily_comparison()
    else:
        render_monthly_statistics()

if __name__ == "__main__":
    main()