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

# --- 主导航 ---
def main():
    st.sidebar.title("🧭 导航")
    page = st.sidebar.radio("跳转至", ["单合约深度分析", "日度多合约对比"])
    
    if page == "单合约深度分析":
        render_single_contract_analysis()
    else:
        render_daily_comparison()

if __name__ == "__main__":
    main()