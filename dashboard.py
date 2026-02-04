import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine

# --- 配置 ---
# 请确保这里的数据库连接字符串与你 main.py 中的一致
DB_URL = "postgresql://postgres:123456@127.0.0.1:5432/nordpool_db?client_encoding=utf8"

# 设置页面布局为宽屏模式
st.set_page_config(page_title="回测交易可视化看板", layout="wide")

# --- 数据库函数 ---
@st.cache_resource
def get_engine():
    return create_engine(DB_URL)

def get_market_data(contract_name, area=None):
    """读取并去重市场行情数据"""
    engine = get_engine()
    query = f"""
    SELECT trade_time, contract_name, price, volume, trade_id, delivery_area 
    FROM trades 
    WHERE contract_name = '{contract_name}'
    """
    if area:
        query += f" AND delivery_area = '{area}'"
    
    query += " ORDER BY trade_time ASC"
    
    try:
        df = pd.read_sql(query, engine)
        if not df.empty:
            # 需求：根据 trade_id 去重
            df = df.drop_duplicates(subset=['trade_id'])
        return df
    except Exception as e:
        st.error(f"读取市场数据失败: {e}")
        return pd.DataFrame()

def get_backtest_trades(contract_name, run_id=None):
    """读取回测交易记录"""
    engine = get_engine()
    query = f"""
    SELECT * FROM backtest_trades 
    WHERE contract_name = '{contract_name}'
    """
    
    # 如果指定了 run_id，则只看该次运行的结果
    if run_id:
        query += f" AND run_id = '{run_id}'"
    
    query += " ORDER BY timestamp ASC"
    
    try:
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        st.error(f"读取回测数据失败: {e}")
        return pd.DataFrame()

def get_distinct_runs(contract_name):
    """获取该合约存在的所有 run_id"""
    engine = get_engine()
    query = f"SELECT DISTINCT run_id FROM backtest_trades WHERE contract_name = '{contract_name}' ORDER BY run_id DESC"
    try:
        df = pd.read_sql(query, engine)
        return df['run_id'].tolist()
    except:
        return []

# --- 侧边栏：查询条件 ---
st.sidebar.header("🔍 查询条件")
# 默认区域
selected_area = st.sidebar.text_input("区域 (Delivery Area)", value="SE3") 
# 默认合约
contract_name_input = st.sidebar.text_input("合约名称 (Contract Name)", value="PH-20250624-12")

search_btn = st.sidebar.button("查询 / 刷新")

if search_btn or contract_name_input:
    st.title(f"📊 交易回测分析: {contract_name_input}")

    # 1. 获取 Run IDs
    run_ids = get_distinct_runs(contract_name_input)
    selected_run_id = None
    if run_ids:
        selected_run_id = st.selectbox("选择回测批次 (Run ID)", run_ids, index=0)
    else:
        st.warning("未在 backtest_trades 表中找到该合约的回测记录。")

    # 2. 加载数据
    with st.spinner('正在从数据库加载数据...'):
        market_df = get_market_data(contract_name_input, selected_area)
        trades_df = get_backtest_trades(contract_name_input, selected_run_id)

    # 3. 数据概览
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("市场数据条数", len(market_df))
    col2.metric("策略交易次数", len(trades_df))
    
    if not trades_df.empty:
        total_pnl = trades_df['pnl'].sum()
        win_rate = (trades_df[trades_df['pnl'] > 0].shape[0] / len(trades_df)) * 100 if len(trades_df) > 0 else 0
        col3.metric("总盈亏 (PnL)", f"{total_pnl:.2f} EUR", delta_color="normal")
        col4.metric("胜率 (Win Rate)", f"{win_rate:.2f}%")

    # 4. 绘制交互式图表
    if not market_df.empty:
        st.subheader("📈 价格走势与买卖点 (支持缩放和滚动)")
        
        fig = go.Figure()

        # A. 绘制市场价格线 + 真实成交点 (Market Trades)
        # 修改点：mode='lines+markers'，并添加 hovertemplate 显示量价
        fig.add_trace(go.Scattergl(
            x=market_df['trade_time'],
            y=market_df['price'],
            mode='lines+markers', # 关键修改：显示线和点
            name='市场成交 (Market)',
            line=dict(color='#636EFA', width=1),
            # 市场点设为小蓝点，半透明，避免喧宾夺主
            marker=dict(symbol='circle', size=4, color='#636EFA', opacity=0.6), 
            text=market_df['volume'], # 将 volume 传入 text 字段供 hover 使用
            hovertemplate=(
                "<b>市场成交</b><br>" +
                "时间: %{x}<br>" +
                "价格: %{y:.2f} EUR<br>" +
                "成交量: %{text:.1f} MW<br>" +
                "<extra></extra>" # 隐藏默认的 trace name 标签
            )
        ))

        # B. 绘制回测买卖点 (Strategy Trades)
        if not trades_df.empty:
            # 买入点：绿色圆点
            buys = trades_df[trades_df['action'] == 'BUY']
            if not buys.empty:
                fig.add_trace(go.Scattergl(
                    x=buys['timestamp'],
                    y=buys['price'],
                    mode='markers',
                    name='策略买入 (BUY)',
                    # 策略点设为大绿点，带黑边，非常醒目
                    marker=dict(symbol='circle', size=12, color='#00CC96', line=dict(width=2, color='DarkSlateGrey')),
                    # 悬停显示策略详情
                    text=buys.apply(lambda row: f"🟢 <b>策略买入</b><br>策略: {row['strategy']}<br>数量: {row['size']}", axis=1),
                    hovertemplate="%{text}<br>价格: %{y:.2f}<br>时间: %{x}<extra></extra>"
                ))

            # 卖出点：红色圆点
            sells = trades_df[trades_df['action'] == 'SELL']
            if not sells.empty:
                fig.add_trace(go.Scattergl(
                    x=sells['timestamp'],
                    y=sells['price'],
                    mode='markers',
                    name='策略卖出 (SELL)',
                    # 策略点设为大红点，带黑边
                    marker=dict(symbol='circle', size=12, color='#EF553B', line=dict(width=2, color='DarkSlateGrey')),
                    # 悬停显示盈亏详情
                    text=sells.apply(lambda row: f"🔴 <b>策略卖出</b><br>策略: {row['strategy']}<br>数量: {row['size']}<br>盈亏: {row['pnl']:.2f}", axis=1),
                    hovertemplate="%{text}<br>价格: %{y:.2f}<br>时间: %{x}<extra></extra>"
                ))

        # C. 布局配置优化
        fig.update_layout(
            title=f'{contract_name_input} 交易详情',
            xaxis=dict(
                title='时间',
                rangeslider=dict(visible=True), # 启用底部时间轴滑块
                type='date'
            ),
            yaxis_title='价格 (EUR)',
            hovermode='closest', # 改为 closest，这样鼠标指哪里显示哪里，避免多个标签重叠
            height=600,
            margin=dict(l=20, r=20, t=50, b=20),
            dragmode='pan', # 默认平移模式
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # 允许滚轮缩放
        config = {
            'scrollZoom': True, 
            'displayModeBar': True,
            'modeBarButtons.add': ['drawline', 'drawopenpath', 'eraseshape']
        }

        st.plotly_chart(fig, use_container_width=True, config=config)
    else:
        st.warning("未找到该合约的市场数据，无法绘图。请检查合约名称或 trades 表数据。")

    # 5. 展示数据表格
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📝 回测交易明细", "📋 市场行情原始数据"])
    
    with tab1:
        st.subheader(f"回测交易记录 ({len(trades_df)} 条)")
        if not trades_df.empty:
            display_cols = ['trade_id', 'timestamp', 'action', 'price', 'size', 'pnl', 'strategy', 'open_strategy', 'delivery_start']
            valid_cols = [c for c in display_cols if c in trades_df.columns]
            st.dataframe(
                trades_df[valid_cols].style.format({
                    'price': '{:.2f}',
                    'size': '{:.1f}',
                    'pnl': '{:.2f}'
                }).applymap(lambda x: 'color: green' if x > 0 else 'color: red' if x < 0 else '', subset=['pnl']),
                use_container_width=True
            )
        else:
            st.info("无回测交易数据")

    with tab2:
        st.subheader(f"市场行情数据 (去重后: {len(market_df)} 条)")
        if not market_df.empty:
            st.dataframe(
                market_df.style.format({'price': '{:.2f}', 'volume': '{:.1f}'}),
                use_container_width=True
            )
        else:
            st.info("无市场数据")