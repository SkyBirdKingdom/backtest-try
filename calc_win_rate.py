import pandas as pd
from sqlalchemy import create_engine
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def calculate_round_trip_stats(db_url):
    """
    通过重演交易流，计算基于“完整交易周期(Round Trip)”的胜率和盈亏
    """
    engine = create_engine(db_url)
    
    # 1. 读取所有交易记录，按交易时间严格排序
    # 注意：这里我们只需要 trades 表，因为 trades 表里的 pnl 已经是交易所计算好的“已实现盈亏”
    # 只要把一个闭环内的所有 pnl 加起来，就是这波操作的总盈亏。
    query = """
    SELECT 
        timestamp,
        contract_name,
        action,
        size,
        price,
        pnl,
        strategy,
        delivery_start
    FROM backtest_trades 
    ORDER BY timestamp ASC
    """
    
    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        logging.error(f"读取数据库失败: {e}")
        return

    if df.empty:
        logging.info("没有交易记录。")
        return

    # 2. 按合约分组处理
    contracts = df['contract_name'].unique()
    
    round_trips = []  # 用于存储所有提取出来的完整交易周期
    
    print(f"开始分析 {len(contracts)} 个合约的交易流...")

    for contract in contracts:
        contract_df = df[df['contract_name'] == contract].copy()
        contract_df = contract_df.sort_values('timestamp')
        
        # --- 状态变量 ---
        current_pos = 0.0          # 当前持仓量
        cycle_pnl = 0.0            # 当前周期的累计盈亏
        cycle_volume = 0.0         # 当前周期交易量
        cycle_start_time = None    # 周期开始时间
        cycle_trades_count = 0     # 周期内交易笔数
        
        # 遍历该合约的每一笔交易
        for _, row in contract_df.iterrows():
            trade_pnl = row['pnl'] if row['pnl'] is not None else 0.0
            trade_size = row['size']
            direction = 1 if row['action'] == 'BUY' else -1
            signed_size = trade_size * direction
            
            # 1. 周期开始判定：如果你当前没持仓，现在发生了交易，这就是新周期的第一笔
            if abs(current_pos) < 0.001:
                cycle_start_time = row['timestamp']
                cycle_pnl = 0.0
                cycle_volume = 0.0
                cycle_trades_count = 0
            
            # 2. 更新状态
            current_pos += signed_size
            cycle_pnl += trade_pnl
            cycle_volume += trade_size
            cycle_trades_count += 1
            
            # 3. 周期结束判定：经过这笔交易后，持仓归零了
            if abs(current_pos) < 0.001:
                # 记录这一个完整的闭环 (Round Trip)
                round_trips.append({
                    'contract_name': contract,
                    'contract_type': contract[:2], # PH or QH
                    'start_time': cycle_start_time,
                    'end_time': row['timestamp'],
                    'delivery_start': row['delivery_start'], # 以前一笔为准，通常合约内是一样的
                    'total_pnl': cycle_pnl,
                    'is_win': cycle_pnl > 0,
                    'total_volume': cycle_volume,
                    'trades_count': cycle_trades_count
                })

    # 3. 转换为 DataFrame 方便统计
    results_df = pd.DataFrame(round_trips)
    
    if results_df.empty:
        print("未检测到任何完整的交易闭环（可能所有仓位都还未平仓）。")
        return

    # 4. 统计输出
    print("\n" + "="*50)
    print("📊 基于完整交易闭环 (Round Trip) 的回测报告")
    print("="*50)
    
    # --- 总体统计 ---
    total_rounds = len(results_df)
    total_wins = results_df['is_win'].sum()
    win_rate = (total_wins / total_rounds) * 100
    total_pnl = results_df['total_pnl'].sum()
    avg_pnl = total_pnl / total_rounds
    
    print(f"总交易闭环数: {total_rounds}")
    print(f"总净利润: {total_pnl:.2f} EUR")
    print(f"总体胜率: {win_rate:.2f}% ({total_wins}/{total_rounds})")
    print(f"平均每轮盈亏: {avg_pnl:.2f} EUR")
    
    # --- 分合约类型统计 (PH vs QH) ---
    print("\n📋 分合约类型统计:")
    print("-" * 65)
    print(f"{'类型':<6} | {'胜率':<10} | {'累计盈亏':<12} | {'闭环次数':<8} | {'单笔均盈'}")
    print("-" * 65)
    
    type_stats = results_df.groupby('contract_type').agg({
        'is_win': ['sum', 'count'],
        'total_pnl': 'sum'
    })
    
    # 整理 groupby 结果
    type_stats.columns = ['win_count', 'total_count', 'total_pnl']
    type_stats['win_rate'] = (type_stats['win_count'] / type_stats['total_count']) * 100
    type_stats['avg_pnl'] = type_stats['total_pnl'] / type_stats['total_count']
    
    for c_type, row in type_stats.iterrows():
        print(f"{c_type:<6} | {row['win_rate']:>6.2f}%    | {row['total_pnl']:>10.2f}   | {int(row['total_count']):>8} | {row['avg_pnl']:>8.2f}")

    print("-" * 65)

    # --- 月度统计 (可选) ---
    if 'delivery_start' in results_df.columns and results_df['delivery_start'].notnull().any():
        print("\n📅 月度盈亏统计 (按交付时间):")
        # 将 timestamp 转换为 period
        results_df['month'] = pd.to_datetime(results_df['delivery_start']).dt.to_period('M')
        
        month_stats = results_df.groupby(['month', 'contract_type'])['total_pnl'].sum().unstack(fill_value=0)
        print(month_stats.round(2))

if __name__ == "__main__":
    # 请替换为您的真实数据库地址
    DB_URL = "postgresql://postgres:123456@192.168.0.179:5432/nordpool_db"
    calculate_round_trip_stats(DB_URL)