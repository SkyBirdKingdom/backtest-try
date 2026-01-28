import sys
import os
import logging
from sqlalchemy import create_engine, text

# 确保能找到 core 和 strategies
sys.path.append(os.getcwd())

from core.engine import BacktestEngine

def clear_tables_except_trades(db_url: str):
    """清空除trades表外的所有回测表"""
    engine = create_engine(db_url)
    
    # 定义需要清空的表（除了trades表）
    tables_to_clear = [
        'backtest_contract_stats',
        'backtest_orders',              
        'backtest_settlements',
        'backtest_signals',
        'backtest_trades'
    ]
    
    for table_name in tables_to_clear:
        try:
            with engine.connect() as conn:
                # 检查表是否存在
                result = conn.execute(text(f"SELECT to_regclass('{table_name}')"))
                table_exists = result.scalar() is not None
                
                if table_exists:
                    conn.execute(text(f"DELETE FROM {table_name}"))
                    conn.commit()
                    print(f"✅ 已清空表: {table_name}")
                else:
                    print(f"⚠️ 表不存在，跳过: {table_name}")
        except Exception as e:
            print(f"❌ 清空表 {table_name} 时出错: {e}")
            if 'conn' in locals():
                conn.rollback()
    
    print("📊 数据清理完成")


def main():
    # 1. 数据库配置
    DB_URL = "postgresql://postgres:123456@127.0.0.1:5432/nordpool_db?client_encoding=utf8"
    
    # 2. 策略配置
    config = {
        "initial_capital": 40000.0,
        "min_price_for_new_position": 10.0, 
        "max_position_size": 6000.0,
        "max_contract_position_size": 4.0, # 默认值，会被 delivery_rules 覆盖
        "daily_loss_limit": 100.0,
        "transaction_cost": 0.23,

        # --- 回测仿真参数 ---
        "execution_wait_trades": 0,    # 成交排队等待笔数 (模拟订单簿深度)
        "order_submission_delay": 5,  # 订单提交延迟秒数 (模拟数据/网络延迟)
        
        # --- 策略参数 ---
        "strategy_params": {
            "forbid_new_open_minutes": 20,
            "signal_cooldown_seconds": 300,
            "price_change_threshold_ratio": 0.1,

            "take_profit_end_minutes": 10,
            "breakeven_end_minutes": 6,
            "stop_loss_end_minutes": 3,
            
            "super_mean_reversion_buy": {
                "use_dynamic_sizing": True,      # 开启
                "liquidity_lookback": 30,        # 看过去30分钟
                "liquidity_participation": 0.20, # 吃掉预测量的 5%
                "liquidity_projection": "60",      # 2. 预测未来1小时总成交 (或填 "till_close")
                "action": "BUY",
                "history_min_len": 10,
                "ma_window": 5,
                "std_ratio_threshold": 0.1,
                "threshold": 2,
                "position_ratio": 0.2,
                "position_split": 3,
                "min_open_size": 0.1
            },
            
            "optimized_extreme_sell": {
                "use_dynamic_sizing": True,      # 开启
                "liquidity_lookback": 30,        # 看过去30分钟
                "liquidity_participation": 0.20, # 吃掉预测量的 5%
                "liquidity_projection": "60",      # 2. 预测未来1小时总成交 (或填 "till_close")
                "action": "SELL",
                "history_min_len": 10,
                "percentile_window": 5,
                "percentile_high": 95,
                "percentile_extreme": 99,
                "threshold": 1.3,
                "position_ratio": 0.6,
                "position_split": 3,
                "min_open_size": 0.1
            },
            
            "high_volatility_dip_buy": { # 新增
                "threshold": 50.0,
                "position_ratio": 0.5,
                "position_split": 3,
                "min_open_size": 0.1
            },
            
            "delivery_time_buy": {       # 新增
                "price_count": 10000,
                "position_ratio": 0.2,
                "position_split": 1,
                "min_open_size": 0.1
            }
        },
        
        "position_constraints": {
            "default_contract_max_position": 1.0,
            "delivery_rules": [
                {
                    "comment": "All Day Rule",
                    "days_of_week": [0, 1, 2, 3, 4, 5, 6],
                    "time_ranges": [
                        {
                            "start": "00:00",
                            "end": "23:59",
                            "max_position": 4.0, # 模拟实盘规则
                            "strategy_params": {
                                "super_mean_reversion_buy": {
                                    "position_ratio": 0.5,
                                    "position_split": 1
                                },
                                "optimized_extreme_sell": {
                                    "position_ratio": 0.5,
                                    "position_split": 1
                                }
                            }
                        }
                    ]
                }
            ]
        }
    }

    # 3. 清空除 trades 表外的所有表
    clear_tables_except_trades(DB_URL)
    
    # 4. 初始化引擎
    engine = BacktestEngine(config, DB_URL)

    # 4. 运行回测
    # 请确保日期范围内你的数据库有数据
    start_date = "2025-12-27"
    end_date = "2026-12-27"
    
    # 可选：只回测特定的合约，填 None 则回测所有
    contract_filter = ["QH-20251227-95"] 
    # contract_filter = None

    try:
        engine.run(start_date, end_date, contract_filter)
        
        # 5. 查看前几笔成交
        if engine.exchange.trades:
            print("\n--- 前 10 笔成交记录 ---")
            for t in engine.exchange.trades[:10]:
                print(f"[{t.timestamp}] {t.contract_name} {t.action} {t.size} @ {t.price}")
        else:
            print("\n没有产生任何交易。")
            
    except Exception as e:
        print(f"回测运行出错: {e}")

if __name__ == "__main__":
    main()