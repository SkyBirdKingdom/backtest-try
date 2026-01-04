import sys
import os

# 确保能找到 core 和 strategies
sys.path.append(os.getcwd())

from core.engine import BacktestEngine

def main():
    # 1. 数据库配置
    DB_URL = "postgresql://postgres:123456@127.0.0.1:5432/nordpool_db"
    
    # 2. 策略配置
    config = {
        "initial_capital": 40000.0,
        "min_price_for_new_position": 10.0, 
        "max_position_size": 15.0,
        "max_contract_position_size": 1.0, # 默认值，会被 delivery_rules 覆盖
        "daily_loss_limit": 150.0,
        "transaction_cost": 0.23,

        # 【新增】成交所需等待的 Tick 数 (模拟排队)
        "execution_wait_trades": 3,
        
        # --- 策略参数 ---
        "strategy_params": {
            "forbid_new_open_minutes": 20,
            "signal_cooldown_seconds": 300,
            "price_change_threshold_ratio": 0.1,
            
            "super_mean_reversion_buy": {
                "ma_window": 5,          # 【关键】原为 20，现改为 5
                "threshold": 2.0,
                "history_min_len": 5,    # 新增
                "std_ratio_threshold": 0.1, # 新增
                "position_ratio": 0.2,   # 原配置是 0.2
                "position_split": 3,
                "min_open_size": 0.1
            },
            
            "optimized_extreme_sell": {
                "percentile_window": 5,  # 【关键】原为 100，现改为 5
                "percentile_high": 95,
                "percentile_extreme": 99,
                "threshold": 1.2,
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
                                    "position_split": 2
                                },
                                "optimized_extreme_sell": {
                                    "position_ratio": 0.5,
                                    "position_split": 2
                                }
                            }
                        }
                    ]
                }
            ]
        }
    }

    # 3. 初始化引擎
    engine = BacktestEngine(config, DB_URL)

    # 4. 运行回测
    # 请确保日期范围内你的数据库有数据
    start_date = "2025-12-29"
    end_date = "2025-12-29"
    
    # 可选：只回测特定的合约，填 None 则回测所有
    # contract_filter = ["PH-20240101-01", "PH-20240101-02"] 
    contract_filter = None

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