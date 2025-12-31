from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

# --- 枚举定义 ---
class ActionType(Enum):
    BUY = "BUY"
    SELL = "SELL"

# --- 核心事件 ---
@dataclass
class TickEvent:
    """
    回测系统的最小时间单位事件。
    """
    timestamp: datetime       
    contract_name: str        
    contract_id: str          
    price: float              
    volume: float             
    delivery_start: datetime  
    delivery_end: datetime    
    contract_type: str        
    trade_id: str             

# --- 交易相关模型 ---

@dataclass
class TradeSignal:
    """策略生成的交易信号"""
    timestamp: datetime
    contract_name: str
    contract_id: str
    action: ActionType
    size: float
    price: float
    strategy_name: str
    delivery_start: datetime
    confidence: float = 1.0
    open_strategy: str = "" # 用于区分是开仓信号还是平仓信号来源于哪个策略
    # --- 新增诊断字段 ---
    z_score: Optional[float] = None       # 记录触发时的Z分
    mean_price: Optional[float] = None    # 记录当时的均价
    std_price: Optional[float] = None     # 记录当时的标准差
    trend_info: str = ""                  # 记录趋势信息 (如 "UP/0.8")
    raw_size: float = 0.0                 # 记录风控调整前的原始计算手数
    # --- 【新增】拦截记录字段 ---
    is_valid: bool = True          # 默认为有效，被拦截时设为 False
    failure_reason: str = ""       # 记录被拦截的具体原因 (如 "趋势拦截", "冷却期")

@dataclass
class Order:
    """挂单"""
    timestamp: datetime
    client_order_id: str
    contract_id: str
    contract_name: str
    side: str # "BUY" or "SELL"
    quantity: float
    remaining_quantity: float
    unit_price: float
    state: str # "NEW", "FILLED", "CANCELLED"
    action: str # "USER_ADDED"
    delivery_start: datetime
    strategy: str
    open_strategy: str = ""
    order_type: str = "LIMIT"
    portfolio_id: str = ""
    delivery_area_id: str = ""
    # --- 新增 ---
    risk_check_passed: bool = True # 标记是否通过风控
    reject_reason: str = ""        # 如果被拒绝，记录原因

@dataclass
class Trade:
    """已成交记录"""
    timestamp: datetime
    trade_id: str
    client_order_id: str
    contract_name: str
    contract_id: str
    action: str # "BUY" or "SELL"
    size: float
    price: float
    strategy: str
    delivery_start: datetime
    pnl: float = 0.0 # 平仓时计算的盈亏
    trade_time: str = ""
    open_strategy: str = ""
    delivery_end: str = ""
    delivery_area: int = 0
    portfolio_id: str = ""

@dataclass
class Position:
    """持仓状态"""
    contract_name: str
    contract_id: str
    size: float
    avg_price: float
    timestamp: datetime
    delivery_start: datetime
    strategy_name: str = ""

@dataclass
class AccountInfo:
    """账户资金快照"""
    timestamp: datetime
    capital: float
    initial_capital: float = 0.0
    total_pnl: float = 0.0