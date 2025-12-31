import logging
from datetime import datetime, timedelta
# --- 关键修改：从 core.models 导入 ---
from core.models import TickEvent, TradeSignal, ActionType
# ---------------------------------

logger = logging.getLogger("PureForceClose")

class PureForceClose:
    """
    纯净版强制平仓策略
    """
    def __init__(self, config: dict = None):
        self.ph_windows = {
            'close': 60, 
            'start_force': 30, 
            'final_force': 10  
        }
        self.qh_windows = {
            'close': 60,
            'start_force': 15,
            'final_force': 3
        }

    def check_force_close(self, 
                          tick: TickEvent, 
                          position_size: float, 
                          current_time: datetime) -> bool:
        if position_size == 0:
            return False

        gate_closure_time = tick.delivery_start - timedelta(minutes=60)
        minutes_to_close = (gate_closure_time - current_time).total_seconds() / 60.0
        
        windows = self.qh_windows if "QH" in tick.contract_type else self.ph_windows
        
        if 0 < minutes_to_close <= windows['start_force']:
            return True
        
        if minutes_to_close <= 0:
            return True
            
        return False

    def generate_close_signal(self, tick: TickEvent, position_size: float, current_time: datetime) -> TradeSignal:
        action = ActionType.SELL if position_size > 0 else ActionType.BUY
        
        return TradeSignal(
            timestamp=current_time,
            contract_name=tick.contract_name,
            contract_id=tick.contract_id,
            action=action,
            size=abs(position_size),
            price=tick.price,
            strategy_name="force_close",
            delivery_start=tick.delivery_start,
            confidence=1.0,
            open_strategy="force_close" # 这里标记一下是强平
        )