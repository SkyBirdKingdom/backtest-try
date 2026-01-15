import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from core.models import TickEvent, TradeSignal, ActionType, Position, Order

logger = logging.getLogger("PureExitManager")

class PureExitManager:
    """
    纯净版平仓管理器 (Lifecycle Manager) - 增强版 V2.4
    1. 锚定 initial_entry_time 进行动态止盈计算
    2. 使用 modify_order 进行订单同步
    3. 目标价格计算：手续费Buffer固定为 0.46
    4. 【严重Bug修复】_find_exit_order 增加策略名称过滤，防止错误修改建仓单
    """
    def __init__(self, config: dict):
        self.config = config
        self.transaction_cost = config.get('transaction_cost', 0.23)
        self.last_order_update_time: Dict[str, datetime] = {}

    def process(self, tick: TickEvent, positions: Dict[str, Position], 
                active_orders: List[Order], exchange) -> None:
        if not tick.delivery_start:
            return

        minutes_to_close = self._get_minutes_to_close(tick.delivery_start, tick.timestamp)
        
        if minutes_to_close > 240 or minutes_to_close <= 0:
            return

        position = positions.get(tick.contract_name)
        if not position or abs(position.size) < 0.001:
            return

        # 获取属于本管理器的平仓单
        existing_exit_order = self._find_exit_order(tick.contract_name, active_orders)

        # 1. 检测数量是否一致 (Sync Check)
        qty_mismatch = False
        if existing_exit_order:
            # 如果持仓 != 挂单剩余，说明发生了部分成交或加仓，需要更新订单
            if abs(abs(position.size) - existing_exit_order.remaining_quantity) > 0.001:
                qty_mismatch = True
        elif not existing_exit_order:
            # 如果没有平仓单，但有持仓，说明需要新挂单
            qty_mismatch = True

        # 2. 计算目标价格
        target_price, is_force_market = self._calculate_target_price(
            minutes_to_close, position, tick
        )

        # 3. 执行管理
        self._manage_exit_order(
            exchange, position, tick, existing_exit_order, 
            target_price, is_force_market, minutes_to_close, qty_mismatch
        )

    def _get_minutes_to_close(self, delivery_start: Union[str, datetime], current_time: datetime) -> float:
        try:
            if isinstance(delivery_start, str):
                delivery_dt = datetime.strptime(delivery_start, '%Y-%m-%d %H:%M:%S')
            else:
                delivery_dt = delivery_start
            gate_closure = delivery_dt - timedelta(hours=1)
            delta = gate_closure - current_time
            return delta.total_seconds() / 60.0
        except Exception:
            return 9999.0

    def _calculate_target_price(self, minutes_to_close: float, position: Position, 
                                tick: TickEvent) -> Tuple[float, bool]:
        """
        计算目标平仓价格
        """
        entry_price = position.avg_price
        
        # 固定手续费缓冲 0.46
        fee_rate = self.transaction_cost 
        cost_padding = 2 * fee_rate      
        
        is_long = position.size > 0
        target_price = tick.price
        is_force_market = False

        # --- 阶段 1: 止盈阶段 ---
        if 20 < minutes_to_close <= 240:
            # 使用初始建仓时间计算进度
            start_time = position.initial_entry_time if position.initial_entry_time else position.timestamp
            start_minutes_to_close = self._get_minutes_to_close(tick.delivery_start, start_time)
            
            if start_minutes_to_close <= 20:
                progress = 1.0
            else:
                total_duration = start_minutes_to_close - 20
                elapsed = start_minutes_to_close - minutes_to_close
                progress = elapsed / total_duration
                progress = max(0.0, min(1.0, progress))
            
            start_margin = 0.50 if entry_price < 50 else 0.20
            end_margin = 0.01
            current_margin = start_margin - (start_margin - end_margin) * progress
            
            decay_price = 0.0
            if is_long:
                decay_price = entry_price * (1 + current_margin) + cost_padding
                # 取优：Max(衰减价, 市场价)
                target_price = max(decay_price, tick.price)
            else:
                decay_price = entry_price / (1 + current_margin) - cost_padding
                # 取优：Min(衰减价, 市场价)
                target_price = min(decay_price, tick.price)

        # --- 阶段 2: 保本阶段 ---
        elif 10 < minutes_to_close <= 20:
            breakeven_price = (entry_price + cost_padding) if is_long else (entry_price - cost_padding)
            if is_long: target_price = max(breakeven_price, tick.price)
            else: target_price = min(breakeven_price, tick.price)

        # --- 阶段 3: 止损阶段 ---
        elif 3 < minutes_to_close <= 10:
            loss_limit = 0.20
            if is_long:
                stop_price = entry_price * (1 - loss_limit) + cost_padding
                target_price = max(stop_price, tick.price)
            else:
                stop_price = entry_price * (1 + loss_limit) - cost_padding
                target_price = min(stop_price, tick.price)

        # --- 阶段 4: 强平阶段 ---
        elif minutes_to_close <= 3:
            target_price = tick.price
            is_force_market = True 

        return target_price, is_force_market

    def modify_order(self, exchange, positions: Dict[str, Position], tick: TickEvent, active_orders: List[Order]) -> bool:
        """
        修改订单的接口占位符
        实际调用应由 Exchange 实现
        """
        position = positions.get(tick.contract_name)
        if not position or abs(position.size) < 0.001:
            return
        now = tick.timestamp

        # 获取属于本管理器的平仓单
        existing_order = self._find_exit_order(tick.contract_name, active_orders)

        minutes_to_close = self._get_minutes_to_close(tick.delivery_start, tick.timestamp)

        target_price, is_force_market = self._calculate_target_price(
            minutes_to_close, position, tick
        )
        target_price = round(target_price, 2)

        # 2. 定时调价 (每分钟)
        last_update = self.last_order_update_time.get(tick.contract_name)
        if (not last_update) or (now - last_update).total_seconds() >= 60:
            if existing_order and abs(existing_order.unit_price - target_price) > 0.05:
                if exchange.modify_order(existing_order.client_order_id, new_price=target_price):
                    self.last_order_update_time[tick.contract_name] = now
                    logger.info(f"调整平仓价 ({minutes_to_close:.1f}m left): {tick.contract_name} 价格->{target_price}")

    def _manage_exit_order(self, exchange, position: Position, tick: TickEvent, 
                           existing_order: Optional[Order], target_price: float, 
                           is_force_market: bool, minutes_to_close: float,
                           qty_mismatch: bool):
        
        now = tick.timestamp
        target_price = round(target_price, 2)
        
        # A. 强平阶段
        if is_force_market:
            if existing_order:
                exchange.cancel_order(existing_order.client_order_id)
            
            action = ActionType.SELL if position.size > 0 else ActionType.BUY
            signal = TradeSignal(
                timestamp=now,
                contract_name=tick.contract_name,
                contract_id=tick.contract_id,
                action=action,
                size=abs(position.size),
                price=tick.price,
                strategy_name="force_close_final",
                delivery_start=tick.delivery_start,
                open_strategy="force_close"
            )
            exchange.submit_order(signal)
            logger.info(f"触发收盘前强平: {tick.contract_name} {action} @ {tick.price}")
            return

        # B. 常规调整
        
        # 1. 数量不一致 (需要修改或新建)
        if qty_mismatch:
            new_qty = abs(position.size)
            if existing_order:
                # 修改现有平仓单
                if exchange.modify_order(existing_order.client_order_id, new_price=target_price, new_quantity=new_qty):
                    self.last_order_update_time[tick.contract_name] = now
                    logger.info(f"同步平仓单 (修改): {tick.contract_name} 数量->{new_qty}, 价格->{target_price}")
            else:
                # 新建平仓单
                self._submit_new_exit_order(exchange, position, tick, target_price, minutes_to_close)
            return

        

    def _submit_new_exit_order(self, exchange, position: Position, tick: TickEvent, target_price: float, minutes_to_close: float):
        action = ActionType.SELL if position.size > 0 else ActionType.BUY
        signal = TradeSignal(
            timestamp=tick.timestamp,
            contract_name=tick.contract_name,
            contract_id=tick.contract_id,
            action=action,
            size=abs(position.size),
            price=target_price,
            strategy_name="auto_profit_taking", # 必须以此开头，以便 _find_exit_order 识别
            delivery_start=tick.delivery_start,
            open_strategy="profit_taking"
        )
        if exchange.submit_order(signal):
            self.last_order_update_time[tick.contract_name] = tick.timestamp
            logger.info(f"挂出平仓单 ({minutes_to_close:.1f}m left): {tick.contract_name} {action} {abs(position.size)}MW @ {target_price}")

    def _find_exit_order(self, contract_name: str, orders: List[Order]) -> Optional[Order]:
        """
        寻找当前合约的活动平仓单
        【关键修改】必须过滤策略名称，只获取由 ExitManager 发起的订单 (auto_profit_taking 或 force_close)
        否则会错误地修改建仓单 (optimized_extreme_sell / super_mean_reversion_buy)
        """
        for order in orders:
            if order.contract_name == contract_name:
                if order.state in ["NEW", "PARTIALLY_FILLED"]:
                    # 检查策略名称前缀
                    # 我们的平仓策略名通常是 "auto_profit_taking", "auto_profit_taking_update", "auto_profit_taking_sync", "force_close_final"
                    if order.strategy.startswith("auto_profit_taking") or order.strategy.startswith("force_close"):
                        return order
        return None