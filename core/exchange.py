import logging
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict
from dataclasses import replace

from core.models import Order, Position, Trade, AccountInfo, TradeSignal, TickEvent

logger = logging.getLogger("VirtualExchange")

class VirtualExchange:
    def __init__(self, initial_capital: float = 100000.0, config: dict = None):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        config = config or {}
        self.slippage_enabled = config.get('slippage_enabled', False)
        self.slippage_rate = config.get('slippage', 0.008) 
        self.transaction_cost_rate = config.get('transaction_cost', 0.23) 
        self.order_timeout_seconds = 300 
        
        self.execution_wait_trades = config.get('execution_wait_trades', 3)
        self.order_submission_delay = config.get('order_submission_delay', 30)
        
        self.current_time = None
        self.positions: Dict[str, Position] = {}
        
        self.active_orders: List[Order] = []
        self.trades: List[Trade] = []
        
        self._order_id_counter = 0
        self._trade_id_counter = 0
        
        self.order_history: List[Order] = [] 

    def on_tick(self, tick: TickEvent):
        self.current_time = tick.timestamp
        self._check_order_timeout()
        self._match_orders(tick)

    def _log_order_snapshot(self, order: Order, event_action: str = None):
        """记录快照"""
        # 创建副本
        snapshot = replace(order)
        # 更新快照时间为当前时间
        snapshot.timestamp = self.current_time
        if event_action:
            snapshot.action = event_action
        self.order_history.append(snapshot)

    def submit_order(self, signal: TradeSignal) -> bool:
        self._order_id_counter += 1
        internal_order_id = f"BK_ORD_{self._order_id_counter}"
        clean_qty = round(signal.size, 1)
        
        order = Order(
            timestamp=self.current_time,
            client_order_id=internal_order_id,
            contract_id=signal.contract_id,
            contract_name=signal.contract_name,
            side=signal.action.value,
            quantity=clean_qty,
            remaining_quantity=clean_qty,
            unit_price=signal.price,
            state="NEW",
            action="USER_SUBMIT",
            delivery_start=signal.delivery_start,
            strategy=signal.strategy_name,
            open_strategy=getattr(signal, 'open_strategy', ''),
            portfolio_id="BACKTEST_PORTFOLIO",
            delivery_area_id="16",
            order_type="LIMIT",
            match_wait_count=0,
            # 【新增】初始化序列号为 1
            event_sequence_no=1 
        )
        
        self.active_orders.append(order)
        # 记录第一条日志
        self._log_order_snapshot(order, "USER_SUBMIT")
        return True

    def modify_order(self, client_order_id: str, new_price: Optional[float] = None, new_quantity: Optional[float] = None) -> bool:
        for order in self.active_orders:
            if order.client_order_id == client_order_id:
                updated = False
                changes = []
                
                if new_price is not None and abs(order.unit_price - new_price) > 0.001:
                    order.unit_price = new_price
                    order.match_wait_count = 0 
                    updated = True
                    changes.append("PRICE")
                
                if new_quantity is not None and abs(order.remaining_quantity - new_quantity) > 0.001:
                    order.quantity = new_quantity
                    order.remaining_quantity = new_quantity
                    order.match_wait_count = 0 
                    updated = True
                    changes.append("QTY")
                
                if updated:
                    # 【新增】序列号 +1
                    order.event_sequence_no += 1
                    action_str = f"USER_MODIFIED_{'_'.join(changes)}"
                    self._log_order_snapshot(order, action_str)
                    return True
        return False

    def cancel_order(self, client_order_id: str) -> bool:
        for order in self.active_orders:
            if order.client_order_id == client_order_id:
                # 【新增】序列号 +1
                order.event_sequence_no += 1
                order.state = "CANCELLED"
                self._log_order_snapshot(order, "USER_CANCELLED")
                self.active_orders.remove(order)
                return True
        return False

    def _check_order_timeout(self):
        for order in list(self.active_orders):
            if not self.current_time or not order.timestamp:
                continue
            time_diff = (self.current_time - order.timestamp).total_seconds()
            if time_diff > self.order_timeout_seconds:
                # 【新增】序列号 +1
                order.event_sequence_no += 1
                order.state = "CANCELLED"
                self._log_order_snapshot(order, "SYSTEM_TIMEOUT")
                self.active_orders.remove(order)
                logger.info(f"订单超时撤单: {order.contract_name}, 剩余: {order.remaining_quantity}")

    def _match_orders(self, tick: TickEvent):
        for order in list(self.active_orders):
            if order.contract_name != tick.contract_name:
                continue
            
            time_since_creation = (tick.timestamp - order.timestamp).total_seconds()
            if time_since_creation < self.order_submission_delay:
                continue

            is_price_match = False
            if order.side == "BUY":
                if tick.price <= order.unit_price: is_price_match = True
            elif order.side == "SELL":
                if tick.price >= order.unit_price: is_price_match = True

            if is_price_match:
                if order.strategy == "force_close_final":
                    exec_price = self._calculate_exec_price(tick.price, order.side)
                    self._execute_trade(order, order.remaining_quantity, exec_price, tick)
                    self.active_orders.remove(order)
                    continue

                order.match_wait_count += 1
                
                if order.match_wait_count >= self.execution_wait_trades:
                    available_volume = tick.volume if tick.volume > 0 else 0.0
                    fill_qty = min(order.remaining_quantity, available_volume)
                    
                    if fill_qty > 0.001: 
                        exec_price = self._calculate_exec_price(tick.price, order.side)
                        self._execute_trade(order, fill_qty, exec_price, tick)
                        
                        # 检查剩余量，如果已经 FULL_FILLED，则移除
                        if order.remaining_quantity <= 0.001:
                            self.active_orders.remove(order)

    def _calculate_exec_price(self, market_price: float, side: str) -> float:
        if not self.slippage_enabled: return market_price
        if side == "BUY": return market_price * (1 + self.slippage_rate)
        else: return market_price * (1 - self.slippage_rate)

    def _execute_trade(self, order: Order, execute_quantity: float, price: float, tick: TickEvent):
        """执行成交"""
        self._trade_id_counter += 1
        trade_id = f"BK_TRD_{self._trade_id_counter}"
        
        contract_type = tick.contract_type
        is_qh = "QH" in contract_type or "QH" in order.contract_name
        fee_rate = (self.transaction_cost_rate / 4.0) if is_qh else self.transaction_cost_rate
        fee = fee_rate * execute_quantity
            
        self.capital -= fee
        pnl = self._update_position(order, execute_quantity, price, tick, is_qh)
        
        # 更新数量
        new_remaining = order.remaining_quantity - execute_quantity
        
        # 【新增】序列号 +1 (每次成交都是一个新的事件)
        order.event_sequence_no += 1
        
        # 严格判断是否完全成交
        if new_remaining <= 0.001:
            order.remaining_quantity = 0.0
            order.state = "FULL_FILLED"  # 【修正】明确的终态
            action_log = "SYSTEM_FILLED"
        else:
            order.remaining_quantity = round(new_remaining, 3)
            order.state = "PARTIALLY_FILLED"
            action_log = "SYSTEM_PARTIAL_FILL"
        
        # 【关键】记录快照
        self._log_order_snapshot(order, action_log)
        
        trade = Trade(
            timestamp=self.current_time,
            trade_id=trade_id,
            client_order_id=order.client_order_id,
            contract_name=order.contract_name,
            contract_id=order.contract_id,
            action=order.side,
            size=execute_quantity, 
            price=price,
            strategy=order.strategy,
            delivery_start=order.delivery_start,
            pnl=pnl, 
            trade_time=self.current_time.strftime("%Y-%m-%d %H:%M:%S"),
            open_strategy=order.open_strategy,
            delivery_end="",
            delivery_area=16,
            portfolio_id=order.portfolio_id
        )
        self.trades.append(trade)
        
        log_msg = f"成交: {order.contract_name} {order.side} {execute_quantity:.2f}MW @ {price:.2f} " \
                  f"(剩余 {order.remaining_quantity:.2f}, Seq={order.event_sequence_no}) | Fee: {fee:.4f}"
        if abs(pnl) > 0.0001:
            log_msg += f" | Realized PnL: {pnl:.2f}"
        logger.info(log_msg)

    def _update_position(self, order: Order, quantity: float, price: float, tick: TickEvent, is_qh: bool) -> float:
        # _update_position 保持不变，已在上一轮修正过 initial_entry_time 逻辑
        key = order.contract_name 
        size_delta = quantity if order.side == "BUY" else -quantity
        realized_pnl = 0.0
        
        if key not in self.positions:
            pos = Position(
                contract_id=order.contract_id,
                contract_name=order.contract_name,
                size=0.0,
                avg_price=0.0,
                timestamp=self.current_time,
                delivery_start=order.delivery_start,
                strategy_name=order.strategy,
                initial_entry_time=self.current_time 
            )
            self.positions[key] = pos

        pos = self.positions[key]
        old_size = pos.size
        new_size = round(old_size + size_delta, 3) 
        
        is_increase = abs(new_size) > abs(old_size)
        is_reversal = (old_size > 0 and new_size < 0) or (old_size < 0 and new_size > 0)
        
        if is_reversal or is_increase:
            pos.initial_entry_time = self.current_time
            
        if (old_size == 0) or (old_size > 0 and size_delta > 0) or (old_size < 0 and size_delta < 0):
            total_val = abs(old_size) * pos.avg_price + abs(size_delta) * price
            if abs(new_size) > 0:
                pos.avg_price = total_val / abs(new_size)
            pos.size = new_size
            pos.strategy_name = order.strategy 
            
        elif (old_size > 0 and size_delta < 0) or (old_size < 0 and size_delta > 0):
            closed_qty = min(abs(old_size), abs(size_delta))
            raw_pnl = 0.0
            if old_size > 0: 
                raw_pnl = (price - pos.avg_price) * closed_qty
            else: 
                raw_pnl = (pos.avg_price - price) * closed_qty
            
            if is_qh: realized_pnl = raw_pnl / 4.0
            else: realized_pnl = raw_pnl
            
            self.capital += realized_pnl
            pos.size = new_size
            
            if is_reversal:
                pos.avg_price = price
                pos.strategy_name = order.strategy
        
        pos.timestamp = self.current_time
        if abs(pos.size) < 0.001:
            if key in self.positions:
                del self.positions[key]
                
        return realized_pnl

    def get_positions(self) -> List[Position]:
        return [p for p in self.positions.values() if abs(p.size) > 0.001]

    def get_account_info(self) -> AccountInfo:
        return AccountInfo(
            timestamp=self.current_time,
            initial_capital=self.initial_capital,
            capital=self.capital,
            total_pnl=self.capital - self.initial_capital
        )