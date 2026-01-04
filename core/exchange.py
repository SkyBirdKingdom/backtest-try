import logging
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict

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
        
        # 撮合限制
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

    def cancel_order(self, client_order_id: str) -> bool:
        for order in self.active_orders:
            if order.client_order_id == client_order_id:
                order.state = "CANCELLED"
                order.action = "USER_CANCELLED"
                self.active_orders.remove(order)
                return True
        return False

    def modify_order(self, client_order_id: str, new_price: Optional[float] = None, new_quantity: Optional[float] = None) -> bool:
        """
        修改订单
        """
        for order in self.active_orders:
            if order.client_order_id == client_order_id:
                updated = False
                
                # 修改价格：通常视为新指令，重置排队计数
                if new_price is not None and abs(order.unit_price - new_price) > 0.001:
                    order.unit_price = new_price
                    order.match_wait_count = 0 
                    updated = True
                
                # 修改数量：更新总量和剩余量
                if new_quantity is not None and abs(order.remaining_quantity - new_quantity) > 0.001:
                    # 如果只是改数量且没改价格，是否重置排队？
                    # 严谨回测通常建议重置，或者如果是减量则不重置，增量则重置。
                    # 这里简化为：任何修改都重置排队，确保保守估计。
                    order.quantity = new_quantity
                    order.remaining_quantity = new_quantity
                    order.match_wait_count = 0
                    updated = True
                
                if updated:
                    order.action = "USER_MODIFIED"
                    return True
        return False

    def _check_order_timeout(self):
        for order in list(self.active_orders):
            if not self.current_time or not order.timestamp:
                continue
            time_diff = (self.current_time - order.timestamp).total_seconds()
            if time_diff > self.order_timeout_seconds:
                order.state = "CANCELLED"
                order.action = "SYSTEM_TIMEOUT"
                self.active_orders.remove(order)
                logger.info(f"订单超时撤单: {order.contract_name}")

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
            action="USER_ADDED",
            delivery_start=signal.delivery_start,
            strategy=signal.strategy_name,
            open_strategy=getattr(signal, 'open_strategy', ''),
            portfolio_id="BACKTEST_PORTFOLIO",
            delivery_area_id="16",
            order_type="LIMIT",
            match_wait_count=0 
        )
        
        self.active_orders.append(order)
        self.order_history.append(order)
        return True

    def get_positions(self) -> List[Position]:
        return [p for p in self.positions.values() if abs(p.size) > 0.001]

    def get_account_info(self) -> AccountInfo:
        return AccountInfo(
            timestamp=self.current_time,
            initial_capital=self.initial_capital,
            capital=self.capital,
            total_pnl=self.capital - self.initial_capital
        )
    
    def _match_orders(self, tick: TickEvent):
        for order in list(self.active_orders):
            if order.contract_name != tick.contract_name:
                continue
            
            # 延迟模拟
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
                        
                        if order.remaining_quantity <= 0.001:
                            self.active_orders.remove(order)
                        else:
                            order.state = "PARTIALLY_FILLED"

    def _calculate_exec_price(self, market_price: float, side: str) -> float:
        if not self.slippage_enabled: return market_price
        if side == "BUY": return market_price * (1 + self.slippage_rate)
        else: return market_price * (1 - self.slippage_rate)

    def _execute_trade(self, order: Order, execute_quantity: float, price: float, tick: TickEvent):
        self._trade_id_counter += 1
        trade_id = f"BK_TRD_{self._trade_id_counter}"
        
        contract_type = tick.contract_type
        is_qh = "QH" in contract_type or "QH" in order.contract_name
        fee_rate = (self.transaction_cost_rate / 4.0) if is_qh else self.transaction_cost_rate
        fee = fee_rate * execute_quantity
            
        self.capital -= fee
        pnl = self._update_position(order, execute_quantity, price, tick, is_qh)
        
        order.remaining_quantity = round(order.remaining_quantity - execute_quantity, 3)
        if order.remaining_quantity <= 0.001: order.state = "FILLED"
        else: order.state = "PARTIALLY_FILLED"
        
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
                  f"(剩余 {order.remaining_quantity:.2f}) | Fee: {fee:.4f}"
        if abs(pnl) > 0.0001:
            log_msg += f" | Realized PnL: {pnl:.2f}"
        logger.info(log_msg)

    def _update_position(self, order: Order, quantity: float, price: float, tick: TickEvent, is_qh: bool) -> float:
        """
        更新持仓并维护 initial_entry_time
        """
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
                initial_entry_time=self.current_time # 初始建仓时间
            )
            self.positions[key] = pos

        pos = self.positions[key]
        old_size = pos.size
        new_size = round(old_size + size_delta, 3) 
        
        # 1. 开仓/加仓 (同向)
        if (old_size == 0) or (old_size > 0 and size_delta > 0) or (old_size < 0 and size_delta < 0):
            total_val = abs(old_size) * pos.avg_price + abs(size_delta) * price
            if abs(new_size) > 0:
                pos.avg_price = total_val / abs(new_size)
            pos.size = new_size
            pos.strategy_name = order.strategy 
            
            # 【关键修正】凡是持仓绝对值增加（买入/做空加仓），都更新建仓时间
            # 这意味着止盈曲线会以这笔新的交易时间为起点重新计算衰减
            pos.initial_entry_time = self.current_time
            
        # 2. 平仓 (反向)
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
            
            # 如果发生了反手 (Pos正转负 或 负转正)
            if (old_size > 0 and new_size < 0) or (old_size < 0 and new_size > 0):
                pos.avg_price = price
                pos.strategy_name = order.strategy
                # 反手意味着新一轮持仓开始，更新时间
                pos.initial_entry_time = self.current_time
            else:
                # 只是部分平仓或完全平仓，不更新 initial_entry_time
                # 这样可以保证剩余持仓的止盈计算基准不变
                pass
        
        pos.timestamp = self.current_time
        if abs(pos.size) < 0.001:
            if key in self.positions:
                del self.positions[key]
                
        return realized_pnl