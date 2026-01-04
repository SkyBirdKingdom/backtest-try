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
        
        # 【新增】成交所需的匹配次数 (默认3)
        self.execution_wait_trades = config.get('execution_wait_trades', 3)
        
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

    def _check_order_timeout(self):
        for order in list(self.active_orders):
            if not self.current_time or not order.timestamp:
                continue
            time_diff = (self.current_time - order.timestamp).total_seconds()
            # 注意：强平单不应该被超时撤销，但强平单通常是立刻成交的
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
            match_wait_count=0 # 初始化计数器
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
        """
        撮合逻辑：
        1. 检查价格是否满足条件。
        2. 如果满足，增加 match_wait_count。
        3. 如果 match_wait_count >= execution_wait_trades (默认3)，则成交。
        4. 强平单 (strategy='force_close_final') 无视此限制，立即成交。
        """
        # 使用副本遍历，因为可能会移除订单
        for order in list(self.active_orders):
            if order.contract_name != tick.contract_name:
                continue
            
            is_price_match = False
            
            # 1. 检查价格条件
            if order.side == "BUY":
                if tick.price <= order.unit_price:
                    is_price_match = True
            elif order.side == "SELL":
                if tick.price >= order.unit_price:
                    is_price_match = True

            # 2. 处理匹配逻辑
            if is_price_match:
                # 特殊通道：强平单立即成交，不需要排队
                if order.strategy == "force_close_final":
                    # 强平单以当前市场价成交，或者加上滑点
                    exec_price = self._calculate_exec_price(tick.price, order.side)
                    self._execute_trade(order, exec_price, tick)
                    self.active_orders.remove(order)
                    continue

                # 常规订单：增加计数器
                order.match_wait_count += 1
                
                # 3. 检查是否满足成交笔数限制
                if order.match_wait_count >= self.execution_wait_trades:
                    exec_price = self._calculate_exec_price(tick.price, order.side)
                    self._execute_trade(order, exec_price, tick)
                    self.active_orders.remove(order)
            else:
                # 如果价格不匹配，计数器是否重置？
                # 模拟逻辑：如果价格移走了，你还在订单簿里，但如果价格很久才回来，
                # 这里的简单模拟通常不需要重置，表示“只要累计有3笔成交在你的价格范围内”就轮到你了。
                # 如果想要更严苛（价格移走就重排），可以在这里 set match_wait_count = 0
                pass

    def _calculate_exec_price(self, market_price: float, side: str) -> float:
        """计算包含滑点的成交价"""
        if not self.slippage_enabled:
            return market_price
            
        if side == "BUY":
            return market_price * (1 + self.slippage_rate)
        else:
            return market_price * (1 - self.slippage_rate)

    def _execute_trade(self, order: Order, price: float, tick: TickEvent):
        self._trade_id_counter += 1
        trade_id = f"BK_TRD_{self._trade_id_counter}"
        
        contract_type = tick.contract_type
        is_qh = "QH" in contract_type or "QH" in order.contract_name
        fee_rate = (self.transaction_cost_rate / 4.0) if is_qh else self.transaction_cost_rate
        fee = fee_rate * order.quantity
            
        self.capital -= fee
        pnl = self._update_position(order, price, tick, is_qh)
        
        order.state = "FILLED"
        order.remaining_quantity = 0.0
        
        trade = Trade(
            timestamp=self.current_time,
            trade_id=trade_id,
            client_order_id=order.client_order_id,
            contract_name=order.contract_name,
            contract_id=order.contract_id,
            action=order.side,
            size=order.quantity,
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
        
        log_msg = f"成交: {order.contract_name} {order.side} {order.quantity:.1f}MW @ {price:.2f} (排队{self.execution_wait_trades}笔) | Fee: {fee:.4f}"
        if abs(pnl) > 0.0001:
            log_msg += f" | Realized PnL: {pnl:.2f}"
        logger.info(log_msg)

    def _update_position(self, order: Order, price: float, tick: TickEvent, is_qh: bool) -> float:
        key = order.contract_name 
        size_delta = order.quantity if order.side == "BUY" else -order.quantity
        realized_pnl = 0.0
        
        if key not in self.positions:
            pos = Position(
                contract_id=order.contract_id,
                contract_name=order.contract_name,
                size=0.0,
                avg_price=0.0,
                timestamp=self.current_time,
                delivery_start=order.delivery_start,
                strategy_name=order.strategy
            )
            self.positions[key] = pos

        pos = self.positions[key]
        old_size = pos.size
        new_size = round(old_size + size_delta, 1)
        
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
            
            if (old_size > 0 and new_size < 0) or (old_size < 0 and new_size > 0):
                pos.avg_price = price
                pos.strategy_name = order.strategy
        
        pos.timestamp = self.current_time
        if abs(pos.size) < 0.001:
            if key in self.positions:
                del self.positions[key]
                
        return realized_pnl