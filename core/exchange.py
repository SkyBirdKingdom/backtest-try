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
        
        # --- 核心撮合限制 ---
        # 1. 成交所需的匹配次数 (模拟排队深度)
        self.execution_wait_trades = config.get('execution_wait_trades', 3)
        # 2. 订单提交延迟秒数 (模拟数据延迟/网络耗时)
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

    def _check_order_timeout(self):
        for order in list(self.active_orders):
            if not self.current_time or not order.timestamp:
                continue
            # 只有非部分成交的订单才检查超时？
            # 或者部分成交后，剩余部分也应该有超时限制？
            # 按照实盘逻辑，通常是订单挂出后多久没“完全成交”就撤。
            # 这里保持原逻辑：针对整个订单的时间检查。
            time_diff = (self.current_time - order.timestamp).total_seconds()
            if time_diff > self.order_timeout_seconds:
                order.state = "CANCELLED"
                order.action = "SYSTEM_TIMEOUT"
                self.active_orders.remove(order)
                logger.info(f"订单超时撤单: {order.contract_name}, 剩余: {order.remaining_quantity}")

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
            remaining_quantity=clean_qty, # 初始化剩余量
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
        """
        撮合逻辑 (含延迟 + 排队 + 流动性限制)
        """
        for order in list(self.active_orders):
            if order.contract_name != tick.contract_name:
                continue
            
            # --- 1. 延迟模拟 ---
            time_since_creation = (tick.timestamp - order.timestamp).total_seconds()
            if time_since_creation < self.order_submission_delay:
                continue

            is_price_match = False
            
            # 2. 价格条件检查
            if order.side == "BUY":
                if tick.price <= order.unit_price:
                    is_price_match = True
            elif order.side == "SELL":
                if tick.price >= order.unit_price:
                    is_price_match = True

            # 3. 撮合处理
            if is_price_match:
                # A. 强平单：无视排队，无视流动性(假设必定成交)，立即全额成交
                if order.strategy == "force_close_final":
                    exec_price = self._calculate_exec_price(tick.price, order.side)
                    # 强平全额成交
                    self._execute_trade(order, order.remaining_quantity, exec_price, tick)
                    self.active_orders.remove(order)
                    continue

                # B. 常规订单：增加排队计数器
                order.match_wait_count += 1
                
                # 检查是否满足排队次数
                if order.match_wait_count >= self.execution_wait_trades:
                    # --- 4. 流动性检查 (Liquidity Check) ---
                    # 能够成交的数量 = min(订单剩余, 当前Tick成交量)
                    # 如果 Tick 没有量 (0 volume)，则无法成交
                    
                    available_volume = tick.volume if tick.volume > 0 else 0.0
                    fill_qty = min(order.remaining_quantity, available_volume)
                    
                    if fill_qty > 0.001: # 只有量足够才成交
                        exec_price = self._calculate_exec_price(tick.price, order.side)
                        
                        # 执行部分或全部成交
                        self._execute_trade(order, fill_qty, exec_price, tick)
                        
                        # 检查是否完全成交
                        if order.remaining_quantity <= 0.001:
                            self.active_orders.remove(order)
                        else:
                            # 留在列表中，状态更新为部分成交
                            order.state = "PARTIALLY_FILLED"
                            # logger.info(f"部分成交: {order.contract_name} 剩余 {order.remaining_quantity:.1f}")

    def _calculate_exec_price(self, market_price: float, side: str) -> float:
        if not self.slippage_enabled:
            return market_price
        if side == "BUY":
            return market_price * (1 + self.slippage_rate)
        else:
            return market_price * (1 - self.slippage_rate)

    def _execute_trade(self, order: Order, execute_quantity: float, price: float, tick: TickEvent):
        """
        执行具体的成交逻辑 (支持部分成交)
        """
        self._trade_id_counter += 1
        trade_id = f"BK_TRD_{self._trade_id_counter}"
        
        contract_type = tick.contract_type
        is_qh = "QH" in contract_type or "QH" in order.contract_name
        fee_rate = (self.transaction_cost_rate / 4.0) if is_qh else self.transaction_cost_rate
        
        # 计算本次成交的费用
        fee = fee_rate * execute_quantity
            
        self.capital -= fee
        
        # 更新持仓 (返回已实现的 PnL)
        pnl = self._update_position(order, execute_quantity, price, tick, is_qh)
        
        # 更新订单剩余量
        order.remaining_quantity = round(order.remaining_quantity - execute_quantity, 3)
        
        if order.remaining_quantity <= 0.001:
            order.state = "FILLED"
        else:
            order.state = "PARTIALLY_FILLED"
        
        # 生成成交记录
        trade = Trade(
            timestamp=self.current_time,
            trade_id=trade_id,
            client_order_id=order.client_order_id,
            contract_name=order.contract_name,
            contract_id=order.contract_id,
            action=order.side,
            size=execute_quantity, # 记录本次实际成交量
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
        更新持仓
        注意：这里的 quantity 是本次成交的数量，不是订单总量
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
                strategy_name=order.strategy
            )
            self.positions[key] = pos

        pos = self.positions[key]
        old_size = pos.size
        new_size = round(old_size + size_delta, 3) # 提高一点精度处理部分成交
        
        # 1. 开仓 (同向)
        if (old_size == 0) or (old_size > 0 and size_delta > 0) or (old_size < 0 and size_delta < 0):
            total_val = abs(old_size) * pos.avg_price + abs(size_delta) * price
            if abs(new_size) > 0:
                pos.avg_price = total_val / abs(new_size)
            pos.size = new_size
            pos.strategy_name = order.strategy 
            
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
            
            # 如果平仓后反手了 (非常少见，但逻辑上要支持)
            if (old_size > 0 and new_size < 0) or (old_size < 0 and new_size > 0):
                pos.avg_price = price
                pos.strategy_name = order.strategy
        
        pos.timestamp = self.current_time
        if abs(pos.size) < 0.001:
            if key in self.positions:
                del self.positions[key]
                
        return realized_pnl