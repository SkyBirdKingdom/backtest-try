import logging
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict
from dataclasses import replace

from core.models import Order, Position, Trade, AccountInfo, TradeSignal, TickEvent, SettlementEvent

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

    # ----------------------------------------------------------------
    # 【核心修改】交割结算逻辑 (关闸清理)
    # ----------------------------------------------------------------
    def settle_expired_positions(self, current_delivery_date: datetime.date) -> List[SettlementEvent]:
        """
        清理所有交付日期小于当前日期的持仓和挂单（模拟关闸/交割）
        """
        # 1. 找出所有过期的合约名称 (包括有持仓的和有挂单的)
        expired_contracts = set()
        settlement_events = []
        
        # 检查持仓
        for contract_name, pos in self.positions.items():
            if pos.delivery_start.date() < current_delivery_date:
                expired_contracts.add(contract_name)
        
        # 检查挂单
        for order in self.active_orders:
            # 只有当 delivery_start 有效时才检查
            if order.delivery_start and order.delivery_start.date() < current_delivery_date:
                expired_contracts.add(order.contract_name)
        
        if not expired_contracts:
            return []

        logger.info(f"⚡ 开始结算过期合约 (关闸清理): {len(expired_contracts)} 个合约")

        for contract_name in expired_contracts:
            # A. 撤销该合约所有未完成的订单
            # 我们不能直接在遍历 active_orders 时移除，要先收集再处理
            orders_to_cancel = [o for o in self.active_orders if o.contract_name == contract_name]
            
            for order in orders_to_cancel:
                order.state = "CANCELLED"
                order.event_sequence_no += 1
                # 【关键】记录一条系统撤单快照，方便回溯
                self._log_order_snapshot(order, "SYSTEM_SETTLEMENT_CANCEL")
                self.active_orders.remove(order)
            
            if orders_to_cancel:
                logger.info(f"  - [{contract_name}] 自动撤单: {len(orders_to_cancel)} 笔")

            # B. 强制平仓/移除持仓
            if contract_name in self.positions:
                pos = self.positions.pop(contract_name)
                
                # 如果持仓不为0，这是一个异常情况，需要详细记录以便分析
                if abs(pos.size) > 0.001:
                    logger.error(
                        f"⚠️ [异常未平仓] 合约: {contract_name} | "
                        f"方向: {'多' if pos.size > 0 else '空'} | "
                        f"数量: {pos.size} MW | "
                        f"成本价: {pos.avg_price:.2f} | "
                        f"建仓时间: {pos.initial_entry_time} | "
                        f"原因分析: 关闸时未能成功平仓，已被强制移除(归零)。"
                    )
                    event = SettlementEvent(
                        timestamp=self.current_time if self.current_time else datetime.now(),
                        contract_name=contract_name,
                        contract_id=pos.contract_id,
                        size=pos.size,
                        avg_price=pos.avg_price,
                        reason="EXPIRED_FORCE_CLOSE"
                    )
                    settlement_events.append(event)
                    # 注意：这里我们直接移除了持仓，没有计算最后一笔盈亏(因为没有结算价)
                    # 资金曲线会维持在最后一次成交的状态。这符合“异常情况”的处理逻辑。
                else:
                    logger.info(f"  - [{contract_name}] 正常交割 (持仓已归零)")
        return settlement_events

    # ----------------------------------------------------------------
    # 基础功能
    # ----------------------------------------------------------------

    def _log_order_snapshot(self, order: Order, event_action: str = None):
        snapshot = replace(order)
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
            event_sequence_no=1 
        )
        
        self.active_orders.append(order)
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
                    if order.remaining_quantity < order.quantity and new_quantity > order.remaining_quantity:
                        logger.warning(f"⚠️ 订单 {order.client_order_id} 在部分成交后被重置数量: {order.remaining_quantity} -> {new_quantity}")
                    order.quantity = new_quantity
                    order.remaining_quantity = new_quantity
                    order.match_wait_count = 0 
                    updated = True
                    changes.append("QTY")
                
                if updated:
                    order.event_sequence_no += 1
                    action_str = f"USER_MODIFIED_{'_'.join(changes)}"
                    self._log_order_snapshot(order, action_str)
                    return True
        return False

    def cancel_order(self, client_order_id: str) -> bool:
        for order in self.active_orders:
            if order.client_order_id == client_order_id:
                order.event_sequence_no += 1
                order.state = "CANCELLED"
                self._log_order_snapshot(order, "USER_CANCELLED")
                self.active_orders.remove(order)
                return True
        return False

    def _check_order_timeout(self):
        for order in list(self.active_orders):
            if not self.current_time or not order.timestamp or order.strategy == "auto_profit_taking" or order.strategy.startswith("force_close") or order.strategy == "consecutive_loss_stop":
                continue
            time_diff = (self.current_time - order.timestamp).total_seconds()
            if time_diff > self.order_timeout_seconds:
                order.event_sequence_no += 1
                order.state = "CANCELLED"
                self._log_order_snapshot(order, "SYSTEM_TIMEOUT")
                self.active_orders.remove(order)
                logger.info(f"订单超时撤单: {order.contract_name}, 剩余: {order.remaining_quantity}")

    def _match_orders(self, tick: TickEvent):
        for order in list(self.active_orders):
            if order.contract_name != tick.contract_name:
                continue

            # 订单提交增加延迟
            time_since_creation = (tick.timestamp - order.timestamp).total_seconds()
            if time_since_creation < self.order_submission_delay:
                continue

            # 【核心修改】如果是强平单，无视价格限制，强制成交 (Market Order)
            is_force_close = (order.strategy == "force_close_final")
            is_price_match = False

            if is_force_close:
                is_price_match = True # 强平单始终匹配
            else:
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
        
        new_remaining = order.remaining_quantity - execute_quantity
        order.event_sequence_no += 1
        
        if new_remaining <= 0.001:
            order.remaining_quantity = 0.0
            order.state = "FULL_FILLED"
            action_log = "SYSTEM_FILLED"
        else:
            order.remaining_quantity = round(new_remaining, 3)
            order.state = "PARTIALLY_FILLED"
            action_log = "SYSTEM_PARTIAL_FILL"
        
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
                open_strategy=order.open_strategy,
                initial_entry_time=self.current_time 
            )
            self.positions[key] = pos

        pos = self.positions[key]
        old_size = pos.size
        new_size = round(old_size + size_delta, 1) 
        
        is_increase = abs(new_size) > abs(old_size)
        is_reversal = (old_size > 0 and new_size < 0) or (old_size < 0 and new_size > 0)
        
        # 1. 初始建仓时间
        if is_reversal or (old_size == 0 and abs(new_size) > 0):
            pos.initial_entry_time = self.current_time
            # 反手时，重置所有状态
            pos.has_triggered_2nd_add = False
            pos.has_reversed = False 
            # 如果这是一个反手策略的成交，标记它已经反手过了
            if "reversal" in order.strategy:
                pos.has_reversed = True

        # 2. 【核心】检测二次加仓 (用于严格模式止损)
        # 如果是加仓行为，且当前不是空仓，且不是反手
        if is_increase and old_size != 0 and not is_reversal:
            # 这里简单处理：只要发生过加仓，就认为触发了
            pos.has_triggered_2nd_add = True
            logger.info(f"[{key}] 触发加仓标记 (Old: {old_size} -> New: {new_size})")
            
        # 3. 成本计算 (加权平均)
        if (old_size == 0) or (old_size > 0 and size_delta > 0) or (old_size < 0 and size_delta < 0):
            # 加仓 or 建仓
            total_val = abs(old_size) * pos.avg_price + abs(size_delta) * price
            if abs(new_size) > 0:
                pos.avg_price = total_val / abs(new_size)
            pos.size = new_size
            # 只有当持仓反转或从0开始时才更新策略名，加仓不改变主要策略名
            if old_size == 0 or is_reversal:
                pos.strategy_name = order.strategy 
                pos.open_strategy = order.open_strategy
            
        elif (old_size > 0 and size_delta < 0) or (old_size < 0 and size_delta > 0):
            # 减仓 or 平仓
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