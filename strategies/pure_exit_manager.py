import logging
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from collections import deque
from scipy.stats import linregress

from core.exchange import VirtualExchange
from core.models import TickEvent, TradeSignal, ActionType, Position, Order

logger = logging.getLogger("PureExitManager")

class PureExitManager:
    """
    纯净版平仓管理器 (Lifecycle Manager) - 增强版 V3.0
    1. 整合连续亏损止损逻辑：检测 `stop_loss_triggered` 标志，激进改价。
    2. 反手策略止盈逻辑：30分钟趋势 + 盈利检查。
    3. 少亏/强平逻辑更新：引入 Avg(10 Ticks) 价格计算。
    """
    def __init__(self, config: dict):
        self.config = config
        self.transaction_cost = config.get('transaction_cost', 0.23)
        self.params = config.get('strategy_params', {})
        self.forbid_new_open_minutes = int(self.params.get('forbid_new_open_minutes', 60))
        self.take_profit_end_minutes = int(self.params.get('take_profit_end_minutes', 10))
        self.breakeven_end_minutes = int(self.params.get('breakeven_end_minutes', 6))
        self.stop_loss_end_minutes = int(self.params.get('stop_loss_end_minutes', 3))
        self.last_order_update_time: Dict[str, datetime] = {}
        
        # 【新增】Tick 历史记录 (用于计算最近10个Tick的均价)
        self.tick_history: Dict[str, deque] = {} 

    def process(self, tick: TickEvent, positions: Dict[str, Position], 
                active_orders: List[Order], exchange, bars: List[dict] = None) -> None:
        if not tick.delivery_start:
            return

        minutes_to_close = self._get_minutes_to_close(tick.delivery_start, tick.timestamp)
        
        # 1. 维护 Tick History (用于少亏阶段均价计算)
        if tick.contract_name not in self.tick_history:
            self.tick_history[tick.contract_name] = deque(maxlen=10)
        self.tick_history[tick.contract_name].append(tick.price)

        position = positions.get(tick.contract_name)
        if not position or abs(position.size) < 0.001:
            existing_exit_order = self._find_exit_order(tick.contract_name, active_orders, include_all=True)
            if existing_exit_order:
                exchange.cancel_order(existing_exit_order.client_order_id)
                logger.info(f"🧹 清理幽灵平仓单: {tick.contract_name} (持仓已归零)")
            return

        # ---------------------------------------------------------------------
        # 【核心逻辑 A】反手策略特殊止盈 (Reverse Strategy Profit Taking)
        # ---------------------------------------------------------------------
        # 如果是反手策略产生的持仓，且在进入少亏阶段之前
        if position.open_strategy == "trend_reversal" and minutes_to_close > self.breakeven_end_minutes:
            if self._check_reverse_profit_exit(tick, position, bars):
                # 如果满足反手止盈条件，直接以当前价挂单/改单
                self._submit_or_modify_reverse_exit(exchange, position, tick, active_orders)
                return

        # ---------------------------------------------------------------------
        # 【核心逻辑 B】止损单接管逻辑 (Stop Loss Chasing)
        # ---------------------------------------------------------------------
        # 如果 Strategy 标记了止损触发，ExitManager 接管所有改价逻辑
        if position.stop_loss_triggered:
            self._handle_stop_loss_chasing(exchange, position, tick, active_orders, minutes_to_close)
            # 止损接管后，不再执行后续常规生命周期
            return

        # ---------------------------------------------------------------------
        # 【核心逻辑 C】常规生命周期 (Profit -> Breakeven -> Reduce Loss -> Force)
        # ---------------------------------------------------------------------
        
        # 0. 关闸前撤销非平仓单
        if minutes_to_close <= self.forbid_new_open_minutes: 
            for order in list(active_orders): 
                if order.contract_name == tick.contract_name:
                    is_exit_strategy = (order.strategy.startswith("auto_profit") or 
                                        order.strategy.startswith("force_close") or
                                        order.strategy.startswith("stop_loss") or 
                                        order.strategy.startswith("exit_"))
                    # is_reversal_strategy = order.strategy.startswith("trend_reversal")
                    if not is_exit_strategy:
                        exchange.cancel_order(order.client_order_id)
                        logger.info(f"🛑 [禁区风控] 强制撤销残留开仓单: {order.client_order_id}")
        
        # if minutes_to_close > 240 or minutes_to_close <= 0:
        #     return
        # FIXED: 放宽时间窗口限制，允许所有时间点的数据进入回测
        if minutes_to_close <= 0:
            return

        # 获取属于本管理器的平仓单
        existing_exit_order = self._find_exit_order(tick.contract_name, active_orders)

        # 1. 检测数量是否一致 (Sync Check)
        qty_mismatch = False
        side_mismatch = False
        if existing_exit_order:
            if abs(abs(position.size) - existing_exit_order.remaining_quantity) > 0.001:
                qty_mismatch = True
            
            # 2. 【核心修复】检查方向差异
            # 如果我是多头(size>0)，我需要卖出(SELL)平仓。如果订单是BUY，说明方向反了（可能是反手成交导致的）
            expected_side = "SELL" if position.size > 0 else "BUY"
            if existing_exit_order.side != expected_side:
                side_mismatch = True
        elif not existing_exit_order:
            qty_mismatch = True

        # 2. 计算目标价格
        target_price, is_force_market = self._calculate_target_price(
            minutes_to_close, position, tick
        )

        # 3. 执行管理
        self._manage_exit_order(
            exchange, position, tick, existing_exit_order, 
            target_price, is_force_market, minutes_to_close, qty_mismatch, side_mismatch
        )

    # ----------------------------------------------------------------
    # 辅助逻辑实现
    # ----------------------------------------------------------------

    def _handle_stop_loss_chasing(self, exchange, position: Position, tick: TickEvent, active_orders: List[Order], minutes_to_close: float):
        """
        处理触发止损后的激进改价
        规则：持续修改价格，不撤销，直到成交。
        """
        target_price = tick.price
        
        # 如果进入了强平阶段，强制市价
        if minutes_to_close <= self.stop_loss_end_minutes:
            is_force_market = True
        else:
            is_force_market = False
            
        # 查找现有订单 (任意类型的平仓单：auto_profit 或 consecutive_loss_stop 或 trend_reversal)
        existing_order = None
        for order in active_orders:
            if order.contract_name == tick.contract_name and \
               (order.strategy.startswith("auto_profit") or order.strategy.startswith("exit_") or order.strategy.startswith("consecutive_loss") or order.strategy.startswith("stop_loss")):
                existing_order = order
                break
        # 【核心修复】方向校验：如果因为反手成交导致持仓方向变了，旧止损单就是毒药，必须撤销
        if existing_order:
            expected_side = "SELL" if position.size > 0 else "BUY"
            if existing_order.side != expected_side:
                exchange.cancel_order(existing_order.client_order_id)
                logger.warning(f"⚠️ [止损修正] 仓位反转/归零，撤销旧方向平仓单: {existing_order.client_order_id}")
                return # 撤销后本轮结束，下一轮如果没有订单且有持仓会重新建单
        
        # 【核心修改】如果是强平阶段，升级订单策略
        if is_force_market:
            if existing_order:
                # 升级现有订单为强平单，不需要撤单
                if existing_order.strategy != "force_close_final":
                    exchange.modify_order(existing_order.client_order_id, new_price=tick.price, new_strategy="force_close_final")
                    logger.info(f"🚨 [止损转强平] 升级订单为强平单: {existing_order.client_order_id}")
            else:
                self._submit_force_close(exchange, position, tick)
            return

        if existing_order:
            # 只有价格偏离时才修改，避免过于频繁
            if abs(existing_order.unit_price - target_price) > 0.01:
                exchange.modify_order(existing_order.client_order_id, new_price=target_price)
                logger.info(f"🚀 [止损追价] {tick.contract_name} 调整价格 -> {target_price}")
        else:
            # 如果没有订单，新建一个
            if abs(position.size) > 0.001:
                self._submit_new_exit_order(exchange, position, tick, target_price, minutes_to_close, strategy_name="consecutive_loss_stop")

    def _check_reverse_profit_exit(self, tick: TickEvent, position: Position, bars: List[dict]) -> bool:
        """
        反手策略止盈检查：
        1. 取最近30分钟(不含当前)的近10个bar
        2. 趋势改变 & 且置信度 > 0.4
        3. 盈利 > 0 (含手续费)
        """
        if not bars: return False
        
        # 1. 准备数据
        cutoff_time = tick.timestamp - timedelta(minutes=30)
        # 排除当前正在生成的Bar (通常 Engine 传进来的是已完成的 Bars, 但为了保险起见，取截止到上一分钟)
        current_minute = tick.timestamp.replace(second=0, microsecond=0)
        
        valid_bars = [b for b in bars if b['start_time'] >= cutoff_time and b['start_time'] < current_minute]
        if len(valid_bars) > 10:
            valid_bars = valid_bars[-10:] # 取最近10个
            
        prices = [float(b['close']) for b in valid_bars]
        if len(prices) < 3: return False
        
        # 2. 趋势计算
        trend_res = self._detect_trend(prices)
        confidence = trend_res['confidence']
        trend = trend_res['trend']
        
        # 判断趋势是否反转 (对于反手策略来说，我们希望顺势。如果趋势反转了，就该跑了)
        # 比如：反手是做空，如果趋势变成上升，且置信度高，则平仓
        should_exit_trend = False
        if position.size > 0: # 当前持多
            if trend == "下降" and confidence > 0.4: should_exit_trend = True
        else: # 当前持空
            if trend == "上升" and confidence > 0.4: should_exit_trend = True
            
        if not should_exit_trend:
            return False
            
        # 3. 盈利检查
        cost = position.avg_price
        fee = self.transaction_cost * 2 # 双边
        is_profitable = False
        
        if position.size > 0:
            if tick.price > (cost + fee): is_profitable = True
        else:
            if tick.price < (cost - fee): is_profitable = True
            
        return is_profitable

    def _submit_or_modify_reverse_exit(self, exchange, position: Position, tick: TickEvent, active_orders: List[Order]):
        target_price = tick.price
        
        # 查找现有订单
        existing = self._find_exit_order(tick.contract_name, active_orders)
        
        if existing:
            if abs(existing.unit_price - target_price) > 0.01:
                exchange.modify_order(existing.client_order_id, new_price=target_price)
                logger.info(f"🔄 [反手止盈] 更新价格 {tick.contract_name} -> {target_price}")
        else:
            self._submit_new_exit_order(exchange, position, tick, target_price, 999, strategy_name="auto_profit_taking_reverse")

    def _detect_trend(self, prices: List[float]) -> Dict:
        """简易线性回归 (复制自 Strategy 以避免循环引用)"""
        x = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = linregress(x, prices)
        r_squared = r_value ** 2
        
        if abs(slope) < 0.1: trend = "平滑"
        elif slope > 0.1: trend = "上升"
        else: trend = "下降"
        
        # 简化的置信度计算
        confidence = r_squared
        if len(prices) < 5: confidence *= 0.5
        
        return {"trend": trend, "confidence": confidence}

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
        fee_rate = self.transaction_cost 
        cost_padding = 2 * fee_rate      
        
        is_long = position.size > 0
        target_price = tick.price
        is_force_market = False

        # --- 阶段 1: 止盈阶段 ---
        if self.take_profit_end_minutes < minutes_to_close:
            start_time = position.last_size_change_time if position.last_size_change_time else position.timestamp
            start_minutes_to_close = self._get_minutes_to_close(tick.delivery_start, start_time)
            
            if start_minutes_to_close <= self.take_profit_end_minutes:
                progress = 1.0
            else:
                total_duration = start_minutes_to_close - self.take_profit_end_minutes
                elapsed = start_minutes_to_close - minutes_to_close
                if total_duration <= 0.001:
                    progress = 1.0
                else:
                    progress = elapsed / total_duration
                progress = max(0.0, min(1.0, progress))
            
            start_margin = 0.50 if entry_price < 50 else 0.30
            end_margin = 0.01
            current_margin = start_margin - (start_margin - end_margin) * progress
            
            decay_price = 0.0
            if is_long:
                decay_price = entry_price * (1 + current_margin) + cost_padding
                if entry_price < 0:
                    decay_price = entry_price + abs(entry_price) * current_margin + cost_padding
                
                target_price = max(decay_price, tick.price)
            else:
                decay_price = entry_price / (1 + current_margin) - cost_padding
                if entry_price < 0:
                    decay_price = entry_price - abs(entry_price) * current_margin - cost_padding
                target_price = min(decay_price, tick.price)

        # --- 阶段 2: 保本阶段 ---
        elif self.breakeven_end_minutes < minutes_to_close <= self.take_profit_end_minutes:
            breakeven_price = (entry_price + cost_padding) if is_long else (entry_price - cost_padding)
            if is_long: target_price = max(breakeven_price, tick.price)
            else: target_price = min(breakeven_price, tick.price)

        # --- 阶段 3: 少亏阶段 (更新逻辑) ---
        elif self.stop_loss_end_minutes < minutes_to_close <= self.breakeven_end_minutes:
            # 计算最近10个Tick的均价
            ticks = list(self.tick_history.get(tick.contract_name, []))
            avg_10 = sum(ticks) / len(ticks) if ticks else tick.price
            
            if is_long:
                # min(avg(最近10个tick), 最新tick - 0.01)
                target_price = min(avg_10, tick.price - 0.01)
            else:
                # max(avg(最近10个tick), 最新tick + 0.01)
                target_price = max(avg_10, tick.price + 0.01)

        # --- 阶段 4: 强平阶段 ---
        elif minutes_to_close <= self.stop_loss_end_minutes:
            target_price = tick.price
            is_force_market = True 

        return target_price, is_force_market

    def modify_order(self, exchange, positions: Dict[str, Position], tick: TickEvent, active_orders: List[Order]) -> bool:
        """
        修改订单的接口占位符 (在Engine中被调用)
        """
        # 注意：这里的逻辑已经集成到了 process 中，这里留空或用于简单的定时更新
        pass

    def _manage_exit_order(self, exchange, position: Position, tick: TickEvent, 
                           existing_order: Optional[Order], target_price: float, 
                           is_force_market: bool, minutes_to_close: float,
                           qty_mismatch: bool, side_mismatch: bool):
        
        now = tick.timestamp
        target_price = round(target_price, 2)

        # --- 【新增】确定当前的平仓策略标签 ---
        current_strategy_name = "exit_unknown"
        if is_force_market:
            current_strategy_name = "exit_force_close"
        elif self.take_profit_end_minutes < minutes_to_close:
            current_strategy_name = "exit_take_profit" # 止盈阶段
        elif self.breakeven_end_minutes < minutes_to_close <= self.take_profit_end_minutes:
            current_strategy_name = "exit_breakeven"   # 保本阶段
        elif self.stop_loss_end_minutes < minutes_to_close <= self.breakeven_end_minutes:
            current_strategy_name = "exit_reduce_loss" # 少亏阶段
        else:
            current_strategy_name = "exit_force_close" # 兜底
        
        # A. 强平阶段
        if is_force_market:
            # 【核心修改】直接升级现有订单，不撤单
            if existing_order:
                # 只有当策略还不是 force_close_final 时才升级，避免重复操作
                if existing_order.strategy != "force_close_final":
                    exchange.modify_order(existing_order.client_order_id, new_price=tick.price, new_strategy="force_close_final")
                    logger.info(f"🚨 [时间到] 升级订单为强平单: {existing_order.client_order_id}")
            else:
                self._submit_force_close(exchange, position, tick)
            return
        
        # 1. 致命错误：方向反了 (Side Mismatch)
        # 这种情况通常发生在反手单成交后，持仓方向变了，但原来的止损单还在
        if side_mismatch and existing_order:
            exchange.cancel_order(existing_order.client_order_id)
            logger.warning(f"⚠️ [方向修正] 仓位反转，撤销旧方向平仓单: {existing_order.client_order_id}")
            # 撤销后，如果有持仓，立即提交新单
            if abs(position.size) > 0.001:
                self._submit_new_exit_order(exchange, position, tick, target_price, minutes_to_close, strategy_name=current_strategy_name)
            return

        # B. 常规调整
        if qty_mismatch:
            new_qty = abs(position.size)
            if existing_order:
                if exchange.modify_order(existing_order.client_order_id, new_price=target_price, new_quantity=new_qty):
                    self.last_order_update_time[tick.contract_name] = now
                    logger.info(f"同步平仓单 (修改): {tick.contract_name} 数量->{new_qty}, 价格->{target_price}")
            else:
                self._submit_new_exit_order(exchange, position, tick, target_price, minutes_to_close, strategy_name=current_strategy_name)
            return
            
        # 定时调价 (每分钟) - 仅在非止损模式下，因为止损模式是实时追价
        last_update = self.last_order_update_time.get(tick.contract_name)
        if (not last_update) or (now - last_update).total_seconds() >= 60:
            if existing_order:
                if exchange.modify_order(existing_order.client_order_id, new_price=target_price):
                    self.last_order_update_time[tick.contract_name] = now
                    logger.info(f"调整平仓价 ({minutes_to_close:.1f}m left): {tick.contract_name} 价格->{target_price}")
            else:
                self._submit_new_exit_order(exchange, position, tick, target_price, minutes_to_close, strategy_name=current_strategy_name)

    def _submit_force_close(self, exchange, position: Position, tick: TickEvent):
        action = ActionType.SELL if position.size > 0 else ActionType.BUY
        signal = TradeSignal(
            timestamp=tick.timestamp,
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

    def _submit_new_exit_order(self, exchange: VirtualExchange, position: Position, tick: TickEvent, target_price: float, minutes_to_close: float, strategy_name="auto_profit_taking"):
        action = ActionType.SELL if position.size > 0 else ActionType.BUY
        signal = TradeSignal(
            timestamp=tick.timestamp,
            contract_name=tick.contract_name,
            contract_id=tick.contract_id,
            action=action,
            size=abs(position.size),
            price=target_price,
            strategy_name=strategy_name, 
            delivery_start=tick.delivery_start,
            open_strategy="profit_taking"
        )
        if exchange.submit_order(signal):
            self.last_order_update_time[tick.contract_name] = tick.timestamp
            logger.info(f"挂出平仓单 ({minutes_to_close:.1f}m left): {tick.contract_name} {action} {abs(position.size)}MW @ {target_price}")

    def _find_exit_order(self, contract_name: str, orders: List[Order], include_all: bool = False) -> Optional[Order]:
        """
        寻找当前合约的活动平仓单
        """
        for order in orders:
            if order.contract_name == contract_name:
                if order.state in ["NEW", "PARTIALLY_FILLED"]:
                    if "trend_reversal" in order.strategy:
                        continue
                    # if include_all:
                    #     return order
                    # 识别所有本管理器相关的策略名
                    if (order.strategy.startswith("auto_profit") or 
                        order.strategy.startswith("exit_") or
                        order.strategy.startswith("force_close") or 
                        order.strategy.startswith("stop_loss") or
                        order.strategy.startswith("consecutive_loss")):
                        return order
        return None