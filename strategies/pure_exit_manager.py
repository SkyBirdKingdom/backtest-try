import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from core.models import TickEvent, TradeSignal, ActionType, Position, Order

logger = logging.getLogger("PureExitManager")

class PureExitManager:
    """
    纯净版平仓管理器 (Lifecycle Manager)
    负责：止盈、保本、止损、强平 四个阶段的订单管理
    """
    def __init__(self, config: dict):
        self.config = config
        self.transaction_cost = config.get('transaction_cost', 0.23)
        
        # 记录上一次调整订单的时间，避免每秒都撤单重挂
        # Key: contract_name, Value: datetime
        self.last_order_update_time: Dict[str, datetime] = {}

    def process(self, tick: TickEvent, positions: Dict[str, Position], 
                active_orders: List[Order], exchange) -> None:
        """
        主处理函数：遍历持仓，根据时间阶段调整平仓挂单
        """
        if not tick.delivery_start:
            return

        # 1. 确定当前所处阶段
        time_to_close_min = self._get_minutes_to_close(tick.delivery_start, tick.timestamp)
        
        # 如果距离收盘超过 4小时(240分钟)，暂不处理（或者是还没到止盈阶段）
        if time_to_close_min > 240:
            return

        # 获取当前合约的持仓
        position = positions.get(tick.contract_name)
        if not position or abs(position.size) < 0.001:
            return

        # 获取当前合约已存在的平仓挂单 (如果有)
        existing_exit_order = self._find_exit_order(tick.contract_name, active_orders)

        # 2. 根据阶段计算目标平仓价格
        target_price, is_force_market = self._calculate_target_price(
            time_to_close_min, position, tick, existing_exit_order
        )

        # 3. 执行订单管理 (挂单、改单、撤单)
        self._manage_exit_order(
            exchange, position, tick, existing_exit_order, 
            target_price, is_force_market, time_to_close_min
        )

    def _get_minutes_to_close(self, delivery_start: Union[str, datetime], current_time: datetime) -> float:
        try:
            if isinstance(delivery_start, str):
                delivery_dt = datetime.strptime(delivery_start, '%Y-%m-%d %H:%M:%S')
            else:
                delivery_dt = delivery_start
            
            # 关闸时间 = 交付时间 - 1小时
            gate_closure = delivery_dt - timedelta(hours=1)
            delta = gate_closure - current_time
            return delta.total_seconds() / 60.0
        except Exception:
            return 9999.0

    def _calculate_target_price(self, minutes_to_close: float, position: Position, 
                                tick: TickEvent, existing_order: Optional[Order]) -> Tuple[float, bool]:
        """
        计算目标价格
        Returns: (target_price, is_force_market)
        """
        entry_price = position.avg_price
        # 费率处理：QH合约除以4
        is_qh = "QH" in tick.contract_type or "QH" in tick.contract_name
        fee_rate = (self.transaction_cost / 4.0) if is_qh else self.transaction_cost
        
        # 我们需要覆盖 双边手续费 (开仓+平仓)，才能算真正的保本/止盈
        # 平仓价至少要优于：Entry +/- (2 * Fee)
        cost_padding = 2 * fee_rate
        
        is_long = position.size > 0
        target_price = tick.price # 默认当前价
        is_force_market = False

        # --- 阶段 1: 止盈阶段 (240m -> 20m) ---
        if 20 < minutes_to_close <= 240:
            total_duration = 240 - 20
            elapsed = 240 - minutes_to_close
            progress = elapsed / total_duration # 0.0 -> 1.0
            
            # 初始止盈比例
            start_margin = 0.50 if entry_price < 50 else 0.20
            end_margin = 0.01
            
            # 线性衰减
            current_margin = start_margin - (start_margin - end_margin) * progress
            
            if is_long:
                # 做多：卖出价 = 成本 + 成本*利润 + 费用
                target_price = entry_price * (1 + current_margin) + cost_padding
            else:
                # 做空：买入价 = 成本 - 成本*利润 - 费用
                target_price = entry_price * (1 - current_margin) - cost_padding

        # --- 阶段 2: 保本阶段 (20m -> 10m) ---
        elif 10 < minutes_to_close <= 20:
            # 目标：最优 (本金 vs 最新成交价)，且覆盖手续费
            # "本金" 在这里理解为保本价 (Entry + Cost)
            breakeven_price = (entry_price + cost_padding) if is_long else (entry_price - cost_padding)
            
            if is_long:
                # 做多平仓(卖)：取 MAX(保本价, 市场价) -> 尽可能多赚，底线是保本
                # 但如果市场价已经低于保本价，为了成交，可能需要贴近市场价？
                # 用户需求："挂单价格调整为最优(本金，最近成交价)"
                # 通常 "最优" 对卖方意味着更高，对买方意味着更低。
                # 但考虑到是急于平仓，这里的 "最优" 可能指 "最容易成交但尽可能不亏"。
                # 策略：尝试挂在保本价，如果市场价更好，就挂市场价。
                target_price = max(breakeven_price, tick.price)
            else:
                # 做空平仓(买)：取 MIN(保本价, 市场价)
                target_price = min(breakeven_price, tick.price)

        # --- 阶段 3: 止损阶段 (10m -> 3m) ---
        elif 3 < minutes_to_close <= 10:
            # 目标：最优 (止损20% vs 最新成交价)，且覆盖手续费
            # 用户特别强调："价格仍然要涵盖手续费" -> 这意味着即使止损，也不能亏手续费？
            # 这其实是一个非常严苛的条件。如果当前价格已经亏损，强行挂 "Entry+Fee" 是无法成交的。
            # 但既然需求如此，我们严格执行：底线是 Entry+Fee。
            
            hard_limit_price = (entry_price + cost_padding) if is_long else (entry_price - cost_padding)
            
            # "止损20%" 逻辑：如果是做多，允许跌到 0.8 * Entry。
            # 但硬限制是 "涵盖手续费"。
            # 如果 Entry=100, Fee=1。HardLimit=101。StopLoss20%=80。
            # 此时必须挂 >= 101。这实际上使得 "Stop Loss 20%" 这一条失效，因为 101 > 80。
            # 我们暂且按 "必须覆盖手续费" 执行。
            
            if is_long:
                target_price = max(hard_limit_price, tick.price)
            else:
                target_price = min(hard_limit_price, tick.price)

        # --- 阶段 4: 强平阶段 (3m -> 0m) ---
        elif minutes_to_close <= 3:
            # 撤销之前挂单，直接以第一档(这里用最新Tick)成交
            target_price = tick.price
            is_force_market = True # 标记为强制市价(模拟)

        return target_price, is_force_market

    def _manage_exit_order(self, exchange, position: Position, tick: TickEvent, 
                           existing_order: Optional[Order], target_price: float, 
                           is_force_market: bool, minutes_to_close: float):
        """
        挂单、改单逻辑
        """
        # A. 如果是强平阶段 (is_force_market)
        if is_force_market:
            # 如果有挂单，先撤单
            if existing_order:
                exchange.cancel_order(existing_order.client_order_id)
            
            # 发送强平信号 (市价单模拟)
            # 在回测中，我们发一个 Limit 单，价格极其激进以确保成交
            # 做多平仓(卖)：价格设为 0 (或极低)
            # 做空平仓(买)：价格设为 9999 (或极高)
            # 但为了记录好看，我们传 tick.price，但在 exchange 中通过 slippage 或直接成交处理
            
            action = ActionType.SELL if position.size > 0 else ActionType.BUY
            # 构造强平信号
            signal = TradeSignal(
                timestamp=tick.timestamp,
                contract_name=tick.contract_name,
                contract_id=tick.contract_id,
                action=action,
                size=abs(position.size),
                price=tick.price, # 强平以此为基准
                strategy_name="force_close_final",
                delivery_start=tick.delivery_start,
                open_strategy="force_close"
            )
            exchange.submit_order(signal)
            logger.info(f"触发收盘前3分钟强平: {tick.contract_name} {action} @ {tick.price}")
            return

        # B. 正常调整阶段 (每分钟调整一次)
        now = tick.timestamp
        last_update = self.last_order_update_time.get(tick.contract_name)
        
        # 只有距离上次调整超过 60秒，或者是新订单，才操作
        if last_update and (now - last_update).total_seconds() < 60:
            return

        # 格式化价格精度
        target_price = round(target_price, 2)

        # 1. 如果没有挂单，直接挂单
        if not existing_order:
            action = ActionType.SELL if position.size > 0 else ActionType.BUY
            signal = TradeSignal(
                timestamp=now,
                contract_name=tick.contract_name,
                contract_id=tick.contract_id,
                action=action,
                size=abs(position.size),
                price=target_price,
                strategy_name="auto_profit_taking",
                delivery_start=tick.delivery_start,
                open_strategy="profit_taking"
            )
            if exchange.submit_order(signal):
                self.last_order_update_time[tick.contract_name] = now
                logger.info(f"挂出平仓单 ({minutes_to_close:.1f}m left): {tick.contract_name} {action} @ {target_price}")

        # 2. 如果有挂单，检查价格是否需要调整
        else:
            # 如果价格差异较大 (例如 > 0.05)，则改单
            if abs(existing_order.unit_price - target_price) > 0.05:
                # 撤单
                exchange.cancel_order(existing_order.client_order_id)
                
                # 重新挂单
                action = ActionType.SELL if position.size > 0 else ActionType.BUY
                signal = TradeSignal(
                    timestamp=now,
                    contract_name=tick.contract_name,
                    contract_id=tick.contract_id,
                    action=action,
                    size=abs(position.size),
                    price=target_price,
                    strategy_name="auto_profit_taking_update",
                    delivery_start=tick.delivery_start,
                    open_strategy="profit_taking"
                )
                if exchange.submit_order(signal):
                    self.last_order_update_time[tick.contract_name] = now
                    logger.info(f"调整平仓单 ({minutes_to_close:.1f}m left): {tick.contract_name} 从 {existing_order.unit_price} -> {target_price}")

    def _find_exit_order(self, contract_name: str, orders: List[Order]) -> Optional[Order]:
        """寻找当前合约的活动平仓单 (反向单)"""
        # 注意：这里简化处理，假设只有一个方向的持仓和挂单
        # 严谨逻辑应该判断挂单方向与持仓方向相反
        for order in orders:
            if order.contract_name == contract_name and order.state == "NEW":
                return order
        return None