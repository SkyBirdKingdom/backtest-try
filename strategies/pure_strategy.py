import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from scipy.stats import linregress 
from collections import deque
from collections import defaultdict
import math

from core.models import TickEvent, TradeSignal, ActionType, Position, Order

logger = logging.getLogger("PureStrategy")

class PureStrategyEngine:
    def __init__(self, config: dict):
        self.config = config
        self.params = config.get('strategy_params', {})
        
        # --- 核心风控参数 ---
        self.min_price_for_new_position = float(config.get('min_price_for_new_position', 10.0))
        self.max_position_size = float(config.get('max_position_size', 15.0))
        self.default_contract_max_position = float(config.get('max_contract_position_size', 1.0))
        
        self.forbid_new_open_minutes = int(self.params.get('forbid_new_open_minutes', 60))
        self.daily_loss_limit = float(config.get('daily_loss_limit', 150.0))
        self.price_change_threshold_ratio = float(self.params.get('price_change_threshold_ratio', 0.1))
        
        self.last_trade_times: Dict[str, datetime] = {}
        
        self.position_constraints = config.get('position_constraints', {})
        self.delivery_rules = self.position_constraints.get('delivery_rules', [])
        
        self.price_history: Dict[str, List[float]] = {}
        self.delivery_time_strategy_executed: Set[str] = set()

        # 【新增】记录上一次发出信号的时间 (Key: Contract_Action)
        self.last_signal_emit_times: Dict[str, datetime] = {}

        # --- 【新增】实盘止损策略参数 ---
        self.high_price_profit_multiplier = 1.02
        self.low_price_profit_multiplier = 1.05
        self.consecutive_loss_count: Dict[str, int] = defaultdict(int)
        self.last_position_avg_price: Dict[str, float] = {}
        self.processed_market_data_ids: Set[str] = set() # 用于K线去重
        self.executed_reverse_strategies: Set[str] = set() # 记录已反手的合约

        # --- 【新增】诊断与生命周期管理 ---
        self.tick_counter = 0
        self.current_date = None
        self.daily_realized_pnl = 0.0
        self.is_risk_triggered = False
        
        # --- 【新增】内置Bar合成器 (Tick -> Bar) ---
        self.bars: Dict[str, List[dict]] = defaultdict(list)
        self.current_bars: Dict[str, dict] = {}

    # ----------------------------------------------------------------
    # 【新增】生命周期方法 (Engine 调用接口)
    # ----------------------------------------------------------------
    
    def on_new_day(self, date_str: str):
        """跨天重置逻辑"""
        logger.info(f"📅 策略收到跨天通知: {date_str} (昨日PnL: {self.daily_realized_pnl:.2f})")
        self.daily_realized_pnl = 0.0
        self.is_risk_triggered = False
        self.current_date = date_str
        
        # 清理单日执行标记
        self.delivery_time_strategy_executed.clear()

        # 清理止损策略状态
        self.consecutive_loss_count.clear()
        self.last_position_avg_price.clear()
        self.processed_market_data_ids.clear()
        # self.executed_reverse_strategies.clear()
        
        # 清理过期的价格缓存
        for k in list(self.price_history.keys()):
            if len(self.price_history[k]) > 500: 
                self.price_history[k] = self.price_history[k][-100:]

    def update_pnl(self, pnl: float):
        """更新策略感知的 PnL (备用接口)"""
        self.daily_realized_pnl += pnl

    def on_tick(self, tick: TickEvent, positions: Dict[str, Position], active_orders: List[Order], account_info) -> List[TradeSignal]:
        """
        主入口函数 (适配 BacktestEngine)
        """
        self.tick_counter += 1
        
        # 1. 自动检测日期变更 (兜底)
        tick_date = tick.timestamp.strftime("%Y-%m-%d")
        if self.current_date != tick_date:
            self.current_date = tick_date
            
        # 2. 【核心】实时合成 K 线 (1分钟 Bar)
        self._update_bars(tick)

        # 3. 诊断心跳 (每 10,000 Tick)
        if self.tick_counter % 10000 == 0:
            logger.info(f"💓 策略运行中... DailyPnL: {self.daily_realized_pnl:.2f} (Limit: -{self.daily_loss_limit})")

        # 4. 全局日内风控检查
        contract_bars = self.bars[tick.contract_name]
        
        # 收集本 Tick 产生的所有信号
        raw_signals = []

        # --- A. 连续亏损止损策略 & 反手 ---
        # --- 【新增】反手状态确认逻辑 ---
        # 如果当前持仓是由反手策略建立的，标记该合约反手已成功执行
        position = positions.get(tick.contract_name)
        if position and "trend_reversal" in position.open_strategy:
             if tick.contract_name not in self.executed_reverse_strategies:
                 self.executed_reverse_strategies.add(tick.contract_name)
                 logger.info(f"✅ [{tick.contract_name}] 检测到反手策略持仓，标记为已执行 (不再触发反手)")
        # -------------------------------

        # 2. 【新增】检查挂单 (防止持仓未形成或部分成交时的遗漏)
        # 只要反手单发生过任何成交 (remaining < quantity)，就视为反手已执行
        if tick.contract_name not in self.executed_reverse_strategies:
            for order in active_orders:
                if order.contract_name == tick.contract_name and "trend_reversal" in order.strategy:
                    # 如果有成交量 (哪怕是 0.1 MW)，就视为“反手订单发生交易”
                    if order.remaining_quantity < order.quantity:
                        self.executed_reverse_strategies.add(tick.contract_name)
                        logger.info(f"✅ [{tick.contract_name}] 检测到反手订单发生交易 (部分成交)，标记为已执行")
                        break

        # 控制止损和反手逻辑
        if position and abs(position.size) > 0.001:
            sl_signals = self._check_consecutive_loss_stop_loss(tick, position, contract_bars, active_orders)
            if sl_signals:
                raw_signals.extend(sl_signals)
        
        # --- 【新增】反手后封锁逻辑 ---
        # 如果该合约已经触发过反手策略，禁止后续一切常规开仓/加仓
        if tick.contract_name in self.executed_reverse_strategies:
            # 仅允许上面的止损/平仓逻辑运行，直接跳过下面的 calculate_signals
            return raw_signals
        # ---------------------------

        # --- B. 调用原有的 calculate_signals (常规开仓/加仓逻辑) ---
        # 即使触发了止损，这里依然执行，因为“开仓除了不能在禁止开仓时间段触发，其他时间段是不受限制的”
        # calculate_signals 内部已经包含了 _check_time_to_close (禁止开仓时间) 的检查
        normal_signals = self.calculate_signals(tick, contract_bars, positions, active_orders, tick.timestamp, self.daily_realized_pnl)
        if normal_signals:
            raw_signals.extend(normal_signals)

        # # 5. 合并与生产环境约束检查
        # final_signals = []
        # for sig in raw_signals:
        #     if not sig.is_valid:
        #         final_signals.append(sig)
        #         continue
            
        #     # 检查：如果有活跃的反手/开仓单，不再发单
        #     sig.is_valid = self._check_production_constraints(sig, active_orders, tick.timestamp)
        #     final_signals.append(sig)

        return raw_signals

    def _update_bars(self, tick: TickEvent):
        """简易的 K 线合成器 (1分钟)"""

        c_name = tick.contract_name
        current_bar = self.current_bars.get(c_name)

        # 如果是新的分钟，归档旧 Bar
        if current_bar and tick.timestamp.minute != current_bar['start_time'].minute:
            self.bars[c_name].append(current_bar)
            # 保持 Bars 长度在合理范围 (只保留最近 300 根用于趋势计算)
            if len(self.bars[c_name]) > 300:
                self.bars[c_name].pop(0)

            del self.current_bars[c_name]
            current_bar = None
            
        # 更新或创建当前 Bar
        if not current_bar:
            self.current_bars[c_name] = {
                'start_time': tick.timestamp,
                'open': tick.price,
                'high': tick.price,
                'low': tick.price,
                'close': tick.price,
                'avg_price': tick.price, 
                'volume': tick.volume,
                'trade_count': 1
            }
        else:
            current_bar['high'] = max(current_bar['high'], tick.price)
            current_bar['low'] = min(current_bar['low'], tick.price)
            current_bar['close'] = tick.price
            current_bar['volume'] += tick.volume
            # 均价计算
            current_bar['avg_price'] = (current_bar['avg_price'] * current_bar['trade_count'] + tick.price) / (current_bar['trade_count'] + 1)
            current_bar['trade_count'] += 1
            
            # 写回字典（如果是引用类型其实不需要，但为了保险）
            self.current_bars[c_name] = current_bar
    

    # =========================================================================
    # 【修改】实盘连续亏损止损策略逻辑 (1分钟K线版本)
    # =========================================================================

    def _get_minutes_to_close(self, delivery_start: datetime, current_time: datetime) -> float:
        gate_closure = delivery_start - timedelta(hours=1)
        delta = gate_closure - current_time
        return delta.total_seconds() / 60.0

    def get_loss_ratio(self, position_size, avg_price, current_price):
        """
        通用亏损率计算方法（支持负价格）

        参数:
        :param position_size: 持仓数量 (正数为多头, 负数为空头)
        :param avg_price: 持仓均价 (可能为负)
        :param current_price: 当前市场价 (可能为负)

        返回:
        :return: float, 亏损率 (例如 0.2 表示亏损 20%, -0.1 表示盈利 10%)
        """
        # 1. 安全检查：无持仓或均价为0（无法计算比例）
        if position_size == 0 or avg_price == 0:
            return 0.0

        # 2. 计算价格差 (Price Difference)
        # 多头 (size > 0): 价格下跌是亏损 (开仓价 - 现价)
        # 空头 (size < 0): 价格上涨是亏损 (现价 - 开仓价)
        if position_size > 0:
            price_diff = avg_price - current_price
        else:
            price_diff = current_price - avg_price

        # 3. 计算亏损率
        # 核心：使用绝对值 abs(avg_price) 作为基准，
        # 这样无论价格在正轴还是负轴，计算出的比例方向都是正确的。
        loss_ratio = price_diff / abs(avg_price)

        return float(loss_ratio)

    def _check_consecutive_loss_stop_loss(self, tick: TickEvent, position: Position, bars: List[dict], active_orders: List[Order]) -> List[TradeSignal]:
        contract_name = tick.contract_name
        
        # 1. 【修改】时间窗口检查: 收盘前4小时 ~ 禁止开仓时间
        minutes_to_close = self._get_minutes_to_close(tick.delivery_start, tick.timestamp)
        actual_forbid_minutes = self.forbid_new_open_minutes
        
        # 窗口：240分钟 >= 剩余时间 > 禁止时间
        if not (actual_forbid_minutes < minutes_to_close):
            return []

        # 2. 必须有足够的历史数据 (至少10根)
        if len(bars) < 10:
            return []
        
        # 取最近10根已完成的K线 (不包含当前正在生成的)
        # bars[-1] 可能是当前分钟未完成的，也可能是刚归档的。
        # 这里逻辑：Engine的bars存的是已归档的。bars[-1]就是最近一根完整的。
        # recent_bars = bars[-10:]

        # ==================== 【修复：K线起始时间过滤】 ====================
        # 确定有效的 K 线起始时间：
        # 如果触发了二次加仓，从最后一次仓位变动时间开始算
        # 如果是普通持仓，从初始建仓时间开始算
        if position.has_triggered_2nd_add and getattr(position, 'second_add_time', None):
            start_time = position.second_add_time
        else:
            start_time = position.initial_entry_time or position.timestamp

        # 过滤出在加仓（或建仓）之后产生的有效 K 线
        valid_bars = [b for b in bars if b['start_time'] >= start_time]

        # 2. 必须在加仓/建仓后，有足够的历史数据 (至少10根)
        # 这样可以确保二次加仓后，至少要观察 10 分钟才会触发严格止损
        if len(valid_bars) < 10:
            return []
        
        # 取加仓后最近的 10 根已完成的有效K线
        recent_bars = valid_bars[-10:]
        # =================================================================

        # 计算最近10根K线的亏损情况
        # 逻辑：假设每一根K线的收盘价/均价 都是一次潜在的平仓机会，计算当时的亏损率
        loss_ratios = []
        for bar in recent_bars:
            bar_price = bar['avg_price'] # 使用K线均价
            loss = self.get_loss_ratio(position.size, position.avg_price, bar_price)
            loss_ratios.append(loss)
            
        strict_threshold = 0.0 # 严格模式：只要亏损 > 0
        normal_threshold = 0.2 if abs(position.avg_price) >= 50 else 0.5 # 普通模式阈值
        
        stop_triggered = False
        trigger_mode = "" # 用于记录触发模式

        # --- 分支 A：严格模式 (Strict Mode) ---
        # 条件：1. 触发过二次加仓; 2. 最近10根K线每一根的亏损都 > 0
        if position.has_triggered_2nd_add:
            is_strict_met = all(l > strict_threshold for l in loss_ratios)
            if is_strict_met:
                stop_triggered = True
                trigger_mode = "Strict"
                logger.warning(f"🔥 [{contract_name}] 严格模式触发: 二次加仓且连续10根K线亏损>0")

        # --- 分支 B：普通模式 (Normal Mode) ---
        # 条件：最近10根K线每一根的亏损都 >= 阈值 (不要求二次加仓)
        if not stop_triggered:
            is_normal_met = all(l >= normal_threshold for l in loss_ratios)
            if is_normal_met:
                stop_triggered = True
                trigger_mode = "Normal"
                logger.warning(f"🚫 [{contract_name}] 普通模式触发: 连续10根K线亏损 >= {normal_threshold*100}%")

        # 止损逻辑全部走严格模式
        # is_strict_met = all(l > strict_threshold for l in loss_ratios)
        # if is_strict_met:
        #     stop_triggered = True
        #     trigger_mode = "Strict"
        #     logger.warning(f"🔥 [{contract_name}] 严格模式触发: 二次加仓且连续10根K线亏损>0")
        
        if stop_triggered:
            position.stop_loss_triggered = True
            return self._create_stop_and_reverse_signals(tick, position, tick.price, active_orders, bars, trigger_mode)
             
        return []
    
    def _create_stop_and_reverse_signals(self, tick: TickEvent, position: Position, market_price: float, active_orders: List[Order], bars: List[dict], trigger_mode: str) -> List[TradeSignal]:
        signals = []

        # --- 1. 检查是否存在【任何平仓单】(止盈 OR 止损) ---
        # 如果存在任何平仓类订单，策略层不再发信号，而是让 ExitManager 接管修改订单
        
        # 1. 检查是否已有平仓单 (止盈单或止损单)
        existing_exit_order = None
        for order in active_orders:
            if order.contract_name == tick.contract_name and \
               (order.strategy.startswith("auto_profit") or order.strategy.startswith("stop_loss") or order.strategy.startswith("consecutive_loss") or order.strategy.startswith("exit_")):
                existing_exit_order = order
                break
        
        # 2. 如果没有现存的平仓单，则生成一个新的止损单
        # 如果有，我们**不**生成新信号，而是依靠 ExitManager 检测 position.stop_loss_triggered 标志来修改现有订单
        if not existing_exit_order:
            action = ActionType.SELL if position.size > 0 else ActionType.BUY
            # 策略名加上模式后缀，如 consecutive_loss_stop_strict
            stop_strategy_name = f"stop_loss_{trigger_mode.lower()}"

            stop_signal = TradeSignal(
                timestamp=tick.timestamp,
                contract_name=tick.contract_name,
                contract_id=tick.contract_id,
                action=action,
                size=abs(position.size),
                price=bars[-1]['avg_price'], # 初始价格，ExitManager 会马上接管并修改
                strategy_name=stop_strategy_name, 
                delivery_start=tick.delivery_start,
                confidence=1.0,
                open_strategy=position.strategy_name,
                failure_reason="StopLoss Triggered" 
            )
            signals.append(stop_signal)
            logger.info(f"[{tick.contract_name}] 生成新的止损信号(Mode: {trigger_mode})")

        # 控制反手逻辑
        # 3. 生成反手信号 (Reverse Strategy)
        # 检查是否已反手 (One-shot Check)
        if tick.contract_name in self.executed_reverse_strategies:
            return signals
        
        # 检查 B: 【新增】是否已有“正在挂单中”的反手单 (防止成交前重复发单)
        # 我们遍历 active_orders，看有没有 strategy 名字里带 "trend_reversal" 的
        for order in active_orders:
            if order.contract_name == tick.contract_name and order.strategy.startswith("trend_reversal"):
                # 如果已经有挂着的反手单，直接返回，不再生成新信号
                logger.info(f"[{tick.contract_name}] 忽略反手信号：已存在活动的反手单 ({order.client_order_id})")
                return signals
        
        reverse_action = ActionType.SELL if position.size > 0 else ActionType.BUY
        # 【修改】反手数量：当前持仓量
        reverse_size = abs(position.size)
        
        # 策略名加上模式后缀，如 trend_reversal_strict
        reverse_strategy_name = f"trend_reversal_{trigger_mode.lower()}"

        reverse_signal = TradeSignal(
            timestamp=tick.timestamp,
            contract_name=tick.contract_name,
            contract_id=tick.contract_id,
            action=reverse_action,
            size=reverse_size,
            price=tick.price, 
            strategy_name=reverse_strategy_name, # 策略名
            delivery_start=tick.delivery_start,
            confidence=0.8,
            open_strategy="trend_reversal", # 标记开仓策略
            trend_info="Reverse after Stop"
        )
        signals.append(reverse_signal)
        logger.info(f"[{tick.contract_name}] 生成反手信号 (Mode: {trigger_mode}, Size: {reverse_size}, Price: {tick.price})")
            
        return signals

    # ----------------------------------------------------------------
    # 以下为您原始的业务逻辑 (calculate_signals 及辅助方法)
    # ----------------------------------------------------------------

    def calculate_signals(self, 
                          tick: TickEvent, 
                          bars: List[dict], 
                          positions: Dict[str, Position], 
                          active_orders: List[Order],
                          current_time: datetime,
                          current_daily_pnl: float = 0.0) -> List[TradeSignal]:
        
        self._update_tick_history(tick)
        self.daily_realized_pnl = current_daily_pnl
        raw_signals = []

        # 0. 基础环境检查
        if abs(tick.price) < self.min_price_for_new_position:
            if tick.contract_name not in positions:
                return []

        # --- 策略 1: 均值回归 ---
        sig_mr = self._check_mean_reversion(tick, bars, positions, current_time)
        if sig_mr:
            self._apply_risk_checks(sig_mr, tick, bars, positions, current_time, current_daily_pnl)
            raw_signals.append(sig_mr)

        # --- 策略 2: 极端价格 ---
        sig_ext = self._check_extreme_sell(tick, bars, positions, current_time)
        if sig_ext:
            self._apply_risk_checks(sig_ext, tick, bars, positions, current_time, current_daily_pnl)
            raw_signals.append(sig_ext)
        
        # 当前价格与持仓价格相比较超过30%，再买一笔
        # CONCLUSION: 该逻辑收益效果不明显，暂时注释掉
        # sig_more = self._can_do_more_positions(tick, positions, current_time)
        # if sig_more:
        #     self._apply_risk_checks(sig_more, tick, bars, positions, current_time, current_daily_pnl)
        #     raw_signals.append(sig_more)
        
        # =========================================================
        # 【新增】生产环境逻辑检查 (信号抑制 + 订单互斥)
        # =========================================================
        signals = []
        for sig in raw_signals:
            if not sig.is_valid: 
                signals.append(sig) # 已经被前面的基础风控拦截了
                continue
            
            # 执行生产环境检查
            sig.is_valid = self._check_production_constraints(sig, active_orders, current_time)
            signals.append(sig)

        return signals
    
    def _check_production_constraints(self, signal: TradeSignal, active_orders: List[Order], current_time: datetime) -> bool:
        """
        生产环境逻辑检查：
        1. 订单互斥：存在同合约同方向的"活跃开仓单"时，禁止发新单
        2. 信号抑制：5秒内同合约同方向抑制
        """
        if signal.strategy_name.startswith("stop_loss") or signal.strategy_name.startswith("trend_reversal"):
            return True  # 止损/反手单不受此限制
        # 1. 订单开仓限制
        for order in active_orders:
            if order.contract_name == signal.contract_name:
                if order.side == signal.action.value:
                    if not (order.strategy.startswith("auto_profit") or order.strategy.startswith("force_close")):
                        signal.is_valid = False
                        signal.failure_reason = f"Active Order Exists ({order.client_order_id})"
                        return False

        # 2. 信号抑制 (5秒防抖)
        key = f"{signal.contract_name}_{signal.action.value}"
        last_emit = self.last_signal_emit_times.get(key)
        if last_emit:
            time_diff = (current_time - last_emit).total_seconds()
            if time_diff < 5.0:
                signal.is_valid = False
                signal.failure_reason = f"Signal Suppressed: <5s ({time_diff:.1f}s)"
                return False
        
        # 更新发射时间
        self.last_signal_emit_times[key] = current_time
        return True
    
    def _apply_risk_checks(self, signal: TradeSignal, tick: TickEvent, bars: List[dict], 
                           positions: Dict[str, Position], current_time: datetime, 
                           current_daily_pnl: float, 
                           skip_trend: bool = False, skip_close_time: bool = False):
        # 0. 基础价格限制
        if abs(tick.price) < self.min_price_for_new_position:
            existing_pos = positions.get(tick.contract_name)
            if not existing_pos or abs(existing_pos.size) < 0.001:
                signal.is_valid = False
                signal.failure_reason = f"Price Limit: {abs(tick.price):.2f} < {self.min_price_for_new_position}"
                return

        # 1. 冷却期检查
        if not self._check_cooldown(signal, current_time):
            signal.is_valid = False
            signal.failure_reason = "Signal Cooldown Active"
            return

        # 2. 趋势过滤
        # if not skip_trend:
        #     if not self._check_trend_analysis(signal, bars):
        #         signal.is_valid = False
        #         if not signal.failure_reason:
        #             signal.failure_reason = "Trend Analysis Failed"
        #         return

        # 3. 通用信号验证
        if not self._validate_signal(signal, positions):
            signal.is_valid = False
            return
        
        if self.is_risk_triggered:
            signal.is_valid = False
            signal.failure_reason = "Global Risk Triggered"
            return

        # 4. 日亏损限制
        if current_daily_pnl < -self.daily_loss_limit:
            self.is_risk_triggered = True
            signal.is_valid = False
            signal.failure_reason = f"Daily Loss Limit Hit: {current_daily_pnl:.2f} < -{self.daily_loss_limit}"
            return

        # 5. 临近关闸限制
        if not skip_close_time:
            if not self._check_time_to_close(tick.delivery_start, current_time):
                signal.is_valid = False
                signal.failure_reason = "Too Close to Gate Closure"
                return
        
        # 6. 更新冷却时间
        if signal.is_valid:
            self.last_trade_times[tick.contract_name + signal.strategy_name] = current_time


    def _validate_ph_signal(self, signal: TradeSignal) -> bool:
        """验证PH信号的特定逻辑"""
        if signal.contract_name.startswith("PH"):
            # 🛡️ 防火墙 1：平仓、止损、反手、强平类策略，跳过除以4！
            # 它们是基于现有持仓的绝对值操作，必须精确匹配。
            is_close_logic = (
                signal.strategy_name.startswith("stop_loss") or 
                signal.strategy_name.startswith("trend_reversal") or 
                signal.strategy_name.startswith("force_close") or
                signal.strategy_name.startswith("exit_") or
                signal.open_strategy == "force_close" # 双重保险
            )
            
            if is_close_logic:
                return True
            original_size = signal.size
            signal.size = round(signal.size / 4, 1)  # 四舍五入到小数点后一位
            if signal.size < 0.1:
                msg = f"信号验证失败 - PH信号调整后仓位过小: 合约={signal.contract_name}, 策略={signal.strategy_name}, 动作={signal.action.value}, 原始数量={original_size}, 调整后数量={signal.size}, 价格={signal.price}, trade_id={getattr(signal, 'trade_id', '')}, trade_time={getattr(signal, 'trade_time', '')}"
                logger.warning(msg)
                return False  # If size is too small, immediately return False
            logger.info(f"PH信号仓位调整: 合约={signal.contract_name}, 原始数量={original_size}, 调整后数量={signal.size}")
            return True  # If it's a PH signal and size is sufficient, return True
        return True

    def _check_cooldown(self, signal: TradeSignal, current_time: datetime) -> bool:
        strategy_name = signal.strategy_name
        cooldown = self.params.get('signal_cooldown_seconds', 5)
        key = signal.contract_name + strategy_name
        last_time = self.last_trade_times.get(key)
        if last_time and (current_time - last_time).total_seconds() < cooldown:
            return False
        return True

    def _update_tick_history(self, tick: TickEvent):
        contract = tick.contract_name
        if contract not in self.price_history:
            self.price_history[contract] = []
        self.price_history[contract].append(tick.price)
        if len(self.price_history[contract]) > 100:
            self.price_history[contract].pop(0)

    def _check_trend_analysis(self, signal: TradeSignal, bars: List[dict]) -> bool:
        if signal.strategy_name not in ["super_mean_reversion_buy", "optimized_extreme_sell"]:
            return True
        cutoff_time = signal.timestamp - timedelta(minutes=30)
        # 确保 bars 不为空
        if not bars: return False
        
        potential_bars = bars[-10:]
        valid_bars = [b for b in potential_bars if b['start_time'] >= cutoff_time]
        price_list = [float(b.get('avg_price', b['close'])) for b in valid_bars]
        if len(price_list) < 3:
            signal.failure_reason = f"Trend Data Insufficient: {len(price_list)} < 3"
            return False
        long_trend_result = self.detect_trend_with_linear_regression(price_list)
        long_trend = long_trend_result["trend"]
        long_confidence = long_trend_result.get("confidence", 0.0)
        signal.trend_info = f"{long_trend} (Conf:{long_confidence:.2f}, R2:{long_trend_result['r_squared']:.2f})"
        if signal.strategy_name == "super_mean_reversion_buy":
            if (long_trend == "下降" and long_confidence >= 0.6):
                signal.failure_reason = f"Trend Intercept: Down Trend (Conf {long_confidence:.2f} >= 0.6)"
                return False 
            elif (long_trend == "下降" and long_confidence < 0.6) or (long_trend != "下降" and long_confidence < 0.6):
                temp_conf = long_confidence
                if long_trend != '下降': temp_conf = 0.6 - long_confidence
                adjustment_factor = (0.6 - temp_conf) / 2
                prev_size = signal.size
                adjusted_size = round(prev_size * adjustment_factor, 1)
                if adjusted_size < 0.1:
                    signal.failure_reason = f"Trend Sizing: {prev_size}->{adjusted_size} < 0.1"
                    return False
                signal.size = adjusted_size
                if signal.contract_name.startswith("QH"):
                    signal.size = signal.size * 2
        elif signal.strategy_name == "optimized_extreme_sell":
            if (long_trend == "上升" and long_confidence >= 0.6):
                signal.failure_reason = f"Trend Intercept: Up Trend (Conf {long_confidence:.2f} >= 0.6)"
                return False
            elif (long_trend == "上升" and long_confidence < 0.6) or (long_trend != "上升" and long_confidence < 0.6):
                temp_conf = long_confidence
                if long_trend != '上升': temp_conf = 0.6 - long_confidence
                adjustment_factor = (0.6 - temp_conf) / 2
                prev_size = signal.size
                adjusted_size = round(prev_size * adjustment_factor, 1)
                if adjusted_size < 0.1:
                    signal.failure_reason = f"Trend Sizing: {prev_size}->{adjusted_size} < 0.1"
                    return False
                signal.size = adjusted_size
                if signal.contract_name.startswith("QH"):
                    signal.size = signal.size * 2
        return True

    def detect_trend_with_linear_regression(self, prices: List[float], window_size: int = 3, slope_threshold: float = 0.1) -> Dict:
        filtered_prices = [float(p) for p in prices if p is not None]
        prices_arr = np.array(filtered_prices, dtype=float)
        if len(prices_arr) < window_size:
            return {"trend": "数据不足", "confidence": 0.0, "r_squared": 0.0}
        if np.all(prices_arr == prices_arr[0]):
            return {"trend": "平滑", "confidence": 1.0, "r_squared": 1.0, "slope": 0.0}
        prices_series = pd.Series(prices_arr)
        smoothed = prices_series.rolling(window=window_size, center=True, min_periods=1).mean()
        x = np.arange(len(smoothed))
        slope, intercept, r_value, p_value, std_err = linregress(x, smoothed.values)
        r_squared = r_value ** 2
        if abs(slope) < slope_threshold: trend = "平滑"
        elif slope > slope_threshold: trend = "上升"
        else: trend = "下降"
        confidence = self.calculate_trend_confidence(r_squared, p_value, len(prices_arr))
        return {"trend": trend, "slope": float(slope), "r_squared": float(r_squared), "confidence": float(confidence)}

    def calculate_trend_confidence(self, r_squared: float, p_value: float, data_points: int) -> float:
        base_confidence = r_squared
        if p_value < 0.001: p_adjustment = 1.0
        elif p_value < 0.01: p_adjustment = 0.9
        elif p_value < 0.05: p_adjustment = 0.8
        elif p_value < 0.1: p_adjustment = 0.6
        else: p_adjustment = 0.3
        if data_points >= 50: data_adjustment = 1.0
        elif data_points >= 40: data_adjustment = 0.9
        elif data_points >= 30: data_adjustment = 0.85
        elif data_points >= 20: data_adjustment = 0.8
        elif data_points >= 5: data_adjustment = 0.7          
        else: data_adjustment = 0.6
        confidence = base_confidence * p_adjustment * data_adjustment
        return min(max(confidence, 0.0), 1.0)

    def _validate_signal(self, signal: TradeSignal, positions: Dict[str, Position]) -> bool:
        existing_position = positions.get(signal.contract_name)
        if existing_position and abs(existing_position.size) > 0.001:
            is_same_direction = (existing_position.size > 0 and signal.action == ActionType.BUY) or \
                                (existing_position.size < 0 and signal.action == ActionType.SELL)
            if is_same_direction:
                position_price = existing_position.avg_price
                price_diff = abs(signal.price - position_price)
                price_threshold = abs(position_price) * self.price_change_threshold_ratio
                if price_diff <= price_threshold:
                    signal.failure_reason = f"Price Diff Insufficient: {price_diff:.2f} <= {price_threshold:.2f}"
                    return False
            five_minutes_ago = signal.timestamp - timedelta(minutes=5)
            if existing_position.timestamp >= five_minutes_ago:
                signal.failure_reason = "Recent Position (<5m)"
                return False
            # 检查是否为导致亏损的平仓信号
            if not self._validate_profit_close(signal, existing_position):
                return False
            # 检查合约在5分钟内是否有持仓
            if not self._validate_recent_position(signal, existing_position):
                return False
        # 验证PH信号
        if not self._validate_ph_signal(signal):
            return False
        return True

    def _validate_recent_position(self, signal: TradeSignal, position: Position) -> bool:
        """验证合约在5分钟内是否有相同策略的持仓，如果有则跳过信号

        Args:
            signal: 交易信号
            positions: 当前持仓列表

        Returns:
            bool: True表示可以继续处理信号，False表示应该跳过信号
        """
        # 计算5分钟前的时间
        five_minutes_ago = signal.timestamp - timedelta(minutes=5)
        if (position.contract_name == signal.contract_name and
                position.strategy_name == signal.strategy_name and  # 添加策略名称验证
                abs(position.size) > 0.001 and
                position.timestamp >= five_minutes_ago):
            # 如果该合约和策略在5分钟内有持仓，跳过信号
            msg = (f"信号验证失败 - 合约5分钟内有相同策略持仓: 合约={signal.contract_name}, 策略={signal.strategy_name}, "
                   f"动作={signal.action.value}, 数量={signal.size}, 价格={signal.price}, trade_id={getattr(signal, 'trade_id', '')}, "
                   f"trade_time={getattr(signal, 'trade_time', '')}, 持仓合约={position.contract_name}, 策略={position.strategy_name}, "
                   f"持仓时间={position.timestamp}, 比较时间={five_minutes_ago}")
            logger.info(msg)
            return False

        return True


    def _validate_profit_close(self, signal: TradeSignal, target_position: Position) -> bool:
        """验证平仓信号是否盈利
        """

        # 2. 找到对应的持仓
        if not target_position:
            # 没有持仓，说明是开仓信号（或持仓已平），不适用此规则
            return True

        # 3. 判断是否为平仓/减仓方向
        is_closing = False
        if target_position.size > 0 and signal.action == ActionType.SELL:
            is_closing = True
        elif target_position.size < 0 and signal.action == ActionType.BUY:
            is_closing = True

        if not is_closing:
            return True

        # 4. 计算预期盈亏
        # 考虑手续费，不考虑size
        fee_per_mw = float(self.config.get('transaction_cost', 0.22))
        total_fee_per_unit = 2 * fee_per_mw  # 开仓+平仓手续费

        is_profitable = False
        if target_position.size > 0:  # 多头，卖出平仓
            # 卖出价格必须高于 (持仓均价 + 双边手续费)
            if signal.price >= (target_position.avg_price + total_fee_per_unit):
                is_profitable = True
        else:  # 空头，买入平仓
            # 买入价格必须低于 (持仓均价 - 双边手续费)
            if signal.price <= (target_position.avg_price - total_fee_per_unit):
                is_profitable = True

        if not is_profitable:
            logger.info(
                f"信号验证失败: 平仓会导致亏损 (策略={signal.strategy_name}), 合约={signal.contract_name}, 持仓均价={target_position.avg_price:.2f}, 信号价格={signal.price:.2f}, 手续费/MW={fee_per_mw:.2f}")
            return False

        return True

    def _get_delivery_rule_config(self, delivery_start: Union[str, datetime]) -> Tuple[float, Dict]:
        current_max_pos = self.default_contract_max_position
        params_override = {}
        if not delivery_start: return current_max_pos, params_override
        try:
            if isinstance(delivery_start, str): dt = datetime.strptime(delivery_start, '%Y-%m-%d %H:%M:%S')
            else: dt = delivery_start
            weekday = dt.weekday() 
            delivery_time = dt.time()
            for rule in self.delivery_rules:
                if weekday in rule.get('days_of_week', []):
                    for time_range in rule.get('time_ranges', []):
                        start_h, start_m = map(int, time_range['start'].split(':'))
                        end_h, end_m = map(int, time_range['end'].split(':'))
                        t_min = delivery_time.hour * 60 + delivery_time.minute
                        start_min = start_h * 60 + start_m
                        end_min = end_h * 60 + end_m
                        if start_min <= t_min < end_min:
                            if 'max_position' in time_range: current_max_pos = float(time_range['max_position'])
                            if 'strategy_params' in time_range: params_override = time_range['strategy_params']
                            return current_max_pos, params_override
        except Exception: pass
        return current_max_pos, params_override

    def _calculate_action_and_size(self, contract_name: str, positions: Dict, max_pos: float, params: Dict, action: ActionType) -> float:
        ratio = params.get('position_ratio', 0.5)
        split = params.get('position_split', 3)
        min_size = params.get('min_open_size', 0.1)
        desired = max(min_size, round(max_pos * ratio / split, 1))
        
        # --- 计算持仓占用 ---
        total_holdings = sum(abs(p.size) for p in positions.values())
        pos = positions.get(contract_name)
        curr_size = abs(pos.size) if pos else 0.0
        global_avail = max(0.0, self.max_position_size - total_holdings)
        contract_avail = max(0.0, max_pos - curr_size)

        final = round(min(desired, global_avail, contract_avail), 1)
        return final if final >= min_size else 0.0

    def _check_time_to_close(self, delivery_start: Union[str, datetime], current_time: datetime) -> bool:
        if not delivery_start: return True
        try:
            if isinstance(delivery_start, str): delivery_dt = datetime.strptime(delivery_start, '%Y-%m-%d %H:%M:%S')
            else: delivery_dt = delivery_start
            gate_closure = delivery_dt - timedelta(hours=1)
            forbid_time = gate_closure - timedelta(minutes=self.forbid_new_open_minutes)
            return current_time < forbid_time
        except Exception as e: 
            logger.error(f"Error in _check_time_to_close: {e}")
            return True

    def _can_do_more_positions(self, tick: TickEvent, positions: Dict, now: datetime) -> Optional[TradeSignal]:
        """
            判断条件：当前市场价与持仓价比较，空仓涨幅30%，多仓跌幅30%，则再做一笔，没有次数限制，仅受限于最大仓位。只对普通开仓判断，反手仓不触发。
	        执行动作：以当前市场价提交加仓订单，仓位大小为min(剩余仓位，单次最大开仓量)
        """
        signal = None
        # 获取当前持仓信息
        contract_name = tick.contract_name
        curr_pos = positions.get(contract_name)
        curr_side = 'BUY' if curr_pos and curr_pos.size > 0 else 'SELL' if curr_pos and curr_pos.size < 0 else 'FLAT'
        entry_price = curr_pos.avg_price if curr_pos else 0.0
        if curr_side == 'FLAT':
            return signal  # 无持仓，直接返回None
        if entry_price == 0.0:
            return signal  # 无持仓，直接返回None
        if curr_side == 'BUY':
            price_increase = (entry_price - tick.price) / abs(entry_price) * 100
            if price_increase >= 30.0:
                max_pos, override = self._get_delivery_rule_config(tick.delivery_start)
                params = self.params.get('super_mean_reversion_buy', {}).copy()
                params.update(override.get('super_mean_reversion_buy', {}))
                
                if not self._check_time_to_close(tick.delivery_start, now):
                    return signal
                    
                size = self._calculate_action_and_size(tick.contract_name, positions, max_pos, params, ActionType.BUY)
                if size > 0.001:
                    signal = TradeSignal(now, tick.contract_name, tick.contract_id, ActionType.BUY, size, tick.price, 'super_mean_reversion_buy', tick.delivery_start)
        elif curr_side == 'SELL':
            price_decrease = (tick.price - entry_price) / abs(entry_price) * 100
            if price_decrease >= 30.0:
                max_pos, override = self._get_delivery_rule_config(tick.delivery_start)
                params = self.params.get('optimized_extreme_sell', {}).copy()
                params.update(override.get('optimized_extreme_sell', {}))
                
                if not self._check_time_to_close(tick.delivery_start, now):
                    return signal
                    
                size = self._calculate_action_and_size(tick.contract_name, positions, max_pos, params, ActionType.SELL)
                if size > 0.001:
                    signal = TradeSignal(now, tick.contract_name, tick.contract_id, ActionType.SELL, size, tick.price, 'optimized_extreme_sell', tick.delivery_start)
        return signal

    # 原始均值回归做多逻辑，暂时注释掉
    # def _check_mean_reversion(self, tick: TickEvent, bars: List[dict], positions: Dict, now: datetime) -> Optional[TradeSignal]:
    #     strategy_name = "super_mean_reversion_buy"
    #     max_pos, override = self._get_delivery_rule_config(tick.delivery_start)
    #     params = self.params.get(strategy_name, {}).copy()
    #     params.update(override.get(strategy_name, {}))
        
    #     window = params.get('ma_window', 20)
    #     threshold = params.get('threshold', 2.0)
    #     # cooldown = params.get('signal_cooldown_seconds', 300) # 移到外面检查
    #     std_ratio = params.get('std_ratio_threshold', 0.05)
        
    #     if len(bars) < params.get('history_min_len', 5): return None
        
    #     prices = [float(b.get('avg_price', b['close'])) for b in bars[-window:]]
    #     if not prices: return None
        
    #     mean = np.mean(prices)
    #     std = np.std(prices)
        
    #     if std <= abs(mean * std_ratio): return None
    #     if std == 0: return None
        
    #     z_score = (tick.price - mean) / std
        
    #     if z_score <= -threshold:
    #         # 检查是否启用了动态仓位
    #         if params.get('use_dynamic_sizing', False):
    #             # max_pos = self._calculate_liquidity_based_size(tick, bars, positions, params)
    #             max_pos = self._calculate_time_based_limit(tick)

    #         size = self._calculate_action_and_size(tick.contract_name, positions, max_pos, params, ActionType.BUY)
    #         # is_valid 由外部 _apply_risk_checks 进一步确认，这里先认为如果是0就是无效
    #         is_valid = size > 0.001
    #         reason = "" if is_valid else "Position Limit Reached (Size=0)"
            
    #         return TradeSignal(now, tick.contract_name, tick.contract_id, ActionType.BUY, size, tick.price, strategy_name, tick.delivery_start, confidence=min(abs(z_score)/threshold, 1.0), open_strategy=strategy_name, z_score=round(z_score,3), mean_price=round(mean,2), std_price=round(std,2), raw_size=max_pos, is_valid=is_valid, failure_reason=reason)
    #     return None

    # 新版均值回归做多逻辑，逻辑与做空逻辑对称，判断保持一致
    def _check_mean_reversion(self, tick: TickEvent, bars: List[dict], positions: Dict, now: datetime) -> Optional[TradeSignal]:
        strategy_name = "super_mean_reversion_buy"
        max_pos, override = self._get_delivery_rule_config(tick.delivery_start)
        params = self.params.get(strategy_name, {}).copy()
        params.update(override.get(strategy_name, {}))
        
        window = params.get('percentile_window', 20)
        percentile = params.get('percentile_low', 5)
        # cooldown = params.get('signal_cooldown_seconds', 300) # 移到外面
        threshold = params.get('threshold', 0.7)
        
        if len(bars) < params.get('history_min_len', 5): return None
        
        prices = [float(b.get('avg_price', b['close'])) for b in bars[-window:]]
        if not prices: return None
        
        lower = np.percentile(prices, percentile)
        mean = np.mean(prices)
        
        condition = False
        if mean < 0:
            if tick.price < 0:
                condition = abs(tick.price) >= abs(mean) * threshold
        else:
            # 正均价情况：当前价格低于5分位且低于均价除以阈值
            if tick.price < 0:
                condition = abs(tick.price) >= mean * threshold
            else:
                # condition = tick.price <= lower and tick.price <= mean / threshold
                condition = tick.price <= mean / threshold
            
        if condition:
            # 检查是否启用了动态仓位
            if params.get('use_dynamic_sizing', False):
                # max_pos = self._calculate_liquidity_based_size(tick, bars, positions, params)
                max_pos = self._calculate_time_based_limit(tick)

            size = self._calculate_action_and_size(tick.contract_name, positions, max_pos, params, ActionType.BUY)
            is_valid = size > 0.001
            reason = "" if is_valid else "Position Limit Reached (Size=0)"
            
            adj_price = tick.price + (mean - tick.price) * 0.02 # 轻微调整价格，增加成交概率
            
            return TradeSignal(now, tick.contract_name, tick.contract_id, ActionType.BUY, size, round(adj_price, 2), strategy_name, tick.delivery_start, open_strategy=strategy_name, z_score=0.0, mean_price=round(mean,2), std_price=0.0, trend_info=f"Lower{percentile}:{round(lower,2)}", raw_size=max_pos, is_valid=is_valid, failure_reason=reason)
        return None

    def _check_extreme_sell(self, tick: TickEvent, bars: List[dict], positions: Dict, now: datetime) -> Optional[TradeSignal]:
        strategy_name = "optimized_extreme_sell"
        max_pos, override = self._get_delivery_rule_config(tick.delivery_start)
        params = self.params.get(strategy_name, {}).copy()
        params.update(override.get(strategy_name, {}))
        
        window = params.get('percentile_window', 20)
        percentile = params.get('percentile_high', 95)
        # cooldown = params.get('signal_cooldown_seconds', 300) # 移到外面
        threshold = params.get('threshold', 1.3)
        
        if len(bars) < params.get('history_min_len', 5): return None
        
        prices = [float(b.get('avg_price', b['close'])) for b in bars[-window:]]
        if not prices: return None
        
        upper = np.percentile(prices, percentile)
        mean = np.mean(prices)
        
        condition = False
        if mean < 0:
            if tick.price > 0: 
                condition = (tick.price - mean) >= abs(mean) * threshold
            else: 
                # condition = tick.price > upper and tick.price > mean / threshold
                condition = tick.price > mean / threshold
        else:
            # condition = tick.price > upper and tick.price > threshold * mean
            condition = tick.price > threshold * mean
        
        logger.info(f"时间：{tick.timestamp} 检查极端卖出: {tick.contract_name} 价格={tick.price:.2f}, 均价={mean:.2f}, 上边界={upper:.2f}, 条件={condition}")
            
        if condition:
            # 检查是否启用了动态仓位
            if params.get('use_dynamic_sizing', False):
                # max_pos = self._calculate_liquidity_based_size(tick, bars, positions, params)
                max_pos = self._calculate_time_based_limit(tick)

            size = self._calculate_action_and_size(tick.contract_name, positions, max_pos, params, ActionType.SELL)
            is_valid = size > 0.001
            reason = "" if is_valid else "Position Limit Reached (Size=0)"
            
            adj_price = max(tick.price * 0.98, mean * 1.3)
            
            return TradeSignal(now, tick.contract_name, tick.contract_id, ActionType.SELL, size, round(adj_price, 2), strategy_name, tick.delivery_start, open_strategy=strategy_name, z_score=0.0, mean_price=round(mean,2), std_price=0.0, trend_info=f"Upper{percentile}:{round(upper,2)}", raw_size=max_pos, is_valid=is_valid, failure_reason=reason)
        return None

    def _calculate_liquidity_based_size(self, 
                                      tick: TickEvent, 
                                      bars: List[dict], 
                                      positions: Dict[str, Position],
                                      params: Dict) -> float:
        """
        🌊 基于流速 (Flow Rate) 的动态开仓量计算
        """
        # 1. 提取参数
        lookback = params.get('liquidity_lookback', 30)
        rate = params.get('liquidity_participation', 0.05)
        projection_mode = params.get('liquidity_projection', '60')
        min_size = params.get('min_open_size', 0.1)

        # 2. 计算流速
        cutoff = tick.timestamp - timedelta(minutes=lookback)
        # 过滤出最近 N 分钟的 Bar
        recent_bars = [b for b in bars if b['start_time'] >= cutoff]
        
        if not recent_bars:
            return 0.0

        total_vol = sum(b['volume'] for b in recent_bars)
        # 即使 bar 不足 30 根，也除以 30，这反映了"时间上的真实稀疏度"
        avg_flow_rate = total_vol / lookback  # MW per minute

        if projection_mode == 'till_close':
            gate_closure = tick.delivery_start - timedelta(hours=1)
            minutes_left = (gate_closure - tick.timestamp).total_seconds() / 60.0
            projection_minutes = max(0, minutes_left)
        else:
            projection_minutes = timedelta(minutes=int(projection_mode)).total_seconds() / 60.0  # 默认 60 分钟

        # 3. 预测未来容量 (默认预测 60 分钟)
        projected_capacity = avg_flow_rate * projection_minutes

        # 4. 计算目标下单量
        target_size = projected_capacity * rate
        target_size = round(target_size, 1)

        # 5. 🛡️ 防火墙 2：全局资金/持仓硬性兜底
        # 即使流速允许买 100MW，我们也不能超过账户允许的最大持仓
        # total_holdings = sum(abs(p.size) for p in positions.values())
        # global_avail = max(0.0, self.max_position_size - total_holdings)
        
        # final_size = min(target_size, global_avail)

        if target_size < min_size:
            return 0.0
            
        # 记录一下，方便调试
        # logger.info(f"🌊 [{tick.contract_name}] 流速: {avg_flow_rate:.2f} MW/m -> 建议: {target_size} MW")
        
        return target_size
    
    def _calculate_time_based_limit(self, tick: TickEvent) -> float:
        """
        ⏳ 基于时间衰减的动态持仓上限
        规则：每剩余1个完整的30分钟 = 1 MW 上限
        """
        # 1. 计算关闸时间 (交付前1小时)
        gate_closure = tick.delivery_start - timedelta(hours=1)
        
        # 2. 如果已经超过关闸时间，上限为0
        if tick.timestamp >= gate_closure:
            return 0.0
        
        # 3. 计算剩余分钟数
        delta = gate_closure - tick.timestamp
        minutes_remaining = delta.total_seconds() / 60.0
        
        # 4. 计算完整的30分钟个数 (向下取整)
        # 比如 65分钟 -> 2 个 30分钟
        # 比如 59分钟 -> 1 个 30分钟
        intervals = int(minutes_remaining // 30)
        
        # 5. 1个区间 = 1MW
        limit = float(intervals * 4.0)
        
        # logger.info(f"⏳ [{tick.contract_name}] 剩余时间: {minutes_remaining:.1f}m -> {intervals}个30分钟 -> 上限: {limit}MW")
        return max(0.0, limit)