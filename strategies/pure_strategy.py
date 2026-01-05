import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from scipy.stats import linregress 
from collections import deque

from core.models import TickEvent, TradeSignal, ActionType, Position, Order

logger = logging.getLogger("PureStrategy")

class PureStrategy:
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

        # --- 【新增】诊断与生命周期管理 ---
        self.tick_counter = 0
        self.current_date = None
        self.daily_realized_pnl = 0.0
        self.is_risk_triggered = False
        
        # --- 【新增】内置Bar合成器 (Tick -> Bar) ---
        # 你的策略依赖 bars 列表，我们需要在 Tick 回测中实时合成它
        self.bars: List[dict] = []
        self.current_bar: Optional[dict] = None

    # ----------------------------------------------------------------
    # 【新增】生命周期方法 (Engine 调用)
    # ----------------------------------------------------------------
    
    def on_new_day(self, date_str: str):
        """跨天重置逻辑"""
        logger.info(f"📅 新的一天: {date_str} (昨日PnL: {self.daily_realized_pnl:.2f})")
        self.daily_realized_pnl = 0.0
        self.is_risk_triggered = False
        self.current_date = date_str
        # 可选：清空当天的 K 线缓存，防止内存溢出，或者保留用于计算长周期趋势
        # self.bars = [] 

    def update_pnl(self, pnl: float):
        """更新策略感知的 PnL"""
        self.daily_realized_pnl += pnl

    def on_tick(self, tick: TickEvent, positions: Dict[str, Position], active_orders: List[Order], account_info) -> Optional[TradeSignal]:
        """
        主入口函数 (适配 BacktestEngine)
        """
        self.tick_counter += 1
        
        # 1. 自动检测日期变更
        tick_date = tick.timestamp.strftime("%Y-%m-%d")
        if self.current_date != tick_date:
            self.on_new_day(tick_date)
            
        # 2. 【核心】实时合成 K 线 (1分钟 Bar)
        # 你的 _check_trend_analysis 需要 bars 数据
        self._update_bars(tick)

        # 3. 诊断心跳 (每 10,000 Tick)
        if self.tick_counter % 10000 == 0:
            logger.info(f"💓 策略心跳: PnL={self.daily_realized_pnl:.2f}, 风控锁={self.is_risk_triggered}, Bars={len(self.bars)}")

        # 4. 全局日内风控检查
        if self.is_risk_triggered:
            return None
            
        if self.daily_realized_pnl <= -self.daily_loss_limit:
            if not self.is_risk_triggered:
                logger.error(f"💥 触发日内亏损限额! PnL: {self.daily_realized_pnl:.2f} <= -{self.daily_loss_limit}. 停止今日开仓。")
                self.is_risk_triggered = True
            return None

        # 5. 调用你原来的逻辑
        # 注意：Engine 期望返回单个 Signal，而 calculate_signals 返回 List
        # 我们取第一个有效的
        signals = self.calculate_signals(tick, self.bars, positions, tick.timestamp, self.daily_realized_pnl)
        
        if signals:
            # 返回优先级最高的信号 (List中的第一个)
            return signals[0]
            
        return None

    def _update_bars(self, tick: TickEvent):
        """简易的 K 线合成器 (1分钟)"""
        # 如果是新的分钟，归档旧 Bar
        if self.current_bar and tick.timestamp.minute != self.current_bar['start_time'].minute:
            self.bars.append(self.current_bar)
            # 保持 Bars 长度在合理范围 (例如只保留最近 300 根用于趋势计算)
            if len(self.bars) > 300:
                self.bars.pop(0)
            self.current_bar = None
            
        # 更新或创建当前 Bar
        if not self.current_bar:
            self.current_bar = {
                'start_time': tick.timestamp,
                'open': tick.price,
                'high': tick.price,
                'low': tick.price,
                'close': tick.price,
                'avg_price': tick.price, # 暂用 Close 代替 VWAP
                'volume': tick.volume
            }
        else:
            self.current_bar['high'] = max(self.current_bar['high'], tick.price)
            self.current_bar['low'] = min(self.current_bar['low'], tick.price)
            self.current_bar['close'] = tick.price
            self.current_bar['volume'] += tick.volume
            # 简单平均价更新 (非严格 VWAP，但足够回测用)
            self.current_bar['avg_price'] = (self.current_bar['avg_price'] + tick.price) / 2

    # ----------------------------------------------------------------
    # 以下是你原来的业务逻辑 (保持不变)
    # ----------------------------------------------------------------

    def calculate_signals(self, 
                          tick: TickEvent, 
                          bars: List[dict], 
                          positions: Dict[str, Position], 
                          current_time: datetime,
                          current_daily_pnl: float = 0.0) -> List[TradeSignal]:
        
        self._update_tick_history(tick)
        signals = []

        # 0. 基础环境检查
        if abs(tick.price) < self.min_price_for_new_position:
            if tick.contract_name not in positions:
                return []

        # --- 策略 1: 均值回归 ---
        sig_mr = self._check_mean_reversion(tick, bars, positions, current_time)
        if sig_mr:
            self._apply_risk_checks(sig_mr, tick, bars, positions, current_time, current_daily_pnl)
            signals.append(sig_mr)

        # --- 策略 2: 极端价格 ---
        sig_ext = self._check_extreme_sell(tick, bars, positions, current_time)
        if sig_ext:
            self._apply_risk_checks(sig_ext, tick, bars, positions, current_time, current_daily_pnl)
            signals.append(sig_ext)
        
        # --- 策略 3: 高波动 ---
        sig_vol = self._high_volatility_dip_buy(tick, positions, current_time)
        if sig_vol:
            self._apply_risk_checks(sig_vol, tick, bars, positions, current_time, current_daily_pnl, skip_trend=True)
            signals.append(sig_vol)

        # --- 策略 4: 交付时间 ---
        sig_del = self._delivery_time_buy_strategy(tick, positions, current_time)
        if sig_del:
            self._apply_risk_checks(sig_del, tick, bars, positions, current_time, current_daily_pnl, skip_trend=True, skip_close_time=True)
            signals.append(sig_del)

        return signals
    
    def _apply_risk_checks(self, signal: TradeSignal, tick: TickEvent, bars: List[dict], 
                           positions: Dict[str, Position], current_time: datetime, 
                           current_daily_pnl: float, 
                           skip_trend: bool = False, skip_close_time: bool = False):
        """
        统一风控检查流程。
        """
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
        if not skip_trend:
            if not self._check_trend_analysis(signal, bars):
                signal.is_valid = False
                if not signal.failure_reason:
                    signal.failure_reason = "Trend Analysis Failed"
                return

        # 3. 通用信号验证
        if not self._validate_signal(signal, positions):
            signal.is_valid = False
            return

        # 4. 日亏损限制
        if current_daily_pnl < -self.daily_loss_limit:
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

    def _check_cooldown(self, signal: TradeSignal, current_time: datetime) -> bool:
        strategy_name = signal.strategy_name
        cooldown = self.params.get('signal_cooldown_seconds', 300)
        
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
        total_holdings = sum(abs(p.size) for p in positions.values())
        global_avail = max(0.0, self.max_position_size - total_holdings)
        pos = positions.get(contract_name)
        curr_size = abs(pos.size) if pos else 0.0
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

    def _check_mean_reversion(self, tick: TickEvent, bars: List[dict], positions: Dict, now: datetime) -> Optional[TradeSignal]:
        strategy_name = "super_mean_reversion_buy"
        max_pos, override = self._get_delivery_rule_config(tick.delivery_start)
        params = self.params.get(strategy_name, {}).copy()
        params.update(override.get(strategy_name, {}))
        
        window = params.get('ma_window', 20)
        threshold = params.get('threshold', 2.0)
        std_ratio = params.get('std_ratio_threshold', 0.05)
        
        if len(bars) < params.get('history_min_len', 5): return None
        
        prices = [float(b.get('avg_price', b['close'])) for b in bars[-window:]]
        if not prices: return None
        
        mean = np.mean(prices)
        std = np.std(prices)
        
        if std <= abs(mean * std_ratio): return None
        if std == 0: return None
        
        z_score = (tick.price - mean) / std
        
        if z_score <= -threshold:
            size = self._calculate_action_and_size(tick.contract_name, positions, max_pos, params, ActionType.BUY)
            is_valid = size > 0.001
            reason = "" if is_valid else "Position Limit Reached (Size=0)"
            
            return TradeSignal(now, tick.contract_name, tick.contract_id, ActionType.BUY, size, tick.price, strategy_name, tick.delivery_start, confidence=min(abs(z_score)/threshold, 1.0), open_strategy=strategy_name, z_score=round(z_score,3), mean_price=round(mean,2), std_price=round(std,2), raw_size=size, is_valid=is_valid, failure_reason=reason)
        return None

    def _check_extreme_sell(self, tick: TickEvent, bars: List[dict], positions: Dict, now: datetime) -> Optional[TradeSignal]:
        strategy_name = "optimized_extreme_sell"
        max_pos, override = self._get_delivery_rule_config(tick.delivery_start)
        params = self.params.get(strategy_name, {}).copy()
        params.update(override.get(strategy_name, {}))
        
        window = params.get('percentile_window', 20)
        percentile = params.get('percentile_high', 95)
        threshold = params.get('threshold', 1.2)
        
        if len(bars) < params.get('history_min_len', 5): return None
        
        prices = [float(b.get('avg_price', b['close'])) for b in bars[-window:]]
        if not prices: return None
        
        upper = np.percentile(prices, percentile)
        mean = np.mean(prices)
        
        condition = False
        if mean < 0:
            if tick.price > 0: condition = (tick.price - mean) >= abs(mean) * threshold
            else: condition = tick.price > upper and tick.price > mean / threshold
        else:
            condition = tick.price > upper and tick.price > threshold * mean
            
        if condition:
            size = self._calculate_action_and_size(tick.contract_name, positions, max_pos, params, ActionType.SELL)
            is_valid = size > 0.001
            reason = "" if is_valid else "Position Limit Reached (Size=0)"
            
            adj_price = max(tick.price * 0.98, mean * 1.2)
            
            return TradeSignal(now, tick.contract_name, tick.contract_id, ActionType.SELL, size, round(adj_price, 2), strategy_name, tick.delivery_start, open_strategy=strategy_name, z_score=0.0, mean_price=round(mean,2), std_price=0.0, trend_info=f"Upper{percentile}:{round(upper,2)}", raw_size=size, is_valid=is_valid, failure_reason=reason)
        return None

    def _high_volatility_dip_buy(self, tick: TickEvent, positions: Dict, now: datetime) -> Optional[TradeSignal]:
        strategy_name = "high_volatility_dip_buy"
        max_pos, override = self._get_delivery_rule_config(tick.delivery_start)
        params = self.params.get(strategy_name, {}).copy()
        params.update(override.get(strategy_name, {}))
        
        prices = self.price_history.get(tick.contract_name, [])
        if len(prices) < 20: return None
        
        recent = prices[-24:]
        vol = np.std(recent)
        min_p = min(recent[-5:])
        
        if vol >= params.get('threshold', 50.0) and tick.price <= min_p:
            size = self._calculate_action_and_size(tick.contract_name, positions, max_pos, params, ActionType.BUY)
            is_valid = size > 0.001
            reason = "" if is_valid else "Position Limit Reached (Size=0)"
            return TradeSignal(now, tick.contract_name, tick.contract_id, ActionType.BUY, size, tick.price, strategy_name, tick.delivery_start, confidence=0.7, open_strategy=strategy_name, std_price=round(vol,2), raw_size=size, is_valid=is_valid, failure_reason=reason)
        return None

    def _delivery_time_buy_strategy(self, tick: TickEvent, positions: Dict, now: datetime) -> Optional[TradeSignal]:
        strategy_name = "delivery_time_buy"
        if tick.contract_name in self.delivery_time_strategy_executed: return None
        max_pos, override = self._get_delivery_rule_config(tick.delivery_start)
        params = self.params.get(strategy_name, {}).copy()
        params.update(override.get(strategy_name, {}))
        if 'delivery_time_buy' in override:
            self.delivery_time_strategy_executed.add(tick.contract_name)
            size = self._calculate_action_and_size(tick.contract_name, positions, max_pos, params, ActionType.BUY)
            is_valid = size > 0.001
            reason = "" if is_valid else "Position Limit Reached (Size=0)"
            return TradeSignal(now, tick.contract_name, tick.contract_id, ActionType.BUY, size, tick.price, strategy_name, tick.delivery_start, confidence=0.7, open_strategy=strategy_name, raw_size=size, is_valid=is_valid, failure_reason=reason)
        return None