import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime, timedelta
from scipy.stats import linregress 

from core.models import TickEvent, TradeSignal, ActionType, Position

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
        
        # 缓存 Tick 级历史数据 (用于高波动策略)
        self.price_history: Dict[str, List[float]] = {}
        # 交付时间策略执行记录
        self.delivery_time_strategy_executed: Set[str] = set()

    def calculate_signals(self, 
                          tick: TickEvent, 
                          bars: List[dict], 
                          positions: Dict[str, Position], 
                          current_time: datetime,
                          current_daily_pnl: float = 0.0) -> List[TradeSignal]:
        
        # 更新 Tick 级历史数据
        self._update_tick_history(tick)
        
        signals = []

        # 0. 基础风控检查
        if abs(tick.price) < self.min_price_for_new_position:
            if tick.contract_name not in positions:
                return []

        # 1. 均值回归策略 (仅做多)
        sig_mr = self._check_mean_reversion(tick, bars, positions, current_time)
        if sig_mr:
            if self._check_trend_analysis(sig_mr, bars):
                if self._validate_signal(sig_mr, positions):
                    if current_daily_pnl >= -self.daily_loss_limit:
                        if self._check_time_to_close(tick.delivery_start, current_time):
                            signals.append(sig_mr)

        # 2. 极端价格策略 (仅做空)
        sig_ext = self._check_extreme_sell(tick, bars, positions, current_time)
        if sig_ext:
            if self._check_trend_analysis(sig_ext, bars):
                if self._validate_signal(sig_ext, positions):
                    if current_daily_pnl >= -self.daily_loss_limit:
                        if self._check_time_to_close(tick.delivery_start, current_time):
                            signals.append(sig_ext)
        
        # 3. 高波动逢低买入策略
        sig_vol = self._high_volatility_dip_buy(tick, positions, current_time)
        if sig_vol:
            if self._validate_signal(sig_vol, positions):
                if current_daily_pnl >= -self.daily_loss_limit:
                    if self._check_time_to_close(tick.delivery_start, current_time):
                        signals.append(sig_vol)

        # 4. 交付时间策略
        sig_del = self._delivery_time_buy_strategy(tick, positions, current_time)
        if sig_del:
            if self._validate_signal(sig_del, positions):
                if current_daily_pnl >= -self.daily_loss_limit:
                    signals.append(sig_del)

        return signals
    
    def _update_tick_history(self, tick: TickEvent):
        contract = tick.contract_name
        if contract not in self.price_history:
            self.price_history[contract] = []
        self.price_history[contract].append(tick.price)
        if len(self.price_history[contract]) > 100:
            self.price_history[contract].pop(0)

    # ==================================================================================
    # 【核心修复】趋势分析逻辑：完全对齐 RiskManager 的数据选取逻辑
    # ==================================================================================

    def _check_trend_analysis(self, signal: TradeSignal, bars: List[dict]) -> bool:
        """
        [RiskManager复刻] 获取最近30分钟窗口内的最近10条数据进行趋势分析
        """
        if signal.strategy_name not in ["super_mean_reversion_buy", "optimized_extreme_sell"]:
            return True
        
        # 1. 确定时间窗口：最近30分钟
        cutoff_time = signal.timestamp - timedelta(minutes=30)
        
        # 2. 选取数据：实盘逻辑是 limit=10
        # 在回测的 bars 列表中，这是最后 10 个 bar
        potential_bars = bars[-10:]
        
        # 3. 过滤时间：确保这 10 个 bar 确实是在 30 分钟内的
        valid_bars = [b for b in potential_bars if b['start_time'] >= cutoff_time]
        
        # 4. 提取价格：使用 avg_price
        price_list = [float(b.get('avg_price', b['close'])) for b in valid_bars]
        
        # 5. 数量检查：少于3条则拦截 (RiskManager逻辑: if len < 3: return False)
        if len(price_list) < 3:
            # logger.info(f"趋势分析拦截: 数据不足 {len(price_list)} < 3")
            return False
            
        # 6. 进行线性回归分析
        long_trend_result = self.detect_trend_with_linear_regression(price_list)
        long_trend = long_trend_result["trend"]
        long_confidence = long_trend_result.get("confidence", 0.0)
        
        signal.trend_info = f"{long_trend} (Conf:{long_confidence:.2f}, R2:{long_trend_result['r_squared']:.2f})"
        
        # 7. 根据策略类型进行拦截或减仓
        if signal.strategy_name == "super_mean_reversion_buy":
            if (long_trend == "下降" and long_confidence >= 0.6):
                return False 
            
            elif (long_trend == "下降" and long_confidence < 0.6) or (long_trend != "下降" and long_confidence < 0.6):
                temp_conf = long_confidence
                if long_trend != '下降':
                    temp_conf = 0.6 - long_confidence
                
                adjustment_factor = (0.6 - temp_conf) / 2
                
                prev_size = signal.size
                adjusted_size = round(prev_size * adjustment_factor, 1)
                
                if adjusted_size < 0.1:
                    return False
                
                signal.size = adjusted_size
                if signal.contract_name.startswith("QH"):
                    signal.size = signal.size * 2
        
        elif signal.strategy_name == "optimized_extreme_sell":
            if (long_trend == "上升" and long_confidence >= 0.6):
                return False
                
            elif (long_trend == "上升" and long_confidence < 0.6) or (long_trend != "上升" and long_confidence < 0.6):
                temp_conf = long_confidence
                if long_trend != '上升':
                    temp_conf = 0.6 - long_confidence
                
                adjustment_factor = (0.6 - temp_conf) / 2
                
                prev_size = signal.size
                adjusted_size = round(prev_size * adjustment_factor, 1)
                
                if adjusted_size < 0.1:
                    return False
                
                signal.size = adjusted_size
                if signal.contract_name.startswith("QH"):
                    signal.size = signal.size * 2
                    
        return True

    def detect_trend_with_linear_regression(self, prices: List[float], 
                                        window_size: int = 3, 
                                        slope_threshold: float = 0.1) -> Dict:
        """[复刻 RiskManager] 线性回归趋势检测"""
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
        
        if abs(slope) < slope_threshold:
            trend = "平滑"
        elif slope > slope_threshold:
            trend = "上升"
        else:
            trend = "下降"
        
        confidence = self.calculate_trend_confidence(r_squared, p_value, len(prices_arr))
        
        return {
            "trend": trend,
            "slope": float(slope),
            "r_squared": float(r_squared),
            "confidence": float(confidence)
        }

    def calculate_trend_confidence(self, r_squared: float, p_value: float, data_points: int) -> float:
        """[复刻 RiskManager] 计算置信度"""
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

    # ----------------------------------------------------------------
    # 策略逻辑部分 (保持不变，已使用正确的参数名和avg_price)
    # ----------------------------------------------------------------

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
                    return False

            five_minutes_ago = signal.timestamp - timedelta(minutes=5)
            if existing_position.timestamp >= five_minutes_ago:
                return False
        return True

    def _get_delivery_rule_config(self, delivery_start: str) -> Tuple[float, Dict]:
        current_max_pos = self.default_contract_max_position
        params_override = {}
        if not delivery_start: return current_max_pos, params_override
        try:
            dt = datetime.strptime(delivery_start, '%Y-%m-%d %H:%M:%S')
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
                            if 'max_position' in time_range:
                                current_max_pos = float(time_range['max_position'])
                            if 'strategy_params' in time_range:
                                params_override = time_range['strategy_params']
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

    def _check_time_to_close(self, delivery_start: str, current_time: datetime) -> bool:
        if not delivery_start: return True
        try:
            delivery_dt = datetime.strptime(delivery_start, '%Y-%m-%d %H:%M:%S')
            gate_closure = delivery_dt - timedelta(hours=1)
            forbid_time = gate_closure - timedelta(minutes=self.forbid_new_open_minutes)
            return current_time < forbid_time
        except Exception: return True

    def _check_mean_reversion(self, tick: TickEvent, bars: List[dict], positions: Dict, now: datetime) -> Optional[TradeSignal]:
        strategy_name = "super_mean_reversion_buy"
        max_pos, override = self._get_delivery_rule_config(tick.delivery_start)
        params = self.params.get(strategy_name, {}).copy()
        params.update(override.get(strategy_name, {}))
        
        window = params.get('ma_window', 20)
        threshold = params.get('threshold', 2.0)
        cooldown = params.get('signal_cooldown_seconds', 300)
        std_ratio = params.get('std_ratio_threshold', 0.05)
        
        if len(bars) < params.get('history_min_len', 5): return None
        
        last = self.last_trade_times.get(tick.contract_name + strategy_name)
        if last and (now - last).total_seconds() < cooldown: return None

        prices = [float(b.get('avg_price', b['close'])) for b in bars[-window:]]
        if not prices: return None
        
        mean = np.mean(prices)
        std = np.std(prices)
        
        if std <= abs(mean * std_ratio): return None
        if std == 0: return None
        
        z_score = (tick.price - mean) / std
        
        if z_score <= -threshold:
            size = self._calculate_action_and_size(tick.contract_name, positions, max_pos, params, ActionType.BUY)
            if size > 0:
                self.last_trade_times[tick.contract_name + strategy_name] = now
                return TradeSignal(now, tick.contract_name, tick.contract_id, ActionType.BUY, size, tick.price, strategy_name, tick.delivery_start, confidence=min(abs(z_score)/threshold, 1.0), open_strategy=strategy_name, z_score=round(z_score,3), mean_price=round(mean,2), std_price=round(std,2), raw_size=size)
        return None

    def _check_extreme_sell(self, tick: TickEvent, bars: List[dict], positions: Dict, now: datetime) -> Optional[TradeSignal]:
        strategy_name = "optimized_extreme_sell"
        max_pos, override = self._get_delivery_rule_config(tick.delivery_start)
        params = self.params.get(strategy_name, {}).copy()
        params.update(override.get(strategy_name, {}))
        
        window = params.get('percentile_window', 20)
        percentile = params.get('percentile_high', 95)
        cooldown = params.get('signal_cooldown_seconds', 300)
        threshold = params.get('threshold', 1.2)
        
        if len(bars) < params.get('history_min_len', 5): return None
        last = self.last_trade_times.get(tick.contract_name + strategy_name)
        if last and (now - last).total_seconds() < cooldown: return None
        
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
            if size > 0:
                self.last_trade_times[tick.contract_name + strategy_name] = now
                adj_price = max(tick.price * 0.98, mean * 1.2) # 简化计算：折扣2%与溢价20%取大
                
                return TradeSignal(now, tick.contract_name, tick.contract_id, ActionType.SELL, size, round(adj_price, 2), strategy_name, tick.delivery_start, open_strategy=strategy_name, z_score=0.0, mean_price=round(mean,2), std_price=0.0, trend_info=f"Upper{percentile}:{round(upper,2)}", raw_size=size)
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
            if size > 0:
                return TradeSignal(now, tick.contract_name, tick.contract_id, ActionType.BUY, size, tick.price, strategy_name, tick.delivery_start, confidence=0.7, open_strategy=strategy_name, std_price=round(vol,2), raw_size=size)
        return None

    def _delivery_time_buy_strategy(self, tick: TickEvent, positions: Dict, now: datetime) -> Optional[TradeSignal]:
        strategy_name = "delivery_time_buy"
        if tick.contract_name in self.delivery_time_strategy_executed: return None
        
        max_pos, override = self._get_delivery_rule_config(tick.delivery_start)
        params = self.params.get(strategy_name, {}).copy()
        params.update(override.get(strategy_name, {}))
        
        # 简单判断：如果delivery_rules里激活了该策略（params不为空），则执行
        if 'delivery_time_buy' in override:
            self.delivery_time_strategy_executed.add(tick.contract_name)
            size = self._calculate_action_and_size(tick.contract_name, positions, max_pos, params, ActionType.BUY)
            if size > 0:
                return TradeSignal(now, tick.contract_name, tick.contract_id, ActionType.BUY, size, tick.price, strategy_name, tick.delivery_start, confidence=0.7, open_strategy=strategy_name, raw_size=size)
        return None