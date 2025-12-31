import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime, timedelta
from scipy.stats import linregress  # 【关键】引入实盘使用的库

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
        
        # 更新 Tick 级历史数据 (用于高波动策略)
        self._update_tick_history(tick)
        
        signals = []

        # 0. 基础风控检查
        # 价格太低且无持仓时，不开新仓
        if abs(tick.price) < self.min_price_for_new_position:
            if tick.contract_name not in positions:
                return []

        # 1. 均值回归策略 (仅做多)
        sig_mr = self._check_mean_reversion(tick, bars, positions, current_time)
        if sig_mr:
            # 趋势过滤
            if self._check_trend_analysis(sig_mr, bars):
                if self._validate_signal(sig_mr, positions):
                    # 只有未达日亏损限额才允许开仓
                    if current_daily_pnl >= -self.daily_loss_limit:
                        # 临近关闸检查
                        if self._check_time_to_close(tick.delivery_start, current_time):
                            signals.append(sig_mr)

        # 2. 极端价格策略 (仅做空)
        sig_ext = self._check_extreme_sell(tick, bars, positions, current_time)
        if sig_ext:
            # 趋势过滤
            if self._check_trend_analysis(sig_ext, bars):
                if self._validate_signal(sig_ext, positions):
                    if current_daily_pnl >= -self.daily_loss_limit:
                        if self._check_time_to_close(tick.delivery_start, current_time):
                            signals.append(sig_ext)
        
        # 3. 【新增】高波动逢低买入策略 (High Volatility Dip Buy)
        sig_vol = self._high_volatility_dip_buy(tick, positions, current_time)
        if sig_vol:
            # 高波动策略通常比较激进，可能不需要趋势过滤或有独立的逻辑
            # 这里保持原逻辑：只做基础验证
            if self._validate_signal(sig_vol, positions):
                if current_daily_pnl >= -self.daily_loss_limit:
                    if self._check_time_to_close(tick.delivery_start, current_time):
                        signals.append(sig_vol)

        # 4. 【新增】交付时间策略 (Delivery Time Buy)
        # 注意：原逻辑中 delivery_time_buy 似乎不受某些风控限制 (如 forbid_new_open_minutes)
        # 但我们这里还是加上基础验证
        sig_del = self._delivery_time_buy_strategy(tick, positions, current_time)
        if sig_del:
            if self._validate_signal(sig_del, positions):
                if current_daily_pnl >= -self.daily_loss_limit:
                    signals.append(sig_del)

        return signals
    
    def _update_tick_history(self, tick: TickEvent):
        """维护 Tick 级别的价格历史，用于计算实时波动率"""
        contract = tick.contract_name
        if contract not in self.price_history:
            self.price_history[contract] = []
        self.price_history[contract].append(tick.price)
        # 只保留最近 100 个 Tick
        if len(self.price_history[contract]) > 100:
            self.price_history[contract].pop(0)

    # ==================================================================================
    # 【核心移植】完全复刻 RiskManager 中的趋势计算逻辑
    # ==================================================================================

    def detect_trend_with_linear_regression(self, prices: List[float], 
                                        window_size: int = 3, 
                                        slope_threshold: float = 0.1) -> Dict:
        """
        [原版逻辑复刻] 使用 scipy.stats.linregress 进行线性回归趋势检测
        包含平滑处理、R方、P值计算
        """
        # 过滤掉None值和非数值数据
        filtered_prices = []
        for price in prices:
            if price is not None:
                try:
                    filtered_prices.append(float(price))
                except (ValueError, TypeError):
                    continue
        
        prices_arr = np.array(filtered_prices, dtype=float)
        
        # 检查数据充足性
        if len(prices_arr) < window_size:
            return {
                "trend": "数据不足",
                "slope": 0.0,
                "r_squared": 0.0,
                "p_value": 1.0,
                "data_points": len(prices_arr),
                "confidence": 0.0
            }
        
        # 检查是否所有价格都相同（无波动）
        if np.all(prices_arr == prices_arr[0]):
            return {
                "trend": "平滑",
                "slope": 0.0,
                "r_squared": 1.0,
                "p_value": 0.0,
                "data_points": len(prices_arr),
                "confidence": 1.0
            }
        
        # 移动窗口平滑 (与原代码一致: center=True, min_periods=1)
        prices_series = pd.Series(prices_arr)
        smoothed = prices_series.rolling(window=window_size, center=True, min_periods=1).mean()
        
        # 线性回归分析
        x = np.arange(len(smoothed))
        # 使用 scipy 的 linregress
        slope, intercept, r_value, p_value, std_err = linregress(x, smoothed.values)
        r_squared = r_value ** 2
        
        # 趋势判断
        if abs(slope) < slope_threshold:
            trend = "平滑"
        elif slope > slope_threshold:
            trend = "上升"
        else:
            trend = "下降"
        
        # 计算置信度
        confidence = self.calculate_trend_confidence(r_squared, p_value, len(prices_arr))
        
        return {
            "trend": trend,
            "slope": float(slope),
            "r_squared": float(r_squared),
            "p_value": float(p_value),
            "data_points": len(prices_arr),
            "confidence": float(confidence)
        }

    def calculate_trend_confidence(self, r_squared: float, p_value: float, data_points: int) -> float:
        """
        [原版逻辑复刻] 计算趋势判断的置信度
        """
        # 基础置信度来自R²值
        base_confidence = r_squared
        
        # P值调整：P值越小，置信度越高
        if p_value < 0.001:
            p_adjustment = 1.0
        elif p_value < 0.01:
            p_adjustment = 0.9
        elif p_value < 0.05:
            p_adjustment = 0.8
        elif p_value < 0.1:
            p_adjustment = 0.6
        else:
            p_adjustment = 0.3
        
        # 数据点数量调整：数据点越多，置信度越高
        if data_points >= 50:
            data_adjustment = 1.0
        elif data_points >= 40:
            data_adjustment = 0.9
        elif data_points >= 30:
            data_adjustment = 0.85
        elif data_points >= 20:
            data_adjustment = 0.8
        elif data_points >= 5:
            data_adjustment = 0.7          
        else:
            data_adjustment = 0.6
        
        # 综合置信度
        confidence = base_confidence * p_adjustment * data_adjustment
        
        # 确保在0-1范围内
        return min(max(confidence, 0.0), 1.0)

    def _check_trend_analysis(self, signal: TradeSignal, bars: List[dict]) -> bool:
        """
        [原版逻辑复刻] 对应 RiskManager._check_trend_analysis
        使用 30分钟的数据 (即最近 30 个 1分钟 Bar，或者由 limit 参数决定)
        """
        # 只处理特定策略的信号
        if signal.strategy_name not in ["super_mean_reversion_buy", "optimized_extreme_sell"]:
            return True
        
        # 实盘逻辑：获取近30分钟的市场数据 (limit=10 5min bars -> 30 1min bars in our corrected generator?)
        # 原 RiskManager 使用 limit=10 的 get_five_min_bars。
        # 如果我们的 bar_generator 生成的是 1min bar，那么 30 分钟就是 30 个点。
        # 但为了稳妥，我们提取最近 30 个数据点进行分析。
        recent_bars = bars[-30:] 
        
        # 提取 avg_price (如果存在) 或 close
        price_list = [float(b.get('avg_price', b['close'])) for b in recent_bars]
        
        # 检查数据点数量：如果数量小于3，则取消信号 (原代码逻辑)
        if len(price_list) < 3:
            # logger.info(f"取消信号: 数据点不足 {len(price_list)}")
            return False
            
        # 进行趋势分析
        long_trend_result = self.detect_trend_with_linear_regression(price_list)
        long_trend = long_trend_result["trend"]
        long_confidence = long_trend_result.get("confidence", 0.0)
        
        # 将趋势信息写入信号用于记录
        signal.trend_info = f"{long_trend} (Conf:{long_confidence:.2f}, R2:{long_trend_result['r_squared']:.2f})"
        
        # 1. 均值回归 (做多)
        if signal.strategy_name == "super_mean_reversion_buy":
            # 如果趋势为下降且置信度>=0.6，取消信号
            if (long_trend == "下降" and long_confidence >= 0.6):
                return False 
            
            # 如果趋势为下降且置信度<0.6，调整开仓量
            elif (long_trend == "下降" and long_confidence < 0.6) or (long_trend != "下降" and long_confidence < 0.6):
                # 原始逻辑：如果不是下降，也可能调整？
                # 原代码：(long_trend == "下降" and long_confidence < 0.6) or (long_trend != "下降" and long_confidence < 0.6)
                # 这里的逻辑是：只要置信度低，就可能调整？
                # 原代码有一段：if (long_trend != '下降'): long_confidence = 0.6 - long_confidence
                
                temp_conf = long_confidence
                if long_trend != '下降':
                    temp_conf = 0.6 - long_confidence # 这是一个非常特殊的调整
                
                adjustment_factor = (0.6 - temp_conf) / 2
                # 防止除以零或负数，原代码逻辑比较复杂，这里尽量简化还原
                # 原代码: adjustment_factor = (0.6 - long_confidence) / 2 (在if内部)
                
                prev_size = signal.size
                adjusted_size = round(prev_size * adjustment_factor, 1)
                
                if adjusted_size < 0.1:
                    return False
                
                signal.size = adjusted_size
                # QH 特殊调整
                if signal.contract_name.startswith("QH"):
                    signal.size = signal.size * 2
        
        # 2. 极端价格 (做空)
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

    # ==================================================================================
    # 下面是常规策略逻辑和辅助函数 (保持不变或微调)
    # ==================================================================================

    def _validate_signal(self, signal: TradeSignal, positions: Dict[str, Position]) -> bool:
        # 对齐原版 _validate_signal
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

    def _calculate_action_and_size(self, 
                                   contract_name: str, 
                                   positions: Dict[str, Position], 
                                   contract_max_position: float,
                                   strategy_params: Dict, 
                                   action_type: ActionType) -> float:
        position_ratio = strategy_params.get('position_ratio', 0.5)
        position_split = strategy_params.get('position_split', 3)
        min_open_size = strategy_params.get('min_open_size', 0.1)
        
        desired_size = contract_max_position * position_ratio / position_split
        desired_size = round(desired_size, 1)
        desired_size = max(desired_size, min_open_size)
        
        total_holdings = sum(abs(p.size) for p in positions.values())
        total_holdings = round(total_holdings, 1) 
        global_available = max(0.0, self.max_position_size - total_holdings)
        global_available = round(global_available, 1)
        
        existing_pos = positions.get(contract_name)
        current_contract_size = abs(existing_pos.size) if existing_pos else 0.0
        current_contract_size = round(current_contract_size, 1)
        contract_available = max(0.0, contract_max_position - current_contract_size)
        contract_available = round(contract_available, 1)
        
        available_space = min(global_available, contract_available)
        final_size = min(desired_size, available_space)
        final_size = round(final_size, 1)
        
        if final_size < min_open_size:
            return 0.0
        return final_size

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
        contract_max_pos, params_override = self._get_delivery_rule_config(tick.delivery_start)
        params = self.params.get(strategy_name, {}).copy()
        params.update(params_override.get(strategy_name, {}))
        
        # 【关键修复】使用 ma_window 而不是 window
        window = params.get('ma_window', 20)
        threshold = params.get('threshold', 2.0)
        cooldown = params.get('signal_cooldown_seconds', 300)
        std_ratio_threshold = params.get('std_ratio_threshold', 0.05) # 新增
        history_min_len = params.get('history_min_len', 5) # 新增
        
        if len(bars) < history_min_len: return None
        
        last_time = self.last_trade_times.get(tick.contract_name + strategy_name)
        if last_time and (now - last_time).total_seconds() < cooldown: return None

        # 使用 avg_price
        price_list = [float(b.get('avg_price', b['close'])) for b in bars[-window:]]
        
        if len(price_list) == 0: return None
        
        mean = np.mean(price_list)
        std = np.std(price_list)
        
        # 【关键修复】标准差阈值过滤
        if std <= abs(mean * std_ratio_threshold):
            return None
            
        if std == 0: return None
        z_score = (tick.price - mean) / std
        
        if z_score <= -threshold:
            calc_size = self._calculate_action_and_size(
                tick.contract_name, positions, contract_max_pos, params, ActionType.BUY
            )
            if calc_size > 0.001:
                self.last_trade_times[tick.contract_name + strategy_name] = now
                return TradeSignal(
                    timestamp=now,
                    contract_name=tick.contract_name,
                    contract_id=tick.contract_id,
                    action=ActionType.BUY, 
                    size=calc_size,
                    price=tick.price, 
                    strategy_name=strategy_name,
                    delivery_start=tick.delivery_start,
                    confidence=min(abs(z_score) / threshold, 1.0),
                    open_strategy=strategy_name,
                    z_score=round(z_score, 3),
                    mean_price=round(mean, 2),
                    std_price=round(std, 2),
                    raw_size=calc_size
                )
        return None

    def _check_extreme_sell(self, tick: TickEvent, bars: List[dict], positions: Dict, now: datetime) -> Optional[TradeSignal]:
        strategy_name = "optimized_extreme_sell"
        contract_max_pos, params_override = self._get_delivery_rule_config(tick.delivery_start)
        params = self.params.get(strategy_name, {}).copy()
        params.update(params_override.get(strategy_name, {}))
        
        # 【关键修复】使用 percentile_window 而不是 window
        window = params.get('percentile_window', 20)
        percentile = params.get('percentile_high', 95)
        cooldown = params.get('signal_cooldown_seconds', 300)
        threshold = params.get('threshold', 1.2)
        history_min_len = params.get('history_min_len', 5)

        last_time = self.last_trade_times.get(tick.contract_name + strategy_name)
        if last_time and (now - last_time).total_seconds() < cooldown: return None
        
        if len(bars) < history_min_len: return None
            
        price_list = [float(b.get('avg_price', b['close'])) for b in bars[-window:]]
        
        if len(price_list) == 0: return None
        
        upper_bound = np.percentile(price_list, percentile)
        mean_price = np.mean(price_list)
        
        extreme_condition = False
        if mean_price < 0:
            if tick.price > 0:
                extreme_condition = (tick.price - mean_price) >= abs(mean_price) * threshold
            else:
                extreme_condition = tick.price > upper_bound and tick.price > mean_price / threshold
        else:
            extreme_condition = tick.price > upper_bound and tick.price > threshold * mean_price
        
        if extreme_condition:
            calc_size = self._calculate_action_and_size(
                tick.contract_name, positions, contract_max_pos, params, ActionType.SELL
            )
            if calc_size > 0.001:
                self.last_trade_times[tick.contract_name + strategy_name] = now
                current_price_adj = abs(tick.price) * 0.02
                current_price_discounted = tick.price - current_price_adj
                mean_price_adj = abs(mean_price) * 0.2
                mean_price_target = mean_price + mean_price_adj
                adjusted_price = max(current_price_discounted, mean_price_target)
                
                return TradeSignal(
                    timestamp=now,
                    contract_name=tick.contract_name,
                    contract_id=tick.contract_id,
                    action=ActionType.SELL,
                    size=calc_size,
                    price=round(adjusted_price, 2),
                    strategy_name=strategy_name,
                    delivery_start=tick.delivery_start,
                    open_strategy=strategy_name,
                    z_score=0.0,
                    mean_price=round(mean_price, 2),
                    std_price=0.0,
                    trend_info=f"Upper{percentile}: {round(upper_bound, 2)}",
                    raw_size=calc_size
                )
        return None
    
    # ----------------------------------------------------------------
    # 策略 C: 【新增】高波动逢低买入 (High Volatility Dip Buy)
    # ----------------------------------------------------------------
    def _high_volatility_dip_buy(self, tick: TickEvent, positions: Dict, now: datetime) -> Optional[TradeSignal]:
        strategy_name = "high_volatility_dip_buy"
        contract_max_pos, params_override = self._get_delivery_rule_config(tick.delivery_start)
        params = self.params.get(strategy_name, {}).copy()
        params.update(params_override.get(strategy_name, {}))
        
        high_vol_threshold = params.get('threshold', 50.0)
        
        # 检查是否有足够的 Tick 历史数据
        prices = self.price_history.get(tick.contract_name, [])
        if len(prices) < 20: # 原逻辑写死 < 20
            return None
            
        # 计算波动率 (使用最近 24 个 Tick)
        recent_prices = prices[-24:]
        volatility = np.std(recent_prices)
        min_price = min(recent_prices[-5:]) # 最近5点最低
        
        if volatility >= high_vol_threshold and tick.price <= min_price:
            calc_size = self._calculate_action_and_size(
                tick.contract_name, positions, contract_max_pos, params, ActionType.BUY
            )
            
            if calc_size > 0.001:
                return TradeSignal(
                    timestamp=now,
                    contract_name=tick.contract_name,
                    contract_id=tick.contract_id,
                    action=ActionType.BUY,
                    size=calc_size,
                    price=tick.price,
                    strategy_name=strategy_name,
                    delivery_start=tick.delivery_start,
                    confidence=0.7,
                    open_strategy=strategy_name,
                    z_score=0.0,
                    std_price=round(volatility, 2), # 借用字段记录波动率
                    trend_info=f"Vol: {round(volatility, 2)} >= {high_vol_threshold}",
                    raw_size=calc_size
                )
        return None
    
    # ----------------------------------------------------------------
    # 策略 D: 【新增】交付时间策略 (Delivery Time Buy)
    # ----------------------------------------------------------------
    def _delivery_time_buy_strategy(self, tick: TickEvent, positions: Dict, now: datetime) -> Optional[TradeSignal]:
        strategy_name = "delivery_time_buy"
        
        # 检查是否已执行过
        if tick.contract_name in self.delivery_time_strategy_executed:
            return None
            
        contract_max_pos, params_override = self._get_delivery_rule_config(tick.delivery_start)
        params = self.params.get(strategy_name, {}).copy()
        params.update(params_override.get(strategy_name, {}))
        
        # 注意：原逻辑是去查 DB 获取最近 N 个数据。回测中我们没有实时 DB，
        # 但我们有 price_history (Tick 缓存)。
        # 如果策略要求 price_count=10000 这么大，内存缓存可能不够。
        # 这里的实现做一个折衷：检查该策略是否在 delivery_rules 里被激活
        # 如果 rules 里配置了 delivery_time_buy 且 position_ratio > 0，则执行。
        
        # 简单实现：如果当前时间在 delivery_rules 指定的范围内，且尚未执行，且有仓位额度，则执行
        # 这种策略通常是在特定时刻“无脑”建仓
        
        # 检查该合约是否匹配任何 delivery_rules
        matched_rule = False
        if tick.delivery_start:
            # 复用 _get_delivery_rule_config 的逻辑来判断是否命中时间窗口
            # 如果 params_override 里有 delivery_time_buy 的配置，说明命中了
            if 'delivery_time_buy' in params_override:
                matched_rule = True
                
        if not matched_rule:
            return None
            
        # 标记已执行
        self.delivery_time_strategy_executed.add(tick.contract_name)
        
        calc_size = self._calculate_action_and_size(
            tick.contract_name, positions, contract_max_pos, params, ActionType.BUY
        )
        
        if calc_size > 0.001:
            return TradeSignal(
                timestamp=now,
                contract_name=tick.contract_name,
                contract_id=tick.contract_id,
                action=ActionType.BUY,
                size=calc_size,
                price=tick.price,
                strategy_name=strategy_name,
                delivery_start=tick.delivery_start,
                confidence=0.7,
                open_strategy=strategy_name,
                raw_size=calc_size
            )
        return None