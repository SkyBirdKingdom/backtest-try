import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from core.models import TickEvent

class BarGenerator:
    """
    K线合成器：接收 Tick，实时合成 K 线。
    【修正】：根据原项目逻辑，虽然叫 FiveMinBars，实际是生成的 1分钟 K线，且价格为简单算术平均。
    """
    def __init__(self):
        # 缓存当前正在生成的 Bar
        self.current_bars: Dict[str, dict] = {} 

    def update_tick(self, tick: TickEvent) -> Optional[dict]:
        contract = tick.contract_name
        
        # ---------------------------------------------------
        # 1. 计算时间归属 (向下取整到 1分钟)
        # ---------------------------------------------------
        tick_minute = tick.timestamp.replace(second=0, microsecond=0)
        # 既然是1分钟K线，start_time 就是当前的分钟整点
        bar_start_time = tick_minute 
        
        finished_bar = None

        if contract in self.current_bars:
            current_bar = self.current_bars[contract]
            
            # 如果新 Tick 的时间跨过了当前 Bar 的时间段 (即进入了下一分钟)
            if bar_start_time > current_bar['start_time']:
                # 结算上一根 Bar
                self._finalize_bar(current_bar)
                finished_bar = current_bar.copy()
                
                # 初始化新 Bar
                self._init_new_bar(contract, bar_start_time, tick)
            else:
                # 更新当前 Bar
                self._update_current_bar(current_bar, tick)
        else:
            # 第一次遇到该合约，初始化
            self._init_new_bar(contract, bar_start_time, tick)

        return finished_bar

    def _init_new_bar(self, contract, start_time, tick):
        self.current_bars[contract] = {
            'contract_name': contract,
            'start_time': start_time,
            'open': tick.price,
            'high': tick.price,
            'low': tick.price,
            'close': tick.price,
            'volume': tick.volume,
            
            # --- 关键修改：用于计算 Simple Mean ---
            'price_sum': tick.price,  # 价格累加
            'trade_count': 1,         # 成交笔数
            'avg_price': tick.price   # 初始均价
        }

    def _update_current_bar(self, bar, tick):
        bar['high'] = max(bar['high'], tick.price)
        bar['low'] = min(bar['low'], tick.price)
        bar['close'] = tick.price
        bar['volume'] += tick.volume
        
        # --- 关键修改：累加价格和笔数 ---
        bar['price_sum'] += tick.price
        bar['trade_count'] += 1

    def _finalize_bar(self, bar):
        """
        K线结束时计算最终的 avg_price
        逻辑：Sum(Prices) / Count(Trades)
        """
        if bar['trade_count'] > 0:
            bar['avg_price'] = bar['price_sum'] / bar['trade_count']
        else:
            bar['avg_price'] = bar['close']
        
        # 格式化精度，保留2位小数
        bar['avg_price'] = round(bar['avg_price'], 2)
        
        # 清理临时字段，防止污染输出（可选）
        # del bar['price_sum']
        # del bar['trade_count']