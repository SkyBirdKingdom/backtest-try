import logging
from collections import defaultdict
from typing import Dict, List
from datetime import datetime

from core.data_loader import DataLoader
from core.exchange import VirtualExchange
from core.bar_generator import BarGenerator
from core.models import TickEvent
from core.recorder import BacktestRecorder

from strategies.pure_strategy import PureStrategyEngine
from strategies.pure_exit_manager import PureExitManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BacktestEngine")

class BacktestEngine:
    def __init__(self, config: dict, db_url: str):
        self.config = config
        
        # 1. 初始化基础设施
        self.loader = DataLoader(db_url)
        self.exchange = VirtualExchange(initial_capital=config.get('initial_capital', 100000.0), config=config)
        self.bar_generator = BarGenerator()
        
        # 2. 初始化记录员
        self.recorder = BacktestRecorder(db_url)
        
        # 3. 初始化策略
        self.strategy = PureStrategyEngine(config)
        self.exit_manager = PureExitManager(config)
        
        # 4. 内存数据库
        self.bars_memory: Dict[str, List[dict]] = defaultdict(list)
        
        # 5. 交付日盈亏计算状态
        self.current_delivery_date = None
        self.current_delivery_pnl = 0.0
        self.last_processed_trade_count = 0 
        
        self.reject_counter = 0 

    def run(self, start_date: str, end_date: str, contract_filter: List[str] = None):
        logger.info(f"=== 启动回测 (按交付日排序): {start_date} 至 {end_date} ===")
        
        tick_stream = self.loader.load_stream(start_date, end_date, contract_filter)
        tick_count = 0
        tick_set = set()
        
        for tick in tick_stream:
            if tick.trade_id in tick_set:
                continue
            tick_set.add(tick.trade_id)
            tick_count += 1
            
            # --- 【核心】交付日变更检测 ---
            tick_delivery_date = tick.delivery_start.date()
            
            if self.current_delivery_date != tick_delivery_date:
                # 在进入新交付日之前，清理旧的过期持仓！
                # 这会释放被占用的 position size
                if self.current_delivery_date is not None:
                    settlement_events = self.exchange.settle_expired_positions(tick_delivery_date)
                    for event in settlement_events:
                        self.recorder.record_settlement(event)

                self.current_delivery_date = tick_delivery_date
                # 重置交付日累计盈亏
                self.current_delivery_pnl = 0.0
                # 通知策略跨日
                if hasattr(self.strategy, 'on_new_day'):
                    self.strategy.on_new_day(str(tick_delivery_date))
                
                logger.info(f"📅 进入新交付日: {tick_delivery_date} (日内盈亏重置, 过期持仓清理)")

            if tick_count % 50000 == 0:
                logger.info(f"进度: {tick.timestamp} | 交付日: {tick_delivery_date} | 当日PnL: {self.current_delivery_pnl:.2f}")

            self.exit_manager.modify_order(self.exchange, self.exchange.positions, tick, self.exchange.active_orders)

            # 1. 交易所层
            self.exchange.on_tick(tick)
            
            # 2. 实时更新交付日盈亏
            current_trade_count = len(self.exchange.trades)
            if current_trade_count > self.last_processed_trade_count:
                new_trades = self.exchange.trades[self.last_processed_trade_count:]
                for trade in new_trades:
                    if trade.delivery_start.date() == self.current_delivery_date:
                        self.current_delivery_pnl += trade.pnl
                self.last_processed_trade_count = current_trade_count

            # 3. 数据层
            new_bar = self.bar_generator.update_tick(tick)
            if new_bar:
                self.bars_memory[tick.contract_name].append(new_bar)
                if len(self.bars_memory[tick.contract_name]) > 500:
                    self.bars_memory[tick.contract_name].pop(0)

            # 4. 策略层：执行平仓管理
            self.exit_manager.process(
                tick, 
                self.exchange.positions, 
                self.exchange.active_orders,
                self.exchange
            )
            
            # 5. 策略层：生成交易信号
            signals = self.strategy.on_tick(
                tick=tick, 
                positions=self.exchange.positions, 
                active_orders=self.exchange.active_orders,
                account_info=None 
            )
            
            self.strategy.daily_realized_pnl = self.current_delivery_pnl

            for sig in signals:
                self.recorder.record_signal(sig)
                if sig.is_valid:
                    self.exchange.submit_order(sig)
                else:
                    self.reject_counter += 1
                    if self.reject_counter % 2000 == 0:
                        logger.info(f"🚫 信号被拒(采样): {sig.contract_name} 原因: [{sig.failure_reason}] DeliveryPnL: {self.current_delivery_pnl:.2f}")

        # 回测结束
        self._on_backtest_finished()

    def _on_backtest_finished(self):
        logger.info("=== 回测结束，正在生成报告 ===")

        for order in self.exchange.order_history:
            self.recorder.record_order(order)
            
        trades = self.exchange.trades
        self.recorder.save_all(trades)
        self.recorder.calculate_and_print_stats(trades)
        
        logger.info(f"最终资金: {self.exchange.capital:.2f}")
        logger.info(f"本次回测 Run ID: {self.recorder.run_id}")