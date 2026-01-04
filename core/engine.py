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
# 【修改】使用新的平仓管理器，不再使用 pure_force_close
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
        # 【修改】初始化 ExitManager
        self.exit_manager = PureExitManager(config)
        
        # 4. 内存数据库
        self.bars_memory: Dict[str, List[dict]] = defaultdict(list)
        
    def run(self, start_date: str, end_date: str, contract_filter: List[str] = None):
        """
        运行回测的主循环
        """
        logger.info(f"=== 启动回测: {start_date} 至 {end_date} ===")
        
        tick_stream = self.loader.load_stream(start_date, end_date, contract_filter)
        tick_count = 0
        
        for tick in tick_stream:
            tick_count += 1
            if tick_count % 50000 == 0:
                logger.info(f"进度: {tick.timestamp} | 已处理 Tick: {tick_count} | 当前资金: {self.exchange.capital:.2f}")

            # 1. 交易所层
            self.exchange.on_tick(tick)
            
            # 2. 数据层
            new_bar = self.bar_generator.update_tick(tick)
            if new_bar:
                self.bars_memory[tick.contract_name].append(new_bar)
                if len(self.bars_memory[tick.contract_name]) > 500:
                    self.bars_memory[tick.contract_name].pop(0)

            # 3. 策略层：执行平仓管理 (优先级最高)
            # 【新增】调用平仓管理器处理止盈、止损、强平
            self.exit_manager.process(
                tick, 
                self.exchange.positions, 
                self.exchange.active_orders,
                self.exchange
            )
            
            # 4. 策略层：开仓信号计算
            account_info = self.exchange.get_account_info()
            current_daily_pnl = account_info.total_pnl 
            
            bars_history = self.bars_memory.get(tick.contract_name, [])
            
            signals = self.strategy.calculate_signals(
                tick=tick, 
                bars=bars_history, 
                positions=self.exchange.positions, 
                current_time=tick.timestamp,
                current_daily_pnl=current_daily_pnl
            )
            
            for sig in signals:
                self.recorder.record_signal(sig)
                if sig.is_valid:
                    self.exchange.submit_order(sig)

        # 回测结束后的处理
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