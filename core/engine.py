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
from strategies.pure_force_close import PureForceClose

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
        self.force_close = PureForceClose(config)
        
        # 4. 内存数据库：存储每个合约的历史 K 线
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

            # 1. 交易所层：更新时间，检查超时订单，撮合
            self.exchange.on_tick(tick)
            
            # 2. 数据层：生成K线
            # 注意：这里的 new_bar 只有在分钟结束时才会生成
            new_bar = self.bar_generator.update_tick(tick)
            if new_bar:
                self.bars_memory[tick.contract_name].append(new_bar)
                # 保持内存不过大，只保留最近 500 根
                if len(self.bars_memory[tick.contract_name]) > 500:
                    self.bars_memory[tick.contract_name].pop(0)

            # 3. 策略层：信号计算
            # 【关键】：这里必须在 if new_bar 之外，确保每个 Tick 都触发计算
            # 实盘逻辑：收到 Tick -> 更新 Tick 历史 -> 用当前 Tick 价格对比历史 K 线均值 -> 触发信号
            
            current_pos = self.exchange.positions.get(tick.contract_name)
            current_size = current_pos.size if current_pos else 0.0
            
            # A. 强制平仓 (Priority 1)
            should_force_close = self.force_close.check_force_close(
                tick, current_size, tick.timestamp
            )
            
            if should_force_close:
                fc_signal = self.force_close.generate_close_signal(tick, current_size, tick.timestamp)
                self.exchange.submit_order(fc_signal)
                continue 
            
            # B. 开仓策略 (Priority 2)
            account_info = self.exchange.get_account_info()
            current_daily_pnl = account_info.total_pnl 
            
            # 获取历史 K 线 (注意：这里取到的是截止上一分钟的完整 K 线)
            # 策略内部会将当前 Tick 价格与这些历史 K 线的统计值(均值/方差)进行比较
            bars_history = self.bars_memory.get(tick.contract_name, [])
            
            signals = self.strategy.calculate_signals(
                tick=tick, 
                bars=bars_history, 
                positions=self.exchange.positions, 
                current_time=tick.timestamp,
                current_daily_pnl=current_daily_pnl
            )
            
            for sig in signals:
                # 记录所有生成的信号 (含被拦截的)
                self.recorder.record_signal(sig)

                # 只有有效信号才下单
                if sig.is_valid:
                    self.exchange.submit_order(sig)

        # 回测结束后的处理
        self._on_backtest_finished()

    def _on_backtest_finished(self):
        logger.info("=== 回测结束，正在生成报告 ===")

        for order in self.exchange.order_history:
            self.recorder.record_order(order)
            
        # 1. 获取所有交易记录
        trades = self.exchange.trades
        
        # 2. 保存到数据库
        self.recorder.save_all(trades)
        
        # 3. 打印详细统计 (含单合约盈亏)
        self.recorder.calculate_and_print_stats(trades)
        
        # 4. 打印最终资金
        logger.info(f"最终资金: {self.exchange.capital:.2f}")
        logger.info(f"本次回测 Run ID: {self.recorder.run_id}")