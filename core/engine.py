import logging
from collections import defaultdict
from typing import Dict, List
from datetime import datetime, timedelta

from core.data_loader import DataLoader
from core.exchange import VirtualExchange
from core.bar_generator import BarGenerator
from core.models import TickEvent, SettlementEvent
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

        # 获取强平窗口阈值 (用于再撮合判断)
        strategy_params = config.get('strategy_params', {})
        self.stop_loss_end_minutes = int(strategy_params.get('stop_loss_end_minutes', 3))
        
        # 4. 内存数据库
        self.bars_memory: Dict[str, List[dict]] = defaultdict(list)
        
        # 5. 交付日盈亏计算状态
        self.current_delivery_date = None
        self.current_delivery_pnl = 0.0
        self.last_processed_trade_count = 0 
        
        self.reject_counter = 0 

        # --- 【新增】单日全局禁开仓标志 ---
        self.daily_trading_blocked = False

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

                # --- 【新增】跨天重置全局禁开仓标志 ---
                self.daily_trading_blocked = False
                if hasattr(self.strategy, 'daily_global_block'):
                    self.strategy.daily_global_block = False
                # -----------------------------------

                # 通知策略跨日
                if hasattr(self.strategy, 'on_new_day'):
                    self.strategy.on_new_day(str(tick_delivery_date))
                
                logger.info(f"📅 进入新交付日: {tick_delivery_date} (日内盈亏重置, 过期持仓清理)")
            
            # --- 【新增】PH合约最后6分钟全局阻断逻辑 ---
            if not self.daily_trading_blocked and tick.contract_name.startswith("PH"):
                # 【关键修复】只有当我们当前持有该 PH 合约的仓位时，才进行时间判断
                pos = self.exchange.positions.get(tick.contract_name)
                if pos and abs(pos.size) > 0.001:
                    if tick.delivery_start:
                        gate_closure = tick.delivery_start - timedelta(hours=1)
                        minutes_to_close = (gate_closure - tick.timestamp).total_seconds() / 60.0
                        
                        # 匹配最后 6 分钟阶段
                        if 0 < minutes_to_close <= 6.0:
                            self.daily_trading_blocked = True
                            if hasattr(self.strategy, 'daily_global_block'):
                                self.strategy.daily_global_block = True
                            
                            logger.warning(f"🛑 [全局风控] 持仓中的 PH 合约 {tick.contract_name} 进入最后6分钟 ({minutes_to_close:.1f}m)！")
                            logger.warning(f"🚫 触发单日全局禁开仓：阻止后续所有新信号，并撤销现有未成交开仓单。")

                            # --- 【新增】将触发记录保存到数据库中 ---
                            block_event = SettlementEvent(
                                timestamp=tick.timestamp,
                                contract_name=tick.contract_name,
                                contract_id=tick.contract_id,
                                size=pos.size,
                                avg_price=pos.avg_price,
                                reason="GLOBAL_BLOCK_6MIN" # 使用特定的 reason 标记
                            )
                            self.recorder.record_settlement(block_event)
                            # ----------------------------------------
                            
                            # 撤销所有未成交的开仓订单 (保留自动止盈、止损、强平单)
                            orders_to_cancel = []
                            for order in self.exchange.active_orders:
                                is_exit_strategy = (order.strategy.startswith("auto_profit") or 
                                                    order.strategy.startswith("force_close") or
                                                    order.strategy.startswith("stop_loss") or 
                                                    order.strategy.startswith("exit_"))
                                if not is_exit_strategy:
                                    orders_to_cancel.append(order.client_order_id)
                                    
                            for oid in orders_to_cancel:
                                self.exchange.cancel_order(oid)
                                logger.info(f"🧹 [全局风控] 已撤销残留开仓单: {oid}")
            # ----------------------------------------------------

            if tick_count % 50000 == 0:
                logger.info(f"进度: {tick.timestamp} | 交付日: {tick_delivery_date} | 当日PnL: {self.current_delivery_pnl:.2f}")


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
            # 【核心修改】传入 bars_memory 供 ExitManager 计算反手趋势
            self.exit_manager.process(
                tick, 
                self.exchange.positions, 
                self.exchange.active_orders,
                self.exchange,
                bars=self.bars_memory.get(tick.contract_name, [])
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
                    current_orders = [order for order in self.exchange.active_orders if order.contract_name == sig.contract_name and order.open_strategy == sig.strategy_name and order.side == sig.action]
                    if current_orders:
                        self.exchange.modify_order(current_orders[0].client_order_id, sig.price, sig.size)
                    else:
                        self.exchange.submit_order(sig)
                else:
                    self.reject_counter += 1
                    if self.reject_counter % 2000 == 0:
                        logger.info(f"🚫 信号被拒(采样): {sig.contract_name} 原因: [{sig.failure_reason}] DeliveryPnL: {self.current_delivery_pnl:.2f}")
            
            # ----------------------------------------------------------------------------------
            # 【核心修正】Engine 时序优化 (Re-Match for Force Close)
            # 如果当前是强平时间窗口(最后几分钟)，再次调用撮合。
            # 这样 ExitManager 刚刚生成的强平单（或升级单）可以立即在当前 Tick 成交，
            # 避免了"最后一分钟生成强平单但因为没有下一个Tick而无法成交"的Bug。
            # ----------------------------------------------------------------------------------
            if tick.delivery_start:
                gate_closure = tick.delivery_start - timedelta(hours=1)
                minutes_to_close = (gate_closure - tick.timestamp).total_seconds() / 60.0
                
                if minutes_to_close <= self.stop_loss_end_minutes:
                    # 再次调用撮合，尝试即时成交刚才生成的强平单
                    self.exchange.on_tick(tick)
                    
                    # 再次同步交易记录(为了盈亏统计准确性)
                    current_trade_count = len(self.exchange.trades)
                    if current_trade_count > self.last_processed_trade_count:
                        new_trades = self.exchange.trades[self.last_processed_trade_count:]
                        for trade in new_trades:
                            if trade.delivery_start.date() == self.current_delivery_date:
                                self.current_delivery_pnl += trade.pnl
                        self.last_processed_trade_count = current_trade_count
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