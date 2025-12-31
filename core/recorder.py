import pandas as pd
import logging
from sqlalchemy import create_engine
from typing import List
from datetime import datetime
from core.models import Trade, TradeSignal, Order

logger = logging.getLogger("BacktestRecorder")

class BacktestRecorder:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") 
        
        # 内存缓存
        self.signals_buffer: List[TradeSignal] = []
        self.orders_buffer: List[Order] = []

    def record_signal(self, signal: TradeSignal):
        """缓存信号 (在 engine 中调用)"""
        self.signals_buffer.append(signal)

    def record_order(self, order: Order):
        """缓存订单 (在 exchange 中调用)"""
        self.orders_buffer.append(order)

    def save_all(self, trades: List[Trade]):
        """一次性保存所有数据"""
        self.save_trades(trades)
        self._save_signals()
        self._save_orders() 

    def save_trades(self, trades: List[Trade]):
        """
        将交易记录保存到数据库表 backtest_trades，并强制处理精度
        """
        if not trades:
            logger.warning("没有交易记录需要保存")
            return

        # 1. 转换为 DataFrame
        data = [t.__dict__ for t in trades]
        df = pd.DataFrame(data)
        
        # 2. 增加回测标识
        df['run_id'] = self.run_id
        df['created_at'] = datetime.now()

        # 3. 处理字段类型
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # ---------------------------------------------------------
        # 【关键修改】 强制精度清洗 (利用 Pandas 的矢量化操作)
        # ---------------------------------------------------------
        # 数量保留 1 位小数
        if 'size' in df.columns:
            df['size'] = df['size'].astype(float).round(1)
            
        # 价格和盈亏保留 2 位小数
        if 'price' in df.columns:
            df['price'] = df['price'].astype(float).round(2)
            
        if 'pnl' in df.columns:
            df['pnl'] = df['pnl'].astype(float).round(2)
            
        # ---------------------------------------------------------
        
        # 4. 写入数据库
        try:
            table_name = 'backtest_trades'
            df.to_sql(table_name, self.engine, if_exists='append', index=False)
            logger.info(f"✅ 成功保存 {len(df)} 条交易记录到表 '{table_name}' (run_id: {self.run_id})")
        except Exception as e:
            logger.error(f"❌ 保存交易记录失败: {e}")
    
    def _save_signals(self):
        """保存所有产生的信号"""
        if not self.signals_buffer:
            return
            
        df = pd.DataFrame([s.__dict__ for s in self.signals_buffer])
        df['run_id'] = self.run_id
        df['created_at'] = datetime.now()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 处理 action 枚举
        if 'action' in df.columns:
            df['action'] = df['action'].apply(lambda x: x.value if hasattr(x, 'value') else x)
            
        try:
            df.to_sql('backtest_signals', self.engine, if_exists='append', index=False)
            logger.info(f"✅ 保存 {len(df)} 条信号记录")
        except Exception as e:
            logger.error(f"❌ 保存信号记录失败: {e}")

    def _save_orders(self):
        """保存所有生成的订单"""
        if not self.orders_buffer:
            return

        df = pd.DataFrame([o.__dict__ for o in self.orders_buffer])
        df['run_id'] = self.run_id
        df['created_at'] = datetime.now()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        try:
            df.to_sql('backtest_orders', self.engine, if_exists='append', index=False)
            logger.info(f"✅ 保存 {len(df)} 条订单记录")
        except Exception as e:
            logger.error(f"❌ 保存订单记录失败: {e}")

    def calculate_and_print_stats(self, trades: List[Trade]):
        """
        计算并打印统计数据
        """
        if not trades:
            return

        df = pd.DataFrame([t.__dict__ for t in trades])
        
        # 统计计算前也先清洗一下，确保打印好看
        df['pnl'] = df['pnl'].astype(float).round(2)
        
        print("\n" + "="*50)
        print(f"📊 回测统计报告 (Run ID: {self.run_id})")
        print("="*50)

        # 1. 总体统计
        total_pnl = df['pnl'].sum()
        
        print(f"💰 总净利润: {total_pnl:.2f} EUR")
        print(f"📝 总交易数: {len(df)}")
        
        # 2. 单合约统计
        print("\n📋 单合约收益统计:")
        print("-" * 65)
        print(f"{'合约名称':<20} | {'盈亏 (EUR)':<12} | {'交易次数':<8} | {'方向'}")
        print("-" * 65)

        # 按合约分组
        if not df.empty:
            contract_stats = df.groupby('contract_name').agg({
                'pnl': 'sum',
                'trade_id': 'count',
                'action': lambda x: ','.join(sorted(set(x))) 
            }).sort_values(by='pnl', ascending=False)

            for contract_name, row in contract_stats.iterrows():
                print(f"{contract_name:<20} | {row['pnl']:>10.2f} | {row['trade_id']:>8} | {row['action']}")
        
        print("-" * 65)
        
        # 3. 保存单合约统计到数据库 (同样进行精度清洗)
        try:
            if not df.empty:
                contract_stats_db = contract_stats.reset_index()
                contract_stats_db['pnl'] = contract_stats_db['pnl'].round(2)
                contract_stats_db['run_id'] = self.run_id
                contract_stats_db.to_sql('backtest_contract_stats', self.engine, if_exists='append', index=False)
        except Exception as e:
            logger.warning(f"保存单合约统计失败: {e}")