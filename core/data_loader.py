import logging
import pandas as pd
import re
from datetime import datetime, timedelta
from typing import Generator, List, Optional
from sqlalchemy import create_engine, text
from core.models import TickEvent

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataLoader")

class DataLoader:
    def __init__(self, db_url: str):
        self.db_url = db_url
        try:
            self.engine = create_engine(self.db_url)
            logger.info("数据库引擎初始化成功")
        except Exception as e:
            logger.error(f"数据库连接配置失败: {e}")
            raise
            
        self.contract_pattern = re.compile(r"^(PH|QH)-(\d{8})-(\d{2})$")

    def load_stream(self, start_date: str, end_date: str, contract_filter: List[str] = None) -> Generator[TickEvent, None, None]:
        """
        流式生成 TickEvent
        【修改】按照 交付日 -> 合约名 -> 交易时间 的顺序读取数据
        """
        # 修改 SQL：筛选 delivery_start，并调整排序逻辑
        query_sql = """
            SELECT 
                trade_time, 
                contract_name, 
                contract_id, 
                price, 
                volume, 
                delivery_start,
                delivery_end,
                contract_type,
                trade_id
            FROM trades 
            WHERE delivery_start >= :start_time 
              AND delivery_start < :end_time
              AND delivery_area = 'SE3'
        """
        
        if contract_filter and len(contract_filter) > 0:
            contracts_str = "', '".join(contract_filter)
            query_sql += f" AND contract_name IN ('{contracts_str}')"
            
        # 【关键修改】排序逻辑：交付日优先 -> 合约名次之 -> 交易时间最后
        # 这保证了回测是 "一个合约接一个合约" 进行的
        query_sql += " ORDER BY trade_time ASC, contract_name ASC"

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        params = {"start_time": start_dt, "end_time": end_dt}
        
        logger.info(f"开始加载数据 (按交付日排序): {start_dt} -> {end_dt}")
        
        chunk_idx = 0
        total_valid = 0
        
        try:
            with self.engine.connect() as conn:
                iterator = pd.read_sql(
                    text(query_sql), 
                    conn, 
                    params=params, 
                    parse_dates=['trade_time', 'delivery_start', 'delivery_end'],
                    chunksize=50000
                )

                for chunk_df in iterator:
                    chunk_idx += 1
                    if chunk_df.empty: continue
                    
                    # 简单日志，避免刷屏
                    if chunk_idx % 10 == 0:
                        min_d = chunk_df['delivery_start'].min()
                        max_d = chunk_df['delivery_start'].max()
                        logger.info(f"处理进度: 交付日 {min_d} ~ {max_d}")

                    for row in chunk_df.itertuples():
                        # 1. 基础正则校验
                        match = self.contract_pattern.match(row.contract_name)
                        if not match: continue
                        c_type_str, date_str, period_str = match.groups()
                        
                        # 2. 校验日期一致性
                        try:
                            contract_date = datetime.strptime(date_str, "%Y%m%d").date()
                            if row.delivery_start.date() != contract_date: continue
                        except ValueError: continue

                        # 3. 校验时长
                        if not row.delivery_end: continue
                        duration_seconds = (row.delivery_end - row.delivery_start).total_seconds()
                        if c_type_str == "PH" and abs(duration_seconds - 3600) > 1: continue
                        elif c_type_str == "QH" and abs(duration_seconds - 900) > 1: continue
                        
                        # 4. 时间窗口校验 (只放行收盘前 4 小时的数据)
                        gate_closure_time = row.delivery_start - timedelta(hours=1)
                        minutes_to_close = (gate_closure_time - row.trade_time).total_seconds() / 60.0
                        
                        if minutes_to_close > 240 or minutes_to_close < 0:
                            continue
                        
                        total_valid += 1
                        final_contract_type = row.contract_type if row.contract_type else c_type_str

                        yield TickEvent(
                            timestamp=row.trade_time,
                            contract_name=row.contract_name,
                            contract_id=row.contract_id,
                            price=float(row.price),
                            volume=float(row.volume),
                            delivery_start=row.delivery_start,
                            delivery_end=row.delivery_end,
                            contract_type=final_contract_type,
                            trade_id=row.trade_id
                        )
            
            logger.info(f"数据加载结束，共产出有效 Tick: {total_valid}")

        except Exception as e:
            logger.error(f"数据读取错误: {e}")
            raise