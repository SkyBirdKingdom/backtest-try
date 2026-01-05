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
        【关键修复】严格按照 (Delivery - 1h) 作为收盘时间，
        仅放行 [收盘前240分钟, 收盘] 区间内的数据。
        """
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
            WHERE trade_time >= :start_time 
              AND trade_time < :end_time
              AND delivery_area = 'SE3'
        """
        
        if contract_filter and len(contract_filter) > 0:
            contracts_str = "', '".join(contract_filter)
            query_sql += f" AND contract_name IN ('{contracts_str}')"
            
        query_sql += " ORDER BY trade_time ASC"

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        params = {"start_time": start_dt, "end_time": end_dt}
        
        logger.info(f"开始加载数据: {start_dt} -> {end_dt}")
        
        total_count = 0
        skipped_count = 0
        skipped_time_window = 0 
        
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
                    for row in chunk_df.itertuples():
                        total_count += 1
                        
                        # 1. 解析合约
                        match = self.contract_pattern.match(row.contract_name)
                        if not match:
                            skipped_count += 1
                            continue
                        c_type_str, date_str, period_str = match.groups()
                        
                        # 2. 校验日期
                        try:
                            contract_date = datetime.strptime(date_str, "%Y%m%d").date()
                            if row.delivery_start.date() != contract_date:
                                skipped_count += 1
                                continue
                        except ValueError:
                            skipped_count += 1
                            continue

                        # 3. 校验时长
                        if not row.delivery_end:
                            skipped_count += 1
                            continue
                        duration_seconds = (row.delivery_end - row.delivery_start).total_seconds()
                        if c_type_str == "PH":
                            if abs(duration_seconds - 3600) > 1: 
                                skipped_count += 1
                                continue
                        elif c_type_str == "QH":
                            if abs(duration_seconds - 900) > 1:
                                skipped_count += 1
                                continue
                        
                        # 4. 【关键修复】4小时时间窗口校验
                        # 收盘时间 = 交付时间 - 1小时
                        gate_closure_time = row.delivery_start - timedelta(hours=1)
                        # 距离收盘还有多少分钟
                        minutes_to_close = (gate_closure_time - row.trade_time).total_seconds() / 60.0
                        
                        # 条件A: 还没进入收盘前4小时 (minutes_to_close > 240)
                        # 条件B: 已经超过收盘时间 (minutes_to_close < 0) -> 【新增】
                        if minutes_to_close > 240 or minutes_to_close < 0:
                            skipped_time_window += 1
                            continue
                        
                        # 5. 确定类型
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
                        
            logger.info(f"数据流加载完成: 总数 {total_count}, 有效 {total_count - skipped_count - skipped_time_window}")
            logger.info(f"过滤统计: 格式错误/不匹配 {skipped_count}, 非交易窗口(>4h或<0m) {skipped_time_window}")

        except Exception as e:
            logger.error(f"数据读取错误: {e}")
            raise