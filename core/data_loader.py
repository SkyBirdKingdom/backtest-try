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
            
        # 预编译正则，提高性能
        # 匹配格式: PH-20250101-02 或 QH-20250227-66
        self.contract_pattern = re.compile(r"^(PH|QH)-(\d{8})-(\d{2})$")

    def load_stream(self, start_date: str, end_date: str, contract_filter: List[str] = None) -> Generator[TickEvent, None, None]:
        """
        流式生成 TickEvent，包含严格的数据清洗逻辑。
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
                        
                        # --- 核心校验逻辑 ---
                        
                        # 1. 解析合约名称
                        match = self.contract_pattern.match(row.contract_name)
                        if not match:
                            # logger.debug(f"跳过: 合约名格式不符 {row.contract_name}")
                            skipped_count += 1
                            continue
                            
                        c_type_str, date_str, period_str = match.groups()
                        
                        # 2. 校验日期一致性
                        # contract_name 中的日期必须等于 delivery_start 的日期
                        try:
                            contract_date = datetime.strptime(date_str, "%Y%m%d").date()
                            if row.delivery_start.date() != contract_date:
                                # logger.warning(f"跳过: 日期不匹配 {row.contract_name} | NameDate: {contract_date} != DelivStart: {row.delivery_start.date()}")
                                skipped_count += 1
                                continue
                        except ValueError:
                            skipped_count += 1
                            continue

                        # 3. 校验时长 (Duration)
                        # PH 必须是 1小时 (3600秒), QH 必须是 15分钟 (900秒)
                        if not row.delivery_end:
                            skipped_count += 1
                            continue
                            
                        duration_seconds = (row.delivery_end - row.delivery_start).total_seconds()
                        
                        if c_type_str == "PH":
                            if abs(duration_seconds - 3600) > 1: # 允许1秒误差
                                # logger.warning(f"跳过: PH合约时长不对 {row.contract_name} duration={duration_seconds}")
                                skipped_count += 1
                                continue
                        elif c_type_str == "QH":
                            if abs(duration_seconds - 900) > 1:
                                # logger.warning(f"跳过: QH合约时长不对 {row.contract_name} duration={duration_seconds}")
                                skipped_count += 1
                                continue
                        
                        # 4. 确定合约类型 (用于后续费率计算)
                        # 优先使用数据库里的 contract_type，如果没有则使用解析出来的 c_type_str
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
                        
            logger.info(f"数据流加载完成: 总数 {total_count}, 有效 {total_count - skipped_count}, 跳过 {skipped_count}")

        except Exception as e:
            logger.error(f"数据读取错误: {e}")
            raise