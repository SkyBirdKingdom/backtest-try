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
        流式生成 TickEvent (含深度诊断日志)
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
            logger.info(f"应用合约过滤器: 包含 {len(contract_filter)} 个合约")
        else:
            logger.info("未应用合约过滤器: 加载全部合约")
            
        query_sql += " ORDER BY trade_time ASC"

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        params = {"start_time": start_dt, "end_time": end_dt}
        
        logger.info(f"【诊断】SQL 参数: {params}")
        logger.info(f"【诊断】生成的 SQL 前200字符: {query_sql[:200]}...")
        
        total_count = 0
        
        # 详细的过滤计数器
        skipped_regex = 0
        skipped_date_mismatch = 0
        skipped_duration = 0
        skipped_too_early = 0  # > 4小时
        skipped_too_late = 0   # < 0小时 (收盘后)
        
        chunk_idx = 0
        
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
                    
                    if chunk_df.empty:
                        logger.warning(f"【诊断】Chunk {chunk_idx} 为空！")
                        continue

                    # 打印当前块的时间范围，确认是否读到了11月24日之后的数据
                    min_t = chunk_df['trade_time'].min()
                    max_t = chunk_df['trade_time'].max()
                    logger.info(f"Chunk {chunk_idx}: 范围 {min_t} -> {max_t} | 行数: {len(chunk_df)}")
                    
                    for row in chunk_df.itertuples():
                        total_count += 1
                        
                        # 1. 解析合约正则
                        match = self.contract_pattern.match(row.contract_name)
                        if not match:
                            skipped_regex += 1
                            # 仅打印前3个错误样本，避免刷屏
                            if skipped_regex <= 3:
                                logger.warning(f"过滤样本(正则不匹配): {row.contract_name}")
                            continue
                        c_type_str, date_str, period_str = match.groups()
                        
                        # 2. 校验日期一致性
                        try:
                            contract_date = datetime.strptime(date_str, "%Y%m%d").date()
                            if row.delivery_start.date() != contract_date:
                                skipped_date_mismatch += 1
                                continue
                        except ValueError:
                            skipped_regex += 1
                            continue

                        # 3. 校验时长
                        if not row.delivery_end:
                            skipped_duration += 1
                            continue
                        duration_seconds = (row.delivery_end - row.delivery_start).total_seconds()
                        if c_type_str == "PH":
                            if abs(duration_seconds - 3600) > 1: 
                                skipped_duration += 1
                                continue
                        elif c_type_str == "QH":
                            if abs(duration_seconds - 900) > 1:
                                skipped_duration += 1
                                continue
                        
                        # 4. 时间窗口校验 (核心疑点)
                        gate_closure_time = row.delivery_start - timedelta(hours=1)
                        minutes_to_close = (gate_closure_time - row.trade_time).total_seconds() / 60.0
                        
                        if minutes_to_close > 240:
                            skipped_too_early += 1
                            if skipped_too_early % 10000 == 0:
                                logger.info(f"过滤样本(太早): {row.trade_time} 距关闸 {minutes_to_close:.1f}m > 240m")
                            continue
                            
                        if minutes_to_close < 0:
                            skipped_too_late += 1
                            if skipped_too_late % 10000 == 0: # 采样打印
                                logger.info(f"过滤样本(太晚/收盘后): {row.trade_time} 距关闸 {minutes_to_close:.1f}m < 0m")
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
                        
            # 最终统计报告
            valid_count = total_count - (skipped_regex + skipped_date_mismatch + skipped_duration + skipped_too_early + skipped_too_late)
            logger.info("="*50)
            logger.info(f"数据加载完成报告:")
            logger.info(f"总读取行数: {total_count}")
            logger.info(f"有效产出: {valid_count}")
            logger.info(f"过滤器统计:")
            logger.info(f"  - 正则/命名错误: {skipped_regex}")
            logger.info(f"  - 日期不匹配: {skipped_date_mismatch}")
            logger.info(f"  - 时长异常: {skipped_duration}")
            logger.info(f"  - 时间窗口太早(>4h): {skipped_too_early}")
            logger.info(f"  - 时间窗口太晚(<0m): {skipped_too_late}")
            logger.info("="*50)

        except Exception as e:
            logger.error(f"数据读取错误: {e}")
            raise