"""
缠论分析模块 v2 — 分析引擎

集成严格算法，支持增量更新
"""

import logging
import time
from typing import Optional, Dict, List, Tuple
import pandas as pd
import os 
import baostock as bs
# 将项目根目录添加到sys.path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.models import (
    AnalysisResult, Fractal, Stroke, Segment, Pivot,
    FractalType, Direction, 
)
from core.storage import FeatherStore
from core.state_manager import AnalysisStateManager
from core.analyzer import (
    identify_fractals,
    build_strokes,
    build_segments,
    detect_pivots,
    detect_divergence,
    compute_status,
)
from config import AnalysisConfig, StorageConfig
from utils.tool import fetch_bs_k_data

logger = logging.getLogger(__name__)

from czsc import CZSC, RawBar, Freq

def czsc_analysis(df: pd.DataFrame) -> dict:
    """
    使用 CZSC 库进行缠论分析
    :param df: 包含 'dt', 'open', 'high', 'low', 'close', 'vol' 列的K线数据
    :return: 包含分析结果的字典
    """
    
    # --- 1. 数据标准化：将DataFrame转换为RawBar对象列表 ---
    bars = []
    # 确保数据是按时间升序排列的
    df = df.sort_values('date')
    for _, row in df.iterrows():
        bar = RawBar(
            symbol="002364",      # 交易标的代码，可根据实际情况设置
            id=row.name,        # bar的唯一标识，通常用索引即可
            freq=Freq.F30,        # K线频率，此处设为日线，可按需修改
            dt=row['date'],
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            vol=row['volume'],
            amount=row['amount']
        )
        bars.append(bar)
    
    # --- 2. 初始化 CZSC 核心对象 ---
    # CZSC 对象在初始化时会自动完成包含关系处理、分型、笔、线段的识别
    c = CZSC(bars)
    
    # --- 3. 获取当前分析结果 ---
    # 获取最后一笔和线段的信息
    last_bi = c.bi_list[-1] if c.bi_list else None
    
    # 获取所有中枢信息
    pivots_info = []
    for zs in c.zss:
        pivots_info.append({
            'zg': zs.zg,    # 中枢上沿
            'zd': zs.zd,    # 中枢下沿
            'gg': zs.gg,    # 中枢最高点
            'dd': zs.dd,    # 中枢最低点
            'bis_count': len(zs.bis)  # 构成中枢的笔数
        })
    
    # --- 5. 结果汇总 ---
    analysis_result = {
        'current_price': bars[-1].close if bars else None,
        'last_bi': {
            'direction': last_bi.direction.value if last_bi else None,
            'start_dt': last_bi.fx_a.dt if last_bi else None,
            'end_dt': last_bi.fx_b.dt if last_bi else None,
            'is_confirm': last_bi.is_confirm if last_bi else False,
        } if last_bi else {},
        'last_xd': {
            'direction': last_xd.direction.value if last_xd else None,
            'start_dt': last_xd.fx_a.dt if last_xd else None,
            'end_dt': last_xd.fx_b.dt if last_xd else None,
        } if last_xd else {},
        'pivots': pivots_info,
    }
    
    return analysis_result

if __name__ == "__main__":
    code = "sz.002364"
    period = "30m"
    bs.login()
    klines = fetch_bs_k_data(code, period)
    bs.logout()
    rs = czsc_analysis(klines)
    print(rs)