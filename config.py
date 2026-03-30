"""
项目全局配置
所有模块统一从此文件导入常量
"""
import os
from pathlib import Path
from typing import List
from dataclasses import dataclass, field, asdict

# ================= 均线周期 =================
MA_PERIODS = [5, 8, 13, 21, 34, 55, 89, 144]

# ================= K线周期 =================
PERIODS = ["1m", "5m", "30m", "day", "week"]

# ================= 批处理 =================
BATCH_SIZE = 200

# ================= 目录路径 =================
BASIC_PATH = "data/basic"
DATA_PATH = "data/raw"
SNAPSHOT_PATH = "data/snapshot"
STRUCTURE_PATH = "data/structure"

STOCK_DATA_PATH = f"{DATA_PATH}/stock"
INDEX_DATA_PATH = f"{DATA_PATH}/index"
STOCK_SNAPSHOT = f"{SNAPSHOT_PATH}/stock"
INDEX_SNAPSHOT = f"{SNAPSHOT_PATH}/index"   
STOCK_STRUCTURE_PATH = f"{STRUCTURE_PATH}/stock"
INDEX_STRUCTURE_PATH = f"{STRUCTURE_PATH}/index"

STOCK_INFO_PATH = f"{BASIC_PATH}/stock_info.feather"
INDEX_INFO_PATH = f"{BASIC_PATH}/index_info.feather"

# ================= 自动创建目录 =================
def init_dirs():
    """启动时调用一次，确保所有目录存在"""
    dirs = [
        BASIC_PATH, DATA_PATH, SNAPSHOT_PATH, STRUCTURE_PATH,
        STOCK_DATA_PATH, INDEX_DATA_PATH,
        STOCK_SNAPSHOT, INDEX_SNAPSHOT,
        STOCK_STRUCTURE_PATH, INDEX_STRUCTURE_PATH,
    ]
    for period in PERIODS:
        dirs.extend([
            f"{STOCK_DATA_PATH}/{period}",
            f"{INDEX_DATA_PATH}/{period}",
            f"{STOCK_STRUCTURE_PATH}/{period}",
            f"{INDEX_STRUCTURE_PATH}/{period}",
        ])
    for d in dirs:
        os.makedirs(d, exist_ok=True)


@dataclass
class AnalysisConfig:
    """缠论分析参数配置"""
    # 笔构建参数
    stroke_merge_bars: int = 5           # 合并K线阈值（不足则合并）
    stroke_gap_ratio: float = 0.0005     # 缺口比例阈值（万分之5）

    # 线段参数
    segment_min_strokes: int = 3         # 线段最少包含笔数

    # 缓存配置
    cache_max_items: int = 50            # 内存缓存最大股票数

    # 分析周期
    default_freqs: List[str] = field(default_factory=lambda: ["day", "30m", "5m"])


@dataclass
class StorageConfig:
    """存储配置"""
    data_dir: str = "data/raw/stock"
    state_dir: str = "data/state/stock"

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def state_path(self) -> Path:
        return Path(self.state_dir)