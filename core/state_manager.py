# 分析状态持久化
"""
缠论分析模块 - 状态管理器

管理增量分析所需的状态持久化，
确保每次增量更新只重新计算必要的部分
"""
import logging
from typing import Optional, List, Dict, Any

from .models import Fractal, Stroke, Segment, Pivot, FractalType, Direction
from .storage import FeatherStore

logger = logging.getLogger(__name__)


class AnalysisStateManager:
    """
    管理单只股票单周期的分析状态

    状态内容：
    - last_bar_count: 上次分析时的K线总数
    - last_confirmed_fractal_idx: 最后一个已确认分型的K线索引
    - confirmed_fractals: 已确认的分型列表
    - confirmed_strokes: 已确认的笔列表
    - confirmed_segments: 已确认的线段列表
    - segment_state: 当前线段构建状态
    """

    def __init__(self, store: FeatherStore, symbol: str, freq: str):
        self.store = store
        self.symbol = symbol
        self.freq = freq
        self._state: Optional[dict] = None

    def load(self) -> Optional[dict]:
        """从磁盘加载状态"""
        self._state = self.store.load_state(self.symbol, self.freq)
        return self._state

    def save(self, state: dict) -> None:
        """保存状态到磁盘"""
        self._state = state
        self.store.save_state(self.symbol, self.freq, state)

    @property
    def has_state(self) -> bool:
        return self._state is not None

    @property
    def last_bar_count(self) -> int:
        return self._state.get("last_bar_count", 0) if self._state else 0

    @property
    def last_confirmed_fractal_idx(self) -> int:
        return self._state.get("last_confirmed_fractal_idx", -1) if self._state else -1

    def build_state(
        self,
        fractals: List[Fractal],
        strokes: List[Stroke],
        segments: List[Segment],
        pivots: List[Pivot],
        bar_count: int,
    ) -> dict:
        """
        从完整分析结果构建可序列化的状态字典

        只保存已确认的数据，未确认的在增量更新时重建
        """
        confirmed_fractals = [f for f in fractals if f.confirmed]
        confirmed_strokes = [s for s in strokes if s.confirmed]
        confirmed_segments = [seg for seg in segments if seg.confirmed]

        last_confirmed_idx = (
            confirmed_fractals[-1].bar_index if confirmed_fractals else -1
        )

        # 当前线段构建状态
        segment_state = None
        if segments:
            last_seg = segments[-1]
            if not last_seg.confirmed:
                segment_state = {
                    "stroke_indices": [
                        s.start_fractal.bar_index for s in last_seg.strokes
                    ],
                    "direction": last_seg.direction.value,
                    "high": last_seg.high,
                    "low": last_seg.low,
                }

        state = {
            "symbol": self.symbol,
            "freq": self.freq,
            "last_bar_count": bar_count,
            "last_confirmed_fractal_idx": last_confirmed_idx,
            "confirmed_fractals": [f.to_dict() for f in confirmed_fractals],
            "confirmed_strokes": [s.to_dict() for s in confirmed_strokes],
            "confirmed_segments": [seg.to_dict() for seg in confirmed_segments],
            "pivots": [p.to_dict() for p in pivots],
            "segment_state": segment_state,
        }

        return state

    def restore_fractals(self) -> List[Fractal]:
        """从状态恢复已确认的分型列表"""
        if not self._state:
            return []
        result = []
        for fd in self._state.get("confirmed_fractals", []):
            result.append(Fractal(
                fractal_type=FractalType(fd["type"]),
                price=fd["price"],
                bar_index=fd["bar_index"],
                timestamp=fd.get("timestamp"),
                confirmed=True,
            ))
        return result
