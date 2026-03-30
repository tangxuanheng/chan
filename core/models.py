import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 数据模型
# ══════════════════════════════════════════════════════════════

class FractalType(Enum):
    TOP = "top"
    BOTTOM = "bottom"


class Direction(Enum):
    UP = "up"
    DOWN = "down"


class StrokeState(Enum):
    """未完成笔的状态"""
    EXTENDING_UP = "向上延伸"
    FORMING_TOP = "构筑顶分型"
    EXTENDING_DOWN = "向下延伸"
    FORMING_BOTTOM = "构筑底分型"


class PivotRelation(Enum):
    ABOVE = "中枢上方"
    BELOW = "中枢下方"
    INSIDE = "中枢内部"


class DivergenceType(Enum):
    BEARISH = "顶背驰"
    BULLISH = "底背驰"


@dataclass
class Fractal:
    fractal_type: FractalType
    price: float            # 顶=最高价, 底=最低价
    bar_index: int
    timestamp: object = None
    confirmed: bool = False

    def to_dict(self):
        return {
            "type": self.fractal_type.value,
            "price": self.price,
            "bar_index": self.bar_index,
            "timestamp": str(self.timestamp) if self.timestamp is not None else None,
            "confirmed": self.confirmed,
        }


@dataclass
class Stroke:
    direction: Direction
    start_fractal: Fractal
    end_fractal: Fractal
    bar_count: int = 0
    confirmed: bool = False

    @property
    def high(self) -> float:
        return max(self.start_fractal.price, self.end_fractal.price)

    @property
    def low(self) -> float:
        return min(self.start_fractal.price, self.end_fractal.price)

    @property
    def amplitude(self) -> float:
        return abs(self.end_fractal.price - self.start_fractal.price)

    def to_dict(self):
        return {
            "direction": self.direction.value,
            "start": self.start_fractal.to_dict(),
            "end": self.end_fractal.to_dict(),
            "bar_count": self.bar_count,
            "amplitude": round(self.amplitude, 4),
            "confirmed": self.confirmed,
        }


@dataclass
class Segment:
    strokes: List[Stroke] = field(default_factory=list)
    direction: Direction = Direction.UP
    confirmed: bool = False

    @property
    def start_price(self) -> float:
        return self.strokes[0].start_fractal.price if self.strokes else 0.0

    @property
    def end_price(self) -> float:
        return self.strokes[-1].end_fractal.price if self.strokes else 0.0

    @property
    def high(self) -> float:
        return max(s.high for s in self.strokes) if self.strokes else 0.0

    @property
    def low(self) -> float:
        return min(s.low for s in self.strokes) if self.strokes else 0.0

    @property
    def amplitude(self) -> float:
        return abs(self.end_price - self.start_price)

    def to_dict(self):
        return {
            "direction": self.direction.value,
            "stroke_count": len(self.strokes),
            "start_price": self.start_price,
            "end_price": self.end_price,
            "high": self.high,
            "low": self.low,
            "amplitude": round(self.amplitude, 4),
            "confirmed": self.confirmed,
        }


@dataclass
class Pivot:
    segments: List[Segment] = field(default_factory=list)
    zg: float = 0.0     # 中枢上沿 = min(各段high)
    zd: float = 0.0     # 中枢下沿 = max(各段low)
    gg: float = 0.0     # 最高点 = max(各段high)
    dd: float = 0.0     # 最低点 = min(各段low)
    direction: Direction = Direction.UP

    @property
    def center(self) -> float:
        return (self.zg + self.zd) / 2

    @property
    def height(self) -> float:
        return self.zg - self.zd

    @property
    def oscillation_count(self) -> int:
        return len(self.segments)

    def contains(self, price: float) -> bool:
        return self.zd <= price <= self.zg

    def to_dict(self):
        return {
            "zg": round(self.zg, 4),
            "zd": round(self.zd, 4),
            "gg": round(self.gg, 4),
            "dd": round(self.dd, 4),
            "center": round(self.center, 4),
            "height": round(self.height, 4),
            "direction": self.direction.value,
            "oscillation_count": self.oscillation_count,
            "segment_count": len(self.segments),
        }


@dataclass
class Divergence:
    """背驰"""
    div_type: DivergenceType
    at_segment_idx: int
    price_current: float
    price_previous: float
    indicator_current: float
    indicator_previous: float
    method: str = "macd"      # "macd" | "amplitude"

    def to_dict(self):
        return {
            "type": self.div_type.value,
            "segment_index": self.at_segment_idx,
            "price_current": self.price_current,
            "price_previous": self.price_previous,
            "indicator_current": round(self.indicator_current, 6),
            "indicator_previous": round(self.indicator_previous, 6),
            "method": self.method,
        }


@dataclass
class AnalysisResult:
    symbol: str
    freq: str
    fractals: List[Fractal] = field(default_factory=list)
    strokes: List[Stroke] = field(default_factory=list)
    segments: List[Segment] = field(default_factory=list)
    pivots: List[Pivot] = field(default_factory=list)
    divergences: List[Divergence] = field(default_factory=list)

    # 多周期状态
    pivot_relation: PivotRelation = PivotRelation.ABOVE
    stroke_state: Optional[StrokeState] = None
    stroke_direction: Optional[Direction] = None
    segment_stroke_count: int = 0
    segment_direction: Optional[Direction] = None

    last_bar_index: int = 0
    is_incremental: bool = False

    def summary(self) -> dict:
        return {
            "symbol": self.symbol,
            "freq": self.freq,
            "fractal_count": len(self.fractals),
            "stroke_count": len(self.strokes),
            "segment_count": len(self.segments),
            "pivot_count": len(self.pivots),
            "divergence_count": len(self.divergences),
            "pivot_relation": self.pivot_relation.value,
            "stroke_state": self.stroke_state.value if self.stroke_state else None,
            "stroke_direction": (
                self.stroke_direction.value if self.stroke_direction else None
            ),
            "segment_stroke_count": self.segment_stroke_count,
            "segment_direction": (
                self.segment_direction.value if self.segment_direction else None
            ),
            "is_incremental": self.is_incremental,
        }

