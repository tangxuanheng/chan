# 核心缠论分析算法
"""
缠论分析模块 - 核心算法

实现分型识别 → 笔构建 → 线段构建 → 中枢检测 完整流水线
所有算法均为增量友好设计，支持从指定位置重新分析
"""
import logging
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from .models import (
    Fractal, Stroke, Segment, Pivot,
    FractalType, Direction, Divergence, PivotRelation,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 第一层：分型识别（含包含关系合并）
# ══════════════════════════════════════════════════════════════

def _merge_included_bars(df: pd.DataFrame) -> pd.DataFrame:
    """
    合并K线包含关系

    缠论定义：K线A包含K线B <=> A.high >= B.high 且 A.low <= B.low
    合并规则（按趋势方向）：
      上升趋势中：取两K线 high 的较大值、low 的较大值（向高看齐）
      下降趋势中：取两K线 high 的较小值、low 的较小值（向低看齐）

    第一根K线视为上升趋势开始。
    """
    if len(df) < 2:
        return df

    rows = df.to_dict("records")
    merged = [rows[0]]
    trend_up = True  # 初始方向

    for i in range(1, len(rows)):
        prev = merged[-1]
        cur = rows[i]

        prev_h, prev_l = prev["high"], prev["low"]
        cur_h, cur_l = cur["high"], cur["low"]

        # 判断包含关系
        if prev_h >= cur_h and prev_l <= cur_l:
            # prev 包含 cur
            if trend_up:
                merged[-1]["high"] = max(prev_h, cur_h)
                merged[-1]["low"] = max(prev_l, cur_l)
            else:
                merged[-1]["high"] = min(prev_h, cur_h)
                merged[-1]["low"] = min(prev_l, cur_l)
            # 收盘价取合并后的中间值
            merged[-1]["close"] = (merged[-1]["high"] + merged[-1]["low"]) / 2
        elif cur_h >= prev_h and cur_l <= prev_l:
            # cur 包含 prev
            if trend_up:
                merged[-1]["high"] = max(prev_h, cur_h)
                merged[-1]["low"] = max(prev_l, cur_l)
            else:
                merged[-1]["high"] = min(prev_h, cur_h)
                merged[-1]["low"] = min(prev_l, cur_l)
            merged[-1]["close"] = (merged[-1]["high"] + merged[-1]["low"]) / 2
        else:
            # 无包含关系，加入并更新趋势方向
            merged.append(cur)
            trend_up = cur_h > prev_h

    result = pd.DataFrame(merged)
    result = result.reset_index(drop=True)
    return result


def identify_fractals(df: pd.DataFrame) -> List[Fractal]:
    """
    严格分型识别

    步骤：
    1. 合并K线包含关系
    2. 在合并后的K线上识别顶/底分型
    3. 合并连续同向分型（保留最极端的）

    顶分型：中间K线 high >= 左右K线 high
    底分型：中间K线 low <= 左右K线 low
    """
    if len(df) < 3:
        return []

    # Step 1: 合并包含关系
    merged = _merge_included_bars(df)
    if len(merged) < 3:
        return []

    highs = merged["high"].values
    lows = merged["low"].values
    timestamps = merged["datetime"].values if "datetime" in merged.columns else [None] * len(merged)

    # Step 2: 识别分型
    raw: List[Fractal] = []
    for i in range(1, len(merged) - 1):
        if highs[i] >= highs[i - 1] and highs[i] >= highs[i + 1]:
            raw.append(Fractal(
                fractal_type=FractalType.TOP,
                price=float(highs[i]),
                bar_index=i,
                timestamp=timestamps[i],
                confirmed=True,  # 合并后K线的分型视为已确认
            ))
        elif lows[i] <= lows[i - 1] and lows[i] <= lows[i + 1]:
            raw.append(Fractal(
                fractal_type=FractalType.BOTTOM,
                price=float(lows[i]),
                bar_index=i,
                timestamp=timestamps[i],
                confirmed=True,
            ))

    if not raw:
        return []

    # Step 3: 合并连续同向分型
    merged_f = [raw[0]]
    for f in raw[1:]:
        prev = merged_f[-1]
        if f.fractal_type == prev.fractal_type:
            # 同向分型，保留更极端的
            if f.fractal_type == FractalType.TOP:
                if f.price > prev.price:
                    merged_f[-1] = f
            else:
                if f.price < prev.price:
                    merged_f[-1] = f
        else:
            merged_f.append(f)

    return merged_f


# ══════════════════════════════════════════════════════════════
# 第二层：笔的构建
# ══════════════════════════════════════════════════════════════

def build_strokes(
    fractals: List[Fractal],
    min_gap: int = 4,
) -> List[Stroke]:
    """
    严格笔构建

    规则：
    1. 一笔必须从一个顶分型到一个底分型（或反之），顶底交替
    2. 顶分型价格必须严格高于底分型价格
    3. 两个分型之间至少间隔 min_gap 根K线
       （缠论原文要求至少5根，含分型左右各1根，中间至少3根）
    4. 相邻笔之间的分型不能被包含：
       例如 向下笔的底分型价格 必须 < 前后向上笔的底分型价格

    算法：
    1. 过滤相邻同向分型（已在分型阶段处理）
    2. 逐对检查是否满足笔的条件
    3. 迭代合并不满足条件的相邻笔

    Args:
        fractals: 合并后的分型序列
        min_gap: 两分型间最小K线间隔（默认4，即含端点共5根）

    Returns:
        笔列表
    """
    if len(fractals) < 2:
        return []

    # Step 1: 构建候选笔
    candidates: List[Stroke] = []
    for i in range(len(fractals) - 1):
        sf = fractals[i]
        ef = fractals[i + 1]

        # 必须顶底交替
        if sf.fractal_type == ef.fractal_type:
            continue

        # K线间隔检查
        gap = ef.bar_index - sf.bar_index
        if gap < min_gap:
            continue

        # 方向判断 + 价格合理性
        if sf.fractal_type == FractalType.TOP and ef.fractal_type == FractalType.BOTTOM:
            if sf.price <= ef.price:
                continue
            direction = Direction.DOWN
        elif sf.fractal_type == FractalType.BOTTOM and ef.fractal_type == FractalType.TOP:
            if sf.price >= ef.price:
                continue
            direction = Direction.UP
        else:
            continue

        candidates.append(Stroke(
            direction=direction,
            start_fractal=sf,
            end_fractal=ef,
            bar_count=gap,
            confirmed=ef.confirmed,
        ))

    if not candidates:
        return []

    # Step 2: 迭代合并不满足条件的笔
    # 条件：相邻两笔同方向时，后一笔不满足最小K线数则合并
    changed = True
    while changed:
        changed = False
        result: List[Stroke] = []
        i = 0
        while i < len(candidates):
            if (
                i + 1 < len(candidates)
                and candidates[i].direction == candidates[i + 1].direction
                and candidates[i + 1].bar_count < min_gap
            ):
                # 合并：跳过中间分型，连接首尾
                merged = Stroke(
                    direction=candidates[i].direction,
                    start_fractal=candidates[i].start_fractal,
                    end_fractal=candidates[i + 1].end_fractal,
                    bar_count=(
                        candidates[i + 1].end_fractal.bar_index
                        - candidates[i].start_fractal.bar_index
                    ),
                    confirmed=candidates[i + 1].end_fractal.confirmed,
                )
                result.append(merged)
                i += 2
                changed = True
            else:
                result.append(candidates[i])
                i += 1
        candidates = result

    return candidates


# ══════════════════════════════════════════════════════════════
# 第三层：线段构建（特征序列分型法）
# ══════════════════════════════════════════════════════════════

def _extract_feature_sequence(strokes: List[Stroke]) -> List[Tuple[float, float]]:
    """
    提取特征序列

    缠论定义：线段中每一笔的价格区间 [low, high] 构成一个特征序列元素。
    不区分笔的方向，所有笔都参与特征序列。

    Returns:
        [(high, low), ...] 每个元素为一笔的价格范围
    """
    return [(s.high, s.low) for s in strokes]


def _check_fractal_in_features(
    features: List[Tuple[float, float]]
) -> Tuple[bool, int]:
    """
    在特征序列中检查分型

    缠论定义（特征序列分型）：
    三个连续特征元素 A, B, C，若 B 的范围被 A 和 C 的范围同时包含，
    即 A.high >= B.high, A.low <= B.low, C.high >= B.high, C.low <= B.low，
    则 B 处形成特征序列分型。

    含义：
    - 向上线段中：特征序列底分型 = 线段结束信号
    - 向下线段中：特征序列顶分型 = 线段结束信号

    该规则自动处理非重叠情况：
    当 B 与 A 不重叠时（B.low > A.high 或 B.high < A.low），
    包含条件不成立，不会误判为分型。

    Returns:
        (found: bool, fractal_at_index: int)
        fractal_at_index 为特征序列中的索引（对应 strokes 中的索引）
    """
    for i in range(1, len(features) - 1):
        prev_h, prev_l = features[i - 1]
        mid_h, mid_l = features[i]
        next_h, next_l = features[i + 1]

        if (
            prev_h >= mid_h
            and prev_l <= mid_l
            and next_h >= mid_h
            and next_l <= mid_l
        ):
            return True, i

    return False, -1


def _find_fractal_in_prefix(
    features: List[Tuple[float, float]],
) -> Tuple[bool, int]:
    """
    在特征序列前缀中查找分型

    用于处理"当前笔超出线段范围"的情况：
    检查 features[:-1] 中是否存在分型（不包含最后一笔）。
    """
    if len(features) < 3:
        return False, -1
    return _check_fractal_in_features(features[:-1])


def build_segments(
    strokes: List[Stroke],
    min_strokes: int = 3,
) -> List[Segment]:
    """
    严格线段构建 — 特征序列分型法

    缠论原文：
    "线段的结束，由特征序列的分型来确定。"

    算法：
    1. 从第一笔开始，累积笔到当前线段
    2. 每加入一笔，构建特征序列并检查分型
    3. 若发现分型 → 当前线段结束，新线段从分型后一笔开始
    4. 特殊处理：当一笔超出当前线段范围时，
       先检查已有笔的特征序列是否存在分型

    Args:
        strokes: 笔列表
        min_strokes: 线段最少笔数

    Returns:
        线段列表
    """
    if len(strokes) < min_strokes:
        return []

    segments: List[Segment] = []
    start = 0

    while start <= len(strokes) - min_strokes:
        seg_strokes: List[Stroke] = []
        seg_high = -float("inf")
        seg_low = float("inf")
        found_end = False

        for j in range(start, len(strokes)):
            stroke = strokes[j]
            s_high = stroke.high
            s_low = stroke.low

            if not seg_strokes:
                # 第一笔，启动新线段
                seg_strokes.append(stroke)
                seg_high = s_high
                seg_low = s_low
                continue

            # 检查该笔是否超出当前线段价格范围
            if s_high > seg_high or s_low < seg_low:
                # 超出范围
                if len(seg_strokes) >= min_strokes:
                    # 检查已有笔的特征序列是否存在分型
                    features = _extract_feature_sequence(seg_strokes)
                    found, frac_idx = _find_fractal_in_prefix(features)

                    if found:
                        # 分型在 frac_idx 处，线段结束到该笔
                        # 线段包含分型所在的笔
                        end_idx = start + frac_idx
                        final_strokes = strokes[start: end_idx + 1]
                        if len(final_strokes) >= min_strokes:
                            segments.append(Segment(
                                strokes=final_strokes,
                                direction=(
                                    Direction.UP
                                    if final_strokes[-1].end_fractal.price
                                    > final_strokes[0].start_fractal.price
                                    else Direction.DOWN
                                ),
                                confirmed=all(s.confirmed for s in final_strokes),
                            ))
                        start = end_idx + 1
                        found_end = True
                        break

            # 加入当前笔
            seg_strokes.append(stroke)
            seg_high = max(seg_high, s_high)
            seg_low = min(seg_low, s_low)

            # 检查特征序列分型（对所有已累积的笔）
            if len(seg_strokes) >= 3:
                features = _extract_feature_sequence(seg_strokes)
                found, frac_idx = _check_fractal_in_features(features)

                if found:
                    # 线段结束在 frac_idx 处
                    end_idx = start + frac_idx
                    final_strokes = strokes[start: end_idx + 1]
                    if len(final_strokes) >= min_strokes:
                        segments.append(Segment(
                            strokes=final_strokes,
                            direction=(
                                Direction.UP
                                if final_strokes[-1].end_fractal.price
                                > final_strokes[0].start_fractal.price
                                else Direction.DOWN
                            ),
                            confirmed=all(s.confirmed for s in final_strokes),
                        ))
                    start = end_idx + 1
                    found_end = True
                    break

        if found_end:
            continue

        # 未找到线段结束点
        # 如果剩余笔数 >= min_strokes，作为未完成线段
        remaining = strokes[start:]
        if len(remaining) >= min_strokes:
            segments.append(Segment(
                strokes=remaining,
                direction=(
                    Direction.UP
                    if remaining[-1].end_fractal.price
                    > remaining[0].start_fractal.price
                    else Direction.DOWN
                ),
                confirmed=False,
            ))
        break

    return segments


# ══════════════════════════════════════════════════════════════
# 第四层：中枢检测
# ══════════════════════════════════════════════════════════════

def detect_pivots(segments: List[Segment]) -> List[Pivot]:
    """
    中枢检测

    缠论定义：
    连续三段线段的价格范围有重叠区域，构成中枢。
    - ZG（中枢上沿）= min(各段 high)
    - ZD（中枢下沿）= max(各段 low)
    - 必须 ZG > ZD（有重叠）
    - GG = max(所有段的 high)
    - DD = min(所有段的 low)

    中枢延伸：后续线段的范围仍与 [ZD, ZG] 有重叠。

    Args:
        segments: 线段列表

    Returns:
        中枢列表
    """
    if len(segments) < 3:
        return []

    pivots: List[Pivot] = []
    i = 0

    while i <= len(segments) - 3:
        s1, s2, s3 = segments[i], segments[i + 1], segments[i + 2]

        zg = min(s1.high, s2.high, s3.high)
        zd = max(s1.low, s2.low, s3.low)

        if zg > zd:
            # 有重叠 → 发现中枢
            pivot_segs = [s1, s2, s3]
            gg = max(s1.high, s2.high, s3.high)
            dd = min(s1.low, s2.low, s3.low)

            # 延伸检查
            j = i + 3
            while j < len(segments):
                sj = segments[j]
                new_zg = min(zg, sj.high)
                new_zd = max(zd, sj.low)

                if new_zg > new_zd:
                    # 仍在中枢内
                    pivot_segs.append(sj)
                    zg = new_zg
                    zd = new_zd
                    gg = max(gg, sj.high)
                    dd = min(dd, sj.low)
                    j += 1
                else:
                    break

            direction = (
                Direction.UP
                if pivot_segs[-1].end_price > pivot_segs[0].start_price
                else Direction.DOWN
            )

            pivots.append(Pivot(
                segments=pivot_segs,
                zg=round(zg, 4),
                zd=round(zd, 4),
                gg=round(gg, 4),
                dd=round(dd, 4),
                direction=direction,
            ))

            i = j
        else:
            i += 1

    return pivots


# ══════════════════════════════════════════════════════════════
# 第五层：背驰检测
# ══════════════════════════════════════════════════════════════

def _compute_macd(
    close: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算 MACD 指标

    Returns:
        (dif, dea, histogram)
    """
    def ema(data, period):
        result = np.zeros_like(data, dtype=float)
        result[0] = data[0]
        multiplier = 2.0 / (period + 1)
        for i in range(1, len(data)):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
        return result

    dif = ema(close, fast) - ema(close, slow)
    dea = ema(dif, signal)
    histogram = 2 * (dif - dea)

    return dif, dea, histogram


def _macd_area(histogram: np.ndarray, start: int, end: int) -> float:
    """
    计算 MACD 柱面积（带方向）

    正柱面积 = 正值求和
    负柱面积 = 负值求和（保留负号）
    返回的是带符号的面积值
    """
    if start >= end or start < 0 or end > len(histogram):
        return 0.0
    segment = histogram[start:end]
    return float(np.sum(segment))


def _macd_abs_area(histogram: np.ndarray, start: int, end: int) -> float:
    """计算 MACD 柱的绝对面积（用于幅值对比）"""
    if start >= end or start < 0 or end > len(histogram):
        return 0.0
    return float(np.sum(np.abs(histogram[start:end])))


def detect_divergence(
    df: pd.DataFrame,
    segments: List[Segment],
    pivots: List[Pivot],
) -> List[Divergence]:
    """
    背驰检测

    缠论定义：
    背驰是线段级别（或笔级别）的力度衰减信号。

    顶背驰（bearish）：
      价格创新高（或接近前高），但 MACD 柱面积 / 力度 比前一段弱。

    底背驰（bullish）：
      价格创新低（或接近前低），但 MACD 柱面积 / 力度 比前一段强（负得更少）。

    检测方法（两种，取其一即可确认）：
    1. MACD 柱面积对比：同方向相邻两段，后段面积 < 前段面积
    2. 幅度对比：同方向相邻两段，后段幅度 < 前段幅度

    检测层级：
    - 线段背驰：两段同向线段的对比
    - 笔背驰（可选）：两笔的对比

    Args:
        df: K线数据（需包含 close 列）
        segments: 线段列表
        pivots: 中枢列表

    Returns:
        背驰列表
    """
    divergences: List[Divergence] = []

    if len(segments) < 2 or len(df) < 30:
        return divergences

    close = df["close"].values.astype(float)
    _, _, histogram = _compute_macd(close)

    # ── 线段级背驰 ──
    # 找同方向的相邻段对
    for i in range(1, len(segments)):
        prev_seg = segments[i - 1]
        curr_seg = segments[i]

        # 必须同方向
        if prev_seg.direction != curr_seg.direction:
            continue

        # 获取段在K线中的位置范围
        prev_start_bar = prev_seg.strokes[0].start_fractal.bar_index
        prev_end_bar = prev_seg.strokes[-1].end_fractal.bar_index
        curr_start_bar = curr_seg.strokes[0].start_fractal.bar_index
        curr_end_bar = curr_seg.strokes[-1].end_fractal.bar_index

        # 确保索引有效
        if (
            prev_start_bar >= prev_end_bar
            or curr_start_bar >= curr_end_bar
            or curr_end_bar > len(histogram)
            or prev_end_bar > len(histogram)
        ):
            continue

        # ── 方法1: MACD 柱面积对比 ──
        prev_area = _macd_abs_area(histogram, prev_start_bar, prev_end_bar)
        curr_area = _macd_abs_area(histogram, curr_start_bar, curr_end_bar)

        # ── 方法2: 幅度对比 ──
        prev_amp = prev_seg.amplitude
        curr_amp = curr_seg.amplitude

        if curr_seg.direction == Direction.UP:
            # 向上段：检查顶背驰
            # 条件：当前段价格 >= 前段价格（创新高或接近），
            #       且力度更弱
            price_rising = curr_seg.end_price >= prev_seg.end_price * 0.98

            if price_rising:
                # MACD 背驰
                if prev_area > 0 and curr_area < prev_area:
                    divergences.append(Divergence(
                        div_type=DivergenceType.BEARISH,
                        at_segment_idx=i,
                        price_current=curr_seg.end_price,
                        price_previous=prev_seg.end_price,
                        indicator_current=curr_area,
                        indicator_previous=prev_area,
                        method="macd",
                    ))
                # 幅度背驰
                elif curr_amp < prev_amp * 0.8:
                    divergences.append(Divergence(
                        div_type=DivergenceType.BEARISH,
                        at_segment_idx=i,
                        price_current=curr_seg.end_price,
                        price_previous=prev_seg.end_price,
                        indicator_current=curr_amp,
                        indicator_previous=prev_amp,
                        method="amplitude",
                    ))
        else:
            # 向下段：检查底背驰
            price_falling = curr_seg.end_price <= prev_seg.end_price * 1.02

            if price_falling:
                # MACD 背驰
                if prev_area > 0 and curr_area < prev_area:
                    divergences.append(Divergence(
                        div_type=DivergenceType.BULLISH,
                        at_segment_idx=i,
                        price_current=curr_seg.end_price,
                        price_previous=prev_seg.end_price,
                        indicator_current=curr_area,
                        indicator_previous=prev_area,
                        method="macd",
                    ))
                # 幅度背驰
                elif curr_amp < prev_amp * 0.8:
                    divergences.append(Divergence(
                        div_type=DivergenceType.BULLISH,
                        at_segment_idx=i,
                        price_current=curr_seg.end_price,
                        price_previous=prev_seg.end_price,
                        indicator_current=curr_amp,
                        indicator_previous=prev_amp,
                        method="amplitude",
                    ))

    # ── 笔级背驰（在没有线段背驰时补充检测） ──
    # 在最近的中枢内或中枢突破时，笔的背驰也有参考价值
    if not divergences and len(pivots) > 0:
        last_pivot = pivots[-1]
        # 在中枢附近检测笔背驰
        recent_strokes = []
        for seg in segments[-3:]:  # 最近3段
            recent_strokes.extend(seg.strokes)

        if len(recent_strokes) >= 4:
            for k in range(2, len(recent_strokes)):
                prev_s = recent_strokes[k - 2]
                curr_s = recent_strokes[k]

                if prev_s.direction != curr_s.direction:
                    continue

                prev_start = prev_s.start_fractal.bar_index
                prev_end = prev_s.end_fractal.bar_index
                curr_start = curr_s.start_fractal.bar_index
                curr_end = curr_s.end_fractal.bar_index

                if (
                    prev_end >= len(histogram)
                    or curr_end >= len(histogram)
                    or prev_start >= prev_end
                    or curr_start >= curr_end
                ):
                    continue

                prev_area = _macd_abs_area(histogram, prev_start, prev_end)
                curr_area = _macd_abs_area(histogram, curr_start, curr_end)
                prev_amp = prev_s.amplitude
                curr_amp = curr_s.amplitude

                if curr_s.direction == Direction.UP:
                    if curr_s.end_price >= prev_s.end_price * 0.98:
                        if (prev_area > 0 and curr_area < prev_area) or curr_amp < prev_amp * 0.8:
                            divergences.append(Divergence(
                                div_type=DivergenceType.BEARISH,
                                at_segment_idx=-1,
                                price_current=curr_s.end_price,
                                price_previous=prev_s.end_price,
                                indicator_current=curr_area if prev_area > 0 and curr_area < prev_area else curr_amp,
                                indicator_previous=prev_area if prev_area > 0 and curr_area < prev_area else prev_amp,
                                method="macd" if prev_area > 0 and curr_area < prev_area else "amplitude",
                            ))
                            break
                else:
                    if curr_s.end_price <= prev_s.end_price * 1.02:
                        if (prev_area > 0 and curr_area < prev_area) or curr_amp < prev_amp * 0.8:
                            divergences.append(Divergence(
                                div_type=DivergenceType.BULLISH,
                                at_segment_idx=-1,
                                price_current=curr_s.end_price,
                                price_previous=prev_s.end_price,
                                indicator_current=curr_area if prev_area > 0 and curr_area < prev_area else curr_amp,
                                indicator_previous=prev_area if prev_area > 0 and curr_area < prev_area else prev_amp,
                                method="macd" if prev_area > 0 and curr_area < prev_area else "amplitude",
                            ))
                            break

    return divergences


# ══════════════════════════════════════════════════════════════
# 多周期状态计算
# ══════════════════════════════════════════════════════════════

def compute_status(
    df: pd.DataFrame,
    strokes: List[Stroke],
    segments: List[Segment],
    pivots: List[Pivot],
    divergences: List[Divergence],
) -> dict:
    """
    计算多维度状态

    返回：
    1. 与最近中枢的关系
    2. 未完成笔的状态
    3. 未完成段的状态
    4. 最近背驰信号
    """
    current_price = float(df["close"].iloc[-1])

    # ── 中枢位置 ──
    pivot_rel = PivotRelation.ABOVE
    latest_pivot = pivots[-1] if pivots else None
    if latest_pivot:
        if latest_pivot.contains(current_price):
            pivot_rel = PivotRelation.INSIDE
        elif current_price > latest_pivot.zg:
            pivot_rel = PivotRelation.ABOVE
        else:
            pivot_rel = PivotRelation.BELOW

    # ── 未完成笔状态 ──
    stroke_state = None
    stroke_dir = None
    if strokes:
        last = strokes[-1]
        if not last.confirmed:
            stroke_dir = last.direction
            n = len(df)
            if n >= 2:
                last_high = float(df["high"].iloc[-1])
                last_low = float(df["low"].iloc[-1])

                if last.direction == Direction.UP:
                    if last_high > last.end_fractal.price:
                        stroke_state = StrokeState.EXTENDING_UP
                    else:
                        stroke_state = StrokeState.FORMING_TOP
                else:
                    if last_low < last.end_fractal.price:
                        stroke_state = StrokeState.EXTENDING_DOWN
                    else:
                        stroke_state = StrokeState.FORMING_BOTTOM

    # ── 未完成段状态 ──
    seg_stroke_count = 0
    seg_dir = None
    if segments:
        last_seg = segments[-1]
        if not last_seg.confirmed:
            seg_stroke_count = len(last_seg.strokes)
            seg_dir = last_seg.direction
        else:
            # 已确认的段之后可能有未纳入段的笔
            last_seg_end_bar = last_seg.strokes[-1].end_fractal.bar_index
            trailing = [
                s for s in strokes
                if s.start_fractal.bar_index > last_seg_end_bar
            ]
            if trailing:
                seg_stroke_count = len(trailing)
                seg_dir = trailing[-1].direction

    return {
        "pivot_relation": pivot_rel,
        "latest_pivot": latest_pivot,
        "stroke_state": stroke_state,
        "stroke_direction": stroke_dir,
        "segment_stroke_count": seg_stroke_count,
        "segment_direction": seg_dir,
        "has_divergence": len(divergences) > 0,
        "latest_divergence": divergences[-1].to_dict() if divergences else None,
    }
