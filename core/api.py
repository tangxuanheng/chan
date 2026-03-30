# FastAPI 路由

"""
缠论分析模块 - API 路由
"""
import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .engine import ChanlunEngine
from .config import AnalysisConfig, StorageConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chanlun", tags=["缠论分析"])

# 全局引擎实例（可通过依赖注入替换）
_engine: Optional[ChanlunEngine] = None


def get_engine() -> ChanlunEngine:
    global _engine
    if _engine is None:
        _engine = ChanlunEngine()
    return _engine


# ── 请求/响应模型 ──────────────────────────────────────

class AnalyzeRequest(BaseModel):
    symbol: str
    freq: str = "D"
    force_full: bool = False


class BatchAnalyzeRequest(BaseModel):
    symbols: List[str]
    freqs: List[str] = ["D", "30m", "5m"]


class StatusResponse(BaseModel):
    symbol: str
    freqs: dict


# ── 路由 ──────────────────────────────────────────────

@router.post("/analyze")
async def analyze_single(req: AnalyzeRequest):
    """单股票单周期分析"""
    engine = get_engine()
    result = engine.analyze(req.symbol, req.freq, force_full=req.force_full)
    if result is None:
        raise HTTPException(404, f"数据不足或不存在: {req.symbol}/{req.freq}")
    return result.summary()


@router.get("/result/{symbol}/{freq}")
async def get_result(symbol: str, freq: str):
    """获取完整分析结果"""
    engine = get_engine()
    # 优先返回缓存
    cached = engine._get_cache(symbol, freq)
    if cached:
        return {
            "summary": cached.summary(),
            "fractals": [f.to_dict() for f in cached.fractals[-20:]],  # 最近20个
            "strokes": [s.to_dict() for s in cached.strokes[-10:]],    # 最近10笔
            "segments": [s.to_dict() for s in cached.segments[-5:]],   # 最近5段
            "pivots": [p.to_dict() for p in cached.pivots[-3:]],       # 最近3个中枢
        }
    # 没有缓存则触发分析
    result = engine.analyze(symbol, freq)
    if result is None:
        raise HTTPException(404, f"数据不足: {symbol}/{freq}")
    return {
        "summary": result.summary(),
        "fractals": [f.to_dict() for f in result.fractals[-20:]],
        "strokes": [s.to_dict() for s in result.strokes[-10:]],
        "segments": [s.to_dict() for s in result.segments[-5:]],
        "pivots": [p.to_dict() for p in result.pivots[-3:]],
    }


@router.get("/status/{symbol}")
async def get_multi_status(
    symbol: str,
    freqs: str = Query("D,30m,5m", description="逗号分隔的周期列表"),
):
    """获取多周期状态摘要（轻量接口）"""
    engine = get_engine()
    freq_list = [f.strip() for f in freqs.split(",")]
    statuses = engine.get_multi_timeframe_status(symbol, freq_list)
    return {"symbol": symbol, "statuses": statuses}


@router.post("/batch-analyze")
async def batch_analyze(req: BatchAnalyzeRequest):
    """批量分析多只股票"""
    engine = get_engine()
    results = {}
    for symbol in req.symbols:
        symbol_results = {}
        for freq in req.freqs:
            try:
                r = engine.analyze(symbol, freq)
                if r:
                    symbol_results[freq] = r.summary()
            except Exception as e:
                logger.error(f"分析失败 {symbol}/{freq}: {e}")
                symbol_results[freq] = {"error": str(e)}
        results[symbol] = symbol_results
    return results


@router.post("/force-reanalyze/{symbol}/{freq}")
async def force_reanalyze(symbol: str, freq: str):
    """强制全量重新分析"""
    engine = get_engine()
    result = engine.force_reanalyze(symbol, freq)
    if result is None:
        raise HTTPException(404, f"数据不足: {symbol}/{freq}")
    return result.summary()
