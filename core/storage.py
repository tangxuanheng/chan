## 读取数据：统一入口
"""
缠论分析模块 - 数据存储层
支持Feather格式K线数据读写，分析状态持久化
"""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any

import pandas as pd
import os 

# 将项目根目录添加到sys.path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import StorageConfig

logger = logging.getLogger(__name__)


class FeatherStore:
    """
    Feather格式K线数据存储

    目录结构:
        data/
        └── {period}/            # week/day/30m/5m
            ├── symbol.feather 
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig()
        self.config.data_path.mkdir(parents=True, exist_ok=True)
        self.config.state_path.mkdir(parents=True, exist_ok=True)

    # ── 路径工具 ─────────────────────────────────────────

    def _feather_path(self, symbol: str, freq: str) -> Path:
        symbol_dir = Path(f"{self.config.data_path}/{freq}")
        symbol_dir.mkdir(parents=True, exist_ok=True)
        return Path(f"{self.config.data_path}/{freq}/{symbol}.feather")

    def _state_path(self, symbol: str, freq: str) -> Path:
        return Path(f"{self.config.state_path}/{freq}/{symbol}.json")

    # ── K线数据操作 ──────────────────────────────────────

    def load_klines(self, symbol: str, freq: str) -> Optional[pd.DataFrame]:
        """加载K线数据，返回标准列：datetime, open, high, low, close, volume"""
        path = self._feather_path(symbol, freq)
        if not path.exists():
            logger.warning(f"数据文件不存在: {path}")
            return None
        try:
            df = pd.read_feather(path)
            # 统一列名
            col_map = {}
            for col in df.columns:
                cl = col.lower().strip()
                if cl in ("datetime", "date", "time", "trade_time"):
                    col_map[col] = "datetime"
                elif cl in ("open", "开盘价"):
                    col_map[col] = "open"
                elif cl in ("high", "最高价"):
                    col_map[col] = "high"
                elif cl in ("low", "最低价"):
                    col_map[col] = "low"
                elif cl in ("close", "收盘价"):
                    col_map[col] = "close"
                elif cl in ("volume", "vol", "成交量"):
                    col_map[col] = "volume"
                elif cl in ("amount", "成交额"):
                    col_map[col] = "amount"
                elif cl in ("code", "symbol", "代码", "股票代码"):
                    col_map[col] = "code"
            df = df.rename(columns=col_map)

            required = {"code", "datetime", "open", "high", "low", "close", "amount"}
            if not required.issubset(set(df.columns)):
                raise ValueError(f"缺少必要列: {required - set(df.columns)}")

            if "volume" not in df.columns:
                df["volume"] = 0.0

            df = df[["code", "datetime", "open", "high", "low", "close", "volume"]]
            df = df.sort_values("datetime").reset_index(drop=True)

            # 确保数值类型
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["open", "high", "low", "close"])

            logger.info(f"加载 {symbol}/{freq}: {len(df)} 条K线")
            return df
        except Exception as e:
            logger.error(f"加载K线失败 {path}: {e}")
            return None

    def save_klines(self, symbol: str, freq: str, df: pd.DataFrame) -> bool:
        """保存K线数据到Feather文件"""
        path = self._feather_path(symbol, freq)
        try:
            df.to_feather(path)
            logger.info(f"保存 {symbol}/{freq}: {len(df)} 条K线 -> {path}")
            return True
        except Exception as e:
            logger.error(f"保存K线失败 {path}: {e}")
            return False

    def update_klines(self, symbol: str, freq: str, new_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        增量更新K线数据
        - 合并新旧数据，去重（以datetime为键），排序
        - 返回完整的合并后DataFrame
        """
        existing = self.load_klines(symbol, freq)
        if existing is None or len(existing) == 0:
            self.save_klines(symbol, freq, new_df)
            return new_df

        # 合并去重
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["datetime"], keep="last")
        combined = combined.sort_values("datetime").reset_index(drop=True)

        self.save_klines(symbol, freq, combined)
        return combined

    def list_symbols(self) -> List[str]:
        """列出已存储的所有股票代码"""
        if not self.config.data_path.exists():
            return []
        codes = [f.stem for f in dir_path.glob("*.feather")]
        return sorted(set(codes))

    # ── 分析状态操作 ──────────────────────────────────────

    def save_state(self, symbol: str, freq: str, state: dict) -> None:
        """保存分析状态到JSON"""
        path = self._state_path(symbol, freq)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, default=str)
            logger.debug(f"保存状态: {path}")
        except Exception as e:
            logger.error(f"保存状态失败 {path}: {e}")

    def load_state(self, symbol: str, freq: str) -> Optional[dict]:
        """加载分析状态"""
        path = self._state_path(symbol, freq)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载状态失败 {path}: {e}")
            return None

    def delete_state(self, symbol: str, freq: str) -> None:
        path = self._state_path(symbol, freq)
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    fs = FeatherStore()
    klines = fs.load_klines("sh.600004", "5m")
    print(klines.head())