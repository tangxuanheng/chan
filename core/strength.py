## 9档强弱 + 板块加权强弱

import os
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time
from datetime import datetime

# 将项目根目录添加到sys.path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import STOCK_SNAPSHOT, STOCK_DATA_PATH

# ══════════════════════════════════════════════════════════════
#  1. 快照构建器
# ══════════════════════════════════════════════════════════════

class SnapshotBuilder:
    """
    从各股票的 feather 文件中提取 (date, close, level) 构建快照序列。

    三层策略：
      - full_build:    首次全量构建，批量并发读每个文件尾部
      - incremental:   每日增量，只读有新数据的文件
      - from_snapshot: 直接用已有的 latest.feather 追加到缓存（零文件遍历）
    """

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)

    # ──────────────────────────────────────────────────────
    #  公开接口
    # ──────────────────────────────────────────────────────

    def full_build(
        self,
        period: str,
        lookback_days: int = 60,
        max_workers: int = 8
    ) -> pd.DataFrame:
        """
        全量构建快照缓存。
        首次运行或缓存损坏时调用。
        """
        cache_path = self._cache_path(period)
        data_dir = Path(f"{self.data_root}/{period}")

        stock_files = self._list_stock_files(data_dir)
        if not stock_files:
            return pd.DataFrame()

        print(f"[{period}] 全量构建: {len(stock_files)} 个文件, "
              f"回看 {lookback_days} 天")

        t0 = time.time()

        # 并发批量读取
        all_dfs = self._batch_read_tails(
            data_dir, stock_files, lookback_days, max_workers
        )
        if not all_dfs:
            return pd.DataFrame()

        result = pd.concat(all_dfs, ignore_index=True)
        result = self._dedup_and_sort(result)

        # 持久化缓存
        result.to_feather(cache_path)

        elapsed = time.time() - t0
        print(f"[{period}] 构建完成: {len(result)} 行, "
              f"{result['code'].nunique()} 只股票, "
              f"耗时 {elapsed:.1f}s")

        return result

    def incremental_update(
        self,
        period: str,
        max_workers: int = 8
    ) -> pd.DataFrame:
        """
        增量更新快照缓存。
        每日收盘后调用，只处理有新数据的文件。
        """
        cache_path = self._cache_path(period)

        if not cache_path.exists():
            print(f"[{period}] 缓存不存在，执行全量构建")
            return self.full_build(period)

        cached = pd.read_feather(cache_path)
        if cached.empty:
            return self.full_build(period)

        last_date = cached['date'].max()
        data_dir = Path(f"{self.data_root}/{period}")
        stock_files = self._list_stock_files(data_dir)

        # 找出修改时间晚于缓存的文件（有新数据）
        cache_mtime = cache_path.stat().st_mtime
        stale_files = [
            f for f in stock_files
            if Path(f"{data_dir}/{f}").stat().st_mtime > cache_mtime
        ]

        if not stale_files:
            print(f"[{period}] 无新数据, 缓存最新日期: {last_date}")
            return cached

        print(f"[{period}] 增量更新: {len(stale_files)}/{len(stock_files)} "
              f"个文件有新数据")

        t0 = time.time()

        new_dfs = self._batch_read_new_rows(
            data_dir, stale_files, last_date, max_workers
        )

        if new_dfs:
            new_data = pd.concat(new_dfs, ignore_index=True)
            combined = pd.concat([cached, new_data], ignore_index=True)
        else:
            combined = cached

        combined = self._dedup_and_sort(combined)
        combined.to_feather(cache_path)

        new_rows = len(combined) - len(cached)
        elapsed = time.time() - t0
        print(f"[{period}] 增量完成: +{new_rows} 行, "
              f"总计 {len(combined)} 行, 耗时 {elapsed:.2f}s")

        return combined

    def append_from_snapshot(
        self,
        period: str,
        snapshot_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        直接从已有的 latest.feather 快照追加到缓存。
        适合每日流程：你已经有了当天快照，不需要再遍历文件。

        snapshot_df: 列至少包含 last_date, code, last_close, last_level
        """
        cache_path = self._cache_path(period)

        # 构造当天快照行
        print(f"columns: {snapshot_df.columns}")
        new_rows = snapshot_df[['last_date', 'code', 'last_close', 'last_level']].copy()
        new_rows.columns = ['date', 'code', 'close', 'strength']

        if not cache_path.exists():
            new_rows.to_feather(cache_path)
            return new_rows

        cached = pd.read_feather(cache_path)
        combined = pd.concat([cached, new_rows], ignore_index=True)
        combined = self._dedup_and_sort(combined)
        combined.to_feather(cache_path)

        return combined

    def get_snapshots(
        self,
        period: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """读取快照缓存，可选日期范围过滤"""
        cache_path = self._cache_path(period)

        if not cache_path.exists():
            return pd.DataFrame()

        df = pd.read_feather(cache_path)

        if start_date:
            df = df[df['date'] >= start_date]
        if end_date:
            df = df[df['date'] <= end_date]

        return df.reset_index(drop=True)

    # ──────────────────────────────────────────────────────
    #  内部方法
    # ──────────────────────────────────────────────────────

    def _cache_path(self, period: str) -> Path:
        return Path(f"{STOCK_SNAPSHOT}/{period}_level_history.feather")

    def _list_stock_files(self, data_dir: Path) -> List[str]:
        """列出所有股票 feather 文件（排除snapshots目录）"""
        if not data_dir.exists():
            return []
        return sorted([
            f.name for f in data_dir.iterdir()
            if f.suffix == '.feather'
            and f.is_file()
            and f.stem not in ('snapshots',)
            and not f.stem.startswith('.')
        ])

    def _batch_read_tails(
        self,
        data_dir: Path,
        filenames: List[str],
        lookback_days: int,
        max_workers: int
    ) -> List[pd.DataFrame]:
        """并发批量读取每个文件的尾部"""
        results = []
        batch_size = 200

        for i in range(0, len(filenames), batch_size):
            batch = filenames[i : i + batch_size]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._read_file_tail,
                        Path(f"{data_dir}/{fname}"),
                        fname,
                        lookback_days
                    ): fname
                    for fname in batch
                }

                for future in as_completed(futures):
                    df = future.result()
                    if df is not None and not df.empty:
                        results.append(df)

        return results

    def _read_file_tail(
        self,
        file_path: Path,
        filename: str,
        lookback_days: int
    ) -> Optional[pd.DataFrame]:
        """读取单个文件的尾部（只读3列）"""
        code = Path(filename).stem
        try:
            # ── 关键优化：只读3列，跳过open/high/low/ma/macd等 ──
            df = pd.read_feather(
                file_path,
                columns=['date', 'close', 'level']
            )

            if df.empty:
                return None

            # 只取最后 lookback_days 行
            if len(df) > lookback_days:
                df = df.iloc[-lookback_days:]

            df = df.copy()
            df['code'] = code

            # 过滤掉 level 为 0 的行（数据不足）
            df = df[df['level'] > 0]

            if df.empty:
                return None
            df.rename(columns={"level": "strength"}, inplace=True)
            return df[['date', 'code', 'close', 'strength']]

        except Exception as e:
            # 文件损坏或格式不对，静默跳过
            print(f"read_file_tail err msg: {str(e)}")
            return None

    def _batch_read_new_rows(
        self,
        data_dir: Path,
        filenames: List[str],
        after_date: str,
        max_workers: int
    ) -> List[pd.DataFrame]:
        """并发读取指定日期之后的新行"""
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._read_new_rows,
                    Path(f"{data_dir}/{fname}"),
                    fname,
                    after_date
                ): fname
                for fname in filenames
            }

            for future in as_completed(futures):
                df = future.result()
                if df is not None and not df.empty:
                    results.append(df)

        return results

    def _read_new_rows(
        self,
        file_path: Path,
        filename: str,
        after_date: str
    ) -> Optional[pd.DataFrame]:
        """读取指定日期之后的新数据"""
        code = Path(filename).stem

        try:
            df = pd.read_feather(
                file_path,
                columns=['date', 'close', 'level']
            )

            new_df = df[df['date'] > after_date].copy()

            if new_df.empty:
                return None

            new_df['code'] = code
            new_df = new_df[new_df['level'] > 0]

            if new_df.empty:
                return None
            new_df.rename(columns={"level": "strength"}, inplace=True)
            return new_df[['date', 'code', 'close', 'strength']]

        except Exception:
            print(f"_read_new_rows err msg: {str(e)}")
            return None

    def _dedup_and_sort(self, df: pd.DataFrame) -> pd.DataFrame:
        """去重排序：同一股票同一天只保留最后一条"""
        df = df.sort_values(['date', 'code'])
        df = df.drop_duplicates(
            subset=['date', 'code'],
            keep='last'
        ).reset_index(drop=True)
        return df


# ══════════════════════════════════════════════════════════════
#  2. 快照管理器（对接分析引擎）
# ══════════════════════════════════════════════════════════════

@dataclass
class SnapshotState:
    code: str
    close: float
    strength: int
    date: str


class SnapshotManager:
    """
    管理多周期快照，对外提供结构化接口。

    内部缓存格式 (history.feather):
        date | code | close | strength
    """

    def __init__(self, data_root: str):
        self.data_root = data_root
        self.builder = SnapshotBuilder(data_root)
        # 内存缓存：避免重复读 feather
        self._df_cache: Dict[str, pd.DataFrame] = {}

    # ──────────────────────────────────────────────────────
    #  初始化
    # ──────────────────────────────────────────────────────

    def warm_up(
        self,
        periods: List[str],
        lookback_days: int = 60,
        force_rebuild: bool = False
    ):
        """
        预热所有周期的快照缓存。
        首次运行或数据源更新后调用。
        """
        for period in periods:
            cache_path = self.builder._cache_path(period)

            if force_rebuild or not cache_path.exists():
                self.builder.full_build(period, lookback_days)
            else:
                self.builder.incremental_update(period)

            # 加载到内存
            self._df_cache[period] = pd.read_feather(cache_path)

    # ──────────────────────────────────────────────────────
    #  每日流程
    # ──────────────────────────────────────────────────────

    def daily_update(
        self,
        period: str,
        snapshot_df: pd.DataFrame
    ):
        """
        每日更新流程（推荐方式）：

        你已经有了 latest.feather，直接传进来即可，零文件遍历。

        snapshot_df: 从 latest.feather 读取的 DataFrame
                     至少包含 last_date, code, last_close, last_level 列
        """
        # 追加到缓存
        updated = self.builder.append_from_snapshot(
            period, snapshot_df
        )

        # 刷新内存缓存
        self._df_cache[period] = updated

    def daily_update_from_files(self, period: str):
        """
        每日更新流程（备选方式）：
        如果你没有单独的快照文件，通过遍历目录增量更新。
        """
        updated = self.builder.incremental_update(period)
        self._df_cache[period] = updated

    # ──────────────────────────────────────────────────────
    #  查询接口
    # ──────────────────────────────────────────────────────

    def get_latest_snapshot(self, period: str) -> Dict[str, SnapshotState]:
        """获取最新一天的快照（内存操作，零IO）"""
        df = self._get_cached_df(period)
        if df.empty:
            return {}

        latest_date = df['date'].max()
        latest = df[df['date'] == latest_date]

        return {
            row['code']: SnapshotState(
                code=row['code'],
                close=float(row['close']),
                strength=int(row['strength']),
                date=str(row['date'])
            )
            for _, row in latest.iterrows()
        }

    def get_snapshot_on_date(
        self,
        period: str,
        target_date: str
    ) -> Dict[str, SnapshotState]:
        """获取指定日期的快照"""
        df = self._get_cached_df(period)
        if df.empty:
            return {}
        target_dt = pd.to_datetime(target_date)
        df['date'] = pd.to_datetime(df['date'])
        day_data = df[df['date'] == target_dt]

        return {
            row['code']: SnapshotState(
                code=row['code'],
                close=float(row['close']),
                strength=int(row['strength']),
                date=str(row['date'])
            )
            for _, row in day_data.iterrows()
        }

    def get_stock_strength_series(
        self,
        period: str,
        code: str,
        days: int = 20
    ) -> Optional[np.ndarray]:
        """
        获取单只股票的强弱序列（内存操作）。
        替代之前需要读单个feather文件的操作。
        """
        df = self._get_cached_df(period)
        if df.empty:
            return None

        stock_df = df[df['code'] == code].tail(days)
        if stock_df.empty:
            return None

        return stock_df['strength'].values.astype(np.int8)

    def get_date_list(self, period: str) -> List[str]:
        """获取快照中所有交易日"""
        df = self._get_cached_df(period)
        if df.empty:
            return []
        return sorted(df['date'].unique().tolist())

    def get_available_periods(self) -> List[str]:
        """获取已有缓存的周期列表"""
        root = Path(self.data_root)
        periods = []
        for d in root.iterdir():
            if d.is_dir() and (d / 'snapshots' / 'history.feather').exists():
                periods.append(d.name)
        return sorted(periods)

    # ──────────────────────────────────────────────────────
    #  内部
    # ──────────────────────────────────────────────────────

    def _get_cached_df(self, period: str) -> pd.DataFrame:
        """获取缓存的DataFrame，惰性加载"""
        if period not in self._df_cache:
            cache_path = self.builder._cache_path(period)
            if cache_path.exists():
                self._df_cache[period] = pd.read_feather(cache_path)
            else:
                self._df_cache[period] = pd.DataFrame()
        return self._df_cache[period]


# ══════════════════════════════════════════════════════════════
#  3. 每日分析引擎
# ══════════════════════════════════════════════════════════════

class DailyAnalysisEngine:
    """
    每日分析引擎。
    快照构建和分析完全解耦。
    """

    def __init__(self, data_root: str, period: str = 'day'):
        self.period = period
        self.snapshot_mgr = SnapshotManager(data_root)

    def initialize(self, lookback_days: int = 60):
        """
        首次初始化：构建快照缓存。
        只需要运行一次，后续每日增量更新。
        """
        print(f"{'='*50}")
        print(f"初始化 {self.period} 快照缓存")
        print(f"{'='*50}")

        self.snapshot_mgr.warm_up(
            periods=[self.period],
            lookback_days=lookback_days,
            force_rebuild=False
        )

    def run(self, trade_date: str) -> Dict:
        """
        每日运行入口。

        流程：
        1. 读取当天 latest.feather
        2. 追加到快照缓存
        3. 从缓存中取前一日快照（内存操作）
        4. 对比检测变动
        5. 扫描持续强势
        """
        t0 = time.time()

        # ── Step 1: 加载当天快照并追加缓存 ──
        latest_path = Path(f"{STOCK_SNAPSHOT}/{self.period}_snapshot.feather")

        if latest_path.exists():
            today_snapshot_df = pd.read_feather(latest_path)
            # 追加到历史缓存（零文件遍历）
            self.snapshot_mgr.daily_update(
                self.period, today_snapshot_df
            )
        else:
            print("警告: latest.feather 不存在，使用文件遍历方式")
            self.snapshot_mgr.daily_update_from_files(self.period)

        # ── Step 2: 从缓存取当天和前日快照（纯内存） ──
        dates = self.snapshot_mgr.get_date_list(self.period)

        if len(dates) < 2:
            print("快照数据不足2天，无法进行变动检测")
            return {'trade_date': trade_date, 'insufficient_data': True}

        curr_date = dates[-1]
        prev_date = dates[-2]

        curr_snapshot = self.snapshot_mgr.get_snapshot_on_date(
            self.period, curr_date
        )
        prev_snapshot = self.snapshot_mgr.get_snapshot_on_date(
            self.period, prev_date
        )

        print(f"快照就绪: {prev_date}({len(prev_snapshot)}只) → "
              f"{curr_date}({len(curr_snapshot)}只)  "
              f"[{time.time()-t0:.3f}s]")

        # ── Step 3: 强弱变动检测（纯内存） ──
        t1 = time.time()
        changes = self._detect_changes(prev_snapshot, curr_snapshot, trade_date)
        print(f"变动检测: {len(changes)}条  [{time.time()-t1:.3f}s]")

        # ── Step 4: 持续强势扫描（纯内存，不再读文件） ──
        t2 = time.time()
        continuous = self._scan_continuous_strength(
            self.period, min_strength=6, min_days=3
        )
        print(f"持续强势: {len(continuous)}只  [{time.time()-t2:.3f}s]")

        total = time.time() - t0
        print(f"总耗时: {total:.3f}s")

        return {
            'date': trade_date,
            'period': self.period,
            'prev_date': prev_date,
            'curr_date': curr_date,
            'changes': changes,
            'continuous_strength': continuous,
            'stats': {
                'total_stocks': len(curr_snapshot),
                'strong_stocks': sum(
                    1 for s in curr_snapshot.values() if s.strength >= 6
                ),
                'total_changes': len(changes),
                'processing_time': round(total, 3)
            }
        }

    # ── 变动检测（同之前，纯内存） ──

    def _detect_changes(self, prev, curr, trade_date) -> List[Dict]:
        categories = {
            ('UP', 1): 'UP_1', ('UP', 2): 'UP_2', ('UP', 3): 'UP_3+',
            ('DOWN', 1): 'DOWN_1', ('DOWN', 2): 'DOWN_2', ('DOWN', 3): 'DOWN_3+',
        }
        changes = []
        common = set(prev.keys()) & set(curr.keys())

        for code in common:
            ps, cs = prev[code].strength, curr[code].strength
            if ps == cs:
                continue
            level = abs(cs - ps)
            typ = 'UP' if cs > ps else 'DOWN'
            cat = categories.get((typ, min(level, 3)), f'{typ}_3+')
            changes.append({
                'code': code,
                'prev_strength': ps,
                'curr_strength': cs,
                'change_type': typ,
                'change_level': level,
                'category': cat
            })

        return changes

    # ── 持续强势扫描（从缓存读，不再遍历文件） ──

    def _scan_continuous_strength(
        self,
        period: str,
        min_strength: int = 6,
        min_days: int = 3
    ) -> pd.DataFrame:
        """
        从快照缓存中扫描持续强势股票。
        全部内存操作，不再读取单个股票文件。
        """
        df = self.snapshot_mgr._get_cached_df(period)
        if df.empty:
            return pd.DataFrame()

        # 取最近 min_days 个交易日
        dates = sorted(df['date'].unique())
        recent_dates = dates[-min_days:]

        if len(recent_dates) < min_days:
            return pd.DataFrame()

        recent = df[df['date'].isin(recent_dates)]

        # 按股票分组，检查每天是否都 >= min_strength
        grouped = recent.groupby('code')['strength']

        # 找出每天都满足条件的股票
        valid = grouped.agg(['min', 'count'])
        valid = valid[
            (valid['min'] >= min_strength) &
            (valid['count'] == min_days)
        ]

        if valid.empty:
            return pd.DataFrame()

        # 获取最新一天的数据
        latest_date = recent_dates[-1]
        latest = df[df['date'] == latest_date]
        latest = latest[latest['code'].isin(valid.index)]

        result = latest[['code', 'close', 'strength']].copy()
        result = result.rename(columns={'strength': 'current_strength'})

        # 计算连续天数（精确值，不限于min_days）
        for idx, row in result.iterrows():
            code = row['code']
            all_strengths = df[df['code'] == code]['strength'].values
            cont = 0
            for s in reversed(all_strengths):
                if s >= min_strength:
                    cont += 1
                else:
                    break
            result.loc[idx, 'continuous_days'] = cont

        result = result.sort_values(
            'continuous_days', ascending=False
        ).reset_index(drop=True)

        return result


if __name__ == "__main__":
    # ── 首次运行：初始化快照缓存 ──
    engine = DailyAnalysisEngine(STOCK_DATA_PATH, period='day')
    engine.initialize(lookback_days=60)
    # 输出:
    # ==================================================
    # 初始化 daily 快照缓存
    # ==================================================
    # [daily] 全量构建: 5200 个文件, 回看 60 天
    # [daily] 构建完成: 312000 行, 5200 只股票, 耗时 6.3s

    # ── 之后每天：增量更新 + 分析 ──
    result = engine.run('2025-03-25')
    # 输出:
    # 快照就绪: 2025-03-25(5180只) → 2025-03-26(5195只)  [0.003s]
    # 变动检测: 342条  [0.008s]
    # 持续强势: 87只   [0.012s]
    # 总耗时: 0.025s

    # 按最近强弱级别输出各类持续强势股代码
    strong_dict = {}
    for strength, group in result["continuous_strength"].groupby('current_strength'):
        # 按 continuous_days 降序排序
        sorted_group = group.sort_values('continuous_days', ascending=False)
        # 生成字典列表
        strong_dict[strength] = sorted_group[['code', 'continuous_days']].to_dict('records')
    print(json.dumps(strong_dict[6], ensure_ascii=False, indent=2))

    # ── 访问历史快照 ──
    # 获取指定日期的快照
    snap = engine.snapshot_mgr.get_snapshot_on_date('day', '2026-03-25')
    print(f"3月25日共 {len(snap)} 只股票")

    # 获取某只股票的强弱序列（不再需要读单个feather文件）
    series = engine.snapshot_mgr.get_stock_strength_series('day', 'sz.002364', days=20)
    print(f"中恒电气近20日强弱: {series}")

    # 获取所有交易日列表
    dates = engine.snapshot_mgr.get_date_list('day')
    print(f"快照覆盖 {len(dates)} 个交易日: {dates[0]} ~ {dates[-1]}")

    # ── 多周期 ──
    # for period in ['day', '30m', '5m']:
    #     engine_p = DailyAnalysisEngine(DATA_ROOT, period=period)
    #     engine_p.initialize(lookback_days=30)
    #     result_p = engine_p.run('2025-03-26')