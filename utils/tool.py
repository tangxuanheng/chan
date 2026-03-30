import os
import glob
from typing import Optional, List, Dict, Any
import baostock as bs
import akshare as ak 
import pandas as pd 
from numba import jit
import numpy as np
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

# 将项目根目录添加到sys.path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BATCH_SIZE, MA_PERIODS, STOCK_INFO_PATH, STOCK_DATA_PATH, STOCK_SNAPSHOT, STOCK_STRUCTURE_PATH, INDEX_INFO_PATH, INDEX_DATA_PATH, INDEX_SNAPSHOT, INDEX_STRUCTURE_PATH

# ================= 数据源抽象 =================

# 数据源注册表
DATA_SOURCE_MAP = {
    "day":   "baostock",
    "week":  "baostock",
    "month": "baostock",
    "30m":   "akshare",
    "5m":    "akshare",
    "1m":    "akshare",
}

# akshare period 参数映射
AKSHARE_PERIOD_MAP = {"1m": "1", "5m": "5", "30m": "30"}
# baostock period 参数映射
BAOSTOCK_PERIOD_MAP = {"5m": "5", "15m": "15", "30m": "30", "60m": "60", "day": "d", "week": "w", "month": "m"}

# 更新顺序：先日周，再分钟（从大周期到小周期）
UPDATE_ORDER = ["day", "week", "30m", "5m"]

# 初始化顺序：先日/周，再分钟
INIT_ORDER = ["day", "week", "30m", "5m"]

MAX_MA = max(MA_PERIODS)

# 日线周线默认拉取年数
DAY_YEARS = 5
WEEK_YEARS = 10
# 5分钟、30分钟默认拉取时间
MINUTE_MONTHS = 6

# ================= 辅助函数 =================
def extract_code_number(code):
    """
    从证券代码中提取数字部分
    例如: 'sh.600000' -> '600000'
         'sz.000001' -> '000001'
         '600000' -> '600000'
    """
    import re
    match = re.search(r'\d+', code)
    return match.group() if match else ''

def convert_to_ak_code(code):
    """
    将sh.600000格式代码转为sh600000
    例如: 'sh.600000' -> 'sh600000'
         'sz.000001' -> 'sz000001'
    """
    return code.replace('.', '')

def _available_mas(data_len):
    """返回当前数据长度可计算的MA周期列表"""
    return sorted(p for p in MA_PERIODS if data_len >= p)


def _data_file(data_dir, period, code):
    return f"{data_dir}/{period}/{code}.feather"


def _file_exists(data_dir, period, code):
    return os.path.exists(_data_file(data_dir, period, code))

def _bs_login():
    """登录 baostock，返回是否成功"""
    lg = bs.login()
    if lg.error_code != '0':
        print(f"❌ baostock 登录失败: {lg.error_msg}")
        return False
    return True


def _bs_logout():
    bs.logout()


# ================= 数据获取 =================

def get_stock_basic():
    """
    通过baostock获取A股证券基本资料
    
    Returns
    -------
    pandas.DataFrame
        A股证券基本资料DataFrame
            code code_name     ipoDate outDate type status
        0  sh.000001    上证综合指数  1991-07-15            2      1
        1  sh.000002    上证A股指数  1992-02-21            2      1

    type : int
        证券类型: 1表示股票, 2表示指数, 3表示基金, 4表示债券, 5表示存托凭证
    status : int
        证券状态: 1表示正常, 0表示退市
    """
    # 登录获取股票列表（需要先登录，但股票列表获取后可以继续用同一会话）
    try:
        bs.login()
        rs = bs.query_stock_basic(code="", code_name="")
        # 将结果转换为DataFrame
        stock_list = []
        while (rs.error_code == '0') and rs.next():
            stock_list.append(rs.get_row_data())
        # 设置列名
        result = pd.DataFrame(stock_list, columns=rs.fields)
        result['type'] = result['type'].astype(int)
        result['status'] = result['status'].astype(int)
        result['ipoDate'] = pd.to_datetime(result['ipoDate'])
        result['outDate'] = pd.to_datetime(result['outDate'])
        result.reset_index(drop=True, inplace=True)
    except Exception as e:
        print(f"⚠️ 获取股票列表失败: {e}，将跳过数据获取")
        return None
    finally:
        bs.logout()
    return result

def get_trading_dates(start_date: str, end_date: str):
    """
    获取交易日列表（用于分钟数据获取时的日期校验）
    
    Parameters
    ----------
    start_date : str
        开始日期，格式 "YYYY-MM-DD"
    end_date : str
        结束日期，格式 "YYYY-MM-DD"
    
    Returns
    -------
    list
        交易日列表，格式 ["YYYY-MM-DD", ...]
    """
    try:
        # 获取交易日历
        trade_date_df = ak.tool_trade_date_hist_sina()
        trade_date_df['trade_date'] = pd.to_datetime(trade_date_df['trade_date'])
        
        # 过滤指定日期范围
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        mask = (trade_date_df['trade_date'] >= start) & (trade_date_df['trade_date'] <= end)
        trading_dates = trade_date_df.loc[mask, 'trade_date'].dt.strftime('%Y-%m-%d').tolist()
        
        return trading_dates
    except Exception as e:
        print(f"⚠️ 获取交易日历失败: {e}，将跳过日期校验")
        return []


def fetch_ak_minute_data(code, period="5", max_retries=3):
    """
    akshare 获取分钟数据，带重试，失败时指数退避
    
    Parameters
    ----------
    code : str
        股票代码，纯数字，如 "600000"
    period : str
        分钟周期："1", "5" 或 "30"
    start_date : str
        开始日期时间，格式 "YYYY-MM-DD HH:MM:SS"
    end_date : str
        结束日期时间，格式 "YYYY-MM-DD HH:MM:SS"
    
    Returns
    -------
    pd.DataFrame
        包含分钟级K线数据的DataFrame
    """    
    for attempt in range(max_retries):
        try:
            df = ak.stock_zh_a_minute(
                symbol=convert_to_ak_code(code),
                period=AKSHARE_PERIOD_MAP[period],
                adjust="qfq",
            )
            if df is None or df.empty:
                return None
            df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount']
            df['date'] = pd.to_datetime(df['date'])
            df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
            df['code'] = code
            return df.sort_values('date').reset_index(drop=True)
        except Exception as e:
            wait = 2 * (attempt + 1)
            print(f"    ⚠️ akshare 重试 {attempt+1}/{max_retries} "
                  f"{code} {period}: {e}，等待{wait}s")
            time.sleep(wait)
    return None

def convert_baostock_time(time_str: str) -> pd.Timestamp:
    """将 baostock 的 time 字段（YYYYMMDDHHMMSSsss）转为 pandas Timestamp"""
    # 补齐毫秒到 6 位
    if len(time_str) == 17:  # 14位日期时间 + 3位毫秒
        time_str = time_str + '000'  # 变成 YYYYMMDDHHMMSSsss000
    # 使用格式解析
    return pd.to_datetime(time_str, format='%Y%m%d%H%M%S%f')

def fetch_bs_k_data(code, period="d", start_date=None, end_date=None):
    """
    通过baostock获取单只股票的K线数据
    
    Parameters
    ----------
    code : str
        bs股票代码，如 "sh.600000"
    period : str
        "d" 日线, "w" 周线, "m" 月线
        "5" 5分钟, "15" 15分钟, "30" 30分钟, "60" 60分钟
    start_date : str
        开始日期 "YYYY-MM-DD"
    end_date : str
        结束日期 "YYYY-MM-DD"
    """    
    # 设置默认日期范围
    if start_date is None:
        if period == "day":
            # 日线5年
            start_date = (datetime.now() - timedelta(days=365*DAY_YEARS)).strftime("%Y-%m-%d")
        elif period in ["week", "month"]:  
            # 周线月线十年
            start_date = (datetime.now() - timedelta(days=365*WEEK_YEARS)).strftime("%Y-%m-%d")
        else:
            # 分钟线半年
            start_date = (datetime.now() - timedelta(days=365*WEEK_YEARS)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    fields = "date,code,open,high,low,close,volume,amount" if period in ['day', 'week', 'month'] else "time,code,open,high,low,close,volume,amount"

    try:
        rs = bs.query_history_k_data_plus(
            code,
            fields,
            start_date=start_date,
            end_date=end_date,
            frequency=BAOSTOCK_PERIOD_MAP[period],
            adjustflag="2"  # 前复权
        )
        
        if rs.error_code != '0':
            print(f"查询失败 {code}: {rs.error_msg}")
            return None
        
        # 获取数据
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            return None
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # 转换数据类型
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if "time" in df.columns:
            df['time'] = df['time'].apply(convert_baostock_time)
            df = df.sort_values('time').reset_index(drop=True)
        else:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        
        # 重命名列为标准格式
        df.columns = ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount']
        return df
        
    except Exception as e:
        print(f"❌ 获取失败 {code}.{period}: {e}")
        return None

def fetch_kline(code, period, start_dt, end_dt, kind="stock"):
    """
    统一数据获取入口

    Parameters
    ----------
    code : str
    period : "day" | "week" | "30m" | "5m" | "1m"
    start_dt : datetime
        快照最后时间 + 1个单位
    end_dt : datetime
    kind : "stock" | "index"

    Returns
    -------
    pd.DataFrame 或 None，只包含 start_dt 之后的新数据
    """
    # baostock 支持指定日期范围
    df = fetch_bs_k_data(
        code, period,
        start_date=start_dt,
        end_date=end_dt,
    )
    return df

# ================= 工具函数 =================
@jit(nopython=True)
def calc_ma_full(close_prices, period):
    """全量计算均线（numba加速）"""
    n = len(close_prices)
    ma = np.full(n, np.nan)
    if n < period:
        return ma
    ma[period - 1] = np.mean(close_prices[:period])
    for i in range(period, n):
        ma[i] = ma[i - 1] + (close_prices[i] - close_prices[i - period]) / period
    return ma

@jit(nopython=True)
def calc_level(close, ma_matrix):
    """
    向量化计算 level

    Parameters
    ----------
    close : array (n,)
    ma_matrix : array (n, 8)，8条均线按列排列

    Returns
    -------
    level : array (n,)，取值 1~9

    Use
    -------
    ma_cols = np.column_stack([day_df[f'ma_{p}'].values for p in MA_PERIODS])
    day_df['level'] = calc_level(close_arr, ma_cols)
    """
    n = len(close)
    level = np.ones(n, dtype=np.int32)
    for i in range(n):
        for j in range(ma_matrix.shape[1]):
            if close[i] >= ma_matrix[i, j]:
                level[i] += 1
    return level

@jit(nopython=True)
def calc_macd_full(close_prices):
    """全量计算MACD，返回所有中间值"""
    n = len(close_prices)
    ema12 = np.zeros(n)
    ema26 = np.zeros(n)
    dif = np.zeros(n)
    dea = np.zeros(n)
    macd = np.zeros(n)

    ema12[0] = close_prices[0]
    ema26[0] = close_prices[0]
    for i in range(1, n):
        ema12[i] = ema12[i-1] * 11/13 + close_prices[i] * 2/13
        ema26[i] = ema26[i-1] * 25/27 + close_prices[i] * 2/27
        dif[i] = ema12[i] - ema26[i]
        dea[i] = dea[i-1] * 8/10 + dif[i] * 2/10
        macd[i] = 2 * (dif[i] - dea[i])

    return dif, dea, macd, ema12, ema26

@jit(nopython=True)
def calc_ma_from_state(close_all, start_idx, ma_start, period):
    """
    从指定位置开始增量计算均线

    Parameters
    ----------
    close_all : array
        完整序列：[历史尾巴(用于取滑出值) + 新数据]
    start_idx : int
        新数据在 close_all 中的起始位置
    ma_start : float
        start_idx - 1 位置的 MA 值（来自快照）
    period : int
        均线周期

    Returns
    -------
    ma_values : array
        仅新数据部分的 MA 值
    """
    n_new = len(close_all) - start_idx
    ma_values = np.zeros(n_new)

    prev_ma = ma_start
    for i in range(n_new):
        idx = start_idx + i
        # 滑出的值：close_all[idx - period]
        # idx - period 一定 >= 0，因为 start_idx >= period（由调用方保证）
        ma_values[i] = prev_ma + (close_all[idx] - close_all[idx - period]) / period
        prev_ma = ma_values[i]

    return ma_values


@jit(nopython=True)
def calc_macd_from_state(close_all, start_idx, ema12_prev, ema26_prev, dea_prev):
    """
    从指定位置开始增量计算MACD

    Returns
    -------
    dif : array, shape (n_new,)
    dea : array, shape (n_new,)
    macd : array, shape (n_new,)
    ema12 : float (最终值)
    ema26 : float (最终值)
    dea : float (最终值)
    """
    n_new = len(close_all) - start_idx
    dif = np.zeros(n_new)
    dea_arr = np.zeros(n_new)
    macd = np.zeros(n_new)

    # 初始状态
    prev_ema12 = ema12_prev
    prev_ema26 = ema26_prev
    prev_dea = dea_prev

    for i in range(n_new):
        c = close_all[start_idx + i]
        # 更新 EMA
        ema12 = prev_ema12 * 11/13 + c * 2/13
        ema26 = prev_ema26 * 25/27 + c * 2/27
        d = ema12 - ema26
        d_ea = prev_dea * 8/10 + d * 2/10

        dif[i] = d
        dea_arr[i] = d_ea
        macd[i] = 2 * (d - d_ea)

        # 更新状态供下一次迭代
        prev_ema12 = ema12
        prev_ema26 = ema26
        prev_dea = d_ea

    # 最终状态（最后一次迭代的值）
    final_ema12 = ema12
    final_ema26 = ema26
    final_dea = d_ea

    return dif, dea_arr, macd, final_ema12, final_ema26, final_dea

# ================= 单只股票增量更新 =================
def update_one(code, period, snap, data_dir, snap_dir, now):
    """
    单只股票单周期增量更新

    Returns
    -------
    (merged_df, new_snap) 或 None（无新数据）
    """
    # ---- 确定拉取起始时间 ----
    last_dt = snap['last_date']  # datetime 对象

    if period in ["day", "week"]:
        # 日/周：日期 +1 天
        start_dt = last_dt + timedelta(days=1)
    else:
        # 分钟：日期时间 + 对应周期
        minute_map = {"1m": 1, "5m": 5, "30m": 30}
        start_dt = last_dt + timedelta(minutes=minute_map[period])

    if start_dt > now:
        return None

    # ---- 拉取新数据 ----
    new_df = fetch_bs_k_data(code, period, start_dt.strftime('%Y-%m-%d'), now.strftime('%Y-%m-%d'))
    if new_df is None or len(new_df) == 0:
        return None

    # ---- 读取历史数据 ----
    data_file = _data_file(data_dir, period, code)
    hist_df = pd.read_feather(data_file)
    hist_df['date'] = pd.to_datetime(hist_df['date'])

    # ---- 合并去重 ----
    merged = pd.concat([hist_df, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset='date', keep='last')
    merged = merged.sort_values('date').reset_index(drop=True)

    # ---- 增量计算 ----
    close_all = merged['close'].values.astype(np.float64)
    tail_len = len(hist_df)
    data_len = len(close_all)

    # ---- 解析快照中的可用MA列表 ----
    prev_avail_str = snap.get('available_mas', '')
    prev_avail = set(int(x) for x in prev_avail_str.split(',') if x) if prev_avail_str else set()
    curr_avail = set(_available_mas(data_len))

    # ---- 1. 之前可用的MA → 增量计算 ----
    for p in sorted(prev_avail):
        ma_prev = snap.get(f'ma_{p}_last')
        if ma_prev is None:
            continue
        ma_new = calc_ma_from_state(close_all, tail_len, ma_prev, p)
        merged.loc[tail_len:, f'ma_{p}'] = ma_new

    # ---- 2. 新变可用的MA → 全量补算 ----
    newly_avail = sorted(curr_avail - prev_avail)
    for p in newly_avail:
        # 用 calc_ma_full 在全量数据上算，覆盖所有行
        # 之前的行（<p）是 NaN，之后的行有值
        merged[f'ma_{p}'] = calc_ma_full(close_all, p)

    # ---- 3. 不可用的MA → 确保列存在且为 NaN ----
    for p in MA_PERIODS:
        if p not in curr_avail:
            if f'ma_{p}' not in merged.columns:
                merged[f'ma_{p}'] = np.nan

    # ---- 4. MACD → 增量计算 ----
    if data_len >= 12:
        dif_new, dea_new, macd_new, ema12_f, ema26_f, dea_f = \
            calc_macd_from_state(
                close_all, tail_len,
                snap['ema12_last'], snap['ema26_last'], snap['dea_last']
            )
        merged.loc[tail_len:, 'dif'] = dif_new
        merged.loc[tail_len:, 'dea'] = dea_new
        merged.loc[tail_len:, 'macd'] = macd_new

    # ---- 5. Level → 直接算（不依赖快照）----
    if curr_avail:
        ma_mat = np.column_stack([merged[f'ma_{p}'].values for p in sorted(curr_avail)])
        merged.loc[tail_len:, 'level'] = calc_level(close_all[tail_len:], ma_mat)
    

    # ---- 6. 构建新快照 ----
    last = merged.iloc[-1]
    close_tail = close_all[-MAX_MA:] if data_len >= MAX_MA else close_all

    ns = {
        'code': code,
        'last_date': last['date'],
        'last_close': float(last['close']),
        'last_level': int(last['level']),
        'ema12_last': float(ema12_f),
        'ema26_last': float(ema26_f),
        'dea_last': float(dea_f),
        'close_tail': ','.join(f'{x:.4f}' for x in close_tail),
        'available_mas': ','.join(str(p) for p in sorted(curr_avail)),
    }
    for p in MA_PERIODS:
        ns[f'ma_{p}_last'] = float(last[f'ma_{p}']) if p in curr_avail else 0.0

    return merged, ns



def incremental_update(kind="stock", periods=None):
    if periods is None:
        periods = UPDATE_ORDER

    data_dir = STOCK_DATA_PATH
    snap_dir = STOCK_SNAPSHOT

    info_df = pd.read_feather(STOCK_INFO_PATH)
    code_name_map = dict(zip(info_df['code'], info_df['code_name']))
    total_stocks = len(code_name_map)

    now = datetime.now()

    for period in periods:
        if not _bs_login():
            continue

        try:
            snap_dict = load_snapshot_dict(period, kind)
            print(f"\n{'='*50}")
            print(f"📊 [{kind}/{period}] 快照: {len(snap_dict)} 只")

            updated = 0
            skipped = 0
            errors = 0
            new_snap_dict = {}

            # 使用 tqdm 创建进度条，初始描述为空
            with tqdm(total=total_stocks, desc=f"{kind}/{period}", unit="股", ncols=80) as pbar:
                for code, name in code_name_map.items():
                    snap = snap_dict.get(code)
                    data_file = _data_file(data_dir, period, code)

                    # 无快照或无数据文件 → 跳过
                    if snap is None or not os.path.exists(data_file):
                        skipped += 1
                        if snap:
                            new_snap_dict[code] = snap
                        # 更新进度条描述，显示当前状态
                        pbar.set_postfix_str(f"更新:{updated} 跳过:{skipped} 错误:{errors}")
                        pbar.update(1)
                        continue

                    try:
                        result = update_one(code, period, snap, data_dir, snap_dir, now)

                        if result is None:
                            new_snap_dict[code] = snap
                            skipped += 1
                        else:
                            merged, ns = result
                            merged.to_feather(data_file)
                            new_snap_dict[code] = ns
                            updated += 1

                        # 更新进度条描述
                        pbar.set_postfix_str(f"更新:{updated} 跳过:{skipped} 错误:{errors}")

                    except Exception as e:
                        errors += 1
                        # 使用 tqdm.write 打印错误，避免破坏进度条
                        tqdm.write(f"⚠️ {code}.{period}: {e}")
                        if code not in new_snap_dict:
                            new_snap_dict[code] = snap
                        pbar.set_postfix_str(f"更新:{updated} 跳过:{skipped} 错误:{errors}")

                    pbar.update(1)

            # 保存快照
            save_snapshot_dict(new_snap_dict, snap_dir, period)
            print(f"✅ [{kind}/{period}] 完成: 更新 {updated} | 跳过 {skipped} | 失败 {errors}")

        finally:
            _bs_logout()


# ================= 全量计算 =================

def full_calc(raw_df):
    """
    全量计算指标，有多少数据算多少均线

    Returns
    -------
    df : DataFrame（含所有MA列，不可算的为NaN）
    snap : dict（快照）
    """
    df = raw_df.copy()
    close = df['close'].values.astype(np.float64)
    data_len = len(close)
    avail = _available_mas(data_len)

    # ---- MA：逐条计算，不可算的保持 NaN ----
    for p in MA_PERIODS:
        df[f'ma_{p}'] = calc_ma_full(close, p)
        # calc_ma_full 内部已经处理了 n < period 的情况，返回全 NaN

    # ---- MACD：EMA12 需要至少 12 条 ----
    if data_len >= 12:
        dif, dea, macd, ema12, ema26 = calc_macd_full(close)
    else:
        dif = np.zeros(data_len)
        dea = np.zeros(data_len)
        macd = np.zeros(data_len)
        ema12 = np.zeros(data_len)
        ema26 = np.zeros(data_len)
    df['dif'] = dif
    df['dea'] = dea
    df['macd'] = macd

    # ---- Level：只对可用MA求和 ----
    if avail:
        ma_matrix = np.column_stack([df[f'ma_{p}'].values for p in avail])
        df['level'] = calc_level(close, ma_matrix)
    else:
        df['level'] = np.ones(data_len, dtype=np.int32)

    # ---- 快照 ----
    last = df.iloc[-1]
    close_tail = close[-MAX_MA:] if data_len >= MAX_MA else close

    snap = {
        'code': raw_df['code'].iloc[0] if 'code' in raw_df.columns else '',
        'last_date': last['date'],
        'last_close': float(last['close']),
        'last_level': int(last['level']),
        'ema12_last': float(ema12[-1]),
        'ema26_last': float(ema26[-1]),
        'dea_last': float(dea[-1]),
        'close_tail': ','.join(f'{x:.4f}' for x in close_tail),
        'available_mas': ','.join(str(p) for p in avail),
    }
    for p in MA_PERIODS:
        snap[f'ma_{p}_last'] = float(last[f'ma_{p}']) if p in avail else 0.0

    return df, snap

def _save_data(df, data_dir, period, code):
    df.to_feather(_data_file(data_dir, period, code))

# ================= 进度管理 =================

def _load_progress(progress_file: str) -> Dict[str, set]:
    """加载进度文件，返回 {period: set(codes)}"""
    if not os.path.exists(progress_file):
        return {}
    with open(progress_file, 'r') as f:
        data = json.load(f)
    # 将列表转换为 set
    return {period: set(codes) for period, codes in data.items()}

def _save_progress(progress_file: str, progress: Dict[str, set]):
    """保存进度文件"""
    # 将 set 转换为 list 以便 JSON 序列化
    data = {period: list(codes) for period, codes in progress.items()}
    with open(progress_file, 'w') as f:
        json.dump(data, f)


# ================= 全量初始化 =================

def full_init(kind="stock", periods=None):
    """
    全量初始化

    - K线不足233的也保留，能算几根算几根
    - 已有数据文件的股票跳过
    - 断点续传：进度文件 + 数据文件双重判断
    """
    if periods is None:
        periods = INIT_ORDER

    if kind == "stock":
        data_dir = STOCK_DATA_PATH
        snap_dir = STOCK_SNAPSHOT
        info_path = STOCK_INFO_PATH
    else:
        data_dir = INDEX_DATA_PATH
        snap_dir = INDEX_SNAPSHOT
        info_path = INDEX_INFO_PATH

    progress_file = f"data/.init_progress_{kind}.json"

    # ---- 加载基础信息 ----
    info_df = pd.read_feather(info_path)
    all_codes = info_df['code'].tolist()
    print(f"📊 共 {len(all_codes)} 只 {kind}")

    # ---- 加载进度 ----
    progress = _load_progress(progress_file)

    now_str = datetime.now().strftime('%Y-%m-%d')

    # ---- 按周期逐个初始化 ----
    for period in periods:
        # ---- 双重判断：进度文件 + 数据文件 ----
        done_from_progress = progress.get(period, set())
        done_from_files = {c for c in all_codes if _file_exists(data_dir, period, c)}
        done = done_from_progress | done_from_files

        remaining = [c for c in all_codes if c not in done]

        # 同步进度文件
        progress[period] = done
        _save_progress(progress_file, progress)

        if not remaining:
            print(f"\n⏭️ [{kind}/{period}] 全部已完成 ({len(done)})")
            continue

        print(f"\n{'='*60}")
        print(f"🚀 [{kind}/{period}] 已完成 {len(done)}/{len(all_codes)}，"
              f"剩余 {len(remaining)}")

        # baostock 登录
        if not _bs_login():
            print(f"❌ [{kind}/{period}] baostock 登录失败，跳过")
            continue

        try:
            # 确定日期范围
            years = DAY_YEARS if period == "day" else WEEK_YEARS
            start_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
            if period in ["5m", "30m"]:
                start_date = (datetime.now() - timedelta(days=30 * MINUTE_MONTHS)).strftime('%Y-%m-%d')
            end_date = now_str

            # ---- 使用 tqdm 创建进度条 ----
            total_remaining = len(remaining)
            with tqdm(total=total_remaining, desc=f"{kind}/{period}", unit="股", ncols=80) as pbar:
                # 分批处理，但进度条按股票更新
                for i, code in enumerate(remaining):
                    try:
                        # 拉取原始数据
                        raw_df = fetch_bs_k_data(code, period, start_date, end_date)
                        if raw_df is None or len(raw_df) == 0:
                            tqdm.write(f"⚠️ {code} {period} 无数据")
                            pbar.update(1)
                            continue

                        # 全量计算 + 保存
                        df, snap = full_calc(raw_df)
                        _save_data(df, data_dir, period, code)

                        # 每完成一只股票，就追加快照到该周期的快照文件
                        # 这里原本是批次最后批量追加，但为了进度条实时性，可以每只都追加
                        # 注意：频繁追加可能影响性能，但为简单起见，这里使用每只追加
                        if snap is not None:
                            append_snapshots([snap], snap_dir, period)

                        # 更新进度文件（每只股票完成后更新进度，确保断点续传）
                        progress.setdefault(period, set()).add(code)
                        _save_progress(progress_file, progress)

                        # 更新进度条
                        pbar.update(1)

                    except Exception as e:
                        tqdm.write(f"❌ {code} 失败: {e}")
                        pbar.update(1)

                    # 限速, 可根据实际情况调整
                    time.sleep(0.5)   

            # 打印周期完成信息
            print(f"✅ [{kind}/{period}] 初始化完成")

        finally:
            _bs_logout()

    print(f"\n🎉 全量初始化完成！")


# ================= 快照工具 =================

def load_snapshot_dict(snap_dir, period):
    path = f"{snap_dir}/{period}_snapshot.feather"
    try:
        df = pd.read_feather(path)
        if 'last_date' in df.columns:
            df['last_date'] = pd.to_datetime(df['last_date'])
        return df.set_index('code').to_dict('index')
    except Exception:
        return {}

def save_snapshot_dict(snap_dict, snap_dir, period):
    if not snap_dict:
        return
    df = pd.DataFrame.from_dict(snap_dict, orient='index').reset_index(drop=False)
    df.to_feather(f"{snap_dir}/{period}_snapshot.feather")

def append_snapshots(new_snaps, snap_dir, period, time_col='last_date'):
    """
    追加快照到周期快照文件，按时间列去重，原子写入。
    """
    snap_file = Path(f"{snap_dir}/{period}_snapshot.feather")
    temp_file = snap_file.with_suffix('.tmp')

    # 读取现有快照（如果存在）
    try:
        existing = pd.read_feather(snap_file)
    except FileNotFoundError:
        existing = pd.DataFrame()
    except Exception as e:
        print(f"读取快照文件失败 {snap_file}: {e}")
        raise

    new_df = pd.DataFrame(new_snaps)

    if existing.empty:
        combined = new_df
    else:
        combined = pd.concat([existing, new_df], ignore_index=True)
        # 假设有时间列，先按 code 和时间排序，再保留每个 code 的最新记录
        if time_col in combined.columns:
            combined = combined.sort_values(['code', time_col])
            combined = combined.drop_duplicates(subset='code', keep='last')
        else:
            # 没有时间列时，保留最后出现的行（不可靠，建议添加时间列）
            combined = combined.drop_duplicates(subset='code', keep='last')

    # 原子写入
    combined.to_feather(temp_file)
    os.replace(temp_file, snap_file)

# ================= 重建快照 =================

def rebuild_snapshots(kind: str = "stock", periods: List[str] = None, clean: bool = True):
    """
    从本地已有的数据文件重建快照文件。

    参数：
        kind: 'stock' 或 'index'
        periods: 要重建的周期列表，默认使用 INIT_ORDER
        clean: 是否在重建前删除原有快照文件并重置进度（默认 True）
    """
    if periods is None:
        periods = INIT_ORDER

    if kind == "stock":
        data_dir = STOCK_DATA_PATH
        snap_dir = STOCK_SNAPSHOT
    else:
        data_dir = INDEX_DATA_PATH
        snap_dir = INDEX_SNAPSHOT

    # 进度文件，用于断点续传
    progress_file = f"data/.rebuild_snapshot_progress_{kind}.json"
    progress = _load_progress(progress_file)

    print(f"📊 开始重建 {kind} 快照")

    for period in periods:
        # ---------- 清除原有快照 ----------
        snap_file = Path(f"{snap_dir}/{period}_snapshot.feather")
        if clean and snap_file.exists():
            try:
                snap_file.unlink()
                print(f"🗑️ 已删除旧快照文件: {snap_file}")
            except Exception as e:
                print(f"⚠️ 删除旧快照文件失败: {e}")

            # 重置该周期进度
            if period in progress:
                del progress[period]
            _save_progress(progress_file, progress)

        # 获取该周期所有有数据文件的股票代码
        all_codes = list_codes(period, kind)
        if not all_codes:
            print(f"⏭️ [{kind}/{period}] 无数据文件")
            continue

        done = progress.get(period, set())
        remaining = [c for c in all_codes if c not in done]

        if not remaining:
            print(f"⏭️ [{kind}/{period}] 快照已全部重建 ({len(done)}/{len(all_codes)})")
            continue

        print(f"\n{'='*60}")
        print(f"🔄 [{kind}/{period}] 重建快照: 总 {len(all_codes)}，已完成 {len(done)}，剩余 {len(remaining)}")

        rebuilded = 0
        skipped = 0
        errors = 0
        # 使用 tqdm 进度条
        with tqdm(total=len(remaining), desc=f"{kind}/{period}", unit="股", ncols=80) as pbar:
            for code in remaining:
                try:
                    file_path = Path(f"{data_dir}/{period}/{code}.feather")
                    if not file_path.exists():
                        tqdm.write(f"⚠️ {code} 数据文件不存在，跳过")
                        pbar.set_postfix_str(f"更新:{rebuilded} 跳过:{skipped} 错误:{errors}")
                        pbar.update(1)
                        skipped += 1
                        continue

                    df = pd.read_feather(file_path)
                    if df.empty:
                        tqdm.write(f"⚠️ {code} 数据为空，跳过")
                        pbar.set_postfix_str(f"更新:{rebuilded} 跳过:{skipped} 错误:{errors}")
                        pbar.update(1)
                        skipped += 1
                        continue

                    _, snap = full_calc(df)
                    # 追加快照
                    append_snapshots([snap], snap_dir, period)

                    # 更新进度
                    progress.setdefault(period, set()).add(code)
                    _save_progress(progress_file, progress)
                    rebuilded += 1
                    pbar.set_postfix_str(f"更新:{rebuilded} 跳过:{skipped} 错误:{errors}")

                except Exception as e:
                    tqdm.write(f"❌ {code} 重建快照失败: {e}")
                    errors += 1
                    pbar.set_postfix_str(f"更新:{rebuilded} 跳过:{skipped} 错误:{errors}")

                pbar.update(1)
                time.sleep(0.1)  # 适度限速，避免 I/O 过载

        print(f"✅ [{kind}/{period}] 快照重建完成")

    print(f"\n🎉 所有快照重建完成！")


# ================= 基础信息读取 =================

def load_stock_info() -> pd.DataFrame:
    """读取股票基础信息"""
    if not os.path.exists(STOCK_INFO_PATH):
        # 1. 获取证券基础信息
        all_basic = get_stock_basic()
        # 2. 只保留未退市的
        list_all = all_basic[all_basic['status'] == 1]
        # 3. 保存股票基础信息
        stock_all = list_all[list_all['type'] == 1]
        stock_all.to_feather(STOCK_INFO_PATH)
        return stock_all
    return pd.read_feather(STOCK_INFO_PATH)


def load_index_info() -> pd.DataFrame:
    """读取指数基础信息"""
    if not os.path.exists(INDEX_INFO_PATH):
        # 1. 获取证券基础信息
        all_basic = get_stock_basic()
        # 2. 只保留未退市的
        list_all = all_basic[all_basic['status'] == 1]
        # 3. 保存股票基础信息
        index_all = list_all[list_all['type'] == 2]
        index_all.to_feather(INDEX_INFO_PATH)
        return index_all
    return pd.read_feather(INDEX_INFO_PATH)


# ================= K线+指标数据读取 =================

def _resolve_path(base: str, code: str, period: str) -> str:
    """
    根据代码和周期定位文件路径
    自动匹配 sh./sz./bj. 前缀或纯数字+后缀格式
    """
    # 如果传入的是 baostock 格式 "sh.600000"，提取纯数字
    if '.' in code:
        code_num = code.split('.')[-1]
    else:
        code_num = code

    # 优先精确匹配，再模糊匹配
    exact = f"{base}/{period}/{code_num}.*.feather"
    matches = glob.glob(exact)
    if matches:
        return matches[0]

    # 兼容 baostock 前缀格式的文件名
    prefix_match = f"{base}/{period}/{code}.feather"
    if os.path.exists(prefix_match):
        return prefix_match

    # 兜底：文件名包含 code_num 即可
    broad = glob.glob(f"{base}/{period}/*{code_num}*.feather")
    if broad:
        return broad[0]

    return f"{base}/{period}/{code_num}.SH.feather"  # 返回默认路径（可能不存在）


def load_stock_data(code: str, period: str = "day") -> Optional[pd.DataFrame]:
    """
    读取单只股票的K线+指标数据

    Parameters
    ----------
    code : str
        股票代码，支持 "600000" / "sh.600000" 两种格式
    period : str
        "1m", "5m", "30m", "day", "week", "month"

    Returns
    -------
    DataFrame 或 None
        列: date, open, high, low, close, volume, amount,
            ma_5 ... ma_144, dif, dea, macd, level
    """
    path = _resolve_path(STOCK_DATA_PATH, code, period)
    if not os.path.exists(path):
        return None

    df = pd.read_feather(path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    return df


def load_index_data(code: str, period: str = "day") -> Optional[pd.DataFrame]:
    """
    读取单个指数的K线+指标数据

    Parameters
    ----------
    code : str
        指数代码，如 "000001"(上证指数) / "sh.000001"
    period : str
        同上
    """
    path = _resolve_path(INDEX_DATA_PATH, code, period)
    if not os.path.exists(path):
        return None

    df = pd.read_feather(path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    return df


def list_codes(period: str = "day", kind: str = "stock") -> List[str]:
    """列出本地已有数据的代码列表"""
    dir_path = Path(f"{STOCK_DATA_PATH}/{period}")
    if kind == "index": 
        dir_path = f"{INDEX_DATA_PATH}/{period}"
    if not dir_path.is_dir():
        return []
    # 获取所有 .feather 文件，提取文件名（不含扩展名）
    codes = [f.stem for f in dir_path.glob("*.feather")]
    return sorted(set(codes))

# ================= 批量读取 =================

def load_all_stock_data(
    period: str = "day",
    codes: Optional[List[str]] = None,
    columns: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    批量读取多只股票数据(慎用)

    Parameters
    ----------
    period : str
        周期
    codes : list[str] 或 None（内存占用极大，慎用）
        指定代码列表；None 则读取本地全部
    columns : list[str] 或 None
        只取指定列，减少内存占用（如 ['date','close','level']）

    Returns
    -------
    dict: {code: DataFrame}
    """
    if codes is None:
        codes = list_codes(period)

    result = {}
    for code in codes:
        df = load_stock_data(code, period)
        if df is not None and not df.empty:
            if columns:
                available = [c for c in columns if c in df.columns]
                df = df[available]
            result[code] = df
    return result


def load_all_index_data(
    period: str = "day",
    codes: Optional[List[str]] = None,
    columns: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """批量读取多个指数数据，用法同上"""
    if codes is None:
        codes = list_codes(period, "index")

    result = {}
    for code in codes:
        df = load_index_data(code, period)
        if df is not None and not df.empty:
            if columns:
                available = [c for c in columns if c in df.columns]
                df = df[available]
            result[code] = df
    return result


# ================= 快照读取 =================

def load_stock_snapshot(period: str = "day") -> Optional[pd.DataFrame]:
    """
    读取股票快照

    Returns
    -------
    DataFrame 或 None
        列: code, last_date, last_close, last_level, 
            ma_5_last, ma_5_tail, ..., ema12_last, ema26_last, dea_last
    """
    snap_path = Path(f"{STOCK_SNAPSHOT}/{period}_snapshot.feather")
    if not snap_path.exists():
        return None
    df = pd.read_feather(snap_path)
    if 'last_date' in df.columns:
        df['last_date'] = pd.to_datetime(df['last_date'])
    return df


def load_index_snapshot(period: str = "day") -> Optional[pd.DataFrame]:
    """读取指数快照"""
    snap_path = Path(f"{INDEX_SNAPSHOT}/{period}_snapshot.feather")
    if not snap_path.exists():
        return None
    df = pd.read_feather(snap_path)
    if 'last_date' in df.columns:
        df['last_date'] = pd.to_datetime(df['last_date'])
    return df


def load_snapshot_dict(period: str = "day", kind: str = "stock") -> Dict:
    """
    快照转为 dict[code] = snapshot_row，方便增量更新时快速查找
    """
    if kind == "stock":
        df = load_stock_snapshot(period)
    else:
        df = load_index_snapshot(period)

    if df is None or df.empty:
        return {}
    if 'code' in df.columns:
        df.reset_index(drop=True, inplace=True)
    else:
        df = df.reset_index(names=['code'])
    df_sorted = df.sort_values(['code', 'last_date'], ascending=[True, False])
    df_sorted.drop_duplicates(subset='code', keep='first', inplace=True)
    return df_sorted.set_index('code').to_dict(orient='index')


# ================= 缠论结构读取 =================

def load_stock_structure(code: str, period: str = "day") -> Optional[pd.DataFrame]:
    """
    读取单只股票的缠论分析结果
    """
    path = _resolve_path(STOCK_STRUCTURE_PATH, code, period)
    if not os.path.exists(path):
        return None

    df = pd.read_feather(path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    return df


def load_index_structure(code: str, period: str = "day") -> Optional[pd.DataFrame]:
    """读取单个指数的缠论分析结果"""
    path = _resolve_path(INDEX_STRUCTURE_PATH, code, period)
    if not os.path.exists(path):
        return None

    df = pd.read_feather(path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    return df


# ================= 按日期切片（高频场景）=================

def load_stock_slice(
    code: str,
    period: str = "day",
    start: Optional[str] = None,
    end: Optional[str] = None,
    last_n: Optional[int] = None
) -> Optional[pd.DataFrame]:
    """
    读取股票数据的日期切片

    Parameters
    ----------
    code : str
    period : str
    start : str, "2024-01-01"
    end : str, "2024-12-31"
    last_n : int, 只取最近N行（与 start/end 互斥）

    Examples
    --------
    load_stock_slice("600000", "day", last_n=144)
    load_stock_slice("600000", "day", start="2024-06-01", end="2024-12-31")
    """
    df = load_stock_data(code, period)
    if df is None:
        return None

    if last_n is not None:
        return df.tail(last_n).reset_index(drop=True)

    if start:
        df = df[df['date'] >= pd.to_datetime(start)]
    if end:
        df = df[df['date'] <= pd.to_datetime(end)]

    return df.reset_index(drop=True)


# ================= 实用查询 =================

def get_latest_date(period: str = "day", kind: str = "stock") -> Optional[str]:
    """获取本地数据的最新日期"""
    snap = load_stock_snapshot(period) if kind == "stock" else load_index_snapshot(period)
    if snap is None or snap.empty:
        return None
    return str(snap['last_date'].max().date())


def get_data_status(period: str = "day") -> pd.DataFrame:
    """
    检查本地数据完整性：各代码的数据量、最新日期、是否缺失指标
    """
    codes = list_stock_codes(period)
    records = []

    for code in codes:
        df = load_stock_data(code, period)
        if df is None:
            continue

        has_ma = all(f'ma_{p}' in df.columns for p in MA_PERIODS)
        has_macd = all(c in df.columns for c in ['dif', 'dea', 'macd'])

        records.append({
            'code': code,
            'rows': len(df),
            'first_date': str(df['date'].min().date()),
            'last_date': str(df['date'].max().date()),
            'has_ma': has_ma,
            'has_macd': has_macd,
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    # snap = load_snapshot_dict(period='5m')
    # print(snap.keys())
    snap = pd.read_feather(f"data/snapshot/stock/5m_snapshot.feather")
    print(snap.head())
    # all_5m_data = load_all_stock_data(period='day', codes=["sz.302132"])
    # print(all_5m_data["sz.302132"].columns)
    # rebuild_snapshots("stock", ["week", "day", "30m"])
    
