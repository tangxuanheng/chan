# 将项目根目录添加到sys.path
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tool import incremental_update
from config import init_dirs

if __name__ == "__main__":
    init_dirs()

    # 全部更新：day → week → 30m → 5m → 1m
    incremental_update(kind="stock", periods=["week", "day", "30m", "5m"])

    # 只更新某个周期
    # incremental_update(kind="stock", periods=["day"])