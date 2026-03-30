# 将项目根目录添加到sys.path
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import init_dirs
from utils.tool import full_init

if __name__ == "__main__":
    init_dirs()

    # 全量初始化：day → week → 30m → 5m → 1m
    full_init(kind="stock", periods=["5m"])

    # 只初始化某个周期
    # full_init(kind="stock", periods=["day"])