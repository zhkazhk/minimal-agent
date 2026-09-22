"""pytest 配置：把项目根目录加入 sys.path，使 `pytest` 无需安装即可运行。"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
