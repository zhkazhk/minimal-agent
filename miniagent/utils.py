"""零依赖工具函数：时间、id、token 估算、文本裁剪、JSON 安全序列化。"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import unicodedata
import uuid
from typing import Any, Iterable

_CJK_RANGES = (
    (0x4E00, 0x9FFF),   # CJK 统一表意文字
    (0x3400, 0x4DBF),   # 扩展 A
    (0x3000, 0x303F),   # CJK 标点
    (0xFF00, 0xFFEF),   # 全角字符
    (0x3040, 0x30FF),   # 日文假名
    (0xAC00, 0xD7AF),   # 谚文
)


def now_ms() -> int:
    return int(time.time() * 1000)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()) + f".{now_ms() % 1000:03d}"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def make_session_id(user_id: str, window_id: str) -> str:
    """Session Key = user_id + window_id（窗口级隔离）。"""
    return f"{user_id}::{window_id}"


def split_session_id(session_id: str) -> tuple[str, str]:
    user_id, _, window_id = session_id.partition("::")
    return user_id, window_id or "default"


def is_cjk(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def estimate_tokens(text: str) -> int:
    """启发式 token 估算（不引入 tiktoken）。

    经验规则：CJK 字符 ≈ 1 token/字（含标点），其余（英文/数字/符号）≈ 1 token/4 字符。
    对上下文裁剪来说，只需要**单调且量级正确**，不要求精确。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if is_cjk(ch))
    other = len(text) - cjk
    return int(cjk + math.ceil(other / 4))


def estimate_messages_tokens(messages: Iterable[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        total += 4  # 每条消息的角色/分隔开销
        total += estimate_tokens(str(msg.get("content", "")))
    return total


def truncate(text: Any, limit: int = 400, *, keep_tail: bool = False) -> str:
    """日志/错误信息用的安全截断，避免单条日志炸掉。"""
    s = text if isinstance(text, str) else safe_json(text)
    if len(s) <= limit:
        return s
    if keep_tail:
        return "…" + s[-limit:]
    return s[:limit] + f"…(截断, 共 {len(s)} 字符)"


def text_width(text: str) -> int:
    """终端显示宽度（中文算 2 列），用于画对齐的表格。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def pad(text: str, width: int) -> str:
    return text + " " * max(0, width - text_width(text))


def safe_json(obj: Any, *, max_depth: int = 6) -> str:
    """永不抛异常的 JSON 序列化（用于日志）。"""
    try:
        return json.dumps(_plain(obj, max_depth), ensure_ascii=False, default=str)
    except Exception:  # pragma: no cover - 兜底
        return repr(obj)


def _plain(obj: Any, depth: int = 6) -> Any:
    if depth <= 0:
        return str(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _plain(v, depth - 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_plain(v, depth - 1) for v in obj]
    if hasattr(obj, "model_dump"):
        return _plain(obj.model_dump(), depth - 1)
    if hasattr(obj, "to_dict"):
        return _plain(obj.to_dict(), depth - 1)
    if hasattr(obj, "__dict__"):
        return _plain(vars(obj), depth - 1)
    return str(obj)


_FENCE_RE = re.compile(r"```(?:json|JSON|javascript|js)?\s*(.*?)```", re.DOTALL)


def strip_code_fence(text: str) -> str:
    """去掉 markdown 代码块包裹，返回块内内容（若无代码块则原样返回）。"""
    if not text:
        return ""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def ensure_utf8_stdio() -> None:
    """Windows 控制台默认 GBK，中文日志会乱码/报错，这里强制 UTF-8。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


def load_dotenv(path: str, *, override: bool = False) -> dict[str, str]:
    """极简 .env 读取（不引入 python-dotenv）。"""
    loaded: dict[str, str] = {}
    if not os.path.isfile(path):
        return loaded
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            loaded[key] = value
            if override or key not in os.environ:
                os.environ[key] = value
    return loaded
