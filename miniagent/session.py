"""Session 管理：窗口级隔离 + 会话存储。

隔离模型（对应交付要求第 3 节）::

    session_id = f"{user_id}::{window_id}"

    用户A ─┬─ 窗口1 ── session_id="userA::win1" ── messages=[...]
           └─ 窗口2 ── session_id="userA::win2" ── messages=[...]
    用户B ─── 窗口1 ── session_id="userB::win1" ── messages=[...]

- 同一用户在**不同窗口**的对话上下文完全独立，互不干扰；
- 存储默认是内存字典（demo），通过 `SessionStore` 抽象可换成 Redis / SQLite / 文件；
- **system prompt 不落库**：每次请求时由 ContextManager 临时拼接，避免历史里堆叠重复指令。
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

from .utils import make_session_id, new_id, now_iso, now_ms, safe_json, split_session_id

#: 上下文消息角色（交付要求第 4 节）
ROLES = ("user", "assistant", "tool_call", "tool_result", "error")


@dataclass
class Message:
    """一条上下文消息。"""

    role: str
    content: str
    ts: int = field(default_factory=now_ms)
    turn: int = 0
    name: str = ""              # 工具名 / 来源标记
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"非法 role: {self.role}，允许 {ROLES}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ts_iso"] = now_iso()
        return data

    def compact(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content, "name": self.name, "turn": self.turn}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(
            role=str(data.get("role", "user")),
            content=str(data.get("content", "")),
            ts=int(data.get("ts") or now_ms()),
            turn=int(data.get("turn") or 0),
            name=str(data.get("name") or ""),
            meta=dict(data.get("meta") or {}),
        )


class Session:
    """一个窗口的完整上下文。"""

    def __init__(
        self,
        session_id: str,
        *,
        max_turn: int = 10,
        max_context_tokens: int = 3000,
        keep_recent_messages: int = 12,
        created_at: Optional[int] = None,
        max_turn_override: Optional[int] = None,
    ) -> None:
        self.session_id = session_id
        self.user_id, self.window_id = split_session_id(session_id)
        self.messages: list[Message] = []
        self.max_turn = max_turn
        #: 会话级「显式覆盖」的轮次上限（用户主动设过才非空）；
        #: Agent 用它区分「配置里的默认值」与「针对这个会话的定制值」。
        self.max_turn_override = max_turn_override
        self.max_context_tokens = max_context_tokens
        self.keep_recent_messages = keep_recent_messages
        self.created_at = created_at or now_ms()
        self.updated_at = self.created_at
        self.turn_count = 0                # 已完成的用户提问轮数
        self.compress_count = 0            # 触发上下文裁剪的次数
        self.dropped_messages = 0          # 累计被裁剪的消息数
        self.meta: dict[str, Any] = {}
        self._msgs_since_compress = 0

    # ------------------------------------------------------------ 写入
    def append(
        self,
        role: str,
        content: str,
        *,
        name: str = "",
        turn: int = 0,
        meta: Optional[dict[str, Any]] = None,
    ) -> Message:
        msg = Message(role=role, content=content if isinstance(content, str) else safe_json(content), name=name, turn=turn, meta=meta or {})
        self.messages.append(msg)
        self.updated_at = now_ms()
        return msg

    # 语义化快捷方法
    def add_user(self, content: str, **kw: Any) -> Message:
        self.turn_count += 1
        return self.append("user", content, **kw)

    def add_assistant(self, content: str, **kw: Any) -> Message:
        return self.append("assistant", content, **kw)

    def add_tool_call(self, name: str, arguments: Any, **kw: Any) -> Message:
        return self.append("tool_call", safe_json(arguments), name=name, **kw)

    def add_tool_result(self, name: str, result: str, **kw: Any) -> Message:
        return self.append("tool_result", result, name=name, **kw)

    def add_error(self, text: str, *, name: str = "", **kw: Any) -> Message:
        return self.append("error", text, name=name, **kw)

    # ------------------------------------------------------------ 读取
    @property
    def size(self) -> int:
        return len(self.messages)

    def last_user_message(self) -> Optional[Message]:
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg
        return None

    def tail(self, n: int) -> list[Message]:
        return self.messages[-n:] if n > 0 else []

    def history_for_llm(self) -> list[Message]:
        """喂给 LLM 的消息（不含 system prompt，也不含最终被渲染的 tool_* 之外的东西）。"""
        return list(self.messages)

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "window_id": self.window_id,
            "messages": self.size,
            "turns": self.turn_count,
            "max_turn": self.max_turn,
            "compress_count": self.compress_count,
            "dropped_messages": self.dropped_messages,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "roles": {role: sum(1 for m in self.messages if m.role == role) for role in ROLES},
        }

    def transcript(self, *, limit: Optional[int] = None) -> str:
        """人可读的会话记录（CLI `/history`、测试断言用）。"""
        msgs = self.messages[-limit:] if limit else self.messages
        lines = []
        for idx, msg in enumerate(msgs, 1):
            name = f"({msg.name})" if msg.name else ""
            lines.append(f"{idx:>3}. [{msg.role}{name}] {msg.content}")
        return "\n".join(lines) if lines else "（空会话）"

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "window_id": self.window_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turn_count": self.turn_count,
            "max_turn": self.max_turn,
            "max_context_tokens": self.max_context_tokens,
            "keep_recent_messages": self.keep_recent_messages,
            "compress_count": self.compress_count,
            "dropped_messages": self.dropped_messages,
            "max_turn_override": self.max_turn_override,
            "meta": self.meta,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        session = cls(
            str(data["session_id"]),
            max_turn=int(data.get("max_turn", 10)),
            max_context_tokens=int(data.get("max_context_tokens", 3000)),
            keep_recent_messages=int(data.get("keep_recent_messages", 12)),
            created_at=int(data.get("created_at") or now_ms()),
            max_turn_override=data.get("max_turn_override"),
        )
        session.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        session.turn_count = int(data.get("turn_count", 0))
        session.compress_count = int(data.get("compress_count", 0))
        session.dropped_messages = int(data.get("dropped_messages", 0))
        session.meta = dict(data.get("meta") or {})
        session.updated_at = int(data.get("updated_at") or session.created_at)
        return session


# ---------------------------------------------------------------------------
# 存储后端
# ---------------------------------------------------------------------------

class SessionStore:
    """会话存储接口（生产可换 Redis）。"""

    def load(self, session_id: str) -> Optional[dict[str, Any]]:
        raise NotImplementedError

    def save(self, session: Session) -> None:
        raise NotImplementedError

    def delete(self, session_id: str) -> None:
        raise NotImplementedError

    def list_ids(self) -> list[str]:
        raise NotImplementedError


class MemorySessionStore(SessionStore):
    """内存字典（demo 默认）。"""

    def __init__(self) -> None:
        self.data: dict[str, dict[str, Any]] = {}

    def load(self, session_id: str) -> Optional[dict[str, Any]]:
        return self.data.get(session_id)

    def save(self, session: Session) -> None:
        self.data[session.session_id] = session.to_dict()

    def delete(self, session_id: str) -> None:
        self.data.pop(session_id, None)

    def list_ids(self) -> list[str]:
        return sorted(self.data)


class JsonFileSessionStore(SessionStore):
    """JSON 文件持久化（进程重启后会话不丢，适合本地调试/演示）。"""

    def __init__(self, path: str = "logs/sessions.json") -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = {}
        self._load_file()

    def _load_file(self) -> None:
        if os.path.isfile(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    self._data = json.load(fh) or {}
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def _flush(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)   # 原子替换，避免半个文件

    def load(self, session_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            return self._data.get(session_id)

    def save(self, session: Session) -> None:
        with self._lock:
            self._data[session.session_id] = session.to_dict()
            self._flush()

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)
            self._flush()

    def list_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._data)


# ---------------------------------------------------------------------------
# Session Manager
# ---------------------------------------------------------------------------

class SessionManager:
    """会话隔离与生命周期管理。"""

    def __init__(
        self,
        max_turn: int = 10,
        *,
        max_context_tokens: int = 3000,
        keep_recent_messages: int = 12,
        store: Optional[SessionStore] = None,
        persist: bool = False,
        persist_path: str = "logs/sessions.json",
    ) -> None:
        self.sessions: dict[str, Session] = {}
        self.max_turn = max_turn
        self.max_context_tokens = max_context_tokens
        self.keep_recent_messages = keep_recent_messages
        self.store = store or (JsonFileSessionStore(persist_path) if persist else MemorySessionStore())
        self._lock = threading.RLock()
        self._loaded: set[str] = set()

    # ------------------------------------------------------------ 生命周期
    def get_or_create(self, session_id: str) -> Session:
        """取会话；不存在则创建（若后端有持久化数据则先恢复）。"""
        with self._lock:
            session = self.sessions.get(session_id)
            if session is not None:
                return session
            raw = self.store.load(session_id)
            if raw and session_id not in self._loaded:
                session = Session.from_dict(raw)
                session.max_turn = session.max_turn_override or self.max_turn
                self.sessions[session_id] = session
                self._loaded.add(session_id)
                return session
            session = Session(
                session_id,
                max_turn=self.max_turn,
                max_context_tokens=self.max_context_tokens,
                keep_recent_messages=self.keep_recent_messages,
            )
            self.sessions[session_id] = session
            self._loaded.add(session_id)
            return session

    def for_window(self, user_id: str, window_id: str = "default") -> Session:
        """按 (user_id, window_id) 取会话 —— 窗口隔离的入口。"""
        return self.get_or_create(make_session_id(user_id, window_id))

    def save(self, session: Session) -> None:
        self.store.save(session)

    def save_all(self) -> None:
        with self._lock:
            for session in self.sessions.values():
                self.store.save(session)

    def reset(self, session_id: str) -> None:
        """清空某个窗口的上下文（保留会话对象）。"""
        with self._lock:
            session = self.sessions.get(session_id)
            if session:
                session.messages.clear()
                session.compress_count = 0
                session.dropped_messages = 0
                session.turn_count = 0
                self.store.save(session)

    def drop(self, session_id: str) -> bool:
        with self._lock:
            existed = self.sessions.pop(session_id, None) is not None
            self.store.delete(session_id)
            return existed

    # ------------------------------------------------------------ 查询
    def list_sessions(self) -> list[Session]:
        with self._lock:
            return [self.sessions[key] for key in sorted(self.sessions)]

    def stats(self) -> dict[str, Any]:
        sessions = self.list_sessions()
        return {
            "session_count": len(sessions),
            "total_messages": sum(s.size for s in sessions),
            "total_turns": sum(s.turn_count for s in sessions),
            "sessions": [s.snapshot() for s in sessions],
        }

    def overview(self) -> str:
        """按用户/窗口分组的可读概览（CLI `/sessions`）。"""
        sessions = self.list_sessions()
        if not sessions:
            return "（当前没有任何会话）"
        by_user: dict[str, list[Session]] = {}
        for session in sessions:
            by_user.setdefault(session.user_id, []).append(session)
        lines = []
        for user_id, items in by_user.items():
            lines.append(f"用户 {user_id}:")
            for session in items:
                lines.append(
                    f"  └─ 窗口 {session.window_id:<10} session_id={session.session_id:<28} "
                    f"消息={session.size:<3} 轮次={session.turn_count}/{session.max_turn} "
                    f"裁剪={session.compress_count}"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------ 便捷写入
    def append_msg(self, session_id: str, role: str, content: str, **kw: Any) -> Message:
        session = self.get_or_create(session_id)
        msg = session.append(role, content, **kw)
        self.store.save(session)
        return msg

    def append_user(self, session_id: str, content: str) -> Message:
        """开始一轮新的用户提问（同时把 turn_count 加一）。"""
        session = self.get_or_create(session_id)
        msg = session.add_user(content)
        self.store.save(session)
        return msg

    def append_tool_result(self, session_id: str, tool_name: str, result: str, *, turn: int = 0) -> Message:
        session = self.get_or_create(session_id)
        msg = session.add_tool_result(tool_name, result, turn=turn)
        self.store.save(session)
        return msg


__all__ = [
    "Message",
    "Session",
    "SessionManager",
    "SessionStore",
    "MemorySessionStore",
    "JsonFileSessionStore",
    "ROLES",
    "new_id",
]
