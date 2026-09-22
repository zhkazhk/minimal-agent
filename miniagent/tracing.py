"""Tracer：结构化 Trace 日志 + 人可读时间线 + 异常捕获。

产出两份文件（默认写在 `logs/`）：
1. `logs/trace.jsonl`   —— 机器可读，一行一个事件（append-only，方便 grep / 倒入 ELK）
2. `logs/traces/<run_id>.log` —— 人可读时间线，含每次 LLM 调用、工具入参出参、耗时

每个事件都带 `session_id` / `run_id` / `turn`，所以可以按窗口（session）或按一次提问（run）重建全过程。
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .utils import now_iso, now_ms, safe_json, truncate

LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}


@dataclass
class TraceEvent:
    """一条 trace 记录。"""

    event: str                      # run_start / llm_call / llm_response / tool_call / tool_result / parse_error / ...
    ts: str = field(default_factory=now_iso)
    ts_ms: int = field(default_factory=now_ms)
    level: str = "INFO"
    session_id: str = ""
    run_id: str = ""
    user_id: str = ""
    window_id: str = ""
    turn: int = 0
    duration_ms: Optional[int] = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


class Tracer:
    """线程安全的 trace 收集器。"""

    def __init__(
        self,
        log_dir: str = "logs",
        *,
        console: bool = True,
        raw_llm_dir: Optional[str] = None,
        max_field_chars: int = 1200,
        max_events: int = 20000,
    ) -> None:
        self.log_dir = log_dir
        self.trace_dir = os.path.join(log_dir, "traces")
        self.raw_dir = raw_llm_dir or os.path.join(log_dir, "raw_llm")
        self.jsonl_path = os.path.join(log_dir, "trace.jsonl")
        self.console = console
        self.max_field_chars = max_field_chars
        self._lock = threading.Lock()
        #: 只保留最近 N 条事件，避免长驻进程（服务端）内存无上限增长
        self._events: deque[TraceEvent] = deque(maxlen=max(1, max_events))
        self._run_files: dict[str, Any] = {}
        for path in (self.log_dir, self.trace_dir, self.raw_dir):
            os.makedirs(path, exist_ok=True)

    # ------------------------------------------------------------------ 记录

    def log(self, event: str, *, level: str = "INFO", **fields: Any) -> TraceEvent:
        known = {
            "session_id": fields.pop("session_id", ""),
            "run_id": fields.pop("run_id", ""),
            "user_id": fields.pop("user_id", ""),
            "window_id": fields.pop("window_id", ""),
            "turn": fields.pop("turn", 0) or 0,
            "duration_ms": fields.pop("duration_ms", None),
            "ts": fields.pop("ts", None),
        }
        ev = TraceEvent(
            event=event,
            level=level,
            data=self._trim(fields),
            ts=known["ts"] or now_iso(),
            session_id=str(known["session_id"] or ""),
            run_id=str(known["run_id"] or ""),
            user_id=str(known["user_id"] or ""),
            window_id=str(known["window_id"] or ""),
            turn=int(known["turn"] or 0),
            duration_ms=known["duration_ms"],
        )
        with self._lock:
            self._events.append(ev)
            self._append_jsonl(ev)
            self._append_run_file(ev)
        if self.console and LEVELS.get(level, 20) >= LEVELS["INFO"]:
            self._print(ev)
        return ev

    # 语义化快捷方法 -------------------------------------------------------
    def log_llm_request(self, messages: list[dict[str, Any]], **meta: Any) -> None:
        self.log(
            "llm_request",
            messages=self._preview_messages(messages),
            message_count=len(messages),
            **meta,
        )

    def log_llm_response(self, content: str, *, usage: Optional[dict[str, Any]] = None, **meta: Any) -> None:
        self.log("llm_response", raw_output=content, usage=usage or {}, **meta)

    def log_tool_call(self, name: str, arguments: Any, **meta: Any) -> None:
        self.log("tool_call", tool_name=name, arguments=arguments, **meta)

    def log_tool_result(self, name: str, result: Any, *, duration_ms: int, ok: bool = True, error: str = "", **meta: Any) -> None:
        self.log(
            "tool_result",
            level="INFO" if ok else "ERROR",
            tool_name=name,
            result=result,
            ok=ok,
            error=error,
            duration_ms=duration_ms,
            **meta,
        )

    def log_exception(self, exc: BaseException, *, where: str, **meta: Any) -> TraceEvent:
        """记录异常。**一定要带 traceback** —— 只有类型和 message 是没法排查的。"""
        import traceback as _traceback

        return self.log(
            "exception",
            level="ERROR",
            where=where,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback="".join(_traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
            **meta,
        )

    # ------------------------------------------------------------------ 输出

    def events(self, *, session_id: Optional[str] = None, run_id: Optional[str] = None) -> list[TraceEvent]:
        with self._lock:
            events = list(self._events)
        if session_id:
            events = [e for e in events if e.session_id == session_id]
        if run_id:
            events = [e for e in events if e.run_id == run_id]
        return events

    def tool_call_stats(self) -> dict[str, dict[str, Any]]:
        """按工具聚合调用次数 / 成功率 / 平均耗时（debug 面板用）。"""
        stats: dict[str, dict[str, Any]] = {}
        for ev in self.events():
            if ev.event != "tool_result":
                continue
            name = str(ev.data.get("tool_name", "?"))
            row = stats.setdefault(name, {"calls": 0, "ok": 0, "error": 0, "total_ms": 0})
            row["calls"] += 1
            row["ok" if ev.data.get("ok") else "error"] += 1
            row["total_ms"] += int(ev.duration_ms or 0)
        for row in stats.values():
            row["avg_ms"] = round(row["total_ms"] / row["calls"], 1) if row["calls"] else 0
        return stats

    def flush(self) -> None:
        with self._lock:
            for fh in self._run_files.values():
                try:
                    fh.flush()
                except Exception:
                    pass

    def close(self) -> None:
        with self._lock:
            for fh in self._run_files.values():
                try:
                    fh.close()
                except Exception:
                    pass
            self._run_files.clear()

    # ------------------------------------------------------------------ 内部

    def _trim(self, fields: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in fields.items():
            if isinstance(value, str) and len(value) > self.max_field_chars:
                marker = f"…(截断, 原始 {len(value)} 字符, 完整内容见 raw_llm/ 或 traces/)"
                out[key] = value[: self.max_field_chars] + marker
            else:
                out[key] = value
        return out

    def _preview_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        preview = []
        for msg in messages:
            content = str(msg.get("content", ""))
            preview.append({"role": str(msg.get("role", "")), "content": truncate(content, 240)})
        return preview

    def _append_jsonl(self, ev: TraceEvent) -> None:
        if not self.jsonl_path:
            return
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(ev.line() + "\n")
        except OSError:
            pass

    def _append_run_file(self, ev: TraceEvent) -> None:
        # 内存模式（Tracer.in_memory）没有 trace_dir：必须直接返回，
        # 否则 os.path.join("", "run_x.log") 会把日志写到进程 CWD。
        if not ev.run_id or not self.trace_dir:
            return
        try:
            fh = self._run_files.get(ev.run_id)
            if fh is None:
                path = os.path.join(self.trace_dir, f"{ev.run_id}.log")
                fh = open(path, "w", encoding="utf-8")
                self._run_files[ev.run_id] = fh
                fh.write(f"# run {ev.run_id}  session={ev.session_id}  started={ev.ts}\n")
            fh.write(self._format(ev) + "\n")
            fh.flush()
            # 一次 run 结束就关掉文件句柄：长驻服务里不能每次提问都泄漏一个 fd
            if ev.event == "run_end":
                fh.close()
                self._run_files.pop(ev.run_id, None)
        except OSError:
            pass

    def _format(self, ev: TraceEvent) -> str:
        dur = f" ({ev.duration_ms}ms)" if ev.duration_ms is not None else ""
        head = f"[{ev.ts}] {ev.level:<7} {ev.event}{dur}"
        if ev.turn:
            head += f" turn={ev.turn}"
        body = safe_json(ev.data) if ev.data else ""
        return f"{head}\n    {body}" if body else head

    def _print(self, ev: TraceEvent) -> None:
        icon = {
            "run_start": "▶",
            "llm_response": "🧠",
            "tool_call": "🔧",
            "tool_result": "📦",
            "parse_error": "⚠",
            "exception": "✖",
            "context_compressed": "🗜",
            "final_answer": "✅",
            "force_final_answer": "⏹",
        }.get(ev.event, "·")
        if ev.event in ("llm_request", "session_snapshot"):
            return
        dur = f" {ev.duration_ms}ms" if ev.duration_ms else ""
        detail = ""
        data = ev.data
        if ev.event == "tool_call":
            detail = f"{data.get('tool_name')}({safe_json(data.get('arguments'))})"
        elif ev.event == "tool_result":
            detail = f"{data.get('tool_name')} -> {truncate(data.get('result'), 120)}"
        elif ev.event == "llm_response":
            detail = truncate(data.get("raw_output"), 160)
        elif ev.event in ("run_start", "final_answer"):
            detail = truncate(data.get("text") or data.get("user_input"), 160)
        elif ev.event == "context_compressed":
            detail = f"dropped={data.get('dropped_messages')} msgs"
        elif ev.event == "exception":
            detail = f"{data.get('where')}: {data.get('error_type')}: {truncate(data.get('error'), 120)}"
        elif ev.event == "parse_error":
            detail = truncate(data.get("error"), 160)
        print(f"  {icon} [{ev.event}]{dur} {detail}", flush=True)

    @classmethod
    def in_memory(cls) -> "Tracer":
        """纯内存 Tracer：**不落任何盘**（测试与单次调试用）。"""
        tracer = cls.__new__(cls)
        tracer.log_dir = ""
        tracer.trace_dir = ""
        tracer.raw_dir = ""
        tracer.jsonl_path = ""
        tracer.console = False
        tracer.max_field_chars = 10**9
        tracer._lock = threading.Lock()
        tracer._events = deque(maxlen=20000)
        tracer._run_files = {}
        return tracer


class Timer:
    """`with Timer() as t: ...` → `t.ms`"""

    def __enter__(self) -> "Timer":
        self._start = now_ms()
        self.ms = 0
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.ms = now_ms() - self._start
        return False
