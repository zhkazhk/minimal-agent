"""Context 上下文管理：组装、轮次限制、基础压缩。

职责边界
--------
- **做什么**：把 session 的历史消息 + 固定 system prompt + 工具 schema 组装成 LLM 请求；
  超阈值时裁剪早期消息并插入压缩提示；把 `tool_call` / `tool_result` 渲染成 LLM 能读的文本。
- **不做什么**：不写回 session（裁剪只影响「本次请求」），不改业务语义 —— 原始历史仍完整保留在
  session 里，便于审计与后续换成「LLM 摘要式压缩」。

渲染协议（为什么不用原生 function calling）
------------------------------------------
本项目刻意把所有消息压成 `system` / `user` / `assistant` 三种角色 + 纯文本 JSON 协议：

    assistant : {"type":"tool_call","tool_name":"weather","arguments":{...}}
    user      : 【用户】上海今天天气
                【工具结果 weather】上海 今天：多云，气温 24°C …
    user      : 【错误】工具参数校验失败…

好处：任何 OpenAI 兼容网关（包括只支持最简 chat 的本地模型）都能跑；
代价：模型可能输出格式不规范的 JSON —— 由 `parser.py` 的多级修复兜住，这也正是本项目的重点演练内容。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .errors import ContextOverflowError
from .parser import Parser, ParsedOutput
from .session import Message, Session
from .tracing import Tracer
from .utils import estimate_messages_tokens, estimate_tokens, safe_json, truncate

#: 裁剪后插入的提示（交付要求原文）
COMPRESSION_NOTICE = "【上下文已裁剪，前面对话已压缩，只保留最近对话】"

MAX_TOOL_RESULT_CHARS = 1500


@dataclass
class BuildResult:
    """一次上下文组装的产物 + 元信息（全部会进 trace）。"""

    messages: list[dict[str, str]]
    system_prompt: str
    token_estimate: int
    compressed: bool = False
    dropped_messages: int = 0
    trimmed_tool_results: int = 0
    notes: list[str] = field(default_factory=list)
    dropped_roles: dict[str, int] = field(default_factory=dict)

    @property
    def message_count(self) -> int:
        return len(self.messages)


# ---------------------------------------------------------------------------
# 渲染：session 消息 → LLM 可读文本
# ---------------------------------------------------------------------------

def render_tool_call(name: str, arguments: Any) -> str:
    """工具调用请求 → 与 System Prompt 协议一致的 JSON 文本。"""
    payload = {"type": "tool_call", "tool_name": name, "arguments": arguments if isinstance(arguments, dict) else {}}
    return safe_json(payload)


def render_tool_result(name: str, result: str, *, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    """工具结果 → 带标记的文本块（长结果截断，防止单条结果吃光上下文）。"""
    body = result if isinstance(result, str) else safe_json(result)
    if len(body) > limit:
        body = body[:limit] + f"\n…（工具结果过长已截断，原始 {len(body)} 字符）"
    return f"【工具结果 {name}】\n{body}"


def render_error(text: str, *, name: str = "") -> str:
    tag = f"【错误 {name}】" if name else "【错误】"
    return f"{tag}\n{text}"


def render_tool_exception(name: str, text: str) -> str:
    return f"【工具执行异常 {name}】\n{text}"


class ContextManager:
    """上下文组装 + 裁剪。"""

    def __init__(
        self,
        *,
        max_context_tokens: int = 3000,
        max_context_messages: int = 40,
        keep_recent_messages: int = 12,
        min_recent_messages: int = 4,
        tool_result_limit: int = MAX_TOOL_RESULT_CHARS,
        compress: bool = True,
        tracer: Optional[Tracer] = None,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.max_context_messages = max_context_messages
        self.keep_recent_messages = keep_recent_messages
        self.min_recent_messages = min_recent_messages
        self.tool_result_limit = tool_result_limit
        self.compress = compress
        self.tracer = tracer

    # ------------------------------------------------------------------ 组装
    def build(
        self,
        session: Session,
        system_prompt: str,
        *,
        extra_rules: Optional[list[str]] = None,
        force_final_answer: bool = False,
        run_id: str = "",
    ) -> BuildResult:
        """组装完整 messages。

        `system_prompt` 每次实时拼接（**不落 session**）；历史消息按需裁剪 + 渲染。
        """
        prompt = system_prompt
        if extra_rules:
            prompt = prompt + "\n\n" + "\n".join(f"- {rule}" for rule in extra_rules)

        history = list(session.messages)
        result = BuildResult(messages=[], system_prompt=prompt, token_estimate=0)

        if self.compress:
            history, notes = self._prune(history, prompt, result)
            result.notes.extend(notes)
            if result.dropped_messages:
                result.compressed = True

        rendered: list[dict[str, str]] = []
        if result.compressed:
            rendered.append({"role": "system", "content": COMPRESSION_NOTICE})
        for msg in history:
            item = self._render_message(msg, result)
            if item is not None:
                rendered.append(item)

        rendered = self._dedupe_consecutive(rendered)
        messages = [{"role": "system", "content": prompt}] + rendered
        result.messages = messages
        result.token_estimate = estimate_messages_tokens(messages)

        if self.tracer and (result.compressed or result.trimmed_tool_results):
            self.tracer.log(
                "context_compressed",
                session_id=session.session_id,
                run_id=run_id,
                dropped_messages=result.dropped_messages,
                dropped_roles=result.dropped_roles,
                trimmed_tool_results=result.trimmed_tool_results,
                token_estimate=result.token_estimate,
                kept_messages=len(rendered),
                notes=result.notes,
            )
        return result

    # ------------------------------------------------------------------ 渲染
    def _render_message(self, msg: Message, result: BuildResult) -> Optional[dict[str, str]]:
        if msg.role == "user":
            return {"role": "user", "content": msg.content}
        if msg.role == "assistant":
            return {"role": "assistant", "content": msg.content}
        if msg.role == "tool_call":
            return {"role": "assistant", "content": render_tool_call(msg.name, _maybe_json(msg.content))}
        if msg.role == "tool_result":
            body = msg.content
            if len(body) > self.tool_result_limit:
                body = body[: self.tool_result_limit] + f"\n…（工具结果过长已截断，原始 {len(body)} 字符）"
                result.trimmed_tool_results += 1
            return {"role": "user", "content": render_tool_result(msg.name, body, limit=self.tool_result_limit + 1)}
        if msg.role == "error":
            return {"role": "user", "content": render_error(msg.content, name=msg.name)}
        return None

    @staticmethod
    def _dedupe_consecutive(rendered: list[dict[str, str]]) -> list[dict[str, str]]:
        """合并连续同角色消息（部分 API 对连续 user 消息不友好）。"""
        merged: list[dict[str, str]] = []
        for item in rendered:
            if merged and merged[-1]["role"] == item["role"] == "user":
                merged[-1] = {"role": "user", "content": merged[-1]["content"] + "\n" + item["content"]}
            else:
                merged.append(dict(item))
        return merged

    # ------------------------------------------------------------------ 裁剪
    def _prune(self, history: list[Message], prompt: str, result: BuildResult) -> tuple[list[Message], list[str]]:
        """按「消息条数 + token 估算」双阈值裁剪：丢弃最早的消息，保留最近 N 条。"""
        notes: list[str] = []
        budget = max(256, self.max_context_tokens - estimate_tokens(prompt) - 64)

        def cost(msgs: list[Message]) -> int:
            return estimate_messages_tokens([{"role": m.role, "content": m.content} for m in msgs])

        over_count = len(history) > self.max_context_messages
        over_tokens = cost(history) > budget
        if not (over_count or over_tokens):
            return history, notes

        keep_from = 0
        if over_count:
            keep_from = max(keep_from, len(history) - self.keep_recent_messages)
        if over_tokens:
            keep_from = max(keep_from, self._fit_from_tail(history, budget))
        keep_from = min(keep_from, max(0, len(history) - self.min_recent_messages))

        kept = history[keep_from:]
        dropped_count = keep_from

        # 边界清理：丢掉「孤儿工具结果」（其 tool_call 已被裁掉）与开头的孤立 tool_call，
        # 避免模型看到没有出处的工具输出而困惑。
        kept, boundary = self._clean_boundary(kept)
        dropped_count += len(boundary)

        if dropped_count == 0:
            return history, notes

        if over_count:
            notes.append(f"消息条数 {len(history)} > 阈值 {self.max_context_messages}，已裁剪最早 {dropped_count} 条")
        if over_tokens:
            notes.append(f"估算 tokens {cost(history)} > 预算 {budget}，已裁剪最早 {dropped_count} 条")

        result.dropped_messages = dropped_count
        for msg in history[:keep_from]:
            result.dropped_roles[msg.role] = result.dropped_roles.get(msg.role, 0) + 1
        for msg in boundary:
            key = f"{msg.role}(边界清理)"
            result.dropped_roles[key] = result.dropped_roles.get(key, 0) + 1
        notes.append(f"被裁剪的消息角色分布: {result.dropped_roles}")
        return kept, notes

    def _fit_from_tail(self, history: list[Message], budget: int) -> int:
        """从尾部往前累加，找到最靠前的可保留下标。"""
        total = 0
        for idx in range(len(history) - 1, -1, -1):
            msg = history[idx]
            total += estimate_messages_tokens([{"role": msg.role, "content": msg.content}])
            if total > budget:
                return idx + 1
        return 0

    @staticmethod
    def _clean_boundary(kept: list[Message]) -> tuple[list[Message], list[Message]]:
        dropped: list[Message] = []
        # 去掉开头孤立的 tool_result / error
        while kept and kept[0].role in ("tool_result", "error"):
            dropped.append(kept.pop(0))
        # 去掉结尾孤立的 tool_call（没有对应结果，模型会重复调用）
        while kept and kept[-1].role == "tool_call":
            dropped.append(kept.pop())
        return kept, dropped

    # ------------------------------------------------------------------ 轮次
    def turn_limit_rules(self, session: Session, turn_no: int, *, max_turns: int) -> list[str]:
        """接近/到达轮次上限时追加到 system prompt 的临时规则。"""
        rules: list[str] = []
        remaining = max_turns - turn_no
        if remaining <= 0:
            rules.append("你已达到本次可用工具调用轮次上限：本轮**禁止**再调用工具，必须直接输出最终回答。")
            rules.append("请基于上文已有的工具结果作答；信息不足时明确说明还缺什么。")
        elif remaining == 1:
            rules.append(f"你只剩 1 次工具调用机会（上限 {max_turns} 次），请优先给出最终回答，非必要不要调工具。")
        return rules

    # ------------------------------------------------------------------ 校验
    def ensure_fits(self, result: BuildResult) -> None:
        """极端情况下（单条消息就超预算）兜底报错，便于上层给出可读提示。"""
        if result.token_estimate > self.max_context_tokens * 2:
            raise ContextOverflowError(
                f"上下文估算 {result.token_estimate} tokens，超过上限 {self.max_context_tokens} 的 2 倍；"
                "请调小 max_context_tokens 之外的输入，或降低 tool_result 截断阈值。"
            )


def _maybe_json(text: str) -> Any:
    import json

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"raw": truncate(text, 200)}


__all__ = [
    "ContextManager",
    "BuildResult",
    "COMPRESSION_NOTICE",
    "render_tool_call",
    "render_tool_result",
    "render_error",
    "render_tool_exception",
    "Parser",
    "ParsedOutput",
]
