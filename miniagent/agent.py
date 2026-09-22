"""Agent Runtime 主循环：Session → Context → LLM → Parser → Tool → 回灌 → 循环。

对应交付要求第 1 节：

    Step1 接收用户消息 + 读取对应 session 上下文
    Step2 组装 prompt（历史对话 + 工具 schema + 当前 query）发给 LLM
    Step3 Parser 解析：分支A 最终回答 → 结束；分支B 工具调用 → 继续
    Step4 执行工具（捕获异常 + 记 trace）
    Step5 工具结果追加进 session 上下文
    Step6 回到 Step2 继续循环，直到 LLM 输出最终答案

在此之上补了 4 个**生产必需**的护栏（README「问题记录」里详细写了为什么）：
1. `max_tool_turns` 轮次上限 → 到顶后强制模型直接总结（不再执行工具）；
2. 解析失败重试上限 → 把解析错误回灌让模型自我修正，连续失败则降级返回纯文本；
3. 重复调用检测 → 同工具同参数重复时插入提示，避免死循环烧 token；
4. 全链路 trace → 每次 LLM 调用 / 工具调用 / 异常都有结构化日志。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import AgentConfig
from .context import ContextManager, render_tool_exception
from .errors import LLMError, MiniAgentError, ParseError
from .llm.base import LLMClient, LLMRequest
from .parser import Answer, Parser, ToolCall
from .prompts import PromptLoader
from .session import Session, SessionManager
from .tools.registry import ToolContext, ToolRegistry
from .tracing import Tracer, Timer
from .utils import ensure_utf8_stdio, new_id, now_iso, safe_json, truncate


@dataclass
class ToolInvocation:
    """一次工具调用的完整记录（用于返回值、trace、测试断言）。"""

    name: str
    arguments: dict[str, Any]
    result: str = ""
    ok: bool = True
    error: str = ""
    duration_ms: int = 0
    turn: int = 0
    attempts: int = 1


@dataclass
class AgentResult:
    """一次 `run()` 的结果。"""

    answer: str
    session_id: str
    run_id: str = ""
    turns_used: int = 0
    tool_calls: list[ToolInvocation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stopped_reason: str = "answer"      # answer / max_turns / parse_failed / llm_error
    latency_ms: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
    compressed: bool = False
    dropped_messages: int = 0

    @property
    def used_tools(self) -> list[str]:
        return [call.name for call in self.tool_calls]

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "turns_used": self.turns_used,
            "tool_calls": [
                {"name": c.name, "arguments": c.arguments, "ok": c.ok, "duration_ms": c.duration_ms, "result": truncate(c.result, 200)}
                for c in self.tool_calls
            ],
            "errors": self.errors,
            "stopped_reason": self.stopped_reason,
            "latency_ms": self.latency_ms,
            "compressed": self.compressed,
            "dropped_messages": self.dropped_messages,
        }


class MinimalAgent:
    """最小可用 Agent Runtime。"""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: Optional[ToolRegistry] = None,
        session_mgr: Optional[SessionManager] = None,
        *,
        config: Optional[AgentConfig] = None,
        tracer: Optional[Tracer] = None,
        context_mgr: Optional[ContextManager] = None,
        prompt_loader: Optional[PromptLoader] = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.llm = llm_client
        self.tool_registry = tool_registry if tool_registry is not None else ToolRegistry()
        self.session_mgr = session_mgr or SessionManager(
            max_turn=self.config.max_tool_turns,
            max_context_tokens=self.config.max_context_tokens,
            keep_recent_messages=self.config.keep_recent_messages,
            persist=self.config.persist_sessions,
            persist_path=self.config.session_path,
        )
        self.tracer = tracer or Tracer(self.config.log_dir, console=self.config.console_trace)
        self.context_mgr = context_mgr or ContextManager(
            max_context_tokens=self.config.max_context_tokens,
            max_context_messages=self.config.max_context_messages,
            keep_recent_messages=self.config.keep_recent_messages,
            tool_result_limit=self.config.tool_result_limit,
            compress=self.config.enable_compression,
            tracer=self.tracer,
        )
        self.prompt_loader = prompt_loader or PromptLoader(self.config.prompt_path)
        self.parser = Parser(self.tool_registry, lenient_plain_text=self.config.lenient_plain_text)

    # ================================================================== 对外入口
    async def run(
        self,
        user_input: str,
        *,
        session_id: str = "",
        user_id: str = "user",
        window_id: str = "default",
        max_tool_turns: Optional[int] = None,
    ) -> AgentResult:
        """处理一次用户提问（含完整工具循环）。

        - `session_id` 优先；未提供时由 `user_id + window_id` 组合而成；
        - 追问直接再次调用本方法即可，历史会自动带上。
        """
        session_id = session_id or f"{user_id}::{window_id}"
        session = self.session_mgr.get_or_create(session_id)
        run_id = new_id("run")
        started = time.perf_counter()
        budget = max_tool_turns if max_tool_turns is not None else session.max_turn or self.config.max_tool_turns

        session.add_user(user_input)
        self.tracer.log(
            "run_start",
            session_id=session_id,
            run_id=run_id,
            user_id=session.user_id,
            window_id=session.window_id,
            user_input=user_input,
            session_messages=session.size,
            max_tool_turns=budget,
        )

        result = AgentResult(answer="", session_id=session_id, run_id=run_id)
        tool_turns = 0
        parse_failures = 0
        call_history: list[str] = []
        compressed_any = False
        dropped_any = 0

        try:
            while True:
                force_final = tool_turns >= budget
                extra_rules = self.context_mgr.turn_limit_rules(session, tool_turns, max_turns=budget)
                if parse_failures:
                    extra_rules.append(
                        f"你上一条输出无法解析（已失败 {parse_failures} 次）。"
                        "请严格只输出一个 JSON 对象，不要有额外文字。"
                    )
                if self._is_repeating(call_history):
                    extra_rules.append(
                        "你已经用完全相同的参数调用过同一个工具，结果不会变化。"
                        "请立刻基于已有工具结果输出 type=answer，或换个参数/换个工具。"
                    )

                build = self.context_mgr.build(
                    session,
                    self._system_prompt(session, budget),
                    extra_rules=extra_rules,
                    force_final_answer=force_final,
                    run_id=run_id,
                )
                # 把「本次请求发生了裁剪」这件事记回 session，供 CLI /sessions 与测试观测
                if build.dropped_messages:
                    session.compress_count += 1
                    session.dropped_messages += build.dropped_messages
                    session.meta["last_compress"] = {
                        "at_turn": tool_turns,
                        "dropped": build.dropped_messages,
                        "notes": build.notes,
                    }
                compressed_any = compressed_any or build.compressed
                dropped_any += build.dropped_messages
                result.compressed = compressed_any
                result.dropped_messages = dropped_any

                self.tracer.log_llm_request(
                    build.messages,
                    session_id=session_id,
                    run_id=run_id,
                    turn=tool_turns,
                    token_estimate=build.token_estimate,
                    compressed=build.compressed,
                    force_final_answer=force_final,
                )

                # ---------------- Step2: 调用 LLM ----------------
                request = LLMRequest(
                    messages=build.messages,
                    model=self.config.model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    force_final_answer=force_final,
                    meta={"session_id": session_id, "run_id": run_id, "turn": tool_turns},
                )
                try:
                    with Timer() as timer:
                        response = await self._chat_with_retry(request, session_id=session_id, run_id=run_id, turn=tool_turns)
                except LLMError as exc:
                    self.tracer.log_exception(exc, where="llm.chat", session_id=session_id, run_id=run_id, turn=tool_turns)
                    result.errors.append(exc.to_observation())
                    result.stopped_reason = "llm_error"
                    result.answer = self.config.llm_error_reply
                    session.add_error(exc.to_observation())
                    session.add_assistant(result.answer, turn=tool_turns)
                    return self._finish(result, started, session, run_id)

                self.tracer.log_llm_response(
                    response.content,
                    usage=response.usage,
                    finish_reason=response.finish_reason,
                    model=response.model,
                    session_id=session_id,
                    run_id=run_id,
                    turn=tool_turns,
                    duration_ms=response.latency_ms or timer.ms,
                )

                # ---------------- Step3: 解析 LLM 输出 ----------------
                try:
                    parsed = self.parser.parse(response.content)
                except ParseError as exc:
                    parse_failures += 1
                    self.tracer.log(
                        "parse_error",
                        level="WARNING",
                        session_id=session_id,
                        run_id=run_id,
                        turn=tool_turns,
                        error=exc.message,
                        raw_output=truncate(response.content, 300),
                        failures=parse_failures,
                    )
                    result.errors.append(f"parse_error: {exc.message}")
                    session.add_error(exc.to_observation(), turn=tool_turns)

                    if parse_failures > self.config.max_parse_retries:
                        result.stopped_reason = "parse_failed"
                        result.answer = self.config.parse_error_reply
                        session.add_assistant(result.answer, turn=tool_turns)
                        return self._finish(result, started, session, run_id)
                    continue   # 把错误回灌上下文，让 LLM 修正

                if isinstance(parsed, Answer):
                    # ---------- 分支A：最终回答，循环结束 ----------
                    session.add_assistant(parsed.content, turn=tool_turns)
                    result.answer = parsed.content
                    result.stopped_reason = "answer"
                    result.turns_used = tool_turns
                    self.tracer.log(
                        "final_answer",
                        session_id=session_id,
                        run_id=run_id,
                        turn=tool_turns,
                        text=truncate(parsed.content, 300),
                        repaired=parsed.repaired,
                    )
                    return self._finish(result, started, session, run_id)

                # ---------- 分支B：工具调用 ----------
                assert isinstance(parsed, ToolCall)
                session.add_tool_call(parsed.tool_name, parsed.arguments, turn=tool_turns)
                signature = f"{parsed.tool_name}:{safe_json(parsed.arguments)}"
                call_history.append(signature)
                self.tracer.log_tool_call(
                    parsed.tool_name,
                    parsed.arguments,
                    session_id=session_id,
                    run_id=run_id,
                    turn=tool_turns,
                    repaired=parsed.repaired,
                )

                if force_final:
                    # ---------------- 轮次上限护栏 ----------------
                    note = (
                        f"已达到工具调用轮次上限（{budget}），本轮不再执行工具 `{parsed.tool_name}`。"
                        "请立即基于上文已有结果输出最终回答。"
                    )
                    self.tracer.log("max_turns_reached", level="WARNING", session_id=session_id, run_id=run_id, turn=tool_turns, detail=note)
                    result.errors.append(note)
                    session.add_error(note, name=parsed.tool_name, turn=tool_turns)
                    parse_failures += 1
                    if parse_failures > self.config.max_parse_retries:
                        result.stopped_reason = "max_turns"
                        result.answer = self.config.max_turns_reply
                        session.add_assistant(result.answer, turn=tool_turns)
                        return self._finish(result, started, session, run_id)
                    continue

                # ---------------- Step4: 执行工具（捕获异常 + trace）----------------
                invocation = await self._execute_tool(parsed, session=session, run_id=run_id, turn=tool_turns)
                result.tool_calls.append(invocation)

                # ---------------- Step5: 结果/异常回灌上下文 ----------------
                if invocation.ok:
                    session.add_tool_result(parsed.tool_name, invocation.result, turn=tool_turns)
                else:
                    session.add_error(invocation.error, name=parsed.tool_name, turn=tool_turns)
                    # 工具结果为空时，也要让模型明确知道「执行了但没数据」
                    session.add_tool_result(
                        parsed.tool_name,
                        render_tool_exception(parsed.tool_name, invocation.error),
                        turn=tool_turns,
                    )
                tool_turns += 1
                parse_failures = 0
                # ---------------- Step6: 回到 Step2 ----------------
        except asyncio.CancelledError:  # pragma: no cover - 上层取消
            self.tracer.log("run_cancelled", level="WARNING", session_id=session_id, run_id=run_id)
            raise
        except Exception as exc:  # noqa: BLE001 - 兜底：绝不让 Runtime 崩到用户界面
            self.tracer.log_exception(exc, where="agent.run_loop", session_id=session_id, run_id=run_id)
            result.errors.append(f"{type(exc).__name__}: {exc}")
            result.stopped_reason = "internal_error"
            result.answer = self.config.internal_error_reply
            session.add_error(f"内部异常: {type(exc).__name__}: {exc}")
            session.add_assistant(result.answer)
            return self._finish(result, started, session, run_id)

    # ================================================================== 内部
    def _system_prompt(self, session: Session, max_turns: int) -> str:
        """每次实时拼接 system prompt（**不落 session**）。"""
        return self.prompt_loader.render(
            tools=self.tool_registry.snapshot(),
            session_id=session.session_id,
            user_id=session.user_id,
            window_id=session.window_id,
            current_time=now_iso(),
            max_turns=max_turns,
            tool_prompt_mode=self.config.tool_prompt_mode,
        )

    async def _chat_with_retry(self, request: LLMRequest, *, session_id: str, run_id: str, turn: int):
        """LLM 调用 + 额外一层重试（客户端内部已处理网络级重试，这里兜住 LLMError）。

        说明：`OpenAICompatibleClient` 已带指数退避；这里只做「不改变语义的二次尝试」，
        并把最终失败交给调用方转成用户可读文案。
        """
        attempts = max(1, self.config.llm_call_retries)
        last: Optional[LLMError] = None
        for attempt in range(attempts):
            try:
                return await self.llm.chat(request)
            except LLMError as exc:
                last = exc
                if attempt < attempts - 1:
                    self.tracer.log(
                        "llm_retry",
                        level="WARNING",
                        session_id=session_id,
                        run_id=run_id,
                        turn=turn,
                        attempt=attempt + 1,
                        error=str(exc),
                    )
                    await asyncio.sleep(min(0.5 * (2**attempt), 4.0))
        assert last is not None
        raise last

    async def _execute_tool(self, call: ToolCall, *, session: Session, run_id: str, turn: int) -> ToolInvocation:
        """执行工具：校验 → 运行 → 异常捕获 → trace。"""
        ctx = ToolContext(
            session_id=session.session_id,
            run_id=run_id,
            user_id=session.user_id,
            window_id=session.window_id,
            turn=turn,
            tracer=self.tracer,
            deps=self.config.tool_deps,
        )
        try:
            with Timer() as timer:
                text = await asyncio.wait_for(
                    self.tool_registry.execute(call.tool_name, call.arguments, ctx),
                    timeout=self.config.tool_timeout,
                )
            invocation = ToolInvocation(
                name=call.tool_name,
                arguments=call.arguments,
                result=text,
                ok=True,
                duration_ms=timer.ms,
                turn=turn,
            )
        except asyncio.TimeoutError:
            message = f"工具 `{call.tool_name}` 执行超时（>{self.config.tool_timeout}s）"
            invocation = ToolInvocation(
                name=call.tool_name,
                arguments=call.arguments,
                ok=False,
                error=message,
                turn=turn,
            )
        except MiniAgentError as exc:
            invocation = ToolInvocation(
                name=call.tool_name,
                arguments=call.arguments,
                ok=False,
                error=exc.to_observation(),
                turn=turn,
            )
        except Exception as exc:  # noqa: BLE001 - 未预期异常也要变成可回灌文本，而不是炸掉循环
            self.tracer.log_exception(exc, where="tool.execute", session_id=session.session_id, run_id=run_id, turn=turn)
            invocation = ToolInvocation(
                name=call.tool_name,
                arguments=call.arguments,
                ok=False,
                error=f"工具 `{call.tool_name}` 内部未预期异常: {type(exc).__name__}: {exc}",
                turn=turn,
            )

        self.tracer.log_tool_result(
            invocation.name,
            invocation.result if invocation.ok else invocation.error,
            duration_ms=invocation.duration_ms,
            ok=invocation.ok,
            error=invocation.error,
            session_id=session.session_id,
            run_id=run_id,
            turn=turn,
            arguments=invocation.arguments,
        )
        return invocation

    @staticmethod
    def _is_repeating(call_history: list[str]) -> bool:
        """连续两次完全相同的调用 → 触发提示（防止模型原地打转）。"""
        return len(call_history) >= 2 and call_history[-1] == call_history[-2]

    def _finish(self, result: AgentResult, started: float, session: Session, run_id: str) -> AgentResult:
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        self.session_mgr.save(session)
        self.tracer.log(
            "run_end",
            session_id=session.session_id,
            run_id=run_id,
            duration_ms=result.latency_ms,
            stopped_reason=result.stopped_reason,
            turns_used=result.turns_used,
            tools=result.used_tools,
            answer=truncate(result.answer, 300),
        )
        self.tracer.flush()
        return result

    # ================================================================== 便捷方法
    def run_sync(self, user_input: str, **kwargs: Any) -> AgentResult:
        """同步调用（给脚本 / 单元测试用）。"""
        return asyncio.run(self.run(user_input, **kwargs))

    def reset(self, session_id: str) -> None:
        self.session_mgr.reset(session_id)

    def new_window(self, user_id: str, window_id: str) -> Session:
        return self.session_mgr.for_window(user_id, window_id)

    async def aclose(self) -> None:
        await self.llm.aclose()
        self.session_mgr.save_all()
        self.tracer.close()


def build_agent(config: Optional[AgentConfig] = None, **overrides: Any) -> MinimalAgent:
    """按配置装配一个完整 Agent（LLM + 工具 + Session + Context + Tracer）。"""
    from .llm.openai_compatible import build_client
    from .tools.registry import build_default_registry

    cfg = config or AgentConfig.from_env()
    registry = build_default_registry()
    llm_config = cfg.llm if hasattr(cfg, "llm") else cfg
    client = build_client(llm_config, **overrides)
    return MinimalAgent(client, registry, config=cfg)


__all__ = ["MinimalAgent", "AgentResult", "ToolInvocation", "build_agent", "ensure_utf8_stdio"]
