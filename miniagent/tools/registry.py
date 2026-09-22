"""工具注册表：注册 / 注销 / 查询 / 生成 schema 清单。

对应交付要求「工具注册机制」：

    {
        "name": str,              # 工具名，LLM 调用标识
        "description": str,       # 给 LLM 看的能力描述
        "parameters": jsonschema, # 参数 schema
        "handler": Callable,      # 实际执行函数
    }

设计说明：
- `ToolRegistry` 只是一个**字典 + 一致性校验**，不掺业务逻辑；
- `handler` 可以是同步函数，也可以是 async 函数（`execute()` 统一 await）；
- `ToolContext` 把 session_id / tracer / 依赖注入传给 handler，handler 不必是全局单例。
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Union

from ..errors import ToolExecutionError, ToolNotFoundError
from ..schema import coerce_arguments, validate_arguments
from ..utils import safe_json

ToolHandler = Callable[..., Union[str, dict, list, Any, Awaitable[Any]]]


@dataclass
class ToolContext:
    """执行工具时注入的上下文（让 handler 可感知会话、可写日志、可复用外部客户端）。"""

    session_id: str = ""
    run_id: str = ""
    user_id: str = ""
    window_id: str = ""
    turn: int = 0
    tracer: Any = None
    deps: dict[str, Any] = field(default_factory=dict)

    def dep(self, key: str, default: Any = None) -> Any:
        return self.deps.get(key, default)


@dataclass
class Tool:
    """一个工具的完整定义。"""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("工具 name 必须是非空字符串")
        if not isinstance(self.parameters, dict) or self.parameters.get("type") != "object":
            raise ValueError(f"工具 `{self.name}` 的 parameters 必须是 type=object 的 JSON Schema")
        if not callable(self.handler):
            raise ValueError(f"工具 `{self.name}` 的 handler 必须可调用（同步/async 均可）")

    def _detect_ctx_param(self) -> bool:
        """handler 是否声明了上下文形参（声明了才注入 ToolContext）。

        兼容 `ctx` / `context` / `tool_ctx` 三种命名，且支持位置参数写法
        `def handler(query, ctx)`。
        """
        try:
            params = inspect.signature(self.handler).parameters
        except (TypeError, ValueError):  # 内置函数等拿不到签名
            return False
        for candidate in ("ctx", "context", "tool_ctx"):
            if candidate in params:
                return True
        positional = [
            name
            for name, param in params.items()
            if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        ]
        return len(positional) >= 2 and positional[1] in ("ctx", "context", "tool_ctx")

    def ctx_param_name(self) -> Optional[str]:
        """返回 handler 声明的上下文形参名（没声明则 None）——注入时必须用这个名字。"""
        try:
            params = inspect.signature(self.handler).parameters
        except (TypeError, ValueError):
            return None
        for candidate in ("ctx", "context", "tool_ctx"):
            if candidate in params:
                return candidate
        positional = [
            name
            for name, param in params.items()
            if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) >= 2 and positional[1] in ("ctx", "context", "tool_ctx"):
            return positional[1]
        return None

    @property
    def accepts_ctx(self) -> bool:
        return self.ctx_param_name() is not None

    def to_schema(self) -> dict[str, Any]:
        """给 LLM 看的工具描述（会写进 system prompt，也会作为 trace 的元信息）。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.handler)


class ToolRegistry:
    """全局（或注入式）工具注册表。"""

    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}

    # ---------------------------------------------------------------- 注册
    def register(
        self,
        tool: Optional[Tool] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        schema: Optional[dict[str, Any]] = None,
        handler: Optional[ToolHandler] = None,
        tags: tuple[str, ...] = (),
        override: bool = True,
    ) -> Tool:
        """注册工具。两种用法：

            registry.register(Tool(...))
            registry.register(name="echo", description="...", schema={...}, handler=fn)
        """
        if tool is None:
            tool = Tool(
                name=name or "",                      # type: ignore[arg-type]
                description=description or "",        # type: ignore[arg-type]
                parameters=schema or {},              # type: ignore[arg-type]
                handler=handler,                      # type: ignore[arg-type]
                tags=tags,
            )
        if not override and tool.name in self.tools:
            raise ValueError(f"工具 `{tool.name}` 已存在；如需覆盖请传 override=True")
        self.tools[tool.name] = tool
        return tool

    def unregister(self, name: str, *, missing_ok: bool = True) -> bool:
        """注销工具。返回是否真的删掉了。"""
        if name in self.tools:
            del self.tools[name]
            return True
        if missing_ok:
            return False
        raise ToolNotFoundError(name, self.names())

    def get(self, name: str) -> Tool:
        try:
            return self.tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(name, self.names()) from exc

    def has(self, name: str) -> bool:
        return name in self.tools

    def names(self) -> list[str]:
        return sorted(self.tools)

    def snapshot(self) -> list[dict[str, Any]]:
        """工具 schema 清单（按名字排序，保证 prompt 稳定 → 命中 KV cache）。"""
        return [self.tools[name].to_schema() for name in self.names()]

    def describe(self) -> str:
        """人类可读的工具清单（CLI `/tools` 用）。"""
        lines = []
        for name in self.names():
            tool = self.tools[name]
            required = tool.parameters.get("required") or []
            props = list((tool.parameters.get("properties") or {}).keys())
            lines.append(f"- {name}: {tool.description}")
            lines.append(f"    参数: {props}  必填: {required or '无'}  异步: {tool.is_async}")
        return "\n".join(lines) if lines else "（未注册任何工具）"

    def __len__(self) -> int:
        return len(self.tools)

    def __contains__(self, name: object) -> bool:
        return name in self.tools

    # ---------------------------------------------------------------- 执行
    async def execute(self, name: str, arguments: Any, ctx: Optional[ToolContext] = None) -> str:
        """参数校验 → 执行 → 结果字符串化。

        - 声明了 `ctx` 形参的 handler 会拿到 `ToolContext`（会话 / trace / 依赖注入）；
        - **同步 handler 放进线程池执行**：否则会阻塞事件循环，让 Agent 的
          `tool_timeout` 形同虚设（详见 README「工具超时」一节）；
        - 异常全部转成 `MiniAgentError` 子类，由 Agent 主循环决定怎么回灌给 LLM。
        """
        tool = self.get(name)
        args = validate_arguments(tool.parameters, arguments, tool_name=name)
        ctx = ctx or ToolContext()
        ctx_name = tool.ctx_param_name()
        if ctx_name:
            args = {**args, ctx_name: ctx}
        try:
            if tool.is_async:
                result = await tool.handler(**args)
            else:
                # 同步 handler 丢到线程池：事件循环保持可响应，wait_for 才能真正生效
                result = await asyncio.to_thread(tool.handler, **args)
                if inspect.isawaitable(result):
                    result = await result
        except (ToolNotFoundError, ToolExecutionError):
            raise
        except Exception as exc:  # noqa: BLE001 - 工具内部任何异常都要变成可回灌文本
            raise ToolExecutionError(
                f"工具 `{name}` 执行失败: {type(exc).__name__}: {exc}",
                detail={"arguments": {k: v for k, v in args.items() if k != ctx_name}},
            ) from exc
        return normalize_result(result)

    def validate_call(self, name: str, arguments: Any) -> dict[str, Any]:
        """只校验不执行（Parser 用，可在真正执行前把错误回灌给 LLM）。"""
        tool = self.get(name)
        return validate_arguments(tool.parameters, arguments, tool_name=name)

    def coerce(self, name: str, arguments: Any) -> Any:
        tool = self.get(name)
        return coerce_arguments(tool.parameters, arguments) if isinstance(arguments, dict) else arguments


def normalize_result(result: Any) -> str:
    """工具返回值统一转成字符串（LLM 只能读文本）。"""
    if result is None:
        return "(工具返回空)"
    if isinstance(result, str):
        return result
    if isinstance(result, (dict, list, tuple)):
        return safe_json(result)
    if isinstance(result, bool):
        return "true" if result else "false"
    return str(result)


def build_default_registry() -> ToolRegistry:
    """内置工具集：calculator / search / weather（+ 可选的 now、session_echo 演示工具）。"""
    from .calculator import calculator_tool
    from .search import search_tool
    from .weather import weather_tool

    registry = ToolRegistry()
    registry.register(calculator_tool())
    registry.register(search_tool())
    registry.register(weather_tool())
    return registry
