"""LLM 客户端接口与数据结构。"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


@dataclass
class LLMRequest:
    """一次 LLM 调用的输入。

    `messages` 中 role 只有 `system` / `user` / `assistant` 三种（工具调用与结果
    在组装阶段就被渲染成文本，见 `context.py`），因此天然兼容各种 chat API。

    `temperature` / `max_tokens` 为 `None` 表示「用客户端的默认值」——
    必须保留 None 语义，否则每次请求都会用 dataclass 默认值覆盖客户端配置。
    """

    messages: list[dict[str, str]]
    model: str = ""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    #: 达到工具轮次上限时为 True —— 客户端可据此调整采样（例如降低随机性）
    force_final_answer: bool = False
    #: 透传字段（不同厂商私有参数）
    extra: dict[str, Any] = field(default_factory=dict)
    #: 元信息，不进请求体，只用于 trace
    meta: dict[str, Any] = field(default_factory=dict)

    def with_messages(self, messages: Sequence[dict[str, str]]) -> "LLMRequest":
        clone = LLMRequest(
            messages=list(messages),
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            force_final_answer=self.force_final_answer,
            extra=dict(self.extra),
            meta=dict(self.meta),
        )
        return clone

    def prompt_text(self) -> str:
        return "\n".join(f"{m.get('role')}: {m.get('content')}" for m in self.messages)


@dataclass
class LLMResponse:
    """一次 LLM 调用的输出。"""

    content: str
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str = ""
    latency_ms: int = 0
    cached: bool = False
    raw: Any = None

    @property
    def total_tokens(self) -> int:
        for key in ("total_tokens", "total"):
            if key in self.usage and isinstance(self.usage[key], (int, float)):
                return int(self.usage[key])
        return 0


class LLMClient(abc.ABC):
    """LLM 客户端基类。"""

    name: str = "llm"

    @abc.abstractmethod
    async def chat(self, request: LLMRequest) -> LLMResponse:
        """发起一次对话补全；失败时抛 `miniagent.errors.LLMError`。"""
        raise NotImplementedError

    async def aclose(self) -> None:
        """释放底层连接（默认无操作）。"""
        return None

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} name={self.name!r}>"
