"""统一异常体系。

设计原则（对应交付要求第 5 节「异常处理」）:
- 所有异常都可以被 `to_observation()` 序列化成一段**给 LLM 看的文本**；
- 异常文本写回上下文后，LLM 有机会自我修正（重试工具 / 修正参数 / 告知用户失败）；
- 只有 `MiniAgentError` 的子类才会被 Agent 主循环「软处理」，其他异常视为 bug 直接抛出。
"""

from __future__ import annotations

from typing import Any, Optional


class MiniAgentError(Exception):
    """本项目所有可预期异常的基类。"""

    #: 给 LLM 的默认修复建议
    hint: str = ""

    def __init__(self, message: str, *, hint: str = "", detail: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.hint = hint or self.hint
        self.detail = detail or {}

    def to_observation(self) -> str:
        """转换成写回上下文的工具结果/错误文本。"""
        parts = [f"{type(self).__name__}: {self.message}"]
        if self.hint:
            parts.append(f"修复建议: {self.hint}")
        if self.detail:
            parts.append(f"细节: {self.detail}")
        return "\n".join(parts)

    def __str__(self) -> str:
        """`str(exc)` 也带上修复建议 —— 排查日志与单测断言都依赖它。"""
        text = self.message
        if self.hint:
            text = f"{text}；{self.hint}"
        return text


class LLMError(MiniAgentError):
    """LLM 网络异常 / 超时 / 非 2xx / 响应结构非法。"""

    hint = "这是模型网关或网络问题，可稍后重试；若持续失败请直接告知用户服务不可用。"


class ParseError(MiniAgentError):
    """LLM 输出无法解析成合法的 answer / tool_call。"""

    hint = (
        "必须只输出一个 JSON 对象，且不要包含任何多余文字。"
        '调用工具时输出 {"type":"tool_call","tool_name":"<工具名>","arguments":{...}}，'
        '给出结论时输出 {"type":"answer","content":"<中文回答>"}。'
    )

    def __init__(self, message: str, *, raw: str = "", hint: str = "", detail: Optional[dict[str, Any]] = None):
        super().__init__(message, hint=hint, detail=detail)
        self.raw = raw


class ToolNotFoundError(MiniAgentError):
    hint = "请从系统提示给出的工具列表中选择工具名，不要臆造工具。"

    def __init__(self, name: str, available: list[str]):
        super().__init__(
            f"工具 `{name}` 不存在",
            hint=f"当前可用工具: {', '.join(available) or '（无）'}",
            detail={"requested": name, "available": available},
        )
        self.name = name


class ToolValidationError(MiniAgentError):
    """工具参数不符合 JSON Schema。"""

    hint = "请严格按照 schema 修正 arguments 后重新调用。"


class ToolExecutionError(MiniAgentError):
    """工具 handler 内部抛错 / 超时。"""

    hint = "可以换一种参数重试；如果同样失败，请改用其他方式或直接告知用户该工具不可用。"


class ContextOverflowError(MiniAgentError):
    """上下文裁剪后仍然超限。（保留给生产环境替换 ContextManager 时使用）"""


class MaxTurnsExceeded(MiniAgentError):
    """超过工具调用轮次上限（正常情况下会被 force_final_answer 兜住，很少外抛）。"""
