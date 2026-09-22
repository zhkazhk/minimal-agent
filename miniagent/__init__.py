"""minimal-agent：不用任何 Agent 框架，从零手写的最小可用 Agent Runtime。

模块地图::

    config.py    运行配置（唯一事实来源）
    session.py   Session Manager：user_id + window_id 窗口级隔离 + 会话存储
    context.py   Context Manager：上下文组装 / 轮次限制 / 基础压缩
    prompts.py   System Prompt 加载与渲染（prompts/system_prompt.md）
    tools/       Tool Registry + calculator / search / weather
    llm/         LLM 客户端（OpenAI 兼容 / 剧本 / 离线规则版）
    parser.py    LLM 输出解析：JSON 提取 → 修复 → schema 校验 → 错误回灌
    tracing.py   Trace 日志与异常捕获
    agent.py     Agent 主循环（Step1~Step6）
"""

from .agent import AgentResult, MinimalAgent, ToolInvocation, build_agent
from .config import AgentConfig
from .context import ContextManager
from .errors import (
    LLMError,
    MiniAgentError,
    ParseError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolValidationError,
)
from .llm import LLMClient, LLMRequest, LLMResponse, OfflineMockClient, OpenAICompatibleClient, ScriptedLLMClient
from .parser import Answer, Parser, ToolCall, parse_llm_output
from .prompts import PromptLoader, load_system_prompt
from .session import Message, Session, SessionManager
from .tools import Tool, ToolContext, ToolRegistry, build_default_registry
from .tracing import Tracer, TraceEvent

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # runtime
    "MinimalAgent",
    "AgentResult",
    "ToolInvocation",
    "AgentConfig",
    "build_agent",
    # session / context
    "SessionManager",
    "Session",
    "Message",
    "ContextManager",
    # tools
    "ToolRegistry",
    "Tool",
    "ToolContext",
    "build_default_registry",
    # llm
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "OpenAICompatibleClient",
    "ScriptedLLMClient",
    "OfflineMockClient",
    # parser / prompt
    "Parser",
    "Answer",
    "ToolCall",
    "parse_llm_output",
    "PromptLoader",
    "load_system_prompt",
    # observability
    "Tracer",
    "TraceEvent",
    # errors
    "MiniAgentError",
    "LLMError",
    "ParseError",
    "ToolNotFoundError",
    "ToolValidationError",
    "ToolExecutionError",
]
