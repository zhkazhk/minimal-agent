"""LLM 客户端抽象层。

- `LLMClient`：所有客户端的接口（只有一个 `chat`）。
- `LLMRequest` / `LLMResponse`：请求与响应的统一结构（带 usage、耗时、模型名）。
- `OpenAICompatibleClient`：真实调用 OpenAI 兼容 `/chat/completions`（DeepSeek / 通义 / vLLM / Ollama …）。
- `ScriptedLLMClient`：按剧本返回预设输出，用来**确定性复现** 7 个测试用例。
- `OfflineMockClient`：规则版「假 LLM」，无 API Key 时也能完整跑通 Agent 循环。

为什么自己写 HTTP 而不直接用 openai SDK：
1. 交付要求「无重型框架依赖」，`urllib` 是标准库，零安装即可跑；
2. Agent 的核心价值在**主循环 + 解析 + 上下文管理**，LLM 调用只是一次 POST，
   把这一层做薄反而更透明（也顺便练了重试/超时/异常处理）。
"""

from .base import LLMClient, LLMRequest, LLMResponse
from .openai_compatible import OpenAICompatibleClient, build_client
from .offline import OfflineMockClient, ScriptedLLMClient

__all__ = [
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "OpenAICompatibleClient",
    "build_client",
    "OfflineMockClient",
    "ScriptedLLMClient",
]
