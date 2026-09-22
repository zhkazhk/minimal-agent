"""OpenAI 兼容 LLM 客户端（DeepSeek / OpenAI / 通义 / Moonshot / vLLM / Ollama / one-api …）。

零依赖实现：
- 默认使用标准库 `urllib.request`（阻塞调用放进线程池，不阻塞事件循环）；
- 若环境里装了 `httpx` 或 `aiohttp`，可通过 `backend=` 切换到原生 async 传输；
- 自动重试（指数退避 + 抖动）、超时、错误分类，全部抛 `LLMError`。

可测试性：`poster=` 允许注入自定义传输层，因此**不需要网络也能测重试/异常/多厂商响应解析**。
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

from ..errors import LLMError
from ..utils import truncate
from .base import LLMClient, LLMRequest, LLMResponse

#: 传输层签名：(url, headers, payload, timeout) -> (status_code, body_text)
Poster = Callable[[str, dict[str, str], dict[str, Any], float], tuple[int, str]]

DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "siliconflow": "https://api.siliconflow.cn/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
}

RETRIABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 522, 524}


def _stdlib_post(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> tuple[int, str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310 - url 由使用者配置
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:  # 4xx/5xx：仍要读出 body，便于错误定位
        raw = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        return exc.code, raw
    except urllib.error.URLError as exc:
        raise LLMError(f"网络异常: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LLMError(f"请求超时（{timeout}s）") from exc


def extract_text(payload: Any) -> tuple[str, str]:
    """从各家响应结构里抠出正文。返回 (text, finish_reason)。

    依次兼容：
    1. OpenAI 风格 `choices[0].message.content`（含数组分片）
    2. OpenAI 流式残留 `choices[0].delta.content`
    3. DeepSeek-R1 风格 `message.reasoning_content`（正文为空时兜底）
    4. Anthropic 风格 `content[].text`
    5. Ollama 风格 `message.content` / `response`
    """
    if isinstance(payload, str):
        return payload, ""
    if not isinstance(payload, dict):
        return "", ""

    finish = ""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0] if isinstance(choices[0], dict) else {}
        finish = str(choice.get("finish_reason") or "")
        message = choice.get("message") if isinstance(choice.get("message"), dict) else choice.get("delta")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, list):  # 多模态分片
                text = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
                if text:
                    return text, finish
            if isinstance(content, str) and content.strip():
                return content, finish
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                return reasoning, finish or "reasoning_only"
        text = choice.get("text")
        if isinstance(text, str):
            return text, finish

    content = payload.get("content")
    if isinstance(content, list):  # Anthropic
        text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if text:
            return text, str(payload.get("stop_reason") or "")
    if isinstance(content, str) and content.strip():
        return content, finish

    for key in ("response", "output_text", "text"):  # Ollama / 简易网关
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value, finish
    message = payload.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"], finish
    return "", finish


def extract_error_message(status: int, body: str) -> str:
    """从错误响应里提取人类可读信息。"""
    try:
        data = json.loads(body)
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or error)
            if isinstance(error, str):
                return error
            for key in ("message", "msg", "detail", "error_msg"):
                if data.get(key):
                    return str(data[key])
    except Exception:
        pass
    return truncate(body, 300)


class OpenAICompatibleClient(LLMClient):
    """对 `/chat/completions` 的最小可用封装。"""

    name = "openai-compatible"

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        provider: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        extra_body: Optional[dict[str, Any]] = None,
        default_headers: Optional[dict[str, str]] = None,
        backend: str = "stdlib",
        poster: Optional[Poster] = None,
    ) -> None:
        provider = provider or os.getenv("MINIAGENT_PROVIDER", "")
        resolved_base = (
            base_url
            or os.getenv("MINIAGENT_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or DEFAULT_BASE_URLS.get(provider, "")
        )
        if not resolved_base:
            raise LLMError(
                "缺少 base_url：请设置环境变量 MINIAGENT_BASE_URL（OpenAI 兼容地址，形如 https://api.deepseek.com/v1），"
                "或使用 provider=openai/deepseek/dashscope/ollama…"
            )
        self.base_url = resolved_base.rstrip("/")
        self.api_key = api_key if api_key is not None else (
            os.getenv("MINIAGENT_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or ""
        )
        self.model = model
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.retry_base_delay = retry_base_delay
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = dict(extra_body or {})
        self.default_headers = dict(default_headers or {})
        self.backend = backend
        self._poster = poster
        self.name = provider or self.name
        self.call_count = 0

    # ------------------------------------------------------------------ 请求
    def build_payload(self, request: LLMRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model or self.model,
            "messages": request.messages,
            "stream": False,
        }
        # 请求级未显式指定时才回退到客户端配置（None 表示「用默认值」）
        temperature = self.temperature if request.temperature is None else request.temperature
        max_tokens = self.max_tokens if request.max_tokens is None else request.max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(self.extra_body)
        payload.update(request.extra)
        return payload

    def build_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "User-Agent": "minimal-agent/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.default_headers)
        return headers

    @property
    def endpoint(self) -> str:
        base = self.base_url
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    async def chat(self, request: LLMRequest) -> LLMResponse:
        payload = self.build_payload(request)
        headers = self.build_headers()
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            started = asyncio.get_event_loop().time()
            try:
                status, body = await self._send(headers, payload)
            except LLMError as exc:      # 网络层异常 → 可重试
                last_error = exc
                status, body = 0, ""

            latency_ms = int((asyncio.get_event_loop().time() - started) * 1000)
            if status and 200 <= status < 300:
                self.call_count += 1
                return self._parse_success(body, latency_ms, payload)

            error_text = extract_error_message(status, body) if status else str(last_error)
            retriable = status in RETRIABLE_STATUS or status == 0
            hint = ""
            if status in (401, 403):
                hint = "（API Key 无效或权限不足，请检查 MINIAGENT_API_KEY）"
            elif status == 404:
                hint = "（接口路径不存在，请检查 MINIAGENT_BASE_URL 是否需要带 /v1）"
            elif status == 400:
                hint = "（请求体被拒绝，可能是模型名不对或消息格式不合法）"

            if not retriable or attempt >= self.max_retries:
                raise LLMError(
                    f"LLM 调用失败: HTTP {status or 'N/A'} {error_text}{hint}",
                    detail={"attempts": attempt + 1, "model": payload.get("model"), "endpoint": self.endpoint},
                )

            delay = self.retry_base_delay * (2**attempt) + random.uniform(0, 0.3)
            await asyncio.sleep(min(delay, 15.0))

        raise LLMError(f"LLM 调用失败（已重试 {self.max_retries} 次）: {last_error}")

    async def _send(self, headers: dict[str, str], payload: dict[str, Any]) -> tuple[int, str]:
        poster = self._poster
        if poster is not None:
            return await asyncio.to_thread(poster, self.endpoint, headers, payload, self.timeout)
        if self.backend == "httpx":
            return await self._send_httpx(headers, payload)
        if self.backend == "aiohttp":
            return await self._send_aiohttp(headers, payload)
        return await asyncio.to_thread(_stdlib_post, self.endpoint, headers, payload, self.timeout)

    async def _send_httpx(self, headers: dict[str, str], payload: dict[str, Any]) -> tuple[int, str]:
        try:
            import httpx  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise LLMError("backend='httpx' 但未安装 httpx，请 `pip install httpx` 或改用 backend='stdlib'") from exc
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self.endpoint, headers=headers, json=payload)
                return resp.status_code, resp.text
        except httpx.HTTPError as exc:  # pragma: no cover
            raise LLMError(f"httpx 网络异常: {exc}") from exc

    async def _send_aiohttp(self, headers: dict[str, str], payload: dict[str, Any]) -> tuple[int, str]:
        try:
            import aiohttp  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise LLMError("backend='aiohttp' 但未安装 aiohttp") from exc
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.endpoint, headers=headers, json=payload) as resp:
                    return resp.status, await resp.text()
        except Exception as exc:  # pragma: no cover
            raise LLMError(f"aiohttp 网络异常: {exc}") from exc

    def _parse_success(self, body: str, latency_ms: int, payload: dict[str, Any]) -> LLMResponse:
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"LLM 返回的不是合法 JSON: {exc}",
                detail={"body_preview": truncate(body, 300)},
            ) from exc
        text, finish = extract_text(data)
        usage = data.get("usage") if isinstance(data, dict) else None
        if not isinstance(usage, dict):
            usage = {}
        if not text.strip():
            raise LLMError(
                "LLM 返回了空内容",
                hint="可能是 max_tokens 太小被截断，或触发了内容过滤。",
                detail={"finish_reason": finish, "usage": usage},
            )
        model = str(data.get("model") or payload.get("model") or "")
        return LLMResponse(
            content=text,
            model=model,
            usage=usage,
            finish_reason=finish,
            latency_ms=latency_ms,
            raw=data,
        )


def build_client(config: Any = None, **overrides: Any) -> LLMClient:
    """根据配置构造客户端。

    provider 取值：
    - `mock` / `offline` → 规则版离线客户端（无需 API Key，用于本地 demo）
    - 其他 → OpenAI 兼容客户端
    """
    if config is None:
        return OpenAICompatibleClient(**overrides)

    provider = (getattr(config, "provider", "") or "").lower()
    if provider in ("mock", "offline", "fake"):
        from .offline import OfflineMockClient

        return OfflineMockClient()
    if provider == "scripted":
        from .offline import ScriptedLLMClient

        return ScriptedLLMClient(getattr(config, "script", []) or [])

    return OpenAICompatibleClient(
        base_url=getattr(config, "base_url", None) or None,
        api_key=getattr(config, "api_key", None),
        model=getattr(config, "model", "deepseek-chat"),
        provider=provider or None,
        timeout=getattr(config, "timeout", 60.0),
        max_retries=getattr(config, "max_retries", 3),
        temperature=getattr(config, "temperature", 0.2),
        max_tokens=getattr(config, "max_tokens", 1024),
        extra_body=getattr(config, "extra_body", None),
        backend=getattr(config, "http_backend", "stdlib"),
    )
