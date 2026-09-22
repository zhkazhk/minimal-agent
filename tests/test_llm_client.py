"""LLM 客户端测试：OpenAI 兼容层（注入假传输，无需网络）、剧本客户端、离线规则客户端。"""

from __future__ import annotations

import asyncio
import json

import pytest

from miniagent.errors import LLMError
from miniagent.llm import (
    LLMRequest,
    LLMResponse,
    OfflineMockClient,
    OpenAICompatibleClient,
    ScriptedLLMClient,
    build_client,
)
from miniagent.llm.openai_compatible import extract_error_message, extract_text


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------- 响应解析
def test_extract_openai_style() -> None:
    payload = {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}
    assert extract_text(payload) == ("hello", "stop")


def test_extract_anthropic_style() -> None:
    payload = {"content": [{"type": "text", "text": "hi there"}], "stop_reason": "end_turn"}
    assert extract_text(payload) == ("hi there", "end_turn")


def test_extract_ollama_style() -> None:
    assert extract_text({"message": {"content": "本地模型"}})[0] == "本地模型"
    assert extract_text({"response": "ollama generate"})[0] == "ollama generate"


def test_extract_reasoning_fallback() -> None:
    payload = {"choices": [{"message": {"content": "", "reasoning_content": "思考内容"}}]}
    text, finish = extract_text(payload)
    assert text == "思考内容"
    assert finish == "reasoning_only"


def test_extract_multimodal_parts() -> None:
    payload = {"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}]}
    assert extract_text(payload)[0] == "ab"


def test_extract_error_message_shapes() -> None:
    assert extract_error_message(400, '{"error":{"message":"bad model"}}') == "bad model"
    assert extract_error_message(429, '{"message":"rate limited"}') == "rate limited"
    assert "oops" in extract_error_message(500, "oops")


# --------------------------------------------------------------------- HTTP 层
def _client(**kwargs) -> OpenAICompatibleClient:
    kwargs.setdefault("base_url", "https://example.com/v1")
    kwargs.setdefault("api_key", "sk-test")
    kwargs.setdefault("model", "m")
    return OpenAICompatibleClient(**kwargs)


def test_payload_and_headers() -> None:
    client = _client(temperature=0.7, max_tokens=99, extra_body={"top_p": 0.9})
    request = LLMRequest(messages=[{"role": "user", "content": "hi"}])
    payload = client.build_payload(request)
    assert payload["model"] == "m"
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 99
    assert payload["top_p"] == 0.9
    assert payload["stream"] is False
    headers = client.build_headers()
    assert headers["Authorization"] == "Bearer sk-test"
    assert client.endpoint == "https://example.com/v1/chat/completions"


def test_endpoint_not_double_suffixed() -> None:
    client = _client(base_url="https://example.com/v1/chat/completions")
    assert client.endpoint == "https://example.com/v1/chat/completions"


def test_request_none_sampling_uses_client_defaults() -> None:
    """请求未显式指定采样参数时，必须回退到客户端配置（None 语义）。"""
    client = _client(temperature=0.9, max_tokens=777)
    payload = client.build_payload(LLMRequest(messages=[{"role": "user", "content": "hi"}]))
    assert payload["temperature"] == 0.9
    assert payload["max_tokens"] == 777


def test_request_level_sampling_overrides_client() -> None:
    client = _client(temperature=0.9, max_tokens=777)
    payload = client.build_payload(
        LLMRequest(messages=[{"role": "user", "content": "hi"}], temperature=0.1, max_tokens=32)
    )
    assert payload["temperature"] == 0.1
    assert payload["max_tokens"] == 32


def test_request_extra_overrides_extra_body() -> None:
    client = _client(extra_body={"top_p": 0.9, "seed": 1})
    payload = client.build_payload(LLMRequest(messages=[], extra={"seed": 42}))
    assert payload["top_p"] == 0.9
    assert payload["seed"] == 42


def test_successful_call_via_injected_transport() -> None:
    calls: list[dict] = []

    def poster(url, headers, payload, timeout):
        calls.append({"url": url, "payload": payload})
        body = json.dumps({"choices": [{"message": {"content": '{"type":"answer","content":"ok"}'}}], "usage": {"total_tokens": 12}, "model": "m"})
        return 200, body

    client = _client(poster=poster)
    response = run(client.chat(LLMRequest(messages=[{"role": "user", "content": "hi"}])))
    assert isinstance(response, LLMResponse)
    assert response.content == '{"type":"answer","content":"ok"}'
    assert response.total_tokens == 12
    assert len(calls) == 1
    assert calls[0]["payload"]["messages"][0]["content"] == "hi"


def test_retry_then_success() -> None:
    attempts = {"n": 0}

    def poster(url, headers, payload, timeout):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return 503, '{"error":{"message":"overloaded"}}'
        return 200, json.dumps({"choices": [{"message": {"content": "finally"}}]})

    client = _client(poster=poster, max_retries=3, retry_base_delay=0.0)
    response = run(client.chat(LLMRequest(messages=[{"role": "user", "content": "hi"}])))
    assert response.content == "finally"
    assert attempts["n"] == 3


def test_non_retriable_error_fails_fast() -> None:
    attempts = {"n": 0}

    def poster(url, headers, payload, timeout):
        attempts["n"] += 1
        return 401, '{"error":{"message":"invalid api key"}}'

    client = _client(poster=poster, max_retries=3, retry_base_delay=0.0)
    with pytest.raises(LLMError) as excinfo:
        run(client.chat(LLMRequest(messages=[{"role": "user", "content": "hi"}])))
    assert "401" in str(excinfo.value)
    assert "API Key" in str(excinfo.value)
    assert attempts["n"] == 1        # 401 不该重试


def test_retry_exhausted_raises() -> None:
    def poster(url, headers, payload, timeout):
        return 500, '{"error":{"message":"boom"}}'

    client = _client(poster=poster, max_retries=2, retry_base_delay=0.0)
    with pytest.raises(LLMError) as excinfo:
        run(client.chat(LLMRequest(messages=[{"role": "user", "content": "hi"}])))
    assert "boom" in str(excinfo.value)


def test_network_exception_is_retried_and_wrapped() -> None:
    attempts = {"n": 0}

    def poster(url, headers, payload, timeout):
        attempts["n"] += 1
        raise LLMError("连接被重置")

    client = _client(poster=poster, max_retries=1, retry_base_delay=0.0)
    with pytest.raises(LLMError):
        run(client.chat(LLMRequest(messages=[{"role": "user", "content": "hi"}])))
    assert attempts["n"] == 2


def test_invalid_json_body() -> None:
    client = _client(poster=lambda *a: (200, "not json at all"))
    with pytest.raises(LLMError) as excinfo:
        run(client.chat(LLMRequest(messages=[{"role": "user", "content": "hi"}])))
    assert "合法 JSON" in str(excinfo.value)


def test_empty_content_raises() -> None:
    client = _client(poster=lambda *a: (200, json.dumps({"choices": [{"message": {"content": "  "}}]})))
    with pytest.raises(LLMError) as excinfo:
        run(client.chat(LLMRequest(messages=[{"role": "user", "content": "hi"}])))
    assert "空内容" in str(excinfo.value)


def test_missing_base_url_raises_clearly() -> None:
    with pytest.raises(LLMError) as excinfo:
        OpenAICompatibleClient(base_url="", provider="unknown-provider", api_key="x")
    assert "base_url" in str(excinfo.value)


def test_provider_default_base_url(monkeypatch) -> None:
    monkeypatch.delenv("MINIAGENT_BASE_URL", raising=False)
    client = OpenAICompatibleClient(provider="deepseek", api_key="x")
    assert client.base_url == "https://api.deepseek.com/v1"


# --------------------------------------------------------------------- 剧本客户端
def test_scripted_returns_in_order() -> None:
    client = ScriptedLLMClient(["first", {"type": "answer", "content": "第二"}])
    first = run(client.chat(LLMRequest(messages=[])))
    second = run(client.chat(LLMRequest(messages=[])))
    assert first.content == "first"
    assert json.loads(second.content)["content"] == "第二"
    assert len(client.calls) == 2


def test_scripted_exhausted_raises_helpfully() -> None:
    client = ScriptedLLMClient(["only"])
    run(client.chat(LLMRequest(messages=[])))
    with pytest.raises(LLMError) as excinfo:
        run(client.chat(LLMRequest(messages=[])))
    assert "剧本已用尽" in str(excinfo.value)


def test_scripted_loop_and_default() -> None:
    looping = ScriptedLLMClient(["a"], on_exhausted="loop")
    assert run(looping.chat(LLMRequest(messages=[]))).content == "a"
    assert run(looping.chat(LLMRequest(messages=[]))).content == "a"

    defaulting = ScriptedLLMClient([], on_exhausted="default", default_response="dflt")
    assert run(defaulting.chat(LLMRequest(messages=[]))).content == "dflt"


def test_scripted_injected_exception() -> None:
    client = ScriptedLLMClient([LLMError("网络炸了")])
    with pytest.raises(LLMError):
        run(client.chat(LLMRequest(messages=[])))


def test_scripted_callable_step() -> None:
    client = ScriptedLLMClient([lambda request: '{"type":"answer","content":"动态"}'])
    assert "动态" in run(client.chat(LLMRequest(messages=[]))).content


def test_scripted_push_and_reset() -> None:
    client = ScriptedLLMClient(["a"])
    run(client.chat(LLMRequest(messages=[])))
    client.push("b")
    assert run(client.chat(LLMRequest(messages=[]))).content == "b"
    client.reset(["c"])
    assert run(client.chat(LLMRequest(messages=[]))).content == "c"


# --------------------------------------------------------------------- 离线规则客户端
def _system_prompt(session_id: str = "userA::win1") -> str:
    return f"你是 Agent。\n- 会话ID：{session_id}（用户=userA，窗口=win1）\n- 本次允许的最大工具调用轮次：10"


def test_offline_mock_calculator_flow() -> None:
    client = OfflineMockClient()
    request = LLMRequest(messages=[{"role": "system", "content": _system_prompt()}, {"role": "user", "content": "123+456*7"}])
    first = json.loads(run(client.chat(request)).content)
    assert first["type"] == "tool_call"
    assert first["tool_name"] == "calculator"

    request.messages.append({"role": "assistant", "content": json.dumps(first, ensure_ascii=False)})
    request.messages.append({"role": "user", "content": "【工具结果 calculator】\n123+456*7 = 3315"})
    second = json.loads(run(client.chat(request)).content)
    assert second["type"] == "answer"
    assert "3315" in second["content"]


def test_offline_mock_weather_flow() -> None:
    client = OfflineMockClient()
    request = LLMRequest(messages=[{"role": "system", "content": _system_prompt()}, {"role": "user", "content": "上海今天天气"}])
    first = json.loads(run(client.chat(request)).content)
    assert first["tool_name"] == "weather"
    assert first["arguments"]["city"] == "上海"

    request.messages.append({"role": "user", "content": "【工具结果 weather】\n上海 今天：多云，气温 28°C"})
    second = json.loads(run(client.chat(request)).content)
    assert second["type"] == "answer"
    assert "上海" in second["content"] and "28" in second["content"]


def test_offline_mock_followup_reuses_city() -> None:
    client = OfflineMockClient()
    request = LLMRequest(messages=[{"role": "system", "content": _system_prompt()}, {"role": "user", "content": "上海今天天气"}])
    run(client.chat(request))
    request.messages.append({"role": "user", "content": "【工具结果 weather】\n上海 今天：多云"})
    run(client.chat(request))
    # 新一轮提问
    request.messages.append({"role": "user", "content": "明天呢"})
    payload = json.loads(run(client.chat(request)).content)
    assert payload["tool_name"] == "weather"
    assert payload["arguments"]["city"] == "上海"
    assert payload["arguments"]["date"] == "明天"


def test_offline_mock_state_is_per_session() -> None:
    client = OfflineMockClient()
    a = LLMRequest(messages=[{"role": "system", "content": _system_prompt("userA::win1")}, {"role": "user", "content": "上海今天天气"}])
    b = LLMRequest(messages=[{"role": "system", "content": _system_prompt("userA::win2")}, {"role": "user", "content": "北京今天天气"}])
    run(client.chat(a))
    run(client.chat(b))
    a.messages.append({"role": "user", "content": "明天呢"})
    payload = json.loads(run(client.chat(a)).content)
    assert payload["arguments"]["city"] == "上海"        # 窗口2 的北京不能污染窗口1


def test_offline_mock_plain_answer() -> None:
    client = OfflineMockClient()
    request = LLMRequest(messages=[{"role": "system", "content": _system_prompt()}, {"role": "user", "content": "你好"}])
    payload = json.loads(run(client.chat(request)).content)
    assert payload["type"] == "answer"
    assert payload["content"]


def test_offline_mock_corrects_invalid_arguments() -> None:
    client = OfflineMockClient()
    request = LLMRequest(
        messages=[
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": "上海今天天气"},
            {"role": "user", "content": "【错误 weather】\nToolValidationError: 工具 `weather` 参数校验失败: arguments 缺少必填字段 `city`"},
        ]
    )
    payload = json.loads(run(client.chat(request)).content)
    assert payload["type"] == "tool_call"
    assert payload["arguments"]["city"] == "上海"


# --------------------------------------------------------------------- 工厂
def test_build_client_selects_offline(monkeypatch) -> None:
    class Cfg:
        provider = "mock"

    assert isinstance(build_client(Cfg()), OfflineMockClient)


def test_build_client_selects_openai_compatible() -> None:
    class Cfg:
        provider = "deepseek"
        model = "deepseek-chat"
        base_url = "https://api.deepseek.com/v1"
        api_key = "sk-x"
        timeout = 30.0
        max_retries = 1
        temperature = 0.1
        max_tokens = 512
        extra_body = None
        http_backend = "stdlib"

    assert isinstance(build_client(Cfg()), OpenAICompatibleClient)
