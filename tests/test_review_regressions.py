"""针对代码评审（adversarial review）发现问题的回归测试。

每一条都对应一个真实缺陷；这些测试的作用是：**同一个坑不许踩第二次**。
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from miniagent import AgentConfig, MinimalAgent, build_default_registry
from miniagent.config import DEFAULT_BASE_URLS
from miniagent.errors import LLMError
from miniagent.llm import ScriptedLLMClient
from miniagent.llm.openai_compatible import OpenAICompatibleClient
from miniagent.prompts import DEFAULT_PROMPT_PATH, PromptLoader
from miniagent.session import SessionManager
from miniagent.tools import Tool, ToolContext, ToolRegistry
from miniagent.tools.calculator import safe_eval
from miniagent.tracing import Tracer


def run(coro):
    return asyncio.run(coro)


def offline_config(**overrides) -> AgentConfig:
    base = dict(provider="mock", model="offline-mock", console_trace=False)
    base.update(overrides)
    return AgentConfig.from_env(**base)


def make_agent(client=None, config: AgentConfig | None = None, registry=None) -> MinimalAgent:
    cfg = config or offline_config()
    return MinimalAgent(
        client or ScriptedLLMClient([]),
        registry or build_default_registry(),
        # 故意用「默认参数」的 SessionManager：曾经它会用 max_turn=10 覆盖 config
        SessionManager(),
        config=cfg,
        tracer=Tracer.in_memory(),
    )


# ===========================================================================
# 缺陷 1：Agent 从未真正加载 prompts/system_prompt.md（静默退化为 fallback）
# ===========================================================================
def test_config_prompt_path_defaults_to_real_prompt_file() -> None:
    assert offline_config().prompt_path == DEFAULT_PROMPT_PATH
    assert os.path.isfile(DEFAULT_PROMPT_PATH)


def test_agent_actually_uses_markdown_system_prompt() -> None:
    """端到端：请求里的 system prompt 必须来自 prompts/system_prompt.md。"""
    client = ScriptedLLMClient(['{"type":"answer","content":"ok"}'])
    agent = make_agent(client=client)
    run(agent.run("你好", session_id="u::w"))
    system_prompt = client.calls[0].messages[0]["content"]

    real = PromptLoader(DEFAULT_PROMPT_PATH).template()
    # fallback 里没有这些分节标题；真实 prompt 有
    assert "工作方式" in system_prompt
    assert "回答要求" in system_prompt
    assert len(system_prompt) > len(PromptLoader("").render(tools=[], session_id="", current_time="", max_turns=1))
    assert "{{" not in system_prompt
    assert real[:20] in system_prompt          # 确实是文件内容渲染出来的


def test_prompt_loader_empty_path_does_not_crash() -> None:
    """空路径仍然要能工作（退化为 fallback），但不能是 Agent 的默认行为。"""
    text = PromptLoader("").template()
    assert "minimal-agent" in text


# ===========================================================================
# 缺陷 2：ToolContext / tool_deps 依赖注入是死代码（handler 拿不到 ctx）
# ===========================================================================
def test_ctx_is_injected_when_handler_declares_it() -> None:
    registry = ToolRegistry()
    seen: dict[str, object] = {}

    def with_ctx(query: str, ctx=None) -> str:
        seen["ctx"] = ctx
        return "ok"

    registry.register(
        name="with_ctx",
        description="x",
        schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        handler=with_ctx,
    )
    out = run(registry.execute("with_ctx", {"query": "hi"}, ToolContext(session_id="u::w", deps={"k": 1})))
    assert out == "ok"
    assert isinstance(seen["ctx"], ToolContext)
    assert seen["ctx"].session_id == "u::w"
    assert seen["ctx"].dep("k") == 1


def test_handler_without_ctx_is_untouched() -> None:
    registry = ToolRegistry()
    registry.register(name="plain", description="x", schema={"type": "object", "properties": {"q": {"type": "string"}}}, handler=lambda q: q.upper())
    assert run(registry.execute("plain", {"q": "abc"})) == "ABC"


def test_ctx_alias_names_supported() -> None:
    registry = ToolRegistry()
    for name in ("context", "tool_ctx"):
        namespace: dict[str, object] = {}
        exec(f"def handler(q, {name}=None):\n    return type({name}).__name__", namespace)  # noqa: S102 - 测试内动态生成
        registry.register(name=f"alias_{name}", description="x", schema={"type": "object", "properties": {"q": {"type": "string"}}}, handler=namespace["handler"])
        assert run(registry.execute(f"alias_{name}", {"q": "x"}, ToolContext())) == "ToolContext"


def test_ctx_injection_end_to_end_through_agent() -> None:
    """工具在真实循环里能拿到当前会话信息与 tracer。"""
    registry = build_default_registry()
    captured: dict[str, object] = {}

    def probe(city: str = "上海", ctx=None) -> str:
        captured["session_id"] = getattr(ctx, "session_id", None)
        captured["has_tracer"] = getattr(ctx, "tracer", None) is not None
        return "probe done"

    registry.register(name="probe", description="x", schema={"type": "object", "properties": {"city": {"type": "string"}}}, handler=probe)
    client = ScriptedLLMClient(['{"type":"tool_call","tool_name":"probe","arguments":{"city":"上海"}}', '{"type":"answer","content":"ok"}'])
    agent = make_agent(client=client, registry=registry)
    run(agent.run("probe", session_id="userA::win9"))
    assert captured["session_id"] == "userA::win9"
    assert captured["has_tracer"] is True


def test_validate_call_still_rejects_bad_args_with_ctx_handler() -> None:
    registry = ToolRegistry()
    registry.register(
        name="ctx_tool",
        description="x",
        schema={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        handler=lambda q, ctx=None: q,
    )
    with pytest.raises(Exception):
        run(registry.execute("ctx_tool", {}))


# ===========================================================================
# 缺陷 3：calculator 可被 pow() / factorial() / 序列重复打爆（DoS）
# ===========================================================================
@pytest.mark.parametrize(
    "expression",
    [
        "pow(2, 5000)",          # 绕过 ast.BinOp 的指数守卫
        "pow(9, 999999999)",
        "factorial(100000)",     # 无参数上限
        "[0]*99999999",          # 一行表达式申请几个 GB
        "2**5000",               # 原有守卫（保持）
        "9**999999999",
        "pow(2, 10, 1000)",      # 三段式 pow（取模）
    ],
)
def test_calculator_resource_guards(expression: str) -> None:
    with pytest.raises(Exception):
        safe_eval(expression)


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("pow(2, 10)", 1024),
        ("factorial(10)", 3628800),
        ("[0]*5", [0] * 5),
        ("2**10", 1024),
    ],
)
def test_calculator_still_works_within_limits(expression: str, expected) -> None:
    assert safe_eval(expression) == expected


# ===========================================================================
# 缺陷 4：tool_timeout 对同步 handler 无效（阻塞事件循环）
# ===========================================================================
def test_sync_handler_times_out() -> None:
    import time

    def slow_sync(q: str = "x") -> str:
        time.sleep(1.2)
        return "slept"

    registry = ToolRegistry()
    registry.register(name="slow_sync", description="x", schema={"type": "object", "properties": {"q": {"type": "string"}}}, handler=slow_sync)
    client = ScriptedLLMClient(['{"type":"tool_call","tool_name":"slow_sync","arguments":{"q":"a"}}', '{"type":"answer","content":"done"}'])
    agent = make_agent(client=client, config=offline_config(tool_timeout=0.2), registry=registry)
    result = run(agent.run("go", session_id="u::w"))
    assert result.tool_calls[0].ok is False
    assert "超时" in result.tool_calls[0].error


def test_sync_handler_does_not_block_other_sessions() -> None:
    """同步 handler 在线程池里跑：同一事件循环上的另一个会话仍能推进（不串行阻塞）。"""
    import time

    def slow(q: str = "x") -> str:
        time.sleep(0.6)
        return "slow done"

    registry = build_default_registry()
    registry.register(name="slow", description="x", schema={"type": "object", "properties": {"q": {"type": "string"}}}, handler=slow)
    # 剧本客户端是全局顺序返回的，因此这里用 lambda 按会话名区分回答，避免依赖调用顺序
    client = ScriptedLLMClient(
        [
            '{"type":"tool_call","tool_name":"slow","arguments":{"q":"a"}}',
            lambda request: json.dumps(
                {
                    "type": "answer",
                    "content": "slow answer"
                    if any("u::slow" in m["content"] for m in request.messages)
                    else "fast answer",
                },
                ensure_ascii=False,
            ),
            lambda request: json.dumps(
                {
                    "type": "answer",
                    "content": "slow answer"
                    if any("u::slow" in m["content"] for m in request.messages)
                    else "fast answer",
                },
                ensure_ascii=False,
            ),
        ]
    )
    agent = make_agent(client=client, registry=registry)
    agent.config.tool_timeout = 5.0

    async def scenario():
        return await asyncio.gather(
            agent.run("slow one", session_id="u::slow"),
            agent.run("fast one", session_id="u::fast"),
        )

    slow_result, fast_result = run(scenario())
    assert slow_result.answer == "slow answer"
    assert fast_result.answer == "fast answer"


# ===========================================================================
# 缺陷 5：轮次上限时留下悬空 tool_call + 误报「解析失败」
# ===========================================================================
def test_rejected_tool_call_is_not_persisted() -> None:
    config = offline_config(max_tool_turns=2)
    client = ScriptedLLMClient(
        [
            '{"type":"tool_call","tool_name":"search","arguments":{"query":"a"}}',
            '{"type":"tool_call","tool_name":"search","arguments":{"query":"b"}}',
            '{"type":"tool_call","tool_name":"search","arguments":{"query":"c"}}',
            '{"type":"answer","content":"总结完成。"}',
        ]
    )
    agent = make_agent(client=client, config=config)
    result = run(agent.run("搜索", session_id="u::w"))
    session = agent.session_mgr.get_or_create("u::w")

    calls = sum(1 for m in session.messages if m.role == "tool_call")
    results = sum(1 for m in session.messages if m.role == "tool_result")
    assert len(result.tool_calls) == 2
    assert calls == results == 2, "被轮次上限拒绝的调用不能写进上下文"
    assert result.turns_used == 2


def test_turn_limit_does_not_report_parse_failure() -> None:
    config = offline_config(max_tool_turns=1)
    client = ScriptedLLMClient(
        [
            '{"type":"tool_call","tool_name":"search","arguments":{"query":"a"}}',
            '{"type":"tool_call","tool_name":"search","arguments":{"query":"b"}}',
            '{"type":"answer","content":"总结完成。"}',
        ]
    )
    agent = make_agent(client=client, config=config)
    run(agent.run("搜索", session_id="u::w"))
    # 被拒绝的那一轮，prompt 里应是「轮次上限」而不是「输出无法解析」
    assert any("轮次上限" in call.messages[0]["content"] for call in client.calls)
    assert not any("无法解析" in call.messages[0]["content"] for call in client.calls)


def test_config_max_tool_turns_is_respected_with_default_session_manager() -> None:
    """外部传入的 SessionManager 默认 max_turn 不能覆盖 config。"""
    config = offline_config(max_tool_turns=2)
    client = ScriptedLLMClient(['{"type":"tool_call","tool_name":"search","arguments":{"query":"%d"}}' % i for i in range(6)])
    agent = make_agent(client=client, config=config)
    agent.config.max_limit_rejections = 1
    result = run(agent.run("搜索", session_id="u::w"))
    assert len(result.tool_calls) == 2
    assert result.stopped_reason == "max_turns"


# ===========================================================================
# 缺陷 6：非 answer 退出时 turns_used 恒为 0
# ===========================================================================
def test_turns_used_reported_on_all_exit_paths() -> None:
    config = offline_config(max_tool_turns=1)
    script = [
        '{"type":"tool_call","tool_name":"search","arguments":{"query":"a"}}',
        '{"type":"tool_call","tool_name":"search","arguments":{"query":"b"}}',
        '{"type":"tool_call","tool_name":"search","arguments":{"query":"c"}}',
    ]
    agent = make_agent(client=ScriptedLLMClient(script), config=config)
    agent.config.max_limit_rejections = 1
    result = run(agent.run("搜索", session_id="u::w"))
    assert result.stopped_reason == "max_turns"
    assert result.turns_used == 1                     # 执行了 1 次工具调用
    assert result.to_dict()["turns_used"] == 1
    assert not result.answer.startswith("（剧本")

    # LLM 失败路径
    failing = make_agent(client=ScriptedLLMClient([LLMError("boom")]))
    failing.config.llm_call_retries = 1
    result2 = run(failing.run("hi", session_id="u::w2"))
    assert result2.stopped_reason == "llm_error"
    assert result2.turns_used == 0


# ===========================================================================
# 缺陷 7：Tracer.in_memory() 往 CWD 写 run_*.log（并且泄漏 fd / 无限增长）
# ===========================================================================
def test_in_memory_tracer_writes_nothing() -> None:
    tracer = Tracer.in_memory()
    tracer.log("run_start", session_id="u::w", run_id="run_should_not_exist")
    tracer.log_tool_result("calculator", "x", duration_ms=1, ok=True, session_id="u::w", run_id="run_should_not_exist")
    assert not os.path.exists("run_should_not_exist.log")
    assert tracer.trace_dir == ""
    assert len(tracer.events()) == 2


def test_real_tracer_closes_run_file_after_run_end() -> None:
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".tmp-tests")
    tmp = os.path.join(root, f"tracer-{os.getpid()}")
    os.makedirs(tmp, exist_ok=True)
    tracer = Tracer(tmp, console=False)
    tracer.log("run_start", session_id="u::w", run_id="run_a")
    tracer.log("run_end", session_id="u::w", run_id="run_a")
    assert tracer._run_files == {}, "run 结束后必须关闭文件句柄"
    assert os.path.isfile(os.path.join(tmp, "traces", "run_a.log"))
    tracer.close()


def test_tracer_events_are_bounded() -> None:
    tracer = Tracer.in_memory()
    tracer._events = type(tracer._events)(maxlen=5)
    for idx in range(50):
        tracer.log("noise", session_id="u::w", run_id="r", idx=idx)
    assert len(tracer.events()) == 5


def test_tracer_exception_records_traceback() -> None:
    tracer = Tracer.in_memory()

    def boom() -> None:
        raise ValueError("inner failure")

    try:
        boom()
    except ValueError as exc:
        event = tracer.log_exception(exc, where="unit-test")
    assert "Traceback" in event.data["traceback"]
    assert "inner failure" in event.data["traceback"]


def test_missing_prompt_file_falls_back(tmp_path) -> None:
    loader = PromptLoader(os.path.join(str(tmp_path), "nope.md"))
    assert "minimal-agent" in loader.template()


# ===========================================================================
# 缺陷 9：宽松回退被引号/花括号打败
# ===========================================================================
def test_prose_with_quotes_is_not_a_parse_error() -> None:
    from miniagent.parser import Answer, Parser

    parser = Parser(None, lenient_plain_text=True)
    for text in ['他说 "hello" 然后走了。', "函数签名是 f(x) { return 1 }"]:
        parsed = parser.parse(text)
        assert isinstance(parsed, Answer)
        assert parsed.content.strip() == text


def test_real_broken_json_still_raises() -> None:
    """明确在写协议 JSON、且有问题的输入必须报错，而不是降级成"回答"。

    注意两种子情况：
    - 能补全的截断 → 补完之后语义仍然不合协议（缺字段 / type 非法）→ 报**语义错误**；
    - 补不回来的伪协议文本 → 报**格式错误**。
    两者都必须是 ParseError：**绝不能让用户收到一个"看起来像回答"的残渣**。
    """
    from miniagent.errors import ParseError
    from miniagent.parser import Parser

    parser = Parser(None, lenient_plain_text=True)
    for text in [
        '{"type": "tool_call", "tool_name": "calculator"',                          # 截断 + 缺 arguments
        '{"type":"tool_call","tool_name":"calculator","arguments":{"expression":',   # 截断在值的位置
        'tool_name: calculator, arguments: {expression: 1+1',                        # 伪协议 + 结构损坏
        '{"type": "unknown_kind", "content": "x"}',                                  # 协议字段非法
    ]:
        with pytest.raises(ParseError):
            parser.parse(text)


def test_truncated_protocol_output_gets_actionable_hint() -> None:
    """截断必须被识别为「没输出完」，而不是让模型以为只是漏填了字段。"""
    from miniagent.errors import ParseError
    from miniagent.parser import Parser

    parser = Parser(build_default_registry(), lenient_plain_text=True)
    with pytest.raises(ParseError) as excinfo:
        parser.parse('{"type":"tool_call","tool_name":"calculator","arguments":{"expression":')
    assert "没有输出完整" in str(excinfo.value)


def test_repairable_truncation_is_still_repaired() -> None:
    """反过来：能补全的截断应该被修复，而不是报错浪费一次调用。"""
    from miniagent.parser import Answer, Parser

    parser = Parser(None, lenient_plain_text=True)
    parsed = parser.parse('{"type":"answer","content":"被截断的回答"')
    assert isinstance(parsed, Answer)
    assert parsed.content == "被截断的回答"


def test_prose_mentioning_field_names_is_still_an_answer() -> None:
    """只是「提到」字段名的正常文本不能被误判成协议 JSON。"""
    from miniagent.parser import Answer, Parser

    parser = Parser(None, lenient_plain_text=True)
    for text in [
        "Use the arguments field to pass values.",
        "你需要把参数放进 arguments 里。",
        "tool_name 是工具名，arguments 是参数对象。",
    ]:
        parsed = parser.parse(text)
        assert isinstance(parsed, Answer), text


# ===========================================================================
# 缺陷 11：数组里的多个工具调用被静默丢弃
# ===========================================================================
def test_multiple_tool_calls_in_array_are_reported() -> None:
    from miniagent.errors import ParseError
    from miniagent.parser import Parser

    parser = Parser(build_default_registry())
    raw = json.dumps(
        [
            {"type": "tool_call", "tool_name": "calculator", "arguments": {"expression": "1+1"}},
            {"type": "tool_call", "tool_name": "weather", "arguments": {"city": "上海"}},
        ],
        ensure_ascii=False,
    )
    with pytest.raises(ParseError) as excinfo:
        parser.parse(raw)
    assert "多个工具调用" in str(excinfo.value)
    assert "calculator" in str(excinfo.value) and "weather" in str(excinfo.value)


def test_single_element_array_still_unwraps() -> None:
    from miniagent.parser import Parser, ToolCall

    parser = Parser(build_default_registry())
    parsed = parser.parse('[{"type":"tool_call","tool_name":"calculator","arguments":{"expression":"1+1"}}]')
    assert isinstance(parsed, ToolCall)
    assert parsed.tool_name == "calculator"


# ===========================================================================
# 缺陷 12：config.check() 对内置 provider 误报「没有 base_url」
# ===========================================================================
def test_check_does_not_warn_for_provider_default_base_url() -> None:
    config = AgentConfig(provider="deepseek", api_key="sk-test", base_url="")
    assert config.check() == []
    client = OpenAICompatibleClient(base_url="", provider="deepseek", api_key="sk-test")
    assert client.endpoint == "https://api.deepseek.com/v1/chat/completions"


def test_default_base_urls_cover_documented_providers() -> None:
    for provider in ("openai", "deepseek", "dashscope", "ollama", "vllm"):
        assert provider in DEFAULT_BASE_URLS


def test_check_still_warns_for_unknown_provider_without_base_url() -> None:
    config = AgentConfig(provider="my-private-gateway", api_key="sk-x", base_url="")
    problems = config.check()
    assert any("base_url" in item for item in problems)


# ===========================================================================
# 其它：长工具结果只截断一次，长度标注正确
# ===========================================================================
def test_long_tool_result_truncation_marker_reports_original_length() -> None:
    from miniagent.context import ContextManager
    from miniagent.session import Session

    session = Session("u::w")
    session.add_user("q")
    session.add_tool_result("search", "X" * 5000)
    build = ContextManager(tool_result_limit=100).build(session, "SYS")
    rendered = next(m["content"] for m in build.messages if "【工具结果" in m["content"])
    assert "原始 5000 字符" in rendered
    assert rendered.count("已截断") == 1
