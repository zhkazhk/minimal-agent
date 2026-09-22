"""Agent Runtime 端到端测试：覆盖交付要求里的 7 个用例 + 关键护栏。

全部使用 `OfflineMockClient` / `ScriptedLLMClient`，**不需要 API Key、不需要网络**，可稳定复跑。
"""

from __future__ import annotations

import asyncio
import json

import pytest

from miniagent import AgentConfig, MinimalAgent, build_default_registry
from miniagent.errors import LLMError
from miniagent.llm import OfflineMockClient, ScriptedLLMClient
from miniagent.session import SessionManager
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
        client or OfflineMockClient(),
        registry or build_default_registry(),
        SessionManager(
            max_turn=cfg.max_tool_turns,
            max_context_tokens=cfg.max_context_tokens,
            keep_recent_messages=cfg.keep_recent_messages,
        ),
        config=cfg,
        tracer=Tracer.in_memory(),
    )


# ===========================================================================
# 用例 1：calculator
# ===========================================================================
def test_case1_calculator() -> None:
    agent = make_agent()
    result = run(agent.run("123+456*7", user_id="userA", window_id="win1"))
    assert result.used_tools == ["calculator"]
    assert result.stopped_reason == "answer"
    assert "3315" in result.answer
    assert result.tool_calls[0].ok is True
    assert result.tool_calls[0].arguments["expression"] == "123+456*7"


def test_case1_calculator_uses_tool_even_for_hard_math() -> None:
    agent = make_agent()
    result = run(agent.run("帮我计算 (123+456)*7/3", user_id="userA", window_id="win1"))
    assert "calculator" in result.used_tools
    assert "1351" in result.answer          # 579*7/3 = 1351


# ===========================================================================
# 用例 2：weather
# ===========================================================================
def test_case2_weather() -> None:
    agent = make_agent()
    result = run(agent.run("上海今天天气", user_id="userA", window_id="win1"))
    assert result.used_tools == ["weather"]
    assert "上海" in result.answer
    assert "°C" in result.answer
    assert result.tool_calls[0].arguments["city"] == "上海"


def test_case2_weather_mock_data_is_deterministic() -> None:
    first = run(make_agent().run("上海明天天气", user_id="userA", window_id="win1"))
    second = run(make_agent().run("上海明天天气", user_id="userA", window_id="win1"))
    assert first.answer == second.answer


# ===========================================================================
# 用例 3：无需工具直接回答
# ===========================================================================
def test_case3_direct_answer_without_tools() -> None:
    agent = make_agent()
    result = run(agent.run("你好", user_id="userA", window_id="win1"))
    assert result.tool_calls == []
    assert result.used_tools == []
    assert result.stopped_reason == "answer"
    assert result.turns_used == 0
    assert result.answer


def test_case3_history_contains_only_user_and_assistant() -> None:
    agent = make_agent()
    run(agent.run("你好", user_id="userA", window_id="win1"))
    session = agent.session_mgr.get_or_create("userA::win1")
    assert [m.role for m in session.messages] == ["user", "assistant"]


# ===========================================================================
# 用例 4：连续追问
# ===========================================================================
def test_case4_followup_reuses_session_context() -> None:
    agent = make_agent()
    session_id = "userA::win1"
    first = run(agent.run("上海今天天气", session_id=session_id))
    second = run(agent.run("明天呢", session_id=session_id))

    assert "weather" in first.used_tools
    assert "weather" in second.used_tools
    assert "上海" in second.answer          # 城市沿用
    assert "明天" in second.answer          # 日期切换
    session = agent.session_mgr.get_or_create(session_id)
    assert session.turn_count == 2
    assert session.size == 8                # 2 轮 × (user + tool_call + tool_result + assistant)


def test_case4_followup_with_explicit_city_switch() -> None:
    """追问里显式换了城市 → 必须用新城市，不能再沿用上海。"""
    agent = make_agent()
    session_id = "userA::win1"
    run(agent.run("上海今天天气", session_id=session_id))
    result = run(agent.run("那北京呢", session_id=session_id))
    assert "weather" in result.used_tools
    assert result.tool_calls[0].arguments["city"] == "北京"
    assert "北京" in result.answer and "上海" not in result.answer.split("：")[0]


def test_case4_multi_turn_history_visible_to_llm() -> None:
    client = ScriptedLLMClient(
        [
            '{"type":"answer","content":"第一答"}',
            lambda request: json.dumps(
                {
                    "type": "answer",
                    "content": "看到历史" if any("第一答" in m["content"] for m in request.messages) else "没看到历史",
                },
                ensure_ascii=False,
            ),
        ]
    )
    agent = make_agent(client=client)
    run(agent.run("第一问", session_id="userA::win1"))
    second = run(agent.run("第二问", session_id="userA::win1"))
    assert second.answer == "看到历史"


# ===========================================================================
# 用例 5：窗口隔离
# ===========================================================================
def test_case5_two_windows_isolated() -> None:
    agent = make_agent()
    calc = run(agent.run("123+456*7", user_id="userA", window_id="win1"))
    weather = run(agent.run("上海今天天气", user_id="userA", window_id="win2"))

    win1 = agent.session_mgr.get_or_create("userA::win1")
    win2 = agent.session_mgr.get_or_create("userA::win2")
    assert calc.used_tools == ["calculator"]
    assert weather.used_tools == ["weather"]
    assert "calculator" in win1.transcript() and "calculator" not in win2.transcript()
    assert "weather" in win2.transcript() and "weather" not in win1.transcript()
    assert "3315" not in win2.transcript()


def test_case5_different_users_same_window_isolated() -> None:
    agent = make_agent()
    run(agent.run("上海今天天气", user_id="userA", window_id="win1"))
    run(agent.run("123+456*7", user_id="userB", window_id="win1"))
    sessions = {s.session_id: s for s in agent.session_mgr.list_sessions()}
    assert set(sessions) == {"userA::win1", "userB::win1"}
    assert "weather" in sessions["userA::win1"].transcript()
    assert "calculator" in sessions["userB::win1"].transcript()


def test_case5_parallel_runs_are_isolated() -> None:
    """并发跑两个窗口（真实场景就是多窗口同时提问）。"""
    agent = make_agent()

    async def scenario() -> list:
        return await asyncio.gather(
            agent.run("123+456*7", session_id="userA::win1"),
            agent.run("上海今天天气", session_id="userA::win2"),
        )

    first, second = run(scenario())
    assert first.used_tools == ["calculator"]
    assert second.used_tools == ["weather"]
    assert "3315" in first.answer
    assert "上海" in second.answer


# ===========================================================================
# 用例 6：工具参数错误 → 回灌 → 修正重试
# ===========================================================================
def test_case6_schema_violation_caught_then_retried() -> None:
    client = ScriptedLLMClient(
        [
            '{"type":"tool_call","tool_name":"weather","arguments":{"date":"今天"}}',           # 缺 city
            '{"type":"tool_call","tool_name":"weather","arguments":{"city":"上海"}}',            # 修正
            '{"type":"answer","content":"上海今天多云（mock）。"}',
        ]
    )
    agent = make_agent(client=client)
    result = run(agent.run("上海今天天气", session_id="userA::win1"))
    session = agent.session_mgr.get_or_create("userA::win1")

    assert result.stopped_reason == "answer"
    assert len(result.tool_calls) == 1                     # 只有修正后的那次真正执行
    assert result.tool_calls[0].ok is True
    errors = [m for m in session.messages if m.role == "error"]
    assert len(errors) == 1
    assert "缺少必填字段" in errors[0].content
    assert "city" in errors[0].content


def test_case6_unknown_tool_fed_back() -> None:
    client = ScriptedLLMClient(
        [
            '{"type":"tool_call","tool_name":"translate","arguments":{"text":"hi"}}',
            '{"type":"tool_call","tool_name":"calculator","arguments":{"expression":"1+1"}}',
            '{"type":"answer","content":"结果是 2。"}',
        ]
    )
    agent = make_agent(client=client)
    result = run(agent.run("1+1 等于几", session_id="userA::win1"))
    session = agent.session_mgr.get_or_create("userA::win1")
    assert "calculator" in result.used_tools
    assert any("不存在" in m.content for m in session.messages if m.role == "error")


def test_case6_tool_internal_error_fed_back() -> None:
    """工具内部报错（例如表达式不合法）→ 错误进上下文 → LLM 换参数重试。"""
    client = ScriptedLLMClient(
        [
            '{"type":"tool_call","tool_name":"calculator","arguments":{"expression":"1++*2"}}',
            '{"type":"tool_call","tool_name":"calculator","arguments":{"expression":"1+2"}}',
            '{"type":"answer","content":"结果是 3。"}',
        ]
    )
    agent = make_agent(client=client)
    result = run(agent.run("帮我算个数", session_id="userA::win1"))
    assert [c.ok for c in result.tool_calls] == [False, True]
    session = agent.session_mgr.get_or_create("userA::win1")
    assert any(m.role == "error" and "语法错误" in m.content for m in session.messages)


def test_case6_invalid_json_then_retry() -> None:
    """LLM 输出「想调工具但写坏了」的文本 → 解析失败回灌 → 第二次修正成功。"""
    client = ScriptedLLMClient(
        [
            '好的，我来计算：{"type":"tool_call","tool_name":"calculator","arguments":{"expression":"123+456*7"',   # 缺右括号
            '{"type":"tool_call","tool_name":"calculator","arguments":{"expression":"123+456*7"}}',
            '{"type":"answer","content":"123+456*7 = 3315。"}',
        ]
    )
    agent = make_agent(client=client)
    result = run(agent.run("123+456*7", session_id="userA::win1"))
    assert "3315" in result.answer
    session = agent.session_mgr.get_or_create("userA::win1")
    assert sum(1 for m in session.messages if m.role == "error") == 1


def test_case6_numbers_instead_of_string_is_rejected() -> None:
    """`expression: 123` （数字而非字符串）必须被拦下，而不是被静默转成 "123"。"""
    client = ScriptedLLMClient(
        [
            '{"type":"tool_call","tool_name":"calculator","arguments":{"expression":123}}',
            '{"type":"tool_call","tool_name":"calculator","arguments":{"expression":"123+456*7"}}',
            '{"type":"answer","content":"123+456*7 = 3315。"}',
        ]
    )
    agent = make_agent(client=client)
    result = run(agent.run("123+456*7", session_id="userA::win1"))
    session = agent.session_mgr.get_or_create("userA::win1")
    assert len(result.tool_calls) == 1                     # 只有修正后那次执行
    assert result.tool_calls[0].arguments["expression"] == "123+456*7"
    assert any(m.role == "error" and "类型应为 string" in m.content for m in session.messages)


# ===========================================================================
# 用例 7：上下文压缩
# ===========================================================================
def test_case7_compression_after_threshold() -> None:
    config = offline_config(max_context_messages=8, keep_recent_messages=4, max_context_tokens=1200)
    agent = make_agent(config=config)
    session_id = "userA::win1"
    for city in ["北京", "上海", "深圳", "广州", "杭州", "成都"]:
        run(agent.run(f"{city}今天天气", session_id=session_id))

    session = agent.session_mgr.get_or_create(session_id)
    build = agent.context_mgr.build(session, agent._system_prompt(session, session.max_turn))
    assert session.size == 24
    assert session.compress_count > 0
    assert session.dropped_messages > 0
    assert build.compressed is True
    assert build.dropped_messages > 0
    assert len(build.messages) < session.size
    assert any("上下文已裁剪" in m["content"] for m in build.messages)
    # 最新一轮必须保留
    assert any("成都" in m["content"] for m in build.messages)


def test_case7_compression_does_not_mutate_session() -> None:
    config = offline_config(max_context_messages=4, keep_recent_messages=2)
    agent = make_agent(config=config)
    session_id = "userA::win1"
    for city in ["北京", "上海", "深圳", "广州"]:
        run(agent.run(f"{city}今天天气", session_id=session_id))
    session = agent.session_mgr.get_or_create(session_id)
    assert session.size == 16                      # 原始历史完整保留，裁剪只影响单次请求
    assert any("北京" in m.content for m in session.messages)


def test_case7_latest_question_always_kept() -> None:
    """即使阈值极小，当前用户问题也不能被裁掉。"""
    config = offline_config(max_context_messages=1, keep_recent_messages=1, max_context_tokens=50)
    agent = make_agent(config=config)
    session_id = "userA::win1"
    for city in ["北京", "上海", "深圳"]:
        run(agent.run(f"{city}今天天气", session_id=session_id))
    result = run(agent.run("广州今天天气", session_id=session_id))
    assert "广州" in result.answer


# ===========================================================================
# 护栏：轮次上限 / 解析失败 / LLM 异常 / 工具超时
# ===========================================================================
def test_max_tool_turns_forces_convergence() -> None:
    client = ScriptedLLMClient(
        [
            '{"type":"tool_call","tool_name":"search","arguments":{"query":"a"}}',
            '{"type":"tool_call","tool_name":"search","arguments":{"query":"b"}}',
            '{"type":"tool_call","tool_name":"search","arguments":{"query":"c"}}',
            '{"type":"tool_call","tool_name":"search","arguments":{"query":"d"}}',
            '{"type":"answer","content":"总结完毕。"}',
        ]
    )
    agent = make_agent(client=client, config=offline_config(max_tool_turns=3))
    result = run(agent.run("帮我搜索一下", session_id="userA::win1"))
    assert len([c for c in result.tool_calls if c.ok]) == 3
    assert result.stopped_reason == "answer"
    assert result.answer == "总结完毕。"
    session = agent.session_mgr.get_or_create("userA::win1")
    assert any("轮次上限" in m.content for m in session.messages if m.role == "error")


def test_max_tool_turns_with_stubborn_model() -> None:
    """模型拒绝收敛（一直调工具）→ 到达上限后返回兜底文案，不能死循环。"""
    script = ['{"type":"tool_call","tool_name":"search","arguments":{"query":"%d"}}' % i for i in range(20)]
    agent = make_agent(client=ScriptedLLMClient(script), config=offline_config(max_tool_turns=2, max_parse_retries=2))
    result = run(agent.run("搜", session_id="userA::win1"))
    assert len(result.tool_calls) == 2
    assert result.stopped_reason in ("max_turns", "answer")
    assert result.answer


def test_repeated_identical_call_gets_hint() -> None:
    """同工具同参数连续调用 → system prompt 里应出现「重复调用」提示，避免模型原地打转。"""
    client = ScriptedLLMClient(
        [
            '{"type":"tool_call","tool_name":"search","arguments":{"query":"same"}}',
            '{"type":"tool_call","tool_name":"search","arguments":{"query":"same"}}',
            '{"type":"answer","content":"不再重复。"}',
        ]
    )
    agent = make_agent(client=client, config=offline_config(max_tool_turns=5))
    result = run(agent.run("搜", session_id="userA::win1"))
    assert result.answer == "不再重复。"
    assert len(client.calls) == 3
    # 第 3 次请求（索引 2）的 system prompt 里应带重复调用提示
    assert "完全相同的参数" in client.calls[2].messages[0]["content"]
    assert "完全相同的参数" not in client.calls[0].messages[0]["content"]


def test_parse_failure_then_give_up_gracefully() -> None:
    agent = make_agent(client=ScriptedLLMClient(["这不是 JSON", "还是不是", "依然不是", "仍然不是"]))
    agent.config.max_parse_retries = 2
    result = run(agent.run("你好", session_id="userA::win1"))
    assert result.stopped_reason in ("answer", "parse_failed")
    assert result.answer


def test_llm_error_is_graceful() -> None:
    agent = make_agent(client=ScriptedLLMClient([LLMError("模拟超时")]))
    agent.config.llm_call_retries = 1
    result = run(agent.run("你好", session_id="userA::win1"))
    assert result.stopped_reason == "llm_error"
    assert "不可用" in result.answer
    session = agent.session_mgr.get_or_create("userA::win1")
    assert any(m.role == "error" for m in session.messages)
    assert session.messages[-1].role == "assistant"


def test_tool_timeout_is_caught() -> None:
    from miniagent.tools import Tool, ToolRegistry

    async def slow(city: str = "上海") -> str:
        await asyncio.sleep(1.5)
        return "too late"

    registry = ToolRegistry()
    registry.register(Tool(name="slow_tool", description="慢工具", parameters={"type": "object", "properties": {"city": {"type": "string"}}}, handler=slow))
    client = ScriptedLLMClient(
        [
            '{"type":"tool_call","tool_name":"slow_tool","arguments":{"city":"上海"}}',
            '{"type":"answer","content":"工具太慢，我先回答。"}',
        ]
    )
    agent = make_agent(client=client, registry=registry, config=offline_config(tool_timeout=0.2))
    result = run(agent.run("试试慢工具", session_id="userA::win1"))
    assert result.tool_calls[0].ok is False
    assert "超时" in result.tool_calls[0].error


def test_unexpected_exception_becomes_readable_reply() -> None:
    class BrokenClient(OfflineMockClient):
        async def chat(self, request):        # type: ignore[override]
            raise ValueError("客户端实现有 bug")

    agent = make_agent(client=BrokenClient())
    result = run(agent.run("你好", session_id="userA::win1"))
    assert result.stopped_reason == "internal_error"
    assert result.answer


# ===========================================================================
# 追加工具 / 动态注册
# ===========================================================================
def test_custom_tool_end_to_end() -> None:
    from miniagent.tools import Tool, ToolRegistry

    registry = build_default_registry()
    registry.register(
        Tool(
            name="now",
            description="返回固定的演示时间",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=lambda: "2026-09-22 17:00:00",
        )
    )
    client = ScriptedLLMClient(
        [
            '{"type":"tool_call","tool_name":"now","arguments":{}}',
            '{"type":"answer","content":"现在是 2026-09-22 17:00:00。"}',
        ]
    )
    agent = make_agent(client=client, registry=registry)
    result = run(agent.run("现在几点", session_id="userA::win1"))
    assert result.tool_calls[0].name == "now"
    assert result.tool_calls[0].ok is True
    assert "2026" in result.answer


def test_unregistered_tool_is_rejected() -> None:
    registry = build_default_registry()
    registry.unregister("weather")
    client = ScriptedLLMClient(['{"type":"tool_call","tool_name":"weather","arguments":{"city":"上海"}}', '{"type":"answer","content":"没有天气工具。"}'])
    agent = make_agent(client=client, registry=registry)
    result = run(agent.run("上海天气", session_id="userA::win1"))
    assert result.tool_calls == []
    assert result.answer == "没有天气工具。"
    assert "calculator" in client.calls[0].messages[0]["content"]      # 工具清单里已无 weather
    assert '"weather"' not in client.calls[0].messages[0]["content"].split("调用示例")[0]


# ===========================================================================
# 系统提示词 & 会话不回写 system
# ===========================================================================
def test_system_prompt_not_persisted_but_present_in_request() -> None:
    client = ScriptedLLMClient(['{"type":"answer","content":"ok"}'])
    agent = make_agent(client=client)
    run(agent.run("你好", session_id="userA::win1"))
    session = agent.session_mgr.get_or_create("userA::win1")
    assert all(m.role != "system" for m in session.messages)
    assert client.calls[0].messages[0]["role"] == "system"
    assert "tool_call" in client.calls[0].messages[0]["content"]


def test_system_prompt_contains_tool_schemas_and_session_info() -> None:
    client = ScriptedLLMClient(['{"type":"answer","content":"ok"}'])
    agent = make_agent(client=client)
    run(agent.run("你好", session_id="userA::win7"))
    prompt = client.calls[0].messages[0]["content"]
    for name in ("calculator", "search", "weather"):
        assert name in prompt
    assert "userA::win7" in prompt
    assert "required" in prompt


def test_result_to_dict_is_serializable() -> None:
    agent = make_agent()
    result = run(agent.run("123+456*7", session_id="userA::win1"))
    payload = json.dumps(result.to_dict(), ensure_ascii=False)
    assert "calculator" in payload
