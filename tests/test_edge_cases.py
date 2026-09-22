"""长尾边界用例：并发、特殊字符、多轮混合追问、大 payload、协议边界。

这些是「主流程已经跑通之后」仍然会咬人的场景。
"""

from __future__ import annotations

import asyncio
import json

from miniagent import AgentConfig, MinimalAgent, build_default_registry
from miniagent.llm import OfflineMockClient, ScriptedLLMClient
from miniagent.session import SessionManager
from miniagent.tools import Tool, ToolRegistry
from miniagent.tracing import Tracer


def make_agent(client=None, config: AgentConfig | None = None, registry=None) -> MinimalAgent:
    cfg = config or AgentConfig.from_env(provider="mock", model="offline-mock", console_trace=False)
    return MinimalAgent(
        client or OfflineMockClient(),
        registry or build_default_registry(),
        SessionManager(max_turn=cfg.max_tool_turns),
        config=cfg,
        tracer=Tracer.in_memory(),
    )


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------- 并发
def test_concurrent_runs_on_same_session_do_not_crash() -> None:
    """同一 session 并发提问（真实前端可能双击发送）不能抛异常或丢消息。"""
    agent = make_agent()

    async def scenario():
        return await asyncio.gather(
            agent.run("上海今天天气", session_id="userA::win1"),
            agent.run("北京今天天气", session_id="userA::win1"),
            agent.run("123+456*7", session_id="userA::win1"),
        )

    results = run(scenario())
    assert len(results) == 3
    session = agent.session_mgr.get_or_create("userA::win1")
    # 三条用户消息都在，且都有对应的 assistant 回答
    users = [m for m in session.messages if m.role == "user"]
    assistants = [m for m in session.messages if m.role == "assistant" and not m.content.startswith('{"type"')]
    assert len(users) == 3
    assert len(assistants) == 3


def test_many_windows_are_all_isolated() -> None:
    agent = make_agent()
    cities = ["北京", "上海", "深圳", "广州", "杭州", "成都", "西安", "南京"]
    for idx, city in enumerate(cities):
        run(agent.run(f"{city}今天天气", user_id="userA", window_id=f"win{idx}"))
    sessions = agent.session_mgr.list_sessions()
    assert len(sessions) == len(cities)
    for idx, city in enumerate(cities):
        session = agent.session_mgr.get_or_create(f"userA::win{idx}")
        assert city in session.transcript()
        # 不能出现其他窗口的城市
        for other in cities:
            if other != city:
                assert f"{other} 今天" not in session.transcript()


# --------------------------------------------------------------------- 特殊字符
def test_user_input_with_json_like_text_is_not_confused() -> None:
    """用户直接粘贴一段 JSON，不应被误解为工具调用。"""
    client = ScriptedLLMClient(['{"type":"answer","content":"这是一段 JSON 配置。"}'])
    agent = make_agent(client=client)
    result = run(agent.run('{"name":"demo","value":1}', session_id="userA::win1"))
    assert result.tool_calls == []
    assert result.answer == "这是一段 JSON 配置。"
    session = agent.session_mgr.get_or_create("userA::win1")
    assert session.messages[0].content == '{"name":"demo","value":1}'


def test_answer_content_with_special_characters_round_trips() -> None:
    tricky = '包含 "引号"、\\反斜杠\\、{花括号}、换行\n与 emoji 🎯 的回答'
    client = ScriptedLLMClient([json.dumps({"type": "answer", "content": tricky}, ensure_ascii=False)])
    agent = make_agent(client=client)
    result = run(agent.run("测试特殊字符", session_id="userA::win1"))
    assert result.answer == tricky


def test_tool_argument_with_quotes_and_newlines() -> None:
    payload = "第一行\n第二行 \"带引号\""
    registry = build_default_registry()
    registry.register(
        Tool(
            name="echo",
            description="原样返回 text",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            handler=lambda text: text,
        )
    )
    client = ScriptedLLMClient(
        [
            json.dumps({"type": "tool_call", "tool_name": "echo", "arguments": {"text": payload}}, ensure_ascii=False),
            '{"type":"answer","content":"已回显。"}',
        ]
    )
    agent = make_agent(client=client, registry=registry)
    result = run(agent.run("回显", session_id="userA::win1"))
    assert result.tool_calls[0].ok is True
    assert result.tool_calls[0].result == payload


def test_huge_tool_result_is_truncated_in_context_not_in_session() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="big",
            description="返回超长文本",
            parameters={"type": "object", "properties": {}},
            handler=lambda: "X" * 20000,
        )
    )
    client = ScriptedLLMClient(['{"type":"tool_call","tool_name":"big","arguments":{}}', '{"type":"answer","content":"已处理。"}'])
    agent = make_agent(client=client, registry=registry)
    run(agent.run("跑大工具", session_id="userA::win1"))

    session = agent.session_mgr.get_or_create("userA::win1")
    raw = next(m.content for m in session.messages if m.role == "tool_result")
    assert len(raw) == 20000                      # session 里保留完整结果
    build = agent.context_mgr.build(session, "SYS")
    rendered = "\n".join(m["content"] for m in build.messages)
    assert "已截断" in rendered                    # 进 prompt 的被截断
    assert len(rendered) < 20000


# --------------------------------------------------------------------- 多轮混合追问
def test_mixed_multi_turn_flow() -> None:
    """计算 → 天气 → 追问 → 换城市追问 → 再来一次计算，五轮上下文连续。"""
    agent = make_agent()
    session_id = "userA::win1"
    answers = []
    for question in ["123+456*7", "上海今天天气", "明天呢", "那北京呢", "(1+2)*3"]:
        answers.append(run(agent.run(question, session_id=session_id)).answer)

    assert "3315" in answers[0]
    assert "上海" in answers[1]
    assert "上海" in answers[2] and "明天" in answers[2]
    assert "北京" in answers[3]
    assert "9" in answers[4]

    session = agent.session_mgr.get_or_create(session_id)
    assert session.turn_count == 5
    assert session.compress_count == 0          # 五轮还没触发裁剪（阈值 40 条）


def test_followup_after_new_topic_does_not_reuse_stale_city() -> None:
    """换话题后再问“明天呢”，不应错误地沿用上上轮的城市。"""
    agent = make_agent()
    session_id = "userA::win1"
    run(agent.run("上海今天天气", session_id=session_id))
    run(agent.run("123+456*7", session_id=session_id))
    result = run(agent.run("明天呢", session_id=session_id))
    # 离线规则客户端会沿用最近一次 weather 的城市（上海），但绝不能串到 calculator
    assert "calculator" not in result.used_tools


# --------------------------------------------------------------------- 协议边界
def test_answer_with_empty_content_is_retried() -> None:
    client = ScriptedLLMClient(['{"type":"answer","content":""}', '{"type":"answer","content":"第二次有内容"}'])
    agent = make_agent(client=client)
    result = run(agent.run("你好", session_id="userA::win1"))
    assert result.answer == "第二次有内容"
    session = agent.session_mgr.get_or_create("userA::win1")
    assert any(m.role == "error" for m in session.messages)


def test_tool_call_with_empty_arguments_uses_defaults() -> None:
    registry = build_default_registry()
    client = ScriptedLLMClient(
        [
            '{"type":"tool_call","tool_name":"weather","arguments":{}}',     # 缺 city → 校验失败
            '{"type":"tool_call","tool_name":"weather","arguments":{"city":"上海"}}',
            '{"type":"answer","content":"上海多云。"}',
        ]
    )
    agent = make_agent(client=client, registry=registry)
    result = run(agent.run("上海天气", session_id="userA::win1"))
    assert [c.ok for c in result.tool_calls] == [True]
    assert result.tool_calls[0].arguments["date"] == "今天"


def test_multiple_json_objects_takes_the_first_valid_one() -> None:
    """模型一次吐出多个 JSON 对象时，取**第一个合法**的（并记录 repaired 痕迹）。

    这是刻意的设计：第一个才是模型的真实决策，后续大概率是「自言自语」的重复输出。
    对应地，第一条是 answer 就直接结束循环；是 tool_call 才继续走工具链。
    """
    client = ScriptedLLMClient(
        [
            '{"type":"answer","content":"先想想"}\n{"type":"tool_call","tool_name":"calculator","arguments":{"expression":"2+2"}}',
            '{"type":"answer","content":"2+2 = 4。"}',
        ]
    )
    agent = make_agent(client=client)
    result = run(agent.run("2+2", session_id="userA::win1"))
    assert result.answer == "先想想"                 # 取第一个合法 JSON
    assert result.tool_calls == []                   # 不执行后面那个 tool_call
    assert len(client.calls) == 1                    # 一轮就结束


def test_tool_call_first_then_answer_is_used() -> None:
    """反过来（第一个是 tool_call）就应该正常走工具链。"""
    client = ScriptedLLMClient(
        [
            '{"type":"tool_call","tool_name":"calculator","arguments":{"expression":"2+2"}}\n{"type":"answer","content":"这是模型多余的草稿"}',
            '{"type":"answer","content":"2+2 = 4。"}',
        ]
    )
    agent = make_agent(client=client)
    result = run(agent.run("2+2", session_id="userA::win1"))
    assert "calculator" in result.used_tools
    assert result.answer == "2+2 = 4。"


def test_tool_result_appended_even_when_tool_fails() -> None:
    """工具失败的轮次也必须留下 tool_result，否则上下文里会出现「无结果的调用」。"""
    client = ScriptedLLMClient(
        [
            '{"type":"tool_call","tool_name":"calculator","arguments":{"expression":"1++"}}',
            '{"type":"answer","content":"表达式有问题，我直接说结果：请检查输入。"}',
        ]
    )
    agent = make_agent(client=client)
    run(agent.run("算一下", session_id="userA::win1"))
    session = agent.session_mgr.get_or_create("userA::win1")
    assert any(m.role == "tool_result" for m in session.messages)
    assert any(m.role == "error" for m in session.messages)


# --------------------------------------------------------------------- 配置边界
def test_tiny_context_budget_does_not_crash() -> None:
    config = AgentConfig.from_env(
        provider="mock",
        console_trace=False,
        max_context_tokens=64,
        max_context_messages=1,
        keep_recent_messages=1,
    )
    agent = make_agent(config=config)
    for question in ["北京今天天气", "上海今天天气", "深圳今天天气"]:
        result = run(agent.run(question, session_id="userA::win1"))
        assert result.answer


def test_session_manager_reuse_across_agents() -> None:
    """共享 SessionManager 的两个 Agent 实例看到同一份历史（多实例部署的最小验证）。

    第二个实例的离线客户端是**全新对象**，必须能从上下文重建状态，
    否则「明天呢」这种省略式追问会失去指代对象。
    """
    manager = SessionManager()
    config = AgentConfig.from_env(provider="mock", console_trace=False)
    first = MinimalAgent(OfflineMockClient(), build_default_registry(), manager, config=config, tracer=Tracer.in_memory())
    run(first.run("上海今天天气", session_id="userA::win1"))

    second = MinimalAgent(OfflineMockClient(), build_default_registry(), manager, config=config, tracer=Tracer.in_memory())
    session = second.session_mgr.get_or_create("userA::win1")
    assert "上海" in session.transcript()

    result = run(second.run("明天呢", session_id="userA::win1"))
    assert "weather" in result.used_tools
    assert result.tool_calls[0].arguments["city"] == "上海"     # 从历史里重建出城市
    assert result.tool_calls[0].arguments["date"] == "明天"


def test_agent_reuse_after_many_runs_keeps_state_consistent() -> None:
    agent = make_agent()
    for idx in range(15):
        run(agent.run(f"第{idx}问：北京今天天气", session_id="userA::win1"))
    session = agent.session_mgr.get_or_create("userA::win1")
    assert session.turn_count == 15
    assert session.size == 60                     # 15 轮 × 4 条
    assert session.compress_count > 0             # 中途触发过裁剪
    # 裁剪不会破坏会话结构：最后一条一定是 assistant
    assert session.messages[-1].role == "assistant"
