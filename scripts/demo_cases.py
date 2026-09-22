"""7 个验收用例的可执行 Demo（`python -m miniagent demo`）。

设计要点：
- 默认使用 `--provider mock`（离线规则客户端），因此**无需 API Key 即可复现全部用例**；
- 每个用例都给出「预期行为」+「实际观测」+「PASS/FAIL」，可以直接贴进 README / 验收报告；
- 与 `tests/test_agent_cases.py` 共用同一套断言思路，但这里面向人类阅读。
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from miniagent import AgentConfig, MinimalAgent, build_agent, build_default_registry
from miniagent.llm import OfflineMockClient, ScriptedLLMClient
from miniagent.session import SessionManager
from miniagent.tools import ToolContext, ToolRegistry
from miniagent.tracing import Tracer
from miniagent.utils import ensure_utf8_stdio, text_width


@dataclass
class CaseOutcome:
    case_id: str
    title: str
    expected: str
    observed: str
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)


def _offline_config(**overrides: Any) -> AgentConfig:
    base = dict(
        provider="mock",
        model="offline-mock",
        console_trace=False,
        log_dir="logs",
        persist_sessions=False,
    )
    base.update(overrides)
    return AgentConfig.from_env(**base)


def _agent(config: Optional[AgentConfig] = None, client: Any = None, registry: Optional[ToolRegistry] = None) -> MinimalAgent:
    cfg = config or _offline_config()
    return MinimalAgent(
        client or OfflineMockClient(),
        registry or build_default_registry(),
        SessionManager(max_turn=cfg.max_tool_turns, max_context_tokens=cfg.max_context_tokens, keep_recent_messages=cfg.keep_recent_messages),
        config=cfg,
        tracer=Tracer.in_memory(),
    )


# ---------------------------------------------------------------------------
# 用例 1：calculator
# ---------------------------------------------------------------------------
async def case_1_calculator() -> CaseOutcome:
    agent = _agent()
    result = await agent.run("123+456*7", user_id="userA", window_id="win_calc")
    ok = (
        "calculator" in result.used_tools
        and result.stopped_reason == "answer"
        and "3315" in result.answer
    )
    return CaseOutcome(
        "用例1",
        "用户输入 `123+456*7`",
        "Agent 调用 calculator，返回结果 3315",
        f"工具链={result.used_tools}，answer={result.answer[:80]}",
        ok,
        {"tool_calls": result.to_dict()["tool_calls"], "answer": result.answer},
    )


# ---------------------------------------------------------------------------
# 用例 2：weather
# ---------------------------------------------------------------------------
async def case_2_weather() -> CaseOutcome:
    agent = _agent()
    result = await agent.run("上海今天天气", user_id="userA", window_id="win_weather")
    answer = result.answer
    ok = (
        "weather" in result.used_tools
        and "上海" in answer
        and "°C" in answer
        and result.stopped_reason == "answer"
    )
    return CaseOutcome(
        "用例2",
        "用户输入 `上海今天天气`",
        "调用 weather 工具，返回 mock 天气（温度/天气状况）",
        f"工具链={result.used_tools}，answer={answer[:110]}",
        ok,
        {"answer": answer},
    )


# ---------------------------------------------------------------------------
# 用例 3：无需工具直接回答
# ---------------------------------------------------------------------------
async def case_3_direct_answer() -> CaseOutcome:
    agent = _agent()
    result = await agent.run("你好，请用一句话说明你是谁", user_id="userA", window_id="win_direct")
    ok = result.stopped_reason == "answer" and not result.tool_calls and bool(result.answer)
    return CaseOutcome(
        "用例3",
        "用户输入 `你好，请用一句话说明你是谁`",
        "无需工具，直接输出最终答案，循环终止",
        f"工具调用={len(result.tool_calls)} 次，stop={result.stopped_reason}，answer={result.answer[:90]}",
        ok,
        {"answer": result.answer},
    )


# ---------------------------------------------------------------------------
# 用例 4：连续追问（复用同一 session）
# ---------------------------------------------------------------------------
async def case_4_followup() -> CaseOutcome:
    agent = _agent()
    session_id = "userA::win_followup"
    first = await agent.run("上海今天天气", session_id=session_id)
    second = await agent.run("明天呢", session_id=session_id)
    session = agent.session_mgr.get_or_create(session_id)
    ok = (
        "weather" in first.used_tools
        and "weather" in second.used_tools
        and "明天" in second.answer
        and "上海" in second.answer
        and session.turn_count == 2
    )
    return CaseOutcome(
        "用例4",
        "连续追问：先问「上海今天天气」，再追问「明天呢」",
        "复用同一 session 上下文，继续对话（城市沿用上海、日期切到明天）",
        f"第1轮={first.answer[:48]}…\n             第2轮={second.answer[:90]}\n"
        f"             session 消息数={session.size}，轮次={session.turn_count}",
        ok,
        {"first": first.answer, "second": second.answer},
    )


# ---------------------------------------------------------------------------
# 用例 5：多窗口隔离
# ---------------------------------------------------------------------------
async def case_5_window_isolation() -> CaseOutcome:
    agent = _agent()
    calc = await agent.run("123+456*7", user_id="userA", window_id="win1")
    weather = await agent.run("上海今天天气", user_id="userA", window_id="win2")
    s1 = agent.session_mgr.get_or_create("userA::win1")
    s2 = agent.session_mgr.get_or_create("userA::win2")

    win1_text = s1.transcript()
    win2_text = s2.transcript()
    isolated = (
        s1.session_id != s2.session_id
        and "calculator" in win1_text
        and "weather" not in win1_text
        and "weather" in win2_text
        and "calculator" not in win2_text
        and "3315" not in win2_text
    )
    ok = isolated and "calculator" in calc.used_tools and "weather" in weather.used_tools
    return CaseOutcome(
        "用例5",
        "用户 A 开两个窗口，分别问计算器和天气",
        "两个 session 完全隔离，互不影响",
        f"win1: 消息={s1.size} 工具={calc.used_tools} | win2: 消息={s2.size} 工具={weather.used_tools} | 交叉污染={'无' if isolated else '有'}",
        ok,
        {"win1": win1_text, "win2": win2_text},
    )


# ---------------------------------------------------------------------------
# 用例 6：工具参数非法 → Parser 捕获 → LLM 修正重试
# ---------------------------------------------------------------------------
async def case_6_bad_arguments_retry() -> CaseOutcome:
    # 第 1 次：weather 缺少必填参数 city → Parser 的 schema 校验失败（在工具执行之前就被拦住）
    # 第 2 次：LLM 读到错误后修正参数 → 执行成功
    # 第 3 次：给出最终回答
    scripted = ScriptedLLMClient([
        '{"type":"tool_call","tool_name":"weather","arguments":{"date":"今天"}}',
        '{"type":"tool_call","tool_name":"weather","arguments":{"city":"上海","date":"今天"}}',
        '{"type":"answer","content":"上海今天多云，气温 28°C 左右（mock 数据）。"}',
    ])
    agent = _agent(client=scripted)
    result = await agent.run("上海今天天气", user_id="userA", window_id="win_retry")
    session = agent.session_mgr.get_or_create("userA::win_retry")

    error_msgs = [m for m in session.messages if m.role == "error"]
    executed = [c for c in result.tool_calls if c.ok]
    first_error = error_msgs[0].content if error_msgs else ""
    ok = (
        len(executed) == 1                              # 只有修正后的那次真正执行了
        and bool(error_msgs)                            # 校验错误确实写回了上下文
        and "缺少必填字段" in first_error
        and "city" in first_error
        and result.stopped_reason == "answer"
    )
    return CaseOutcome(
        "用例6",
        "LLM 输出错误工具参数（weather 缺必填参数 city）",
        "Parser 捕获参数错误 → 错误写入上下文 → LLM 修正参数、重试成功",
        f"第1次调用被 Parser 拦截（未执行工具），错误已回灌上下文："
        f"{first_error.splitlines()[0][:70]}\n"
        f"             第2次调用参数={executed[0].arguments if executed else 'N/A'} → 执行成功；"
        f"context 中的 error 消息={len(error_msgs)} 条；最终 stop={result.stopped_reason}",
        ok,
        {"errors": [m.content for m in error_msgs], "answer": result.answer},
    )


# ---------------------------------------------------------------------------
# 用例 7：多轮对话触发上下文裁剪
# ---------------------------------------------------------------------------
async def case_7_context_compression() -> CaseOutcome:
    config = _offline_config(
        max_context_messages=8,
        keep_recent_messages=4,
        max_context_tokens=1200,
    )
    agent = _agent(config=config)
    session_id = "userA::win_compress"
    questions = [
        "北京今天天气",
        "上海今天天气",
        "深圳今天天气",
        "广州今天天气",
        "杭州今天天气",
        "成都今天天气",
    ]
    answers = []
    for question in questions:
        result = await agent.run(question, session_id=session_id)
        answers.append(result.answer)

    session = agent.session_mgr.get_or_create(session_id)
    build = agent.context_mgr.build(session, agent._system_prompt(session, session.max_turn))
    ok = (
        session.size >= 12
        and session.compress_count > 0
        and build.compressed
        and 0 < build.dropped_messages < session.size   # 本次请求确实丢了历史，但保留了一部分
        and any("上下文已裁剪" in m["content"] for m in build.messages)
        and len(build.messages) < session.size
    )
    return CaseOutcome(
        "用例7",
        "持续 6 轮对话，超过消息阈值（max_context_messages=8）",
        "自动裁剪早期上下文，保留最近对话，并追加压缩提示",
        f"session 消息数={session.size} → 本次请求 messages={len(build.messages)}（本次丢弃 {build.dropped_messages} 条）| "
        f"累计裁剪 {session.compress_count} 次、丢弃 {session.dropped_messages} 条 | "
        f"压缩提示={'已插入' if any('上下文已裁剪' in m['content'] for m in build.messages) else '未插入'}",
        ok,
        {"dropped": session.dropped_messages, "notes": build.notes},
    )


# ---------------------------------------------------------------------------
# 补充用例（超出 7 个必测项，验证护栏）
# ---------------------------------------------------------------------------
async def case_8_session_isolation_two_users() -> CaseOutcome:
    agent = _agent()
    await agent.run("上海今天天气", user_id="userA", window_id="win1")
    await agent.run("123+456*7", user_id="userB", window_id="win1")
    sessions = agent.session_mgr.list_sessions()
    ids = [s.session_id for s in sessions]
    texts = {s.session_id: s.transcript() for s in sessions}
    ok = (
        ids == ["userA::win1", "userB::win1"]
        and all(s.size >= 2 for s in sessions)
        # 跨用户同名窗口也不能互相污染
        and "weather" in texts["userA::win1"] and "calculator" not in texts["userA::win1"]
        and "calculator" in texts["userB::win1"] and "weather" not in texts["userB::win1"]
    )
    return CaseOutcome(
        "补充用例8",
        "不同用户同名窗口（userA::win1 / userB::win1）",
        "session_id 含 user_id，跨用户同样隔离",
        f"会话列表={ids}，各自消息数={[s.size for s in sessions]}，交叉污染={'无' if ok else '有'}",
        ok,
        {"overview": agent.session_mgr.overview()},
    )


async def case_9_max_turns_guard() -> CaseOutcome:
    """模型一直想调工具 → 达到轮次上限后强制收敛，不再执行工具。"""
    calls = [
        '{"type":"tool_call","tool_name":"search","arguments":{"query":"循环测试1"}}',
        '{"type":"tool_call","tool_name":"search","arguments":{"query":"循环测试2"}}',
        '{"type":"tool_call","tool_name":"search","arguments":{"query":"循环测试3"}}',
        '{"type":"tool_call","tool_name":"search","arguments":{"query":"还想再搜"}}',
        '{"type":"answer","content":"基于已有检索结果，我给出总结性回答。"}',
    ]
    config = _offline_config(max_tool_turns=3)
    agent = _agent(config=config, client=ScriptedLLMClient(calls))
    result = await agent.run("帮我搜索一下大模型 Agent", user_id="userA", window_id="win_turns")
    executed = [c for c in result.tool_calls if c.ok]
    ok = len(executed) == 3 and result.stopped_reason == "answer" and "总结" in result.answer
    return CaseOutcome(
        "补充用例9",
        "模型持续调用工具（脚本故意给 4 次 tool_call）",
        "达到 max_tool_turns=3 后停止执行工具，直接让 LLM 总结返回",
        f"实际执行工具={len(executed)} 次（脚本给了 4 次），stop={result.stopped_reason}，answer={result.answer[:60]}",
        ok,
        {"tool_calls": result.to_dict()["tool_calls"]},
    )


async def case_10_parse_error_recovery() -> CaseOutcome:
    """LLM 先输出坏 JSON → 错误回灌 → 第二次修正成功。"""
    scripted = ScriptedLLMClient([
        "好的，我来计算一下：{\"type\":\"tool_call\",\"tool_name\":\"calculator\",\"arguments\":{\"expression\":123+456*7}}",  # 非法 JSON（缺引号）
        '{"type":"tool_call","tool_name":"calculator","arguments":{"expression":"123+456*7"}}',
        '{"type":"answer","content":"123+456*7 = 3315。"}',
    ])
    agent = _agent(client=scripted)
    result = await agent.run("123+456*7", user_id="userA", window_id="win_parse")
    session = agent.session_mgr.get_or_create("userA::win_parse")
    ok = "calculator" in result.used_tools and "3315" in result.answer and any(m.role == "error" for m in session.messages)
    return CaseOutcome(
        "补充用例10",
        "LLM 输出坏 JSON（arguments 里 123+456*7 未加引号）",
        "解析失败 → 错误回灌上下文 → LLM 第二次输出合法 JSON 并成功执行",
        f"工具链={result.used_tools}，answer={result.answer[:60]}，error 消息="
        f"{sum(1 for m in session.messages if m.role == 'error')} 条",
        ok,
        {"answer": result.answer},
    )


async def case_11_llm_failure() -> CaseOutcome:
    """LLM 网络异常 → 重试后仍失败 → 返回可读文案，不崩。"""
    from miniagent.errors import LLMError

    agent = _agent(client=ScriptedLLMClient([LLMError("模拟网络超时")]))
    agent.config.llm_call_retries = 1
    agent.config.llm_error_reply = "（模拟）模型服务不可用，请稍后重试。"
    result = await agent.run("你好", user_id="userA", window_id="win_llm_error")
    session = agent.session_mgr.get_or_create("userA::win_llm_error")
    ok = result.stopped_reason == "llm_error" and "不可用" in result.answer and any(m.role == "error" for m in session.messages)
    return CaseOutcome(
        "补充用例11",
        "LLM 网络异常",
        "异常被捕获并写入 trace + 上下文，向用户返回可读失败文案",
        f"stop={result.stopped_reason}，answer={result.answer}",
        ok,
        {"errors": result.errors},
    )


CASES: list[Callable[[], Awaitable[CaseOutcome]]] = [
    case_1_calculator,
    case_2_weather,
    case_3_direct_answer,
    case_4_followup,
    case_5_window_isolation,
    case_6_bad_arguments_retry,
    case_7_context_compression,
    case_8_session_isolation_two_users,
    case_9_max_turns_guard,
    case_10_parse_error_recovery,
    case_11_llm_failure,
]


def _rule(width: int = 78) -> str:
    return "─" * width


async def run_demo(*, provider: str = "", verbose: bool = True) -> int:
    ensure_utf8_stdio()
    selected = CASES
    if provider and provider not in ("mock", "offline"):
        print(f"⚠ demo 用例为保证可复现使用离线 mock 客户端；--provider {provider} 仅影响标注。")

    print("=" * 78)
    print(" minimal-agent 验收用例（不依赖 API Key，全部可复现）")
    print(f" 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}   用例数: {len(selected)}")
    print("=" * 78)

    outcomes: list[CaseOutcome] = []
    for runner in selected:
        outcome = await runner()
        outcomes.append(outcome)
        status = "✅ PASS" if outcome.passed else "❌ FAIL"
        print(f"\n{_rule()}")
        print(f"{status}  {outcome.case_id}：{outcome.title}")
        print(f"  预期：{outcome.expected}")
        print(f"  实测：{outcome.observed}")
    print(f"\n{_rule()}")
    passed = sum(1 for o in outcomes if o.passed)
    print(f" 结果：{passed}/{len(outcomes)} 通过")
    for outcome in outcomes:
        mark = "✅" if outcome.passed else "❌"
        print(f"   {mark} {outcome.case_id:<10} {outcome.title}")
    print(_rule())
    return 0 if passed == len(outcomes) else 1


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="运行 minimal-agent 验收用例")
    parser.add_argument("--provider", default="mock")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)
    return asyncio.run(run_demo(provider=args.provider, verbose=not args.quiet))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
