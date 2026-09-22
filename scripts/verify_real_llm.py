"""真实 LLM 端到端验证：用真模型跑一遍核心链路（不是 mock）。

    python scripts/verify_real_llm.py

覆盖：
1. calculator：模型是否按协议输出 tool_call（而不是自己心算）
2. weather：跨工具选择能力
3. 追问：多轮上下文 + 指代消解（"明天呢"）
4. 多窗口隔离：同用户两个窗口不串味
5. 无需工具直接回答：不该调工具时不乱调
6. 错误参数自修正：给一个 schema 不合法的工具，看模型能否按错误提示改正
7. 轮次上限：上限设 1，观察是否被强制收敛
8. 上下文压缩：小阈值 + 多轮，观察裁剪是否发生

每条都打印：工具链 / 停止原因 / 关键断言结果。真实模型有随机性，
所以断言只检查"结构不变量"（调了哪个工具、是否终止、是否包含关键数字），
不比对具体措辞。
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from miniagent import AgentConfig, MinimalAgent, build_default_registry  # noqa: E402
from miniagent.llm import OpenAICompatibleClient  # noqa: E402
from miniagent.session import SessionManager  # noqa: E402
from miniagent.tools import Tool, ToolRegistry  # noqa: E402
from miniagent.tracing import Tracer  # noqa: E402
from miniagent.utils import ensure_utf8_stdio  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"{'✅ PASS' if ok else '❌ FAIL'}  {name}\n        {detail}\n", flush=True)


def build(config: AgentConfig, registry: ToolRegistry | None = None) -> MinimalAgent:
    client = OpenAICompatibleClient(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model,
        timeout=config.timeout,
        max_retries=config.max_retries,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    return MinimalAgent(
        client,
        registry or build_default_registry(),
        SessionManager(max_turn=config.max_tool_turns),
        config=config,
        tracer=Tracer(config.log_dir, console=config.console_trace),
    )


async def main() -> int:
    ensure_utf8_stdio()
    config = AgentConfig.from_env()
    problems = config.check()
    if problems:
        print("配置有问题，先修好再跑：")
        for problem in problems:
            print("  -", problem)
        return 2
    print("=" * 78)
    print(f"真实 LLM 验证  provider={config.provider} model={config.model}")
    print("=" * 78 + "\n")

    started = time.perf_counter()

    # ---------------- 1. calculator ----------------
    agent = build(config)
    r = await agent.run("123+456*7 等于多少？请用工具精确计算。", session_id="real::calc")
    check(
        "1 calculator 工具调用",
        "calculator" in r.used_tools and "3315" in r.answer and r.stopped_reason == "answer",
        f"工具={r.used_tools} stop={r.stopped_reason} 参数={r.tool_calls[0].arguments if r.tool_calls else None}\n        answer={r.answer[:100]}",
    )

    # ---------------- 2. weather ----------------
    r = await agent.run("上海今天天气怎么样？", session_id="real::weather")
    check(
        "2 weather 工具调用",
        "weather" in r.used_tools and "上海" in r.answer,
        f"工具={r.used_tools} 参数={r.tool_calls[0].arguments if r.tool_calls else None}\n        answer={r.answer[:110]}",
    )

    # ---------------- 3. 追问（指代消解） ----------------
    r2 = await agent.run("那明天呢？", session_id="real::weather")
    ok = "weather" in r2.used_tools and any(
        str(c.arguments.get("city", "")).startswith("上海") for c in r2.tool_calls
    )
    check(
        "3 追问沿用城市 + 切日期",
        ok and ("明天" in str(r2.tool_calls[0].arguments) or "明天" in r2.answer),
        f"工具={r2.used_tools} 参数={r2.tool_calls[0].arguments if r2.tool_calls else None}\n        answer={r2.answer[:110]}",
    )

    # ---------------- 4. 多窗口隔离 ----------------
    a1 = await agent.run("北京今天天气", session_id="real::win1")
    a2 = await agent.run("(2+3)*4 等于多少", session_id="real::win2")
    s1 = agent.session_mgr.get_or_create("real::win1").transcript()
    s2 = agent.session_mgr.get_or_create("real::win2").transcript()
    check(
        "4 多窗口隔离",
        "weather" in s1 and "calculator" not in s1 and "calculator" in s2 and "weather" not in s2,
        f"win1 工具={a1.used_tools} win2 工具={a2.used_tools}\n        交叉污染={'无' if ('calculator' not in s1 and 'weather' not in s2) else '有'}",
    )

    # ---------------- 5. 无需工具 ----------------
    r = await agent.run("用一句话解释什么是递归。不要调用任何工具。", session_id="real::direct")
    check(
        "5 该直接回答时不调工具",
        r.tool_calls == [] and r.stopped_reason == "answer" and len(r.answer) > 8,
        f"工具调用={len(r.tool_calls)} stop={r.stopped_reason}\n        answer={r.answer[:110]}",
    )

    # ---------------- 6. 工具参数校验 + 模型自修正 ----------------
    registry = build_default_registry()
    registry.register(
        Tool(
            name="strict_echo",
            description=(
                "回显文本。参数 text 必填且必须是字符串，mode 必须是 slow 或 fast 之一。"
                "当用户要求回显文本时调用。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "mode": {"type": "string", "enum": ["slow", "fast"]},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            handler=lambda text, mode="fast": f"echo[{mode}]: {text}",
        )
    )
    strict_agent = build(config, registry)
    r = await strict_agent.run("请用 strict_echo 工具回显这句话：你好世界", session_id="real::strict")
    session = strict_agent.session_mgr.get_or_create("real::strict")
    errors = [m for m in session.messages if m.role == "error"]
    executed = [c for c in r.tool_calls if c.ok]
    check(
        "6 工具参数校验（模型一次填对）",
        bool(executed) and r.stopped_reason == "answer" and "你好世界" in r.answer,
        f"成功执行={len(executed)} 次 参数={executed[0].arguments if executed else None} "
        f"上下文中 error={len(errors)} 条\n        answer={r.answer[:100]}",
    )

    # ---------------- 6b. 带取值约束的参数抽取 ----------------
    # 说明：我们**试过**故意让模型首轮失败（漏必填字段 → 它用空串凑数；加 minLength → 它照样填对），
    # deepseek-flash 每次都能一次填对 —— 真实模型太强，无法稳定构造"首轮非法"。
    # 所以「参数非法 → 错误回灌 → 自修正」这条链路由**离线剧本客户端**做确定性覆盖
    # （tests/test_agent_cases.py::test_case6_schema_violation_caught_then_retried）；
    # 这里改为验证真实模型在有约束（minLength）时能否正确抽取参数。
    retry_registry = build_default_registry()
    retry_registry.register(
        Tool(
            name="unlock_report",
            description="生成一份报告。必须提供 report_name（报告名）与 access_code（6 位以上访问码）。",
            parameters={
                "type": "object",
                "properties": {
                    "report_name": {"type": "string", "description": "报告名"},
                    "access_code": {"type": "string", "description": "6 位以上访问码", "minLength": 6},
                },
                "required": ["report_name", "access_code"],
                "additionalProperties": False,
            },
            handler=lambda report_name, access_code: f"报告 {report_name} 已生成（access_code={access_code}）",
        )
    )
    retry_agent = build(config, retry_registry)
    r = await retry_agent.run(
        "请调用 unlock_report 生成报告，报告名是 月度总结，访问码是 ABC123XYZ。",
        session_id="real::retry",
    )
    rsession = retry_agent.session_mgr.get_or_create("real::retry")
    rerrors = [m for m in rsession.messages if m.role == "error"]
    rexecuted = [c for c in r.tool_calls if c.ok]
    params = rexecuted[0].arguments if rexecuted else {}
    check(
        "6b 带约束参数（minLength=6）被正确抽取",
        bool(rexecuted)
        and params.get("report_name") == "月度总结"
        and params.get("access_code") == "ABC123XYZ"
        and not rerrors,
        f"参数={params} 被拦下={len(rerrors)} 次（说明一次填对）\n"
        f"        answer={r.answer[:100]}",
    )

    # ---------------- 6c. 让工具**执行期**报错 → 错误回灌 → 模型可读交代 ----------------
    # 这条能稳定构造：除以 0 一定会抛异常，错误被包装后回灌，模型必须据此给出可读回答。
    fail_registry = build_default_registry()
    fail_registry.register(
        Tool(
            name="divide",
            description="计算 a 除以 b。当用户要求做除法时调用。",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
                "additionalProperties": False,
            },
            handler=lambda a, b: f"{a} / {b} = {a / b}",   # b=0 → ZeroDivisionError
        )
    )
    fail_agent = build(config, fail_registry)
    r = await fail_agent.run("请用 divide 工具计算 10 除以 0。", session_id="real::fail")
    fsession = fail_agent.session_mgr.get_or_create("real::fail")
    ferrors = [m for m in fsession.messages if m.role == "error"]
    failed_calls = [c for c in r.tool_calls if not c.ok]
    check(
        "6c 工具执行期报错 → 错误回灌 → 模型可读交代",
        bool(failed_calls) and bool(ferrors) and bool(r.answer) and r.stopped_reason == "answer",
        f"失败调用={len(failed_calls)} 次 错误回灌={len(ferrors)} 条 stop={r.stopped_reason}\n"
        f"        工具错误={(failed_calls[0].error.splitlines()[0][:80]) if failed_calls else '(无)'}\n"
        f"        answer={r.answer[:110]}",
    )

    # ---------------- 7. 轮次上限强制收敛 ----------------
    limited = config.merged(max_tool_turns=1)
    limited_agent = build(limited)
    r = await limited_agent.run(
        "先查北京天气，再查上海天气，然后把两个城市的温度相加告诉我。",
        session_id="real::limit",
    )
    check(
        "7 轮次上限（max_tool_turns=1）仍能收敛",
        len([c for c in r.tool_calls if c.ok]) <= 1 and r.stopped_reason in ("answer", "max_turns") and bool(r.answer),
        f"执行工具={len([c for c in r.tool_calls if c.ok])} 次 stop={r.stopped_reason}\n        answer={r.answer[:110]}",
    )

    # ---------------- 8. 上下文压缩 ----------------
    compressed = config.merged(max_context_messages=6, keep_recent_messages=4, max_context_tokens=900)
    c_agent = build(compressed)
    answers = []
    for city in ["北京", "上海", "深圳", "广州", "杭州"]:
        rr = await c_agent.run(f"{city}今天天气", session_id="real::compress")
        answers.append(rr.answer)
    cs = c_agent.session_mgr.get_or_create("real::compress")
    check(
        "8 多轮触发上下文裁剪",
        cs.size >= 12 and cs.compress_count > 0 and all(answers),
        f"session 消息={cs.size} 裁剪次数={cs.compress_count} 累计丢弃={cs.dropped_messages} 条",
    )

    # ---------------- 统计 ----------------
    stats = c_agent.tracer.tool_call_stats()
    print("-" * 78)
    print("工具调用统计（真实模型）：")
    for name, row in sorted(stats.items()):
        print(f"  {name:12} 调用 {row['calls']}  成功 {row['ok']}  失败 {row['error']}  平均 {row['avg_ms']}ms")

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print("-" * 78)
    print(f"真实 LLM 验证结果：{passed}/{len(RESULTS)} 通过   总耗时 {time.perf_counter() - started:.1f}s")
    for name, ok, _ in RESULTS:
        print(f"  {'✅' if ok else '❌'} {name}")
    print("=" * 78)
    for agent_ in (agent, strict_agent, retry_agent, limited_agent, c_agent):
        await agent_.aclose()
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
