"""示例：如何新增一个自定义工具（可直接运行）。

    python examples/custom_tool.py

演示三件事：
1. 自定义工具（同步 handler）注册进 ToolRegistry；
2. 依赖注入（ctx / tool_deps）与异步 handler；
3. 工具抛异常时，错误如何被回灌给 LLM 并由 LLM 自行修正。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniagent import AgentConfig, MinimalAgent, Tool, ToolRegistry, build_default_registry  # noqa: E402
from miniagent.llm import OfflineMockClient, ScriptedLLMClient  # noqa: E402
from miniagent.session import SessionManager  # noqa: E402
from miniagent.tracing import Tracer  # noqa: E402
from miniagent.utils import ensure_utf8_stdio  # noqa: E402

# ---------------------------------------------------------------------------
# 1) 自定义工具 A：字数统计（同步 handler）
# ---------------------------------------------------------------------------
WORD_COUNT_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "要统计的文本", "minLength": 1},
    },
    "required": ["text"],
    "additionalProperties": False,
}


def word_count(text: str) -> str:
    """中英混排字数统计。返回 str / dict / list 都可以，框架会自动序列化。"""
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    words = len(re.findall(r"[A-Za-z0-9']+", text))
    return f"字符数={len(text)}，中文字数={cjk}，英文单词数={words}"


# ---------------------------------------------------------------------------
# 2) 自定义工具 B：内部汇率表（异步 handler + 依赖注入 + 会抛错）
# ---------------------------------------------------------------------------
RATE_SCHEMA = {
    "type": "object",
    "properties": {
        "base": {"type": "string", "description": "基础货币代码，如 CNY / USD / EUR（大写三字母）", "minLength": 3, "maxLength": 3},
        "target": {"type": "string", "description": "目标货币代码，如 USD / JPY", "minLength": 3, "maxLength": 3},
    },
    "required": ["base", "target"],
    "additionalProperties": False,
}

#: 演示用固定汇率表；真实场景应替换成 API 调用（放进 tool_deps 注入进来）
FAKE_RATES = {
    ("CNY", "USD"): 0.14,
    ("CNY", "JPY"): 21.3,
    ("USD", "CNY"): 7.12,
    ("USD", "JPY"): 151.8,
    ("EUR", "CNY"): 7.75,
}


async def exchange_rate(base: str, target: str, ctx=None) -> str:
    """异步 handler：注意故意对未知货币对抛异常，用来演示错误回灌。

    `ctx` 参数会由框架注入 `ToolContext`（可取 session_id / tracer / deps）。
    """
    base, target = base.upper(), target.upper()
    if ctx is not None and ctx.tracer is not None:
        ctx.tracer.log("exchange_rate_lookup", session_id=ctx.session_id, pair=f"{base}/{target}")
    await asyncio.sleep(0)  # 模拟一次 IO
    rate = FAKE_RATES.get((base, target))
    if rate is None:
        supported = ", ".join(f"{b}->{t}" for b, t in sorted(FAKE_RATES))
        raise ValueError(f"没有 {base}->{target} 的汇率数据（支持: {supported}）")
    return json.dumps({"base": base, "target": target, "rate": rate, "source": "内部演示汇率表"}, ensure_ascii=False)


def build_registry() -> ToolRegistry:
    """在内置三件套基础上追加两个自定义工具。"""
    registry = build_default_registry()
    registry.register(
        Tool(
            name="word_count",
            description=(
                "统计一段文本的字符数、中文字数和英文单词数。"
                "当用户问「这段话多少字 / 帮我数一下字数 / 字数统计」时调用。"
            ),
            parameters=WORD_COUNT_SCHEMA,
            handler=word_count,
            tags=("text", "demo"),
        )
    )
    registry.register(
        Tool(
            name="exchange_rate",
            description=(
                "查询两种货币之间的汇率（演示用固定汇率表）。"
                "当用户询问汇率、货币换算时调用。货币用三字母代码，例如 CNY、USD、JPY。"
            ),
            parameters=RATE_SCHEMA,
            handler=exchange_rate,
            tags=("finance", "demo"),
        )
    )
    return registry


def make_agent(client=None, registry: ToolRegistry | None = None) -> MinimalAgent:
    config = AgentConfig.from_env(provider="mock", model="offline-mock", console_trace=False)
    return MinimalAgent(
        client or OfflineMockClient(),
        registry or build_registry(),
        SessionManager(),
        config=config,
        tracer=Tracer.in_memory(),
    )


async def demo_new_tool() -> None:
    print("=" * 74)
    print("示例 1：注册自定义工具后，LLM 可以自主调用它")
    print("=" * 74)
    registry = build_registry()
    print(registry.describe())

    client = ScriptedLLMClient(
        [
            '{"type":"tool_call","tool_name":"word_count","arguments":{"text":"今天天气真好 hello world"}}',
            '{"type":"answer","content":"统计完成：共 19 个字符，其中中文 6 个，英文单词 2 个。"}',
        ]
    )
    agent = make_agent(client=client, registry=registry)
    result = await agent.run("帮我数一下字数：今天天气真好 hello world", session_id="demo::win1")
    print(f"\n工具链: {result.used_tools}")
    print(f"工具结果: {result.tool_calls[0].result}")
    print(f"最终回答: {result.answer}")


async def demo_async_tool() -> None:
    print("\n" + "=" * 74)
    print("示例 2：异步 handler + ToolContext 注入")
    print("=" * 74)
    client = ScriptedLLMClient(
        [
            '{"type":"tool_call","tool_name":"exchange_rate","arguments":{"base":"CNY","target":"USD"}}',
            '{"type":"answer","content":"1 人民币约等于 0.14 美元。"}',
        ]
    )
    agent = make_agent(client=client)
    result = await agent.run("人民币兑美元汇率是多少", session_id="demo::win1")
    print(f"工具链: {result.used_tools}")
    print(f"工具结果: {result.tool_calls[0].result}")
    print(f"最终回答: {result.answer}")


async def demo_error_feedback() -> None:
    print("\n" + "=" * 74)
    print("示例 3：工具内部抛异常 → 错误回灌 → LLM 换成合法参数重试")
    print("=" * 74)
    client = ScriptedLLMClient(
        [
            '{"type":"tool_call","tool_name":"exchange_rate","arguments":{"base":"CNY","target":"KRW"}}',   # 汇率表里没有
            '{"type":"tool_call","tool_name":"exchange_rate","arguments":{"base":"CNY","target":"JPY"}}',   # 换成支持的
            '{"type":"answer","content":"1 人民币约等于 21.3 日元。"}',
        ]
    )
    agent = make_agent(client=client)
    result = await agent.run("人民币兑韩元汇率", session_id="demo::win2")
    session = agent.session_mgr.get_or_create("demo::win2")
    print(f"第 1 次调用 ok={result.tool_calls[0].ok}")
    print(f"  错误信息: {result.tool_calls[0].error.splitlines()[0]}")
    print(f"第 2 次调用 ok={result.tool_calls[1].ok}  参数={result.tool_calls[1].arguments}")
    print(f"写回上下文的 error 消息数: {sum(1 for m in session.messages if m.role == 'error')}")
    print(f"最终回答: {result.answer}")


async def demo_unregister() -> None:
    print("\n" + "=" * 74)
    print("示例 4：注销工具后，LLM 调用它会收到明确错误")
    print("=" * 74)
    registry = build_registry()
    registry.unregister("exchange_rate")
    print(f"注销后可用工具: {registry.names()}")
    client = ScriptedLLMClient(
        [
            '{"type":"tool_call","tool_name":"exchange_rate","arguments":{"base":"CNY","target":"USD"}}',
            '{"type":"answer","content":"汇率工具当前不可用，我无法回答该问题。"}',
        ]
    )
    agent = make_agent(client=client, registry=registry)
    result = await agent.run("人民币兑美元汇率", session_id="demo::win3")
    session = agent.session_mgr.get_or_create("demo::win3")
    errors = [m.content for m in session.messages if m.role == "error"]
    print(f"LLM 收到的错误: {errors[0].splitlines()[0] if errors else '(无)'}")
    print(f"最终回答: {result.answer}")


async def main() -> int:
    ensure_utf8_stdio()
    await demo_new_tool()
    await demo_async_tool()
    await demo_error_feedback()
    await demo_unregister()
    print("\n全部示例执行完成。新增工具只需要：写 handler → 定义 schema → registry.register(...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
