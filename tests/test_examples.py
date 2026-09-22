"""示例脚本的集成测试：确保 examples/ 里的代码不会腐烂。"""

from __future__ import annotations

import asyncio

import pytest

from examples.custom_tool import FAKE_RATES, build_registry, exchange_rate, word_count


def run(coro):
    return asyncio.run(coro)


def test_word_count_handler() -> None:
    assert word_count("hello world") == "字符数=11，中文字数=0，英文单词数=2"
    assert word_count("今天天气真好") == "字符数=6，中文字数=6，英文单词数=0"
    assert word_count("今天天气真好 hello world") == "字符数=18，中文字数=6，英文单词数=2"


def test_custom_tools_registered_alphabetically() -> None:
    registry = build_registry()
    assert registry.names() == ["calculator", "exchange_rate", "search", "weather", "word_count"]


def test_exchange_rate_handler() -> None:
    out = run(exchange_rate("cny", "usd"))
    assert "0.14" in out


def test_exchange_rate_raises_for_unsupported_pair() -> None:
    with pytest.raises(ValueError) as excinfo:
        run(exchange_rate("CNY", "KRW"))
    assert "没有 CNY->KRW" in str(excinfo.value)


def test_unsupported_pair_error_is_fed_back_and_retried() -> None:
    """示例 3 的核心断言：工具抛错 → 错误回灌 → LLM 换参数重试成功。"""
    from miniagent.llm import ScriptedLLMClient

    from examples.custom_tool import make_agent

    client = ScriptedLLMClient(
        [
            '{"type":"tool_call","tool_name":"exchange_rate","arguments":{"base":"CNY","target":"KRW"}}',
            '{"type":"tool_call","tool_name":"exchange_rate","arguments":{"base":"CNY","target":"JPY"}}',
            '{"type":"answer","content":"1 人民币约等于 21.3 日元。"}',
        ]
    )
    agent = make_agent(client=client)
    result = run(agent.run("人民币兑韩元汇率", session_id="demo::test"))
    assert [call.ok for call in result.tool_calls] == [False, True]
    assert result.tool_calls[1].arguments["target"] == "JPY"
    assert "21.3" in result.answer
    assert any(call.name == "exchange_rate" for call in result.tool_calls)


def test_unregister_makes_tool_unavailable() -> None:
    from miniagent.llm import ScriptedLLMClient

    from examples.custom_tool import make_agent

    registry = build_registry()
    registry.unregister("exchange_rate")
    client = ScriptedLLMClient(
        [
            '{"type":"tool_call","tool_name":"exchange_rate","arguments":{"base":"CNY","target":"USD"}}',
            '{"type":"answer","content":"汇率工具不可用。"}',
        ]
    )
    agent = make_agent(client=client, registry=registry)
    result = run(agent.run("汇率", session_id="demo::test2"))
    assert result.tool_calls == []
    session = agent.session_mgr.get_or_create("demo::test2")
    errors = [m.content for m in session.messages if m.role == "error"]
    assert errors and "不存在" in errors[0]


def test_example_script_runs_end_to_end() -> None:
    from examples.custom_tool import main

    assert run(main()) == 0


def test_fake_rates_table_is_sane() -> None:
    assert all(isinstance(rate, float) and rate > 0 for rate in FAKE_RATES.values())
