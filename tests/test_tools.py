"""工具层测试：注册/注销、参数校验、三个内置工具、安全边界。

重点验证 calculator 的**安全 eval**：确认它真的挡住了代码注入（这是最容易出事的地方）。
"""

from __future__ import annotations

import asyncio

import pytest

from miniagent.errors import ToolExecutionError, ToolNotFoundError, ToolValidationError
from miniagent.tools import (
    Tool,
    ToolContext,
    ToolRegistry,
    build_default_registry,
    calculator,
    mock_search,
    mock_weather,
    safe_eval,
)


@pytest.fixture()
def registry() -> ToolRegistry:
    return build_default_registry()


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------- 注册表
def test_default_registry_has_three_tools(registry: ToolRegistry) -> None:
    assert registry.names() == ["calculator", "search", "weather"]


def test_register_and_unregister() -> None:
    reg = ToolRegistry()
    schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
    reg.register(name="echo", description="原样返回", schema=schema, handler=lambda text: text)
    assert "echo" in reg
    assert reg.get("echo").description == "原样返回"
    assert reg.unregister("echo") is True
    assert "echo" not in reg
    assert reg.unregister("echo") is False


def test_register_rejects_bad_schema() -> None:
    with pytest.raises(ValueError):
        Tool(name="bad", description="x", parameters={"type": "string"}, handler=lambda: None)


def test_register_rejects_non_callable_handler() -> None:
    with pytest.raises(ValueError):
        Tool(name="bad", description="x", parameters={"type": "object"}, handler="not-callable")  # type: ignore[arg-type]


def test_duplicate_register_control() -> None:
    reg = ToolRegistry()
    schema = {"type": "object", "properties": {}}
    reg.register(name="a", description="d", schema=schema, handler=lambda: "1")
    reg.register(name="a", description="d2", schema=schema, handler=lambda: "2")
    assert reg.get("a").description == "d2"          # 默认覆盖
    with pytest.raises(ValueError):
        reg.register(name="a", description="d3", schema=schema, handler=lambda: "3", override=False)


def test_unknown_tool_raises(registry: ToolRegistry) -> None:
    with pytest.raises(ToolNotFoundError) as excinfo:
        run(registry.execute("nope", {}))
    assert "calculator" in str(excinfo.value)


def test_async_handler_supported() -> None:
    reg = ToolRegistry()

    async def handler(text: str) -> str:
        return text.upper()

    reg.register(name="upper", description="大写", schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}, handler=handler)
    assert run(reg.execute("upper", {"text": "abc"})) == "ABC"


def test_validation_error_is_wrapped(registry: ToolRegistry) -> None:
    with pytest.raises(ToolValidationError):
        run(registry.execute("calculator", {"expression": 123}))     # 类型不对（string）


def test_handler_exception_becomes_tool_execution_error() -> None:
    reg = ToolRegistry()

    def boom(**kwargs):
        raise RuntimeError("内部炸了")

    reg.register(name="boom", description="x", schema={"type": "object", "properties": {}}, handler=boom)
    with pytest.raises(ToolExecutionError) as excinfo:
        run(reg.execute("boom", {}))
    assert "RuntimeError" in str(excinfo.value)


def test_dict_result_is_serialized() -> None:
    reg = ToolRegistry()
    reg.register(name="d", description="x", schema={"type": "object", "properties": {}}, handler=lambda: {"a": 1})
    assert run(reg.execute("d", {})) == '{"a": 1}'


def test_snapshot_is_sorted_and_stable(registry: ToolRegistry) -> None:
    names = [item["name"] for item in registry.snapshot()]
    assert names == sorted(names) == ["calculator", "search", "weather"]


# --------------------------------------------------------------------- calculator
@pytest.mark.parametrize(
    "expression,expected",
    [
        ("123+456*7", "3315"),
        ("(3.5+1.5)*2", "10"),
        ("2**10", "1024"),
        ("sqrt(16)", "4"),
        ("sqrt(16)+2**10", "1028"),
        ("10/4", "2.5"),
        ("7//2", "3"),
        ("7%3", "1"),
        ("-3+1", "-2"),
        ("max(1,5,3)", "5"),
        ("round(3.14159, 2)", "3.14"),
        ("10 > 3", "true"),
        ("pi", None),           # 只验证可计算，不比对文本
    ],
)
def test_calculator_math(expression: str, expected) -> None:
    result = calculator(expression)
    if expected is None:
        assert result
    else:
        assert result.endswith(expected), result


def test_calculator_normalizes_llm_style_input() -> None:
    assert calculator("123 + 456 × 7").endswith("3315")
    assert calculator("（1+2）*3").endswith("9")
    assert calculator("2^10").endswith("1024")


def test_calculator_rejects_code_injection() -> None:
    for evil in [
        "__import__('os').system('echo hi')",
        "open('/etc/passwd').read()",
        "eval('1+1')",
        "exec('a=1')",
        "(lambda: 1)()",
        "().__class__.__bases__",
        "[].__class__",
        "globals()",
        "print(1)",
        "1 if True else 2",
        "x := 5",
    ]:
        with pytest.raises(Exception):
            safe_eval(evil)


def test_calculator_rejects_huge_power() -> None:
    with pytest.raises(Exception):
        safe_eval("9**999999999")


def test_calculator_rejects_overlong_expression() -> None:
    with pytest.raises(Exception):
        safe_eval("1+" * 400 + "1")


def test_calculator_error_is_readable() -> None:
    with pytest.raises(Exception) as excinfo:
        safe_eval("1+")
    assert "语法错误" in str(excinfo.value)


def test_calculator_via_registry(registry: ToolRegistry) -> None:
    out = run(registry.execute("calculator", {"expression": "123+456*7"}))
    assert out == "123+456*7 = 3315"


# --------------------------------------------------------------------- search
def test_search_hits_knowledge_base() -> None:
    out = mock_search("介绍一下大模型")
    assert "大语言模型" in out or "Transformer" in out
    assert "mock://" in out


def test_search_is_deterministic() -> None:
    assert mock_search("量子计算") == mock_search("量子计算")


def test_search_unknown_query_generates_results() -> None:
    out = mock_search("蓝鲸的迁徙路线", top_k=2)
    assert "共 2 条" in out
    assert "mock 引擎生成" in out


def test_search_top_k_bounds(registry: ToolRegistry) -> None:
    with pytest.raises(ToolValidationError):
        run(registry.execute("search", {"query": "x", "top_k": 99}))


def test_search_requires_query(registry: ToolRegistry) -> None:
    with pytest.raises(ToolValidationError):
        run(registry.execute("search", {}))


# --------------------------------------------------------------------- weather
def test_weather_known_city() -> None:
    out = mock_weather("上海")
    assert "上海" in out
    assert "°C" in out
    assert "mock-weather-api" in out


def test_weather_deterministic_and_date_sensitive() -> None:
    assert mock_weather("上海", "今天") == mock_weather("上海", "今天")
    assert mock_weather("上海", "今天") != mock_weather("上海", "明天")


def test_weather_unknown_city_still_works() -> None:
    out = mock_weather("克拉玛依")
    assert "克拉玛依" in out
    assert "°C" in out


def test_weather_fahrenheit() -> None:
    assert "°F" in mock_weather("上海", "今天", "fahrenheit")


def test_weather_invalid_unit(registry: ToolRegistry) -> None:
    with pytest.raises(ToolValidationError):
        run(registry.execute("weather", {"city": "上海", "unit": "kelvin"}))


def test_weather_missing_city(registry: ToolRegistry) -> None:
    with pytest.raises(ToolValidationError) as excinfo:
        run(registry.execute("weather", {"date": "今天"}))
    assert "city" in str(excinfo.value)


def test_tool_context_deps_passed() -> None:
    reg = ToolRegistry()
    reg.register(
        name="ctx_probe",
        description="读取注入依赖",
        schema={"type": "object", "properties": {}},
        handler=lambda ctx=None: "no-ctx",
    )
    # ToolContext 通过 deps 注入：这里直接验证 dataclass 行为
    ctx = ToolContext(session_id="u::w", deps={"api": 42})
    assert ctx.dep("api") == 42
    assert ctx.dep("missing", "fallback") == "fallback"
