"""Parser 鲁棒性测试：JSON 提取、脏数据修复、协议/schema 校验、错误信息质量。

这些用例对应 README「问题记录」里最真实的坑：LLM 输出的 JSON 千奇百怪。
"""

from __future__ import annotations

import pytest

from miniagent.errors import ParseError
from miniagent.parser import Answer, Parser, ToolCall, extract_json_blocks, try_json
from miniagent.tools import build_default_registry


@pytest.fixture()
def parser() -> Parser:
    return Parser(build_default_registry())


# --------------------------------------------------------------------- 正常路径
def test_plain_answer(parser: Parser) -> None:
    parsed = parser.parse('{"type":"answer","content":"你好，我是 Agent"}')
    assert isinstance(parsed, Answer)
    assert parsed.content == "你好，我是 Agent"


def test_plain_tool_call(parser: Parser) -> None:
    parsed = parser.parse('{"type":"tool_call","tool_name":"calculator","arguments":{"expression":"1+1"}}')
    assert isinstance(parsed, ToolCall)
    assert parsed.tool_name == "calculator"
    assert parsed.arguments["expression"] == "1+1"


# --------------------------------------------------------------------- markdown / 噪音
def test_markdown_fence(parser: Parser) -> None:
    raw = '好的，我来计算。\n```json\n{"type":"tool_call","tool_name":"calculator","arguments":{"expression":"2*3"}}\n```\n以上。'
    parsed = parser.parse(raw)
    assert isinstance(parsed, ToolCall)
    assert parsed.arguments["expression"] == "2*3"


def test_fence_without_language(parser: Parser) -> None:
    parsed = parser.parse('```\n{"type":"answer","content":"OK"}\n```')
    assert isinstance(parsed, Answer)


def test_leading_and_trailing_prose(parser: Parser) -> None:
    raw = '我先思考一下。{"type":"answer","content":"结论是 42"}希望有帮助！'
    parsed = parser.parse(raw)
    assert isinstance(parsed, Answer)
    assert parsed.content == "结论是 42"


def test_json_inside_longer_text_with_braces(parser: Parser) -> None:
    raw = 'note {not json} then {"type":"answer","content":"含 } 花括号的内容"} end'
    parsed = parser.parse(raw)
    assert isinstance(parsed, Answer)
    assert "花括号" in parsed.content


# --------------------------------------------------------------------- 脏数据修复
def test_full_width_punctuation(parser: Parser) -> None:
    raw = '｛“type”：“answer”，“content”：“全角也能解析”｝'
    parsed = parser.parse(raw)
    assert isinstance(parsed, Answer)
    assert parsed.content == "全角也能解析"


def test_single_quotes(parser: Parser) -> None:
    raw = "{'type':'answer','content':'单引号'}"
    parsed = parser.parse(raw)
    assert isinstance(parsed, Answer)
    assert parsed.content == "单引号"


def test_trailing_comma(parser: Parser) -> None:
    raw = '{"type":"tool_call","tool_name":"calculator","arguments":{"expression":"1+1",},}'
    parsed = parser.parse(raw)
    assert isinstance(parsed, ToolCall)


def test_unquoted_keys(parser: Parser) -> None:
    raw = '{type:"answer",content:"无引号 key"}'
    parsed = parser.parse(raw)
    assert isinstance(parsed, Answer)


def test_arguments_as_json_string(parser: Parser) -> None:
    raw = '{"type":"tool_call","tool_name":"calculator","arguments":"{\\"expression\\":\\"7*6\\"}"}'
    parsed = parser.parse(raw)
    assert isinstance(parsed, ToolCall)
    assert parsed.arguments["expression"] == "7*6"


def test_truncated_json_is_repaired(parser: Parser) -> None:
    # 模拟 max_tokens 被截断
    raw = '{"type":"answer","content":"这是一段被截断的回'
    parsed = parser.parse(raw)
    assert isinstance(parsed, Answer)
    assert parsed.content.startswith("这是一段被截断的回")


def test_missing_type_inferred_from_fields(parser: Parser) -> None:
    parsed = parser.parse('{"tool_name":"weather","arguments":{"city":"上海"}}')
    assert isinstance(parsed, ToolCall)
    assert "按字段推断" in " ".join(parsed.repaired)


def test_argument_type_coercion(parser: Parser) -> None:
    # top_k 声明为 integer，LLM 给了字符串 "2"
    parsed = parser.parse('{"type":"tool_call","tool_name":"search","arguments":{"query":"大模型","top_k":"2"}}')
    assert isinstance(parsed, ToolCall)
    assert parsed.arguments["top_k"] == 2


def test_default_value_filled(parser: Parser) -> None:
    parsed = parser.parse('{"type":"tool_call","tool_name":"weather","arguments":{"city":"北京"}}')
    assert isinstance(parsed, ToolCall)
    assert parsed.arguments["date"] == "今天"        # schema default
    assert parsed.arguments["unit"] == "celsius"


# --------------------------------------------------------------------- 错误路径
def test_empty_output_raises(parser: Parser) -> None:
    with pytest.raises(ParseError):
        parser.parse("")


def test_unknown_tool_raises_with_available_list(parser: Parser) -> None:
    with pytest.raises(ParseError) as excinfo:
        parser.parse('{"type":"tool_call","tool_name":"translate","arguments":{"text":"hi"}}')
    message = str(excinfo.value)
    assert "不存在" in message
    assert "calculator" in message          # 错误信息里要给出可用工具，LLM 才知道怎么改


def test_missing_required_argument_raises(parser: Parser) -> None:
    with pytest.raises(ParseError) as excinfo:
        parser.parse('{"type":"tool_call","tool_name":"weather","arguments":{"date":"今天"}}')
    assert "缺少必填字段" in str(excinfo.value)


def test_unexpected_argument_raises(parser: Parser) -> None:
    with pytest.raises(ParseError) as excinfo:
        parser.parse('{"type":"tool_call","tool_name":"calculator","arguments":{"expression":"1+1","foo":1}}')
    assert "未声明字段" in str(excinfo.value)


def test_unknown_type_raises(parser: Parser) -> None:
    with pytest.raises(ParseError) as excinfo:
        parser.parse('{"type":"thinking","content":"..."}')
    assert "未知的 type" in str(excinfo.value)


def test_answer_missing_content_raises(parser: Parser) -> None:
    with pytest.raises(ParseError):
        parser.parse('{"type":"answer"}')


def test_plain_text_falls_back_to_answer(parser: Parser) -> None:
    """宽松模式：模型直接说人话（没按协议输出 JSON）时，不要把内容丢掉。"""
    parsed = parser.parse("抱歉，我无法回答这个问题。")
    assert isinstance(parsed, Answer)
    assert "无法回答" in parsed.content


def test_half_json_is_not_treated_as_answer(parser: Parser) -> None:
    with pytest.raises(ParseError):
        parser.parse('{"type": "tool_call", "tool_name": "calc')


def test_prose_with_quotes_and_braces_is_still_an_answer(parser: Parser) -> None:
    """真实模型经常输出含引号/花括号的自然语言 —— 不能误判成坏 JSON。"""
    for text in ['他说 "hello" 然后走了。', "函数签名是 f(x) { return 1 }", "集合 {1,2,3} 是有限的。"]:
        parsed = parser.parse(text)
        assert isinstance(parsed, Answer), text
        assert parsed.content.strip() == text


def test_error_message_contains_hint(parser: Parser) -> None:
    # 明确的 JSON 意图 + 结构损坏（缺右括号 → arguments 也缺失）→ 必须报错而不是降级成回答
    with pytest.raises(ParseError) as excinfo:
        parser.parse('{"type": "tool_call", "tool_name": "calculator", "arguments": {"expression": "1+1"')
    assert isinstance(excinfo.value, ParseError)
    assert "修复建议" in excinfo.value.to_observation()


def test_broken_protocol_json_is_not_downgraded_to_answer(parser: Parser) -> None:
    """结构损坏的协议 JSON 绝不能被当成"回答"返回给用户。"""
    for raw in ('{"type": "tool_call", "tool_name": "calculator"', '{"type":"answer","content"'):
        with pytest.raises(ParseError):
            parser.parse(raw)


def test_unknown_tool_error_is_specific(parser: Parser) -> None:
    """工具名不存在时，应给「工具不存在 + 可用列表」而不是笼统的格式错误。"""
    with pytest.raises(ParseError) as excinfo:
        parser.parse('{"type": "tool_call", "tool_name": "calc", "arguments": {}}')
    message = str(excinfo.value)
    assert "不存在" in message
    assert "calculator" in message
    assert "格式不合法" not in message


def test_try_parse_never_raises(parser: Parser) -> None:
    parsed, error = parser.try_parse("乱七八糟")
    assert parsed is not None or error is not None


# --------------------------------------------------------------------- 工具函数
@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a":1}', "a"),
        ('```json\n{"a":1}\n```', "a"),
    ],
)
def test_extract_json_blocks(raw: str, expected: str) -> None:
    blocks = extract_json_blocks(raw)
    assert any(expected in block for block in blocks)


def test_try_json_repairs_recorded() -> None:
    data, repairs = try_json("{'a': 1,}")
    assert data == {"a": 1}
    assert repairs


def test_answer_type_flag() -> None:
    assert Answer(content="x").is_final is True
    assert ToolCall(tool_name="calculator", arguments={}).is_final is False
