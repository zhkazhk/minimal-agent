"""LLM 输出解析器：把模型自由文本解析成【最终回答】或【工具调用】。

协议（System Prompt 里约束 LLM 必须这么输出）::

    {"type":"answer","content":"最终回答给用户"}
    {"type":"tool_call","tool_name":"calculator","arguments":{"expression":"1+1"}}

Parser 的 4 件事（对应交付要求「LLM 输出解析逻辑」）:
1. **文本中提取 JSON** —— 处理 markdown 代码块、前后废话、多个 JSON 候选；
2. **JSON 合法性校验** —— 常见脏数据自修复（全角符号 / 单引号 / 尾随逗号 / 无引号 key / 截断）；
3. **协议与 schema 校验** —— type 是否合法、工具名是否存在、arguments 是否符合工具 schema；
4. **捕获解析失败** —— 抛 `ParseError`，由 Agent 把错误文本回灌上下文，让 LLM 自我修正。

「脆弱但可修的输入」是本项目最真实的踩坑点，因此这里刻意做了**多级回退**而不是 try/except 一把梭。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from .errors import ParseError, ToolNotFoundError, ToolValidationError
from .utils import safe_json, strip_code_fence, truncate

ANSWER = "answer"
TOOL_CALL = "tool_call"


@dataclass
class Answer:
    """类型 1：最终回答 → 终止循环。"""

    content: str
    raw: str = ""
    repaired: list[str] = field(default_factory=list)

    type = ANSWER

    @property
    def is_final(self) -> bool:
        return True


@dataclass
class ToolCall:
    """类型 2：工具调用 → 进入工具执行。"""

    tool_name: str
    arguments: dict[str, Any]
    raw: str = ""
    reason: str = ""
    repaired: list[str] = field(default_factory=list)

    type = TOOL_CALL
    is_final = False

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.tool_name}({safe_json(self.arguments)})"


ParsedOutput = Union[Answer, ToolCall]

# ---------------------------------------------------------------------------
# JSON 提取
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*(.*?)```", re.DOTALL)


_OUTSIDE_REPLACEMENTS = {
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "｛": "{", "｝": "}", "［": "[", "］": "]",
    "：": ":", "，": ",",
}
_INSIDE_REPLACEMENTS = {"“": '"', "”": '"', "„": '"'}


def _normalize(text: str) -> str:
    """统一全角/特殊字符，减少后续正则与 json.loads 的失败面。

    **关键约束**：字符串值内部的字符必须原样保留 ——
    中文回答里的 `，` 属于内容，被替换成 `,` 会直接污染给用户的答案。
    因此这里用一个小状态机区分「字符串外」与「字符串内」：
    - 字符串外：全角引号/括号/冒号/逗号 → 半角（LLM 常见毛病）；
    - 字符串内：只把全角双引号视作字符串定界符（即视为一个 `"`），其余不动。
    """
    if not text:
        return ""
    text = text.replace("\ufeff", "").replace("\u200b", "")
    out: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
                out.append(ch)
                continue
            if ch == "\\":
                escaped = True
                out.append(ch)
                continue
            if ch == '"':
                in_string = False
                out.append(ch)
                continue
            out.append(_INSIDE_REPLACEMENTS.get(ch, ch))
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            continue
        out.append(_OUTSIDE_REPLACEMENTS.get(ch, ch))
    return "".join(out)


def extract_json_blocks(text: str, *, max_candidates: int = 8) -> list[str]:
    """按「markdown 代码块 → 花括号/方括号配对 → 整段文本」的顺序收集 JSON 候选串。"""
    if not text:
        return []
    candidates: list[str] = []
    normalized = _normalize(text)
    for match in _FENCE_RE.finditer(normalized):
        inner = match.group(1).strip()
        if inner:
            candidates.append(inner)
    # 方括号优先：`[{...},{...}]` 这种「多个工具调用」必须先被整体看到，
    # 否则会被花括号扫描先切出第一个对象，从而**静默丢掉**后面的调用。
    candidates.extend(_bracket_candidates(normalized, max_candidates=max_candidates))
    candidates.extend(_brace_candidates(normalized, max_candidates=max_candidates))
    stripped = normalized.strip()
    if stripped and stripped not in candidates:
        candidates.append(stripped)
    # 去重且保持顺序
    seen: set[str] = set()
    unique: list[str] = []
    for item in candidates:
        key = item.strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(key)
    return unique[:max_candidates]


def _scan_candidates(text: str, open_ch: str, close_ch: str, *, skip_nested_in: str = "", max_candidates: int = 8) -> list[str]:
    """通用的「配对扫描」：正确跳过字符串内部与转义字符。

    `skip_nested_in` 用于避免重复：扫描方括号时忽略花括号内部的 `[`（那些属于某个对象的
    arguments 字段，已经被 `_brace_candidates` 覆盖）。
    """
    results: list[str] = []
    depth = 0
    outer = 0
    start = -1
    in_string = False
    escaped = False
    for idx, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if skip_nested_in:
            if ch == skip_nested_in:
                outer += 1
            elif ch == {"{": "}"}.get(skip_nested_in, ""):
                outer = max(0, outer - 1)
        if ch == open_ch:
            if depth == 0:
                start = idx
            depth += 1
        elif ch == close_ch and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0 and outer == 0:
                results.append(text[start : idx + 1])
                if len(results) >= max_candidates:
                    return results
    return results


def _brace_candidates(text: str, *, max_candidates: int = 8) -> list[str]:
    """扫描出所有「花括号配对」的片段（正确跳过字符串内的括号与转义）。

    只在**字符串外**统计深度：`{"content":"a{b}c"}` 里的 `{` `}` 不能被算进深度，
    否则会把一个合法 JSON 切碎。
    """
    return _scan_candidates(text, "{", "}", max_candidates=max_candidates)


def _bracket_candidates(text: str, *, max_candidates: int = 4) -> list[str]:
    """扫描出所有「方括号配对」的片段（处理 `[{...},{...}]` 这类数组输出）。"""
    return _scan_candidates(text, "[", "]", skip_nested_in="{", max_candidates=max_candidates)


# ---------------------------------------------------------------------------
# JSON 修复
# ---------------------------------------------------------------------------

_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_UNQUOTED_KEY_RE = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)(\s*:)")
_PY_LITERAL_RE = re.compile(r"\b(None|True|False)\b")


#: `try_json` 在「结构被截断、靠补全括号才解析成功」时打的标记
TRUNCATION_REPAIR = "补齐被截断的结构"


def try_json(text: str) -> tuple[Optional[Any], list[str]]:
    """尽最大努力把候选串解析成 Python 对象，返回 (对象, 修复记录)。"""
    repairs: list[str] = []
    attempt = text.strip()
    if not attempt:
        return None, repairs

    try:
        return json.loads(attempt), repairs
    except json.JSONDecodeError:
        pass

    # 修复 1：去尾随逗号 + 单引号字符串 + Python 字面量 + 无引号 key
    fixed = _TRAILING_COMMA_RE.sub(r"\1", attempt)
    if fixed != attempt:
        repairs.append("去掉尾随逗号")
    replaced = _PY_LITERAL_RE.sub(lambda m: {"None": "null", "True": "true", "False": "false"}[m.group()], fixed)
    if replaced != fixed:
        repairs.append("Python 字面量转 JSON")
        fixed = replaced
    if "'" in fixed:
        converted = _single_to_double_quotes(fixed)
        if converted != fixed:
            repairs.append("单引号转双引号")
            fixed = converted
    if _UNQUOTED_KEY_RE.search(fixed):
        fixed = _UNQUOTED_KEY_RE.sub(r'\1"\2"\3', fixed)
        repairs.append("补全无引号的 key")
    try:
        return json.loads(fixed), repairs
    except json.JSONDecodeError:
        pass

    # 修复 2：截断的 JSON（max_tokens 用完）→ 补齐引号/括号
    # 注意：这条修复会「凭空补出结构」，因此必须打标记，让调用方知道
    # **这份 JSON 并不是模型完整输出的**（不能当成功的工具调用来执行）。
    closed = _close_truncated(fixed)
    if closed and closed != fixed:
        try:
            return json.loads(closed), repairs + [TRUNCATION_REPAIR]
        except json.JSONDecodeError:
            pass

    # 修复 3：整个串里再找一次花括号片段
    for candidate in _brace_candidates(_normalize(attempt)):
        try:
            return json.loads(_TRAILING_COMMA_RE.sub(r"\1", candidate)), repairs + ["从文本中二次提取花括号片段"]
        except json.JSONDecodeError:
            continue
    return None, repairs


def _single_to_double_quotes(text: str) -> str:
    """把作为字符串定界符的单引号换成双引号（保留字符串内部的单引号）。"""
    out: list[str] = []
    in_single = False
    in_double = False
    escaped = False
    for ch in text:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            out.append(ch)
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            out.append('"')
            continue
        out.append(ch)
    return "".join(out)


def _close_truncated(text: str) -> str:
    """对截断的 JSON 做括号/引号补齐。"""
    if not text:
        return text
    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
    result = text
    if in_string:
        result += '"'
    # 截断处可能停在 `"key":` 或 `"key": {`，补一个 null 占位
    if re.search(r'[:[,]\s*$', result):
        result += "null"
    result += "".join(reversed(stack))
    return result


# ---------------------------------------------------------------------------
# 解析主入口
# ---------------------------------------------------------------------------

_TOOL_NAME_KEYS = ("tool_name", "tool", "name", "toolName")
_ARG_KEYS = ("arguments", "args", "parameters", "params", "input", "argument")
_CONTENT_KEYS = ("content", "answer", "text", "response", "final_answer", "message")
_TYPE_KEYS = ("type", "kind", "action", "output_type")
_TOOL_TYPE_WORDS = {"tool_call", "tool", "call_tool", "function_call", "action", "invoke", "tool_use"}
_ANSWER_TYPE_WORDS = {"answer", "final", "final_answer", "response", "reply", "message", "text", "conclusion"}


class Parser:
    """有状态的解析器（可选持有工具注册表用于校验工具名/schema）。"""

    def __init__(self, registry: Any = None, *, lenient_plain_text: bool = True) -> None:
        self.registry = registry
        self.lenient_plain_text = lenient_plain_text

    # ------------------------------------------------------------------ 入口
    def parse(self, raw: str) -> ParsedOutput:
        """解析 LLM 输出。失败抛 `ParseError`（错误文本会被回灌给 LLM）。"""
        text = (raw or "").strip()
        if not text:
            raise ParseError("LLM 输出为空，无法解析", raw=raw)

        # 候选分两档，这个区分非常重要：
        #   ① 权威候选：整段文本 / 代码块内容 —— 它们代表模型的**完整意图**；
        #   ② 兜底候选：花括号/方括号切片 —— 只在①连 JSON 都解析不出来时才用（提取"脏文本里的 JSON"）。
        #
        # 绝不能把②和①混在一个循环里"谁先成功用谁"：那会让语义校验形同虚设 ——
        # 例如 `[{call1},{call2}]` 明明触发了「一次只能一个工具调用」的校验错误，
        # 却因为切片 `{call1}` 能解析成功而被静默放行。
        primary: list[str] = []
        fence_stripped = strip_code_fence(text).strip()
        if fence_stripped:
            primary.append(fence_stripped)
        fallback: list[str] = []
        for candidate in extract_json_blocks(text):
            if candidate not in primary and candidate not in fallback:
                fallback.append(candidate)

        errors: list[str] = []
        #: 解析成功但违反协议/schema 的错误 —— 这类信息对模型最有用，优先级最高
        validation_errors: list[tuple[ParseError, list[str]]] = []

        def attempt(candidate: str) -> Optional[ParsedOutput]:
            """尝试一个候选：成功返回结果；语义错误记录后返回 None（不抛）。"""
            data, repairs = try_json(candidate)
            if data is None:
                errors.append(f"候选片段不是合法 JSON: {truncate(candidate, 120)}")
                return None
            truncated = TRUNCATION_REPAIR in repairs
            try:
                parsed = self._from_object(data, raw=raw, repairs=repairs)
            except ParseError as exc:
                if truncated:
                    # 是「没输出完」而不是「参数写错」，提示要说清楚，否则模型会以为只是漏填字段
                    exc = ParseError(
                        f"你的输出看起来**没有输出完整**（{exc.message}）",
                        raw=raw,
                        hint="请重新输出完整的一个 JSON 对象，不要中途截断。",
                        detail=exc.detail or {"raw_output": truncate(raw, 400)},
                    )
                validation_errors.append((exc, repairs))
                errors.append(exc.message)
                return None
            # 关键安全阀：靠「补全结构」才解析成功的 **工具调用** 不能执行。
            # 否则截断会把 `{"tool_name":"calculator","arguments":{"expression":` 这种残片
            # 补成 `{"expression": null}` 甚至 `{}`，然后带着默认值真的去执行一次工具。
            if truncated and isinstance(parsed, ToolCall):
                exc = ParseError(
                    "你的输出**没有输出完整**（工具调用被截断），因此这次调用没有执行。",
                    raw=raw,
                    hint="请重新输出完整的一个 JSON 对象，不要中途截断。",
                    detail={"raw_output": truncate(raw, 400)},
                )
                validation_errors.append((exc, repairs))
                errors.append(exc.message)
                return None
            return parsed

        # 阶段一：权威候选 —— 只要 JSON 合法，它的校验结论就是最终结论，不再降级尝试切片
        for candidate in primary:
            parsed = attempt(candidate)
            if parsed is not None:
                return parsed
            if validation_errors:
                break

        # 阶段二：兜底候选（仅在权威候选连 JSON 都不合法时才走到这里）
        if not validation_errors:
            for candidate in fallback:
                parsed = attempt(candidate)
                if parsed is not None:
                    return parsed

        # 1) schema / 协议错误优先：它告诉模型「哪里填错了」，比「格式不合法」有用得多
        #    但「输出不完整」的提示比「缺字段」更准确，所以先挑带截断修复的那条。
        if validation_errors:
            chosen = next(
                (item for item in validation_errors if TRUNCATION_REPAIR in item[1]),
                validation_errors[0],
            )[0]
            raise ParseError(
                chosen.message,
                raw=raw,
                hint=chosen.hint,
                detail=chosen.detail or {"raw_output": truncate(raw, 400)},
            )

        # 2) 完全没解析出 JSON：区分两种情况
        #    a) 模型在说人话（没按协议输出）→ 宽松模式下降级为最终回答，别丢掉用户可见内容
        #    b) 模型想输出 JSON 但写坏了 → 必须报错回灌，让模型修正，否则会静默吞掉一次工具调用
        if self.lenient_plain_text and self._looks_like_plain_answer(text):
            return Answer(content=strip_code_fence(text), raw=raw, repaired=["未发现 JSON，按纯文本回答处理"])

        if self._looks_like_broken_json(text):
            raise ParseError(
                "LLM 想输出 JSON 但格式不合法（花括号不配对或缺少引号）。"
                "原因：" + ("; ".join(errors[:3]) or "未找到合法 JSON 对象"),
                raw=raw,
                detail={"raw_output": truncate(raw, 400)},
            )

        raise ParseError(
            "无法从 LLM 输出中解析出合法 JSON。原因：" + ("; ".join(errors[:3]) or "未找到 JSON 对象"),
            raw=raw,
            detail={"raw_output": truncate(raw, 400)},
        )

    def try_parse(self, raw: str) -> tuple[Optional[ParsedOutput], Optional[ParseError]]:
        """不抛异常的版本，返回 (结果, 错误)。"""
        try:
            return self.parse(raw), None
        except ParseError as exc:
            return None, exc

    # ------------------------------------------------------------------ 校验
    def _from_object(self, data: Any, *, raw: str, repairs: list[str]) -> ParsedOutput:
        if isinstance(data, list):
            if not data:
                raise ParseError("LLM 输出了空数组", raw=raw)
            # 注意：不能对元素递归调用 _from_object 并让异常直接冒泡 ——
            # 数组里混着一个写坏的片段时，应该报「数组里有非法元素」，而不是半路中断。
            items: list[ParsedOutput] = []
            element_errors: list[str] = []
            for item in data:
                if not isinstance(item, (dict, list)):
                    continue
                try:
                    items.append(self._from_object(item, raw=raw, repairs=repairs))
                except ParseError as exc:
                    element_errors.append(exc.message)
            calls = [item for item in items if isinstance(item, ToolCall)]
            if not calls:
                answers = [item for item in items if isinstance(item, Answer)]
                if answers:
                    return Answer(
                        content="\n".join(a.content for a in answers),
                        raw=raw,
                        repairs=repairs + ["数组合并为回答"],
                    )
                raise ParseError(
                    f"数组里没有可用的 answer/tool_call: {truncate(safe_json(data), 160)}"
                    + (f"；元素错误: {'; '.join(element_errors[:2])}" if element_errors else ""),
                    raw=raw,
                )
            if len(calls) > 1:
                # 本项目一次只执行一个工具调用。**不能静默丢掉后面的** ——
                # 明确告诉模型「只执行了第一个」，它下次就会只发一个。
                names = ", ".join(f"`{c.tool_name}`" for c in calls)
                raise ParseError(
                    f"你一次输出了多个工具调用（{names}）。本项目一次只执行**一个**工具调用，"
                    "请只输出第一个（或最必要的那个）。",
                    raw=raw,
                    detail={"tool_calls": [c.tool_name for c in calls]},
                )
            first = calls[0]
            return ToolCall(
                tool_name=first.tool_name,
                arguments=first.arguments,
                raw=raw,
                reason=first.reason,
                repaired=first.repaired + ["数组解包"],
            )
        if isinstance(data, str):
            # 形如 {"answer": "..."} 里再套一层 JSON 字符串
            nested, nested_repairs = try_json(data)
            if nested is not None:
                return self._from_object(nested, raw=raw, repairs=repairs + nested_repairs + ["嵌套 JSON 解包"])
            return Answer(content=data, raw=raw, repairs=repairs)
        if not isinstance(data, dict):
            raise ParseError(f"JSON 顶层必须是对象(object)，收到 {type(data).__name__}", raw=raw)

        # ---- 判定类型
        declared = str(self._pick(data, _TYPE_KEYS) or "").strip().lower()
        if declared in _TOOL_TYPE_WORDS:
            return self._build_tool_call(data, raw=raw, repairs=repairs)
        if declared in _ANSWER_TYPE_WORDS:
            return self._build_answer(data, raw=raw, repairs=repairs)
        if declared:
            raise ParseError(
                f'未知的 type=`{declared}`，只允许 "answer" 或 "tool_call"',
                raw=raw,
                detail={"keys": list(data)},
            )
        # 未声明 type：靠字段推断
        if self._looks_like_tool_call(data):
            return self._build_tool_call(data, raw=raw, repairs=repairs + ["缺少 type，按字段推断为 tool_call"])
        if any(key in data for key in _CONTENT_KEYS):
            return self._build_answer(data, raw=raw, repairs=repairs + ["缺少 type，按字段推断为 answer"])
        raise ParseError(
            f"JSON 里既没有 type，也看不出是回答还是工具调用；字段={list(data)}",
            raw=raw,
        )

    @staticmethod
    def _pick(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
        for key in keys:
            if key in data:
                return data[key]
        return None

    @staticmethod
    def _looks_like_tool_call(data: dict[str, Any]) -> bool:
        return any(key in data for key in _TOOL_NAME_KEYS) and any(key in data for key in _ARG_KEYS)

    def _build_answer(self, data: dict[str, Any], *, raw: str, repairs: list[str]) -> Answer:
        content = self._pick(data, _CONTENT_KEYS)
        if content is None:
            raise ParseError(f'type="answer" 但缺少 content 字段；字段={list(data)}', raw=raw)
        if isinstance(content, (dict, list)):
            content = safe_json(content)
        content = str(content).strip()
        if not content:
            raise ParseError('type="answer" 但 content 为空字符串', raw=raw)
        return Answer(content=content, raw=raw, repaired=repairs)

    def _build_tool_call(self, data: dict[str, Any], *, raw: str, repairs: list[str]) -> ToolCall:
        name = self._pick(data, _TOOL_NAME_KEYS)
        if not isinstance(name, str) or not name.strip():
            raise ParseError(f'type="tool_call" 但 tool_name 缺失或非法: {name!r}', raw=raw)
        name = name.strip()

        arguments = self._pick(data, _ARG_KEYS)
        if arguments is None:
            arguments = {}
        if isinstance(arguments, str):
            parsed, sub_repairs = try_json(arguments)
            if parsed is None:
                raise ParseError(
                    f"tool_call.arguments 是字符串但不是合法 JSON: {truncate(arguments, 120)}",
                    raw=raw,
                )
            arguments = parsed
            repairs = repairs + sub_repairs + ["arguments 字符串转对象"]
        if not isinstance(arguments, dict):
            raise ParseError(
                f"tool_call.arguments 必须是 JSON 对象，收到 {type(arguments).__name__}: {truncate(safe_json(arguments), 80)}",
                raw=raw,
            )

        reason = str(self._pick(data, ("reason", "thought", "thinking", "why")) or "")

        if self.registry is not None:
            try:
                arguments = self.registry.validate_call(name, arguments)
            except ToolNotFoundError as exc:
                raise ParseError(
                    f"工具 `{name}` 不存在。{exc.hint}",
                    raw=raw,
                    hint=exc.hint,
                    detail={"tool_name": name, "available": exc.detail.get("available")},
                ) from exc
            except ToolValidationError as exc:
                raise ParseError(
                    exc.message,
                    raw=raw,
                    hint=exc.hint,
                    detail={"tool_name": name, "received": arguments},
                ) from exc
        return ToolCall(tool_name=name, arguments=arguments, raw=raw, reason=reason, repaired=repairs)

    @staticmethod
    def _looks_like_plain_answer(text: str) -> bool:
        """判断一段没有 JSON 的文本是不是「模型直接说的话」。

        失败模式很关键：**不能因为文本里出现了引号或花括号就当它是坏 JSON** ——
        真实模型经常输出「他说 "hello" 然后走了。」「函数签名是 f(x) { return 1 }」这类
        含标点的正常回答，误判会让用户拿到「解析失败」而不是答案。

        判据只看两件事：
        1. 是不是以结构字符开头（`{` / `[`）→ 明显在写 JSON；
        2. 有没有「JSON 结构 + 协议词汇」的组合 → 明显在写协议 JSON。
        其余一律当自然语言。
        """
        stripped = strip_code_fence(text).strip()
        if not stripped:
            return False
        if stripped[0] in "{[":
            return False
        if re.search(r'["\'](type|tool_name|arguments|content|answer|tool_call)["\']\s*:', stripped):
            return False
        if re.search(r'"(tool_name|arguments|tool_call)"', stripped):
            return False
        has_structure = any(ch in stripped for ch in "{}")
        mentions_protocol = re.search(r"\b(tool_name|tool_call)\b", stripped) or "arguments" in stripped or '"type"' in stripped
        has_json_key_shape = re.search(r'"\s*[\w\-]+\s*"\s*:', stripped) is not None
        if has_structure and (mentions_protocol or has_json_key_shape):
            return False
        if not has_structure and has_json_key_shape:
            return False
        return True

    @staticmethod
    def _looks_like_broken_json(text: str) -> bool:
        """判断文本是不是「意图输出 JSON 但写坏了」。

        与 `_looks_like_plain_answer` 互补：只要不是自然语言，就是在写 JSON（只是写坏了）。
        """
        stripped = strip_code_fence(text).strip()
        if not stripped:
            return False
        if stripped[0] in "{[":
            return True
        return not Parser._looks_like_plain_answer(text)


def parse_llm_output(raw: str, registry: Any = None) -> ParsedOutput:
    """无状态便捷入口。"""
    return Parser(registry).parse(raw)
