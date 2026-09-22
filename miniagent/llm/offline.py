"""离线 LLM 客户端：让 Agent 框架在**没有 API Key / 没有网络**时也能被完整测试。

1. `ScriptedLLMClient` —— 剧本客户端。按预设序列逐条返回，用来**确定性复现** 7 个测试用例
   （例如「第一次调用故意返回坏 JSON，第二次返回正确 tool_call」）。

2. `OfflineMockClient` —— 规则版「假 LLM」。它不是模型，而是一个**行为近似 LLM 的决策表**：
   读上下文 → 决定「调工具 or 直接回答」→ 工具结果回来后组织自然语言答案。
   用来跑通 demo、录屏、CI。真实场景请把 provider 换成 deepseek/openai（同一套接口）。

两个客户端的内部状态都按 `session_id`（从 messages 里解析出的会话标记）隔离，
因此天然支持「用户 A 开两个窗口」的并发隔离验证。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence, Union

from ..errors import LLMError
from ..utils import make_session_id, safe_json, truncate
from .base import LLMClient, LLMRequest, LLMResponse

# ---------------------------------------------------------------------------
# 1) 剧本客户端
# ---------------------------------------------------------------------------

Step = Union[str, dict, BaseException, Callable[[LLMRequest], Any]]


class ScriptedLLMClient(LLMClient):
    """按剧本返回预设内容。

    用法::

        client = ScriptedLLMClient([
            '{"type":"tool_call","tool_name":"calculator","arguments":{"expression":"123+456*7"}}',
            '{"type":"answer","content":"123+456*7 = 3315"}',
        ])
        client.push('{"type":"answer","content":"追答"}')   # 支持运行中追加

    剧本元素可以是：
    - `str`  → 原样作为 LLM 输出（用来喂坏 JSON、缺字段、markdown 包裹等脏数据）
    - `dict` → 自动 `json.dumps`（省得手写转义）
    - `Exception` 实例 → 直接抛出（模拟网络异常/超时）
    - `callable(request) -> str|dict|Exception` → 动态生成
    剧本用尽后：`on_exhausted="raise"`（默认）抛错，`"loop"` 从头循环，`"default"` 返回 `default_response`。
    """

    name = "scripted"

    def __init__(
        self,
        script: Iterable[Step] = (),
        *,
        on_exhausted: str = "raise",
        default_response: str = '{"type":"answer","content":"（剧本已用尽）"}',
    ) -> None:
        self.script: list[Step] = list(script)
        self.on_exhausted = on_exhausted
        self.default_response = default_response
        self.cursor = 0
        self.calls: list[LLMRequest] = []

    def push(self, *steps: Step) -> None:
        self.script.extend(steps)

    def reset(self, script: Optional[Iterable[Step]] = None) -> None:
        if script is not None:
            self.script = list(script)
        self.cursor = 0
        self.calls.clear()

    async def chat(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if self.cursor >= len(self.script):
            if self.on_exhausted == "loop" and self.script:
                self.cursor = 0
            elif self.on_exhausted == "default":
                return LLMResponse(content=self.default_response, model="scripted", finish_reason="stop")
            else:
                raise LLMError(
                    f"剧本已用尽（第 {self.cursor + 1} 次调用无内容）。"
                    f"这通常意味着 Agent 循环比预期多跑了一轮 —— 请检查上一轮是否为 tool_call 且没有终止性 answer。"
                )
        step = self.script[self.cursor]
        self.cursor += 1

        if callable(step) and not isinstance(step, (str, dict, BaseException)):
            step = step(request)  # type: ignore[operator]
        if isinstance(step, BaseException):
            if isinstance(step, LLMError):
                raise step
            raise LLMError(f"剧本注入的异常: {type(step).__name__}: {step}") from step
        if isinstance(step, dict):
            content = json.dumps(step, ensure_ascii=False)
        else:
            content = str(step)
        return LLMResponse(content=content, model="scripted", finish_reason="stop", usage={"scripted": True})


# ---------------------------------------------------------------------------
# 2) 规则版离线客户端
# ---------------------------------------------------------------------------

_MATH_CHARS = set("0123456789+-*/%^().= ")
_WORD_NUM = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}
_CJK_NUM = re.compile(r"[零一二两三四五六七八九十百千万]+")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

_TOOL_CALL_RE = re.compile(r'"type"\s*:\s*"tool_call".*?"tool_name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}', re.DOTALL)
_INVALID_ARG_RE = re.compile(r"`([^`]+)` 参数校验失败[:：]\s*(.+)")
_MISSING_FIELD_RE = re.compile(r"缺少必填字段 `([^`]+)`")

_WEATHER_WORDS = ("天气", "气温", "温度", "下雨", "会热", "会冷", "空气质量", "aqi", "weather", "穿什么", "带伞", "冷不冷", "热不热")
_TEMPORAL_WORDS = ("明天", "后天", "大后天", "今天", "现在", "tomorrow", "today", "呢", "那")
_CALC_HINT_WORDS = ("计算", "算", "等于", "多少", "calculate", "=", "?")

KNOWN_CITIES = [
    "上海", "北京", "深圳", "广州", "杭州", "成都", "西安", "哈尔滨", "南京", "武汉", "重庆", "天津",
    "苏州", "长沙", "青岛", "厦门", "昆明", "三亚", "拉萨", "乌鲁木齐", "香港", "台北", "澳门",
    "shenzhen", "shanghai", "beijing", "tokyo", "singapore", "london", "new york", "paris", "berlin",
]


@dataclass
class _SessionState:
    """按 session 隔离的极简状态机。"""

    pending_tool: Optional[str] = None
    pending_arguments: dict[str, Any] = field(default_factory=dict)
    last_error: str = ""
    corrected_arguments: dict[str, Any] = field(default_factory=dict)
    last_city: str = ""
    last_expression: str = ""
    seen_tools: list[str] = field(default_factory=list)
    answered: bool = False


def _zh_number_to_int(text: str) -> Optional[int]:
    """把「三千八百」「五十」这类中文数字转成 int（够 demo 用）。"""
    if not text:
        return None
    if "万" in text:
        head, _, tail = text.partition("万")
        high = _zh_number_to_int(head) or 0
        low = _zh_number_to_int(tail) or 0
        return high * 10000 + low
    total, section, number = 0, 0, 0
    for ch in text:
        if ch in _WORD_NUM:
            number = _WORD_NUM[ch]
        elif ch == "十":
            section += (number or 1) * 10
            number = 0
        elif ch == "百":
            section += (number or 1) * 100
            number = 0
        elif ch == "千":
            section += (number or 1) * 1000
            number = 0
        else:
            return None
    return total + section + number


def normalize_math(text: str) -> str:
    """"帮我计算 123 + 456 × 7 等于多少" → "123+456*7" """
    expr = text
    for token in ("帮我计算", "帮我算一下", "帮我算", "请计算", "计算一下", "计算", "算一下", "等于多少", "等于几",
                  "是多少", "结果是多少", "的结果", "等于", "多少", "请问"):
        expr = expr.replace(token, "")
    expr = (
        expr.replace("×", "*").replace("÷", "/").replace("（", "(").replace("）", ")")
        .replace("，", ",").replace("^", "**").replace("？", "").replace("?", "")
        .replace("＝", "=").replace("加", "+").replace("减", "-").replace("乘", "*").replace("除以", "/")
        .replace("的平方", "**2").replace("平方", "**2")
    )
    expr = _CJK_NUM.sub(lambda m: str(_zh_number_to_int(m.group()) or "") , expr)
    expr = re.sub(r"[^0-9+\-*/%^().,\s]", "", expr)
    return re.sub(r"\s+", "", expr).strip()


def looks_like_math(text: str) -> bool:
    expr = normalize_math(text)
    if not expr or not any(op in expr for op in "+-*/%^"):
        return False
    if not re.search(r"\d", expr):
        return False
    return all(ch in _MATH_CHARS or ch == "," for ch in expr)


def extract_city(text: str) -> Optional[str]:
    for city in KNOWN_CITIES:
        if city in text.lower() or city in text:
            return city
    match = re.search(r"([\u4e00-\u9fff]{2,6}?)(?:今天|明天|后天)?(?:的)?(?:天气|气温|温度|下雨)", text)
    if match:
        candidate = match.group(1)
        for noise in ("帮我查", "查一下", "查查", "查询", "请问", "想知道", "看看", "我想", "问一下"):
            candidate = candidate.replace(noise, "")
        if 2 <= len(candidate) <= 6:
            return candidate
    match = re.search(r"([A-Za-z][A-Za-z\s]{2,20}?)\s*(?:weather|temperature)", text, re.I)
    if match:
        return match.group(1).strip().lower()
    return None


def extract_date(text: str) -> str:
    iso = _ISO_DATE.search(text)
    if iso:
        return iso.group()
    for word in ("大后天", "后天", "明天", "今天", "tomorrow", "today"):
        if word in text.lower():
            return {"tomorrow": "明天", "today": "今天"}.get(word, word)
    return "今天"


class OfflineMockClient(LLMClient):
    """规则版离线客户端：无需 API Key，跑通完整 Agent 循环。

    决策优先级（读上下文，不读未来）：
      0. 上一轮工具参数非法（上下文里有 `参数校验失败`）→ 修正参数后重试同一工具
      1. 上一轮工具已返回结果  → 组织自然语言 answer（终止循环）
      2. 追问（"明天呢"）      → 复用上一次 weather 的城市，重新调 weather
      3. 天气意图 / 数学表达式 / 检索意图 → 对应 tool_call
      4. 其他                 → 直接 answer（模拟模型自身知识回答）
    """

    name = "offline-mock"

    def __init__(self) -> None:
        self._states: dict[str, _SessionState] = {}
        self.calls = 0

    # ------------------------------------------------------------ 会话隔离
    @staticmethod
    def session_key(request: LLMRequest) -> str:
        for msg in reversed(request.messages):
            content = str(msg.get("content", ""))
            match = re.search(r"会话(?:ID)?[:：]\s*(\S+)", content)
            if match:
                return match.group(1)
        return request.messages[0].get("content", "default")[:64] if request.messages else "default"

    def state(self, request: LLMRequest) -> _SessionState:
        key = self.session_key(request)
        state = self._states.get(key)
        if state is None:
            state = _SessionState()
            # 进程重启 / 换个实例（SessionStore 里已有历史）时，从上下文里重建状态，
            # 这样「明天呢」这类省略式追问仍然能沿用上一轮的城市。
            self._rehydrate(state, request.messages)
            self._states[key] = state
        return state

    @staticmethod
    def _rehydrate(state: "_SessionState", messages: Sequence[dict[str, str]]) -> None:
        for msg in messages:
            content = str(msg.get("content", ""))
            if msg.get("role") == "assistant" and '"type"' in content and "tool_call" in content:
                match = _TOOL_CALL_RE.search(content)
                if not match:
                    continue
                name = match.group(1)
                state.seen_tools.append(name)
                try:
                    arguments = json.loads(match.group(2))
                except json.JSONDecodeError:
                    continue
                if isinstance(arguments, dict):
                    state.pending_tool = name
                    state.pending_arguments = arguments
                    city = arguments.get("city")
                    if isinstance(city, str) and city:
                        state.last_city = city

    # ------------------------------------------------------------------ 主逻辑
    async def chat(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        state = self.state(request)
        messages = request.messages

        last_user_idx = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=len(messages) - 1)
        user_text = str(messages[last_user_idx].get("content", "")) if messages else ""
        # 工具结果是以 user 角色回灌的，需要剥掉前缀拿到「真正的用户问题」
        tool_result_idx = user_text.find("【工具结果")
        if tool_result_idx >= 0:
            user_text = user_text[:tool_result_idx].strip() or user_text

        # 只认「本次用户消息及其之后」的观察，避免拿上一次提问的旧工具结果作答
        observation, observation_tool = self._last_observation(messages[last_user_idx:])
        has_error = any(
            str(msg.get("content", "")).startswith(("【错误", "【工具执行异常"))
            for msg in messages[last_user_idx:]
        )

        # 0) 上一轮工具参数非法/执行失败 → 修正参数后重试（模拟 LLM 看到错误后的自我修正）
        next_call = self._detect_next_tool(messages)
        if next_call is not None:
            name, args = next_call
            state.pending_tool, state.pending_arguments = name, args
            state.seen_tools.append(name)
            return self._tool_call(name, args)

        # 1) 已经拿到本次提问的工具结果 → 组织自然语言答案，终止循环（关键：不能再重复调同一工具）
        if observation and not has_error and not request.force_final_answer:
            state.answered = True
            return self._answer_response(self._answer(user_text, state, observation, observation_tool))

        # 2) 轮次上限：不再调工具，直接总结
        if request.force_final_answer:
            return self._answer_response(self._answer(user_text, state, observation, observation_tool, forced=True))

        if state.corrected_arguments:
            name = state.pending_tool or "weather"
            args = state.corrected_arguments
            state.corrected_arguments = {}
            state.pending_arguments = args
            state.seen_tools.append(name)
            return self._tool_call(name, args)

        # 3) 追问（"明天呢" / "那北京呢"）→ 沿用上一次的城市，除非用户显式换了城市
        if self._is_followup(user_text) and (state.last_city or extract_city(user_text)) and "weather" in state.seen_tools:
            explicit_city = extract_city(user_text)
            city = explicit_city or state.last_city
            date = extract_date(user_text)
            state.last_city = city                     # 城市切换要记住，供下一次追问使用
            state.pending_tool = "weather"
            state.pending_arguments = {"city": city, "date": date}
            return self._tool_call("weather", state.pending_arguments)

        if any(word in user_text.lower() for word in _WEATHER_WORDS):
            city = extract_city(user_text) or state.last_city or "北京"
            date = extract_date(user_text)
            state.last_city = city
            state.pending_tool, state.pending_arguments = "weather", {"city": city, "date": date}
            state.seen_tools.append("weather")
            return self._tool_call("weather", state.pending_arguments)

        if looks_like_math(user_text):
            expr = normalize_math(user_text) or state.last_expression
            state.last_expression = expr
            state.pending_tool, state.pending_arguments = "calculator", {"expression": expr}
            state.seen_tools.append("calculator")
            return self._tool_call("calculator", {"expression": expr})

        if self._needs_search(user_text) and observation_tool != "search":
            query = self._search_query(user_text)
            state.pending_tool, state.pending_arguments = "search", {"query": query, "top_k": 3}
            state.seen_tools.append("search")
            return self._tool_call("search", {"query": query, "top_k": 3})

        return self._answer_response(self._answer(user_text, state, observation, observation_tool))

    # ------------------------------------------------------------ 上下文解析
    @staticmethod
    def _last_observation(messages: Sequence[dict[str, str]]) -> tuple[str, str]:
        """从上下文尾部往前找最近一条工具结果，返回 (去掉标记头的结果正文, 工具名)。"""
        for msg in reversed(list(messages)):
            content = str(msg.get("content", ""))
            if not content.startswith(("【工具结果", "【工具执行异常", "【错误")):
                continue
            tool = ""
            match = re.search(r"【工具(?:结果|执行异常)[:：]?\s*([\w\-]+)", content)
            if match:
                tool = match.group(1)
            # 结果正文在标记头的下一行
            body = content.split("\n", 1)[1].strip() if "\n" in content else content
            return body or content, tool
        return "", ""

    def _detect_next_tool(self, messages: Sequence[dict[str, str]]) -> Optional[tuple[str, dict[str, Any]]]:
        """检测上下文里是否有「工具参数非法/执行失败」需要修正后重试。"""
        for msg in reversed(list(messages)[-4:]):
            content = str(msg.get("content", ""))
            if not content.startswith(("【错误", "【工具执行异常")):
                continue
            bad = _INVALID_ARG_RE.search(content)
            if bad:
                tool, reason = bad.group(1), bad.group(2)
                corrected = self._correct(tool, reason, messages, content)
                if corrected is not None:
                    return tool, corrected
            missing = _MISSING_FIELD_RE.search(content)
            if missing:
                tool = self._tool_from_json(content) or "weather"
                corrected = self._correct(tool, f"缺少必填字段 {missing.group(1)}", messages, content)
                if corrected is not None:
                    return tool, corrected
        return None

    @staticmethod
    def _tool_from_json(text: str) -> Optional[str]:
        match = _TOOL_CALL_RE.search(text)
        return match.group(1) if match else None

    def _correct(
        self,
        tool: str,
        reason: str,
        messages: Sequence[dict[str, str]],
        error_text: str,
    ) -> Optional[dict[str, Any]]:
        """模拟 LLM 看到校验错误后「修正参数」：从**原始用户问题**里补齐。

        注意：上下文末尾的 user 消息是回灌的错误文本，不是用户说的话；
        必须跳过 `【错误` / `【工具结果` 前缀，并从最早的追问里找原始问题。
        """
        candidates: list[str] = []
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = str(msg.get("content", ""))
            if content.startswith(("【错误", "【工具结果", "【工具执行异常")):
                continue
            candidates.append(content)
        # 最早的候选最可能是「原始问题」（追问里往往只有『明天呢』这类省略句）
        original = candidates[0] if candidates else (candidates[-1] if candidates else "")
        if not original:
            original = next(
                (str(m.get("content", "")) for m in messages if m.get("role") == "user"),
                "",
            )

        if tool == "weather":
            city = extract_city(original)
            if not city:
                for candidate in reversed(candidates):
                    city = extract_city(candidate)
                    if city:
                        break
            city = city or "北京"
            for state in self._states.values():
                state.last_city = city
            return {"city": city, "date": extract_date(original)}
        if tool == "calculator":
            expr = normalize_math(original)
            if expr:
                return {"expression": expr}
            numbers = _NUMBER.findall(original)
            if numbers:
                return {"expression": "+".join(numbers)}
        if tool == "search":
            return {"query": original.strip()[:60] or "大模型", "top_k": 3}
        return None

    # ------------------------------------------------------------ 决策细节
    @staticmethod
    def _is_followup(text: str) -> bool:
        stripped = re.sub(r"[\s。！？!?，,]", "", text)
        if len(stripped) <= 6 and any(w in stripped for w in ("呢", "那", "同上", "继续", "然后")):
            return True
        return bool(re.fullmatch(r"(那)?(明天|后天|大后天|今天|昨天)?(呢|呢\?|如何|怎么样)?", stripped))

    @staticmethod
    def _needs_search(text: str) -> bool:
        triggers = ("搜索", "搜一下", "查一下", "帮我查", "检索", "最新", "资料", "文档", "是什么", "什么是",
                    "介绍一下", "讲讲", "解释", "原理", "怎么做", "区别", "对比", "search", "how to")
        return any(word in text.lower() for word in triggers)

    @staticmethod
    def _search_query(text: str) -> str:
        query = text
        for noise in ("帮我搜索一下", "帮我搜一下", "搜索一下", "搜一下", "帮我查一下", "帮我查", "查一下",
                      "检索一下", "请介绍一下", "介绍一下", "讲讲", "解释一下", "解释", "请问", "谢谢"):
            query = query.replace(noise, "")
        return query.strip("？?。！!，, ") or text.strip()[:40]

    # ------------------------------------------------------------ 输出构造
    @staticmethod
    def _tool_call(name: str, arguments: dict[str, Any]) -> LLMResponse:
        payload = {"type": "tool_call", "tool_name": name, "arguments": arguments}
        return LLMResponse(
            content=json.dumps(payload, ensure_ascii=False),
            model="offline-mock",
            finish_reason="tool_calls",
        )

    @staticmethod
    def _answer_response(content: str) -> LLMResponse:
        payload = {"type": "answer", "content": content}
        return LLMResponse(content=json.dumps(payload, ensure_ascii=False), model="offline-mock", finish_reason="stop")

    def _answer(
        self,
        user_text: str,
        state: _SessionState,
        observation: str,
        tool: str,
        *,
        forced: bool = False,
    ) -> str:
        """把工具结果组织成自然语言最终答案。"""
        prefix = "（已达工具调用轮次上限，以下是基于现有信息的总结）\n" if forced else ""

        if tool == "calculator" and observation:
            payload = self._extract_json(observation)
            if payload and "temperature" not in payload:
                expr = payload.get("expression") or state.last_expression
                result = payload.get("result")
                if result is not None:
                    return prefix + f"计算完成：{expr} = {result}。" if expr else prefix + f"计算结果是 {result}。"
            # calculator 的纯文本形态是 "123+456*7 = 3315"
            match = re.search(r"(.+?)\s*=\s*([^\s=]+)\s*$", observation.strip())
            if match:
                return prefix + f"计算完成：{match.group(1).strip()} = {match.group(2).strip()}。"
            line = self._first_meaningful_line(observation)
            if line:
                return prefix + f"计算完成：{line}"
            return prefix + "计算已完成，但没能从工具结果中解析出数值。"

        if tool == "weather" and observation:
            payload = self._extract_json(observation)
            city = (payload or {}).get("city") or state.last_city or "该城市"
            date = (payload or {}).get("date") or "今天"
            if payload and payload.get("temperature"):
                tips = self._weather_tip(str(payload.get("condition", "")), payload)
                return (
                    prefix
                    + f"{city}{date}天气：{payload.get('condition')}，气温 {payload.get('temperature')}"
                    f"（全天 {payload.get('temp_range')}），湿度 {payload.get('humidity')}，"
                    f"风力 {payload.get('wind')}，AQI {payload.get('aqi')}。{tips}"
                )
            line = self._first_meaningful_line(observation)
            return prefix + f"{city}{date}天气查询结果：{line}"

        if tool == "search" and observation:
            lines = [ln.strip() for ln in observation.splitlines() if ln.strip().startswith("[")]
            if lines:
                bullets = "\n".join(f"  {ln}" for ln in lines[:3])
                return (
                    prefix
                    + f"我检索到以下资料（来自 mock 搜索工具）：\n{bullets}\n"
                    "综合来看，它们指向同一结论：该主题的核心在于概念边界、典型实现与工程取舍三方面。"
                )
            return prefix + self._first_meaningful_line(observation)

        if observation:
            return prefix + self._first_meaningful_line(observation)

        topic = re.sub(r"^(请|帮我|麻烦)?(介绍一下|讲讲|解释一下|什么是|说明)", "", user_text).strip("？?。！! ") or "这个话题"
        if any(word in user_text for word in ("你好", "hi", "hello", "在吗")):
            return "你好！我是 minimal-agent 的最小可用 Agent。我可以调用 calculator（计算）、search（检索）、weather（天气）三个工具，也可以直接回答常识问题。"
        return (
            f"关于「{topic}」：这是一个不需要调用工具就能回答的问题。"
            "大模型（LLM）本质上是一个自回归的下一个 token 预测器，通过海量语料预训练获得通用语言与推理能力；"
            "在工程上，它通常被拆成「预训练 → 指令微调 → 偏好对齐」三个阶段，"
            "再通过上下文工程与工具调用扩展成 Agent。（本回答由离线规则客户端生成，非真实模型输出）"
        )

    @staticmethod
    def _weather_tip(condition: str, payload: dict[str, Any]) -> str:
        if "雨" in condition:
            return "建议带伞。"
        try:
            aqi = int(str(payload.get("aqi", "0")).strip() or 0)
        except ValueError:
            aqi = 0
        if aqi > 100:
            return "空气质量一般，敏感人群减少户外活动。"
        if "晴" in condition:
            return "适合外出。"
        return "体感舒适。"

    @staticmethod
    def _extract_json(text: str) -> Optional[dict[str, Any]]:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _first_meaningful_line(text: str) -> str:
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("{"):
                return line
        return truncate(text, 200)
