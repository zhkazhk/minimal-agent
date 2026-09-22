"""search：mock 搜索引擎。

为了可测试 / 可复现：
- 不联网，内置一个小型「知识库」（若干主题的模拟网页摘要）；
- 命中知识库 → 返回对应摘要；未命中 → 由 query 的哈希**确定性**生成 2~4 条模拟结果；
- 同一个 query 永远返回同一批结果（用例可断言），不同 query 结果不同。

替换成真实搜索：把 `_search_backend` 换成 HTTP 调用即可（见 README「工具扩展方式」）。
"""

from __future__ import annotations

import hashlib
from typing import Any

from .registry import Tool

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "搜索关键词，尽量精简（例如 `大模型 Agent 架构`、`上海 天气`）。",
            "minLength": 1,
        },
        "top_k": {
            "type": "integer",
            "description": "返回的结果条数，1~5，默认 3。",
            "default": 3,
            "minimum": 1,
            "maximum": 5,
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}

#: 内置知识库：主题 → [(标题, 摘要, 来源)]
KNOWLEDGE_BASE: dict[str, list[tuple[str, str, str]]] = {
    "大模型": [
        ("大语言模型（LLM）概述", "大语言模型是基于 Transformer 的神经网络，通过海量文本自监督预训练获得通用语言能力，典型代表有 GPT、Claude、Gemini、DeepSeek 等。", "mock://encyclopedia/llm"),
        ("Scaling Law 与涌现能力", "参数量、数据量与算力同步放大时，模型损失呈幂律下降；当规模越过某个阈值，会出现少量样本学习、思维链推理等涌现能力。", "mock://paper/scaling-law"),
        ("预训练 / 微调 / 对齐三阶段", "现代 LLM 训练通常分三步：预训练打底、SFT 指令微调、RLHF/DPO 人类偏好对齐，最后再做安全与幻觉治理。", "mock://blog/llm-pipeline"),
    ],
    "大模型 agent": [
        ("Agent = LLM + 规划 + 记忆 + 工具", "主流 Agent 架构把 LLM 当作推理内核，外围补上任务规划、记忆管理与工具调用能力，形成「感知-决策-行动」闭环。", "mock://blog/agent-architecture"),
        ("ReAct 范式", "ReAct 让模型交替产出「思考(Thought)」与「行动(Action)」，根据工具返回的观察(Observation)继续推理，是工具调用型 Agent 的经典范式。", "mock://paper/react"),
        ("上下文工程比 Prompt 工程更关键", "工具 schema、历史裁剪策略、错误回灌格式共同决定 Agent 的稳定性，工程上要优先保证输出协议可解析。", "mock://blog/context-engineering"),
    ],
    "python": [
        ("Python 官方文档", "Python 是一门解释型、动态类型的通用编程语言，标准库覆盖 asyncio、json、dataclasses 等常用能力。", "mock://docs.python.org/3"),
        ("asyncio 入门", "asyncio 通过事件循环实现单线程并发；`async def` 定义协程，`await` 挂起等待，适合 I/O 密集的 LLM 调用场景。", "mock://docs.python.org/3/library/asyncio.html"),
    ],
    "json": [
        ("JSON 规范", "JSON 仅支持双引号字符串，不允许注释与尾随逗号；因此 LLM 输出常需要做容错解析（见 miniagent/parser.py）。", "mock://www.json.org"),
    ],
    "天气": [
        ("天气预报是怎么做出来的", "数值天气预报把大气分层网格化，用流体力学方程做时间积分，再结合统计订正输出气温、降水、风力等要素。", "mock://encyclopedia/weather-forecast"),
    ],
    "注意力机制": [
        ("Self-Attention", "自注意力把序列中每个位置的表示投影为 Query/Key/Value，用 QK^T 缩放点积计算权重后加权求和，复杂度 O(n²)。", "mock://paper/attention-is-all-you-need"),
    ],
}

_TEMPLATES = [
    ("{q} —— 快速入门指南", "围绕「{q}」的入门材料：先厘清核心概念与术语，再给一个可运行的最小示例，最后列出 3 个常见坑。", "mock://guide/{slug}"),
    ("{q} 常见问题 FAQ", "关于「{q}」被问得最多的 5 个问题：它解决什么问题、适用边界、性能开销、与替代方案的差异、如何上手。", "mock://faq/{slug}"),
    ("{q} 实践笔记", "工程实践记录：在真实项目中引入「{q}」的收益与代价，包含踩坑清单和调优参数建议。", "mock://notes/{slug}"),
    ("{q} 深度解析（上）", "从原理出发拆解「{q}」的内部机制，配有对比表格，并讨论常见误解。", "mock://deepdive/{slug}"),
    ("{q} 相关开源项目汇总", "整理了 6 个与「{q}」相关的开源项目，覆盖轻量实现与生产级方案，附活跃度对比。", "mock://awesome/{slug}"),
]


def _slug(text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    ascii_part = "".join(ch for ch in text.lower() if ch.isalnum() and ch.isascii())[:16]
    return f"{ascii_part or 'topic'}-{digest}"


def _kb_hits(query: str) -> list[tuple[str, str, str]]:
    """在知识库里做朴素子串/词命中匹配。"""
    normalized = query.lower().replace(" ", "")
    hits: list[tuple[str, str, str]] = []
    for topic, entries in KNOWLEDGE_BASE.items():
        topic_key = topic.lower().replace(" ", "")
        tokens = [t for t in _split_tokens(normalized) if len(t) >= 2]
        if topic_key in normalized or normalized in topic_key or any(t in topic_key or topic_key in t for t in tokens):
            hits.extend(entries)
    return hits


def _split_tokens(text: str) -> list[str]:
    """中英混排的粗暴切分：英文按空白/标点，中文按 2~4 字滑窗。"""
    import re

    tokens = re.findall(r"[a-zA-Z0-9_\-\.]{2,}", text)
    cjk = re.findall(r"[\u4e00-\u9fff]+", text)
    for run in cjk:
        for size in (4, 3, 2):
            for i in range(0, max(1, len(run) - size + 1)):
                tokens.append(run[i : i + size])
    return tokens


def mock_search(query: str, top_k: int = 3) -> str:
    """返回给 LLM 的模拟搜索摘要（确定性）。"""
    query = (query or "").strip()
    if not query:
        return "搜索关键词为空，未获得结果。"
    top_k = max(1, min(int(top_k), 5))

    hits = _kb_hits(query)
    results: list[tuple[str, str, str]] = list(hits)

    if len(results) < top_k:
        seed = int(hashlib.sha1(query.encode("utf-8")).hexdigest(), 16)
        pool = list(_TEMPLATES)
        slug = _slug(query)
        for i in range(len(results), top_k):
            title_tpl, summary_tpl, url_tpl = pool[(seed + i * 7) % len(pool)]
            results.append((
                title_tpl.format(q=query),
                summary_tpl.format(q=query),
                url_tpl.format(slug=f"{slug}-{i + 1}"),
            ))

    lines = [f"模拟搜索结果（query={query!r}，共 {min(len(results), top_k)} 条）:"]
    for idx, (title, summary, url) in enumerate(results[:top_k], 1):
        lines.append(f"[{idx}] {title}\n    摘要: {summary}\n    来源: {url}")
    if not hits:
        lines.append("（注：本次结果由 mock 引擎生成，非真实互联网检索结果。）")
    return "\n".join(lines)


def search_tool() -> Tool:
    return Tool(
        name="search",
        description=(
            "网页搜索工具（mock 实现）。当问题需要外部事实、百科知识、最新资料、"
            "或你不确定准确答案时调用。输入搜索关键词，返回若干条网页标题+摘要+来源链接。"
        ),
        parameters=SCHEMA,
        handler=mock_search,
        tags=("retrieval", "mock"),
    )
