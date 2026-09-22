"""Prompt 加载与渲染。

System Prompt 存在 `prompts/system_prompt.md`（可编辑、可复现、可 diff），
这里负责读取并把占位符替换成运行时信息（工具 schema、会话信息、时间、轮次上限）。

**关键设计**：System Prompt 固定每次请求时拼接在最前面，**不持久化进 session.messages**，
避免历史里堆叠几十份重复指令（既浪费 token，也会让模型行为漂移）。
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional

from .utils import safe_json

DEFAULT_PROMPT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts", "system_prompt.md")

#: 找不到 prompt 文件时的兜底（保证程序仍可运行）
FALLBACK_SYSTEM_PROMPT = """你是 minimal-agent，一个自研 Runtime 驱动的最小可用 Agent。

每次回复必须只输出一个 JSON 对象：
- 调用工具：{"type":"tool_call","tool_name":"<工具名>","arguments":{...}}
- 给出回答：{"type":"answer","content":"<给用户看的回答>"}

当前可用工具：
{{tools_json}}

会话ID：{{session_id}}  本地时间：{{current_time}}  最大工具轮次：{{max_turns}}
"""


class PromptLoader:
    """读取并渲染 system prompt。"""

    def __init__(self, path: str = DEFAULT_PROMPT_PATH, *, cache: bool = True) -> None:
        self.path = path
        self.cache = cache
        self._template: Optional[str] = None

    def template(self) -> str:
        if self._template is not None and self.cache:
            return self._template
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                self._template = fh.read()
        except OSError:
            self._template = FALLBACK_SYSTEM_PROMPT
        return self._template

    def render(
        self,
        *,
        tools: Iterable[dict[str, Any]],
        session_id: str = "",
        user_id: str = "",
        window_id: str = "",
        current_time: str = "",
        max_turns: int = 10,
        tool_prompt_mode: str = "json",
    ) -> str:
        """把工具列表与运行时信息渲染进模板。"""
        tools = list(tools)
        if tool_prompt_mode == "json":
            tools_doc = safe_json(tools) if tools else "（当前没有注册任何工具，请直接用 answer 回答）"
            if tools:
                # 缩进美化，方便模型读 schema；同时保持确定性（排序在注册表里做）
                import json

                tools_doc = json.dumps(tools, ensure_ascii=False, indent=2)
        else:  # pragma: no cover - 预留自然语言描述模式
            lines = []
            for tool in tools:
                lines.append(f"- {tool.get('name')}: {tool.get('description')}\n  参数: {safe_json(tool.get('parameters'))}")
            tools_doc = "\n".join(lines) or "（无工具）"

        replacements = {
            "{{tools_json}}": tools_doc,
            "{{tool_count}}": str(len(tools)),
            "{{session_id}}": session_id or "(未提供)",
            "{{user_id}}": user_id or "(未提供)",
            "{{window_id}}": window_id or "(未提供)",
            "{{current_time}}": current_time,
            "{{max_turns}}": str(max_turns),
        }
        text = self.template()
        for key, value in replacements.items():
            text = text.replace(key, value)
        return text


def load_system_prompt(**kwargs: Any) -> str:
    """便捷函数：加载 + 渲染。"""
    return PromptLoader(kwargs.pop("path", DEFAULT_PROMPT_PATH)).render(**kwargs)
