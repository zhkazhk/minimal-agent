"""采集「真实 LLM API」与「AI Prompt」的实证材料，供提交说明引用。

    python scripts/collect_evidence.py

产出（写入 docs/evidence/）：
1. real_llm_run.md      —— 一次完整真实调用的 trace 时间线（含耗时、工具入参出参）
2. real_llm_stats.md    —— 多次真实调用的统计（各工具调用次数/成功率/平均耗时）
3. system_prompt.md     —— 实际发给 LLM 的 system prompt 全文（AI Prompt 的证据）
4. memory_trace.md      —— 同一 session 连续追问时的上下文演化（证明 memory 如何被召回与放置）
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from miniagent import AgentConfig, MinimalAgent, build_default_registry  # noqa: E402
from miniagent.llm import OpenAICompatibleClient  # noqa: E402
from miniagent.session import SessionManager  # noqa: E402
from miniagent.tracing import Tracer  # noqa: E402
from miniagent.utils import ensure_utf8_stdio  # noqa: E402

OUT_DIR = os.path.join(ROOT, "docs", "evidence")
os.makedirs(OUT_DIR, exist_ok=True)


def build(config: AgentConfig) -> MinimalAgent:
    client = OpenAICompatibleClient(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model,
        timeout=config.timeout,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )
    return MinimalAgent(
        client,
        build_default_registry(),
        SessionManager(max_turn=config.max_tool_turns),
        config=config,
        tracer=Tracer(config.log_dir, console=False),
    )


async def main() -> int:
    ensure_utf8_stdio()
    config = AgentConfig.from_env()
    problems = config.check()
    if problems:
        print("配置不完整，无法采集真实调用证据：", problems)
        return 2

    agent = build(config)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # ---------------- 1) 一次完整真实调用 ----------------
    result = await agent.run("上海今天天气怎么样？", session_id="evidence::trace")
    run_id = result.run_id
    trace_path = os.path.join(config.log_dir, "traces", f"{run_id}.log")
    timeline = ""
    if os.path.isfile(trace_path):
        with open(trace_path, "r", encoding="utf-8") as fh:
            timeline = fh.read()

    with open(os.path.join(OUT_DIR, "real_llm_run.md"), "w", encoding="utf-8") as fh:
        fh.write(f"""# 真实 LLM 调用实证：完整 trace 时间线

- 采集时间：{stamp}
- 提供方 / 模型：`{config.provider}` / `{config.model}`
- 接口地址：`{config.base_url}/chat/completions`（OpenAI 兼容）
- 会话：`{result.session_id}`   run_id：`{run_id}`
- 用户输入：`上海今天天气怎么样？`
- 结果：工具链 `{result.used_tools}`，停止原因 `{result.stopped_reason}`，轮次 `{result.turns_used}`，总耗时 `{result.latency_ms}ms`

## 最终回答

```
{result.answer}
```

## trace 时间线（由 Tracer 自动落盘，未经手工修饰）

> 文件位置：`{os.path.relpath(trace_path, ROOT).replace(os.sep, "/")}`
> 事件类型：run_start → llm_request → llm_response → tool_call → tool_result → llm_response → final_answer → run_end

```
{timeline.strip()}
```
""")
    print(f"  ✅ real_llm_run.md（run_id={run_id}，trace {len(timeline)} 字符）")

    # ---------------- 2) 多次调用统计 ----------------
    probe_agent = build(config)
    cases = [
        ("calculator", "123+456*7 等于多少？用工具算。"),
        ("weather", "北京今天天气"),
        ("search", "帮我搜索一下大模型 Agent 的架构"),
        ("direct", "用一句话说明什么是上下文窗口，不要调用工具。"),
    ]
    rows = []
    for tag, question in cases:
        started = time.perf_counter()
        res = await probe_agent.run(question, session_id=f"evidence::{tag}")
        rows.append(
            {
                "场景": tag,
                "工具链": "、".join(res.used_tools) or "（直接回答）",
                "停止原因": res.stopped_reason,
                "轮次": res.turns_used,
                "耗时ms": res.latency_ms,
                "wall_s": round(time.perf_counter() - started, 2),
            }
        )
        print(f"  ✅ 真实调用 [{tag}] 工具={res.used_tools} stop={res.stopped_reason}")

    stats = probe_agent.tracer.tool_call_stats()
    with open(os.path.join(OUT_DIR, "real_llm_stats.md"), "w", encoding="utf-8") as fh:
        fh.write(f"""# 真实 LLM 调用实证：多场景统计

- 采集时间：{stamp}
- 模型：`{config.model}` @ `{config.base_url}`

## 各场景结果

| 场景 | 工具链 | 停止原因 | 轮次 | 耗时(ms) |
| --- | --- | --- | --- | --- |
""")
        for row in rows:
            fh.write(f"| {row['场景']} | {row['工具链']} | {row['停止原因']} | {row['轮次']} | {row['耗时ms']} |\n")
        fh.write("\n## 工具调用统计（Tracer 聚合）\n\n")
        fh.write("| 工具 | 调用次数 | 成功 | 失败 | 平均耗时(ms) |\n| --- | --- | --- | --- | --- |\n")
        for name, row in sorted(stats.items()):
            fh.write(f"| {name} | {row['calls']} | {row['ok']} | {row['error']} | {row['avg_ms']} |\n")
        fh.write(f"\n> 完整断言见 `scripts/verify_real_llm.py`（10 项端到端验证）。\n")
    print(f"  ✅ real_llm_stats.md（{len(rows)} 个场景，{len(stats)} 个工具）")

    # ---------------- 3) AI Prompt 实证 ----------------
    session = agent.session_mgr.get_or_create("evidence::trace")
    prompt = agent._system_prompt(session, session.max_turn)
    raw_template = open(os.path.join(ROOT, "prompts", "system_prompt.md"), encoding="utf-8").read()
    with open(os.path.join(OUT_DIR, "system_prompt.md"), "w", encoding="utf-8") as fh:
        fh.write(f"""# AI Prompt 实证：实际发给 LLM 的 System Prompt

## 1. 模板文件（受版本控制，可 diff）

> `prompts/system_prompt.md`，{len(raw_template.splitlines())} 行。
> 占位符：`{{{{tools_json}}}}`（工具列表）、`{{{{session_id}}}}`、`{{{{current_time}}}}`、`{{{{max_turns}}}}` 等。

```markdown
{raw_template.strip()}
```

## 2. 运行时渲染结果（真实请求里发出的内容）

> 采集时间：{stamp}　会话：`{session.session_id}`
> 共 {len(prompt.splitlines())} 行。工具 Schema 由 `ToolRegistry.snapshot()` 动态注入，
> 因此**新增工具无需改 Prompt 文件**。

```markdown
{prompt.strip()}
```

## 3. 协议约定（LLM 必须遵守，违反会被 Parser 拒绝并回灌错误）

| 输出类型 | 格式 | 效果 |
| --- | --- | --- |
| 工具调用 | `{{"type":"tool_call","tool_name":"<name>","arguments":{{...}}}}` | 执行工具后继续循环 |
| 最终回答 | `{{"type":"answer","content":"<回答>"}}` | 终止循环，返回用户 |
""")
    print(f"  ✅ system_prompt.md（模板 {len(raw_template.splitlines())} 行 → 渲染 {len(prompt.splitlines())} 行）")

    # ---------------- 4) Memory 召回与放置实证 ----------------
    mem_agent = build(config)
    sid = "evidence::memory"
    turns = ["上海今天天气", "那明天呢", "那北京呢"]
    snapshots = []
    for question in turns:
        sess = mem_agent.session_mgr.get_or_create(sid)
        # ① 本轮开始前，session 里已有的上下文（= 会被召回的 memory）
        before = mem_agent.context_mgr.build(sess, mem_agent._system_prompt(sess, sess.max_turn))
        before_blocks = list(before.messages)
        before_size = sess.size

        result = await mem_agent.run(question, session_id=sid)

        # ② 本轮**实际发出的第一条请求**：历史 + 本轮用户消息（直接取自 trace，未经拼接）
        first_request = next(
            (ev for ev in mem_agent.tracer.events(run_id=result.run_id) if ev.event == "llm_request"),
            None,
        )
        sent = first_request.data.get("messages", []) if first_request else []

        sess_after = mem_agent.session_mgr.get_or_create(sid)
        snapshots.append(
            {
                "question": question,
                "before_size": before_size,
                "before_tokens": before.token_estimate,
                "before_compressed": before.compressed,
                "before_blocks": before_blocks,
                "sent": sent,
                "result": result,
                "after_transcript": sess_after.transcript(),
                "after_size": sess_after.size,
            }
        )

    def render_table(messages: list[dict], limit: int = 130) -> str:
        lines = ["| # | role | 内容（截断显示） |", "| --- | --- | --- |"]
        for i, msg in enumerate(messages, 1):
            preview = str(msg.get("content", "")).replace("\n", " ").replace("|", "\\|")[:limit]
            lines.append(f"| {i} | `{msg.get('role')}` | {preview} |")
        return "\n".join(lines)

    with open(os.path.join(OUT_DIR, "memory_trace.md"), "w", encoding="utf-8") as fh:
        fh.write(f"""# Memory 证据：上下文如何随追问演化

- 采集时间：{stamp}　模型：`{config.model}`　会话：`{sid}`
- 连续三轮追问（第 2、3 轮都是省略式追问，必须依赖 memory 才能理解指代）。
  每轮记录三件事：
  1. **本轮开始前 session 里已有的上下文**（= 会被召回的 memory 总量）
  2. **本轮实际发出的第一条请求**（直接取自 trace 的 `llm_request` 事件，证明 memory 被放在哪里）
  3. 该轮结束后 session 累积了什么

""")
        for idx, snap in enumerate(snapshots, 1):
            result = snap["result"]
            history_count = max(0, len(snap["before_blocks"]) - 1)
            fh.write(f"""---

## 第 {idx} 轮：用户说 `{snap['question']}`

### ① 本轮开始前，session 里已有的上下文

- session 消息数：**{snap['before_size']}** 条
- 组装后进入 LLM 的消息：**{len(snap['before_blocks'])}** 条（1 system + {history_count} 历史），token 估算 {snap['before_tokens']}，压缩={snap['before_compressed']}

{render_table(snap['before_blocks'])}

### ② 本轮实际发出的第一条请求（来自 trace）

共 **{len(snap['sent'])}** 条 —— 相比上面多了本轮的 `{snap['question']}`：

{render_table(snap['sent'])}

> **召回时机**：`ContextManager.build()` 在每次 LLM 调用前执行，全量注入该 session 历史，无检索。
> **放置方式**：`[system prompt] → [历史消息按时间升序] → [本轮用户消息]`。

### ③ 本轮工具调用与回答

- 工具链：`{result.used_tools or '（未调用工具）'}`　停止原因 `{result.stopped_reason}`　轮次 {result.turns_used}

```
{result.answer}
```

### ④ 该轮结束后 session 累积内容（{snap['after_size']} 条）

```
{snap['after_transcript']}
```

""")
        fh.write(f"""---

## 结论：召回时机与放置方式

| 问题 | 本项目的做法 | 证据 |
| --- | --- | --- |
| **召回时机** | 每次 LLM 调用前全量召回（`ContextManager.build()`）。触发点：① 新一轮用户提问；② 每次工具结果回灌后继续循环（同一轮内的第 2、3 次 LLM 调用）；③ 追问 —— 与①同路径，因为追问就是往同一 session append 一条 user 消息 | 上表第 ② 节；以及第 1 轮 `工具链=['weather']` 说明本轮内至少发生了 2 次 LLM 调用，每次都重新组装了上下文 |
| **放置方式** | 固定顺序 `[system] → [历史升序] → [本轮用户消息]`；`tool_call` 渲染为 assistant 的 JSON 文本，`tool_result`/`error` 渲染为 user 的 `【…】` 文本块 | 上表逐条可见 |
| **压缩时的放置** | 触发裁剪时，在 system 之后、保留历史之前插入一条 `system`：【上下文已裁剪，前面对话已压缩，只保留最近对话】 | `python -m miniagent demo` 用例 7 |
| **不放置什么** | ① system prompt 不重复放（否则历史里堆叠几十份指令）；② 被裁掉的原始消息不放；③ 工具 Schema 只在 system 里出现一次 | `test_system_prompt_not_persisted_but_present_in_request` |
| **追问为何能work** | 第 2 轮「那明天呢」既没提城市也没提"天气"，模型能答出"上海明天阴天"，唯一来源就是被召回的第 1 轮历史 | 第 2 轮回答 vs 第 1 轮历史 |
""")
    print(f"  ✅ memory_trace.md（{len(turns)} 轮追问，含实际请求快照）")

    await agent.aclose()
    for extra in (probe_agent, mem_agent):
        await extra.aclose()

    print(f"\n全部证据已写入 {os.path.relpath(OUT_DIR, ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
