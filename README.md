# minimal-agent · 从零手写的最小可用 Agent Runtime

> **不使用 LangGraph / LangChain / OpenClaw 等任何 Agent 框架**，自己手写 Agent 主循环。
> 工具注册、LLM 输出解析、Session 隔离、上下文管理、Trace 日志、异常处理全部自研。
> **运行期零第三方依赖**（只用 Python 标准库），克隆下来即可跑通全部验收用例。

```
用户 ──> Agent Runtime ──> LLM API（OpenAI 兼容）
              │
              ├── Session Manager   窗口级会话隔离（user_id + window_id）
              ├── Context Manager   上下文组装 / 轮次限制 / 基础压缩
              ├── Tool Registry     工具注册 · schema 校验 · 执行 · 异常包装
              ├── LLM Parser        JSON 提取 → 容错修复 → schema 校验 → 错误回灌
              ├── Agent Main Loop   Step1~Step6 核心循环 + 4 道护栏
              └── Tracer            结构化 trace + 人可读时间线 + 异常捕获
```

| 项目 | 值 |
| --- | --- |
| 语言 / 版本 | Python 3.9+（开发验证于 3.13.14 / Windows） |
| 运行期依赖 | **无**（`urllib` + `asyncio`；可选 `httpx` / `aiohttp` / `jsonschema`） |
| 代码规模 | 约 9300 行（含注释/文档串）；其中**有效代码约 6000 行**：核心运行时 ≈3550 行，测试 ≈1900 行，示例/Demo ≈600 行 |
| 自动化测试 | **208 passed / 0 failed**（9 个测试模块，含 11 个端到端验收用例） |
| 验收用例 | **11 / 11 通过**（交付要求 7 个必测 + 4 个补充护栏用例） |
| LLM 接口 | 任意 OpenAI 兼容端点（DeepSeek / OpenAI / 通义 / Moonshot / vLLM / Ollama…） |

---

## 目录

- [1. 快速开始](#1-快速开始)
- [2. 系统架构](#2-系统架构)
- [3. 核心 Agent Loop（Step1~Step6）](#3-核心-agent-loopstep1step6)
- [4. 工具层：注册机制与 3 个内置工具](#4-工具层注册机制与-3-个内置工具)
- [5. LLM 输出解析：JSON 协议与容错](#5-llm-输出解析json-协议与容错)
- [6. Session 设计：窗口级隔离](#6-session-设计窗口级隔离)
- [7. Context 设计：组装、轮次限制、基础压缩](#7-context-设计组装轮次限制基础压缩)
- [8. Memory 召回策略](#8-memory-召回策略)
- [9. 异常处理与 Trace 日志](#9-异常处理与-trace-日志)
- [10. 验收用例与实测结果](#10-验收用例与实测结果)
- [11. 测试](#11-测试)
- [12. 工具扩展方式](#12-工具扩展方式)
- [13. 配置项](#13-配置项)
- [14. 问题记录：开发踩坑](#14-问题记录开发踩坑)
- [15. 项目结构](#15-项目结构)
- [16. 已知限制与后续可做](#16-已知限制与后续可做)

---

## 1. 快速开始

### 1.1 零配置跑通（不需要 API Key）

```bash
git clone <your-repo-url> minimal-agent
cd minimal-agent

# 跑一遍 11 个验收用例（离线规则客户端，结果确定性可复现）
python -m miniagent demo

# 跑全部自动化测试（内置零依赖测试运行器，不需要 pip install）
python tests/run_all.py
```

`demo` 的输出（实测节选）：

```
==============================================================================
 minimal-agent 验收用例（不依赖 API Key，全部可复现）
 开始时间: 2026-09-22 17:49:39   用例数: 11
==============================================================================

──────────────────────────────────────────────────────────────────────────────
✅ PASS  用例1：用户输入 `123+456*7`
  预期：Agent 调用 calculator，返回结果 3315
  实测：工具链=['calculator']，answer=计算完成：123+456*7 = 3315。

──────────────────────────────────────────────────────────────────────────────
✅ PASS  用例2：用户输入 `上海今天天气`
  预期：调用 weather 工具，返回 mock 天气（温度/天气状况）
  实测：工具链=['weather']，answer=上海今天天气：多云，气温 28°C（全天 23°C ~ 32°C），
        湿度 75%，风力 4 级，AQI 154。空气质量一般，敏感人群减少户外活动。

──────────────────────────────────────────────────────────────────────────────
✅ PASS  用例4：连续追问：先问「上海今天天气」，再追问「明天呢」
  实测：第1轮=上海今天天气：多云，气温 28°C …
        第2轮=上海明天天气：阴，气温 23°C（全天 19°C ~ 26°C），湿度 68%，风力 5 级，AQI 151。
        session 消息数=8，轮次=2

──────────────────────────────────────────────────────────────────────────────
✅ PASS  用例6：LLM 输出错误工具参数（weather 缺必填参数 city）
  实测：第1次调用被 Parser 拦截（未执行工具），错误已回灌上下文：
        ParseError: 工具 `weather` 参数校验失败: arguments 缺少必填字段 `city`
        第2次调用参数={'city': '上海', 'date': '今天', 'unit': 'celsius'} → 执行成功

──────────────────────────────────────────────────────────────────────────────
 结果：11/11 通过
```

### 1.2 接真实 LLM

```bash
cp .env.example .env
# 编辑 .env，填入 base_url 与 api_key（任何 OpenAI 兼容端点都可）
```

```ini
MINIAGENT_PROVIDER=deepseek
MINIAGENT_BASE_URL=https://api.deepseek.com/v1
MINIAGENT_API_KEY=sk-xxxxxxxx
MINIAGENT_MODEL=deepseek-chat
```

然后：

```bash
python -m miniagent config                       # 自检：打印生效配置 + 缺失项提示
python -m miniagent ask "123+456*7 等于多少"      # 单次提问
python -m miniagent ask "上海今天天气" -u userA -w win1
python -m miniagent ask "上海今天天气" -u userA -w win2 --json   # 另一个窗口（独立会话）
python -m miniagent repl -u userA -w win1        # 交互式多轮（支持追问）
```

### 1.3 作为库使用

```python
import asyncio
from miniagent import MinimalAgent, AgentConfig, build_default_registry
from miniagent.llm import OpenAICompatibleClient
from miniagent.session import SessionManager

async def main():
    config = AgentConfig.from_env()                       # 读 .env / 环境变量
    agent = MinimalAgent(
        OpenAICompatibleClient(
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
            model="deepseek-chat",
        ),
        build_default_registry(),                         # calculator / search / weather
        SessionManager(max_turn=config.max_tool_turns),
        config=config,
    )

    # 第一次提问
    r1 = await agent.run("上海今天天气", user_id="userA", window_id="win1")
    print(r1.answer, r1.used_tools)

    # 追问：同一个 session_id 自带历史，直接再调一次即可
    r2 = await agent.run("明天呢", session_id=r1.session_id, user_id="userA", window_id="win1")
    print(r2.answer)

    # 另一个窗口：完全隔离
    r3 = await agent.run("123+456*7", user_id="userA", window_id="win2")
    print(r3.answer)

    await agent.aclose()

asyncio.run(main())
```

### 1.4 可选依赖（不装也能跑）

| 用途 | 包 | 启用方式 |
| --- | --- | --- |
| 原生 async HTTP | `httpx` | `MINIAGENT_HTTP_BACKEND=httpx` |
| 原生 async HTTP | `aiohttp` | `MINIAGENT_HTTP_BACKEND=aiohttp` |
| 更严格的参数校验 | `jsonschema` | 装了即自动优先使用 |
| 测试（可选） | `pytest` | `pytest -q`（等价于 `python tests/run_all.py`） |

> 默认走标准库 `urllib`（放进线程池执行），因此在**完全离线的环境**里也能 import、能跑 demo、能跑测试。

---

## 2. 系统架构

### 2.1 分层视图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              用户 / 前端                                  │
│               user_id + window_id  →  session_id = "userA::win1"          │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │ user_input
┌────────────────────────────────▼─────────────────────────────────────────┐
│                          Agent Runtime (agent.py)                        │
│                                                                          │
│   ┌────────────────┐   ┌─────────────────┐   ┌───────────────────────┐   │
│   │ SessionManager │   │ ContextManager  │   │    ToolRegistry       │   │
│   │ 窗口级隔离      │──▶│ 组装/裁剪/渲染   │   │ 注册·校验·执行·包装    │   │
│   │ 内存 / JSON 落盘│   │ 轮次规则注入     │   │ calculator/search/... │   │
│   └────────────────┘   └────────┬────────┘   └───────────▲───────────┘   │
│                                 │ messages                │ tool_call     │
│                        ┌────────▼────────┐                │               │
│                        │   LLM Client    │                │               │
│                        │ OpenAI 兼容/剧本 │                │               │
│                        └────────┬────────┘                │               │
│                                 │ raw text                │               │
│                        ┌────────▼────────┐   tool_call    │               │
│                        │     Parser      │────────────────┘               │
│                        │ 提取→修复→校验   │                                │
│                        └────────┬────────┘                                │
│                                 │ answer / ParseError                     │
│   ┌─────────────────────────────▼────────────────────────────────────┐    │
│   │ Tracer：run_start / llm_request / llm_response / tool_call /      │    │
│   │         tool_result / parse_error / context_compressed / exception│    │
│   └──────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 一次带工具调用的完整数据流

```
用户: "上海今天天气"
  │
  ├─(1) SessionManager.get_or_create("userA::win1")
  │      └─ session.messages += [user] "上海今天天气"
  │
  ├─(2) ContextManager.build(session, system_prompt)
  │      ├─ system_prompt 现场拼接（工具 schema + 会话信息 + 时间 + 轮次上限）
  │      ├─ session 历史 → 消息列表（tool_call / tool_result 渲染成文本）
  │      └─ 超阈值 → 丢弃最早消息 + 插入「【上下文已裁剪…】」
  │
  ├─(3) LLMClient.chat()  ──HTTPS──▶  /chat/completions
  │      ◀── '{"type":"tool_call","tool_name":"weather","arguments":{"city":"上海"}}'
  │
  ├─(4) Parser.parse()
  │      ├─ 提取 JSON（代码块 / 花括号配对 / 整段兜底）
  │      ├─ 修复脏数据（全角 / 单引号 / 尾随逗号 / 截断补全）
  │      └─ schema 校验 → ToolCall(tool_name="weather", arguments={...})
  │
  ├─(5) ToolRegistry.execute() → weather(city="上海") → "上海 今天：多云，气温 28°C …"
  │      └─ 异常 → ToolExecutionError → 错误文本
  │
  ├─(6) session.messages += [tool_call], [tool_result]
  │      └─ 回到 (2)，重新调用 LLM
  │
  ├─(7) LLM 输出 '{"type":"answer","content":"上海今天多云，气温 28°C…"}'
  │      └─ Parser → Answer → 写入 session → 返回用户，循环结束
  │
  └─ Tracer 落盘：logs/trace.jsonl（机器可读）+ logs/traces/<run_id>.log（人可读）
```

---

## 3. 核心 Agent Loop（Step1~Step6）

实现位于 `miniagent/agent.py::MinimalAgent.run()`，与交付要求的 Step1~Step6 一一对应：

| 步骤 | 交付要求 | 代码位置 | 说明 |
| --- | --- | --- | --- |
| Step1 | 接收用户消息 + 读取对应 session 上下文 | `session_mgr.get_or_create()` + `session.add_user()` | session_id 由 `user_id + window_id` 决定 |
| Step2 | 组装 prompt（历史 + 工具 schema + query）发给 LLM | `context_mgr.build()` → `_system_prompt()` → `llm.chat()` | system prompt **每次实时拼接，不落库** |
| Step3 | Parser 解析：answer → 结束；tool_call → 继续 | `parser.parse()` | 分支判断 + 解析失败回灌 |
| Step4 | 执行工具（捕获异常 + 记录 trace） | `_execute_tool()` | 超时/校验/内部异常全部转成可回灌文本 |
| Step5 | 工具结果追加到 session 上下文 | `session.add_tool_result()` / `add_error()` | 错误也进上下文，让 LLM 有机会自修 |
| Step6 | 回到 Step2 继续循环，直到输出最终答案 | `while True:` | 循环由 answer / 护栏终止 |

### 3.1 四道生产护栏（超出最小实现，但不做会踩坑）

```
                   ┌──────────────────────────────────────────┐
   while True:     │  ① max_tool_turns 轮次上限                │
      build ───────┤     到顶后禁止再执行工具，强制模型总结      │
        │          │  ② max_parse_retries 解析失败上限          │
        ▼          │     错误回灌让模型自我修正，连续失败则降级   │
   llm.chat        │  ③ 重复调用检测                            │
        │          │     同工具同参数连续出现 → 提示模型换策略    │
        ▼          │  ④ 全链路 trace + 兜底异常处理              │
   parser.parse    │     任何未预期异常都不能把 Runtime 打崩     │
        │          └──────────────────────────────────────────┘
   ┌────┴────┐
   ▼         ▼
 answer   tool_call ──▶ execute ──▶ 回灌上下文 ──┐
   │                                              │
   └──────────────▶ 返回用户                       └──▶ 回到 while 顶部
```

1. **轮次上限**：`max_tool_turns`（默认 10）。达到上限后不再执行工具，向 system prompt 追加
   「本轮**禁止**再调用工具，必须直接输出最终回答」，若模型仍然试图调工具，
   则回灌提示并计入失败次数，超过 `max_parse_retries` 后返回兜底文案。
   实测（`补充用例9`）：脚本故意给 4 次 tool_call，实际只执行 3 次后正常收敛。
2. **解析失败重试**：解析错误作为 `error` 消息回灌上下文，模型看到错误后修正输出。
   连续失败超过 `max_parse_retries`（默认 3）则停止，返回 `parse_error_reply`，绝不静默吞掉。
3. **重复调用检测**：连续两次「同工具 + 同参数」时，向 system prompt 注入
   「你已经用完全相同的参数调用过同一个工具，结果不会变化」，避免烧 token 的死循环。
4. **兜底异常处理**：`run()` 最外层捕获所有未预期异常，写 trace + 返回可读文案，
   保证 Runtime 不会把栈回溯甩到用户界面上。

---

## 4. 工具层：注册机制与 3 个内置工具

### 4.1 工具结构（`miniagent/tools/registry.py`）

```python
{
    "name": "weather",                    # 工具名，LLM 调用标识
    "description": "天气查询工具…",         # 给 LLM 看的能力描述
    "parameters": { ... jsonschema ... }, # 参数 schema
    "handler": callable,                  # 实际执行函数（同步 / async 均可）
}
```

注册表是全局字典，支持注册 / 注销 / 查询 / 执行：

```python
registry = ToolRegistry()
registry.register(Tool(name="echo", description="原样返回",
                       parameters={"type": "object",
                                   "properties": {"text": {"type": "string"}},
                                   "required": ["text"]},
                       handler=lambda text: text))
registry.unregister("echo")        # 注销后 LLM 再调用会收到「工具不存在 + 可用工具列表」
registry.snapshot()                # 工具 schema 清单（按名字排序，保证 prompt 稳定）
registry.describe()                # 人类可读清单（CLI `miniagent tools`）
```

### 4.2 三个内置工具

| 工具 | 参数（schema 摘要） | 实现要点 |
| --- | --- | --- |
| `calculator` | `expression: string`（必填）、`precision: integer`（默认 6） | **安全 eval**：`ast.parse` + 白名单遍历 + 受限命名空间，**全程不使用 eval/exec**；拒绝 `__` / `import` / `lambda` / 属性访问 / 下标 / 超大指数 |
| `search` | `query: string`（必填）、`top_k: integer`（1~5，默认 3） | mock 搜索引擎：内置小知识库命中优先，未命中则由 query 哈希**确定性**生成 2~4 条模拟结果（同 query 恒等结果，便于断言） |
| `weather` | `city: string`（必填）、`date: string`（默认「今天」）、`unit: enum[celsius,fahrenheit]` | mock 天气：内置城市气候基线 + 未收录城市按名字哈希确定性生成；支持今天/明天/后天/ISO 日期，温度按日期偏移，便于验证追问场景 |

`calculator` 的安全策略（`tests/test_tools.py` 有 11 条注入攻击用例）：

```python
safe_eval("123+456*7")                       # → 3315
safe_eval("sqrt(16)+2**10")                  # → 1028
safe_eval("123 + 456 × 7")                   # → 3315（自动归一化全角/中文运算符）
safe_eval("__import__('os').system('ls')")   # ✗ UnsafeExpression
safe_eval("open('/etc/passwd').read()")      # ✗ UnsafeExpression
safe_eval("().__class__.__bases__")          # ✗ UnsafeExpression
safe_eval("9**999999999")                    # ✗ 指数过大，拒绝计算
```

### 4.3 参数校验

内置零依赖 JSON Schema 校验器（`miniagent/schema.py`），支持
`type / properties / required / additionalProperties / items / enum / const /
minimum / maximum / minLength / maxLength / minItems / maxItems / anyOf / oneOf / allOf / default`。

**关键设计**：`coerce_arguments()` 只做**无损**修正
（`"3"`→`3`、`"true"`→`True`、单值→单元素数组、补 `default`），
**不做** number→string 这类反向宽松化 —— 否则 `{"expression": 123}` 会被静默变成 `"123"`，
把模型的理解错误藏起来。实测该场景会被拦下并把「类型应为 string」回灌给模型（见 `用例6` 系列测试）。

---

## 5. LLM 输出解析：JSON 协议与容错

### 5.1 固定 JSON 协议（写在 System Prompt 里）

```jsonc
// 类型 1：最终回答 → 终止循环
{"type":"answer","content":"最终回答给用户"}

// 类型 2：工具调用 → 执行工具
{"type":"tool_call","tool_name":"calculator","arguments":{"expression":"123+456*7"}}
```

### 5.2 Parser 的四件事（`miniagent/parser.py`）

```
raw LLM text
   │
   ├─ ① 提取 JSON
   │     · markdown 代码块（```json … ```，或无语言标记）
   │     · 花括号配对扫描（**字符串内的 {} 与转义字符不参与计数**）
   │     · 整段文本兜底；最多 8 个候选，逐个尝试
   │
   ├─ ② 合法性修复（多级回退，每级都记录到 ToolCall.repaired）
   │     · 全角 → 半角（**仅在字符串外**，避免污染中文答案的「，」）
   │     · 尾随逗号、Python 字面量（None/True/False）、单引号字符串、无引号 key
   │     · 截断补全（max_tokens 用尽：补齐引号 / 括号 / `"key":` 后补 null）
   │     · 从坏串里二次提取花括号片段
   │
   ├─ ③ 协议 + schema 校验
   │     · type ∈ {answer, tool_call}；缺 type 时按字段推断
   │     · 兼容别名：tool_name/tool/name、arguments/args/parameters/input、content/answer/text
   │     · tool_name 必须存在；arguments 必须满足工具 schema（含类型矫正与 default 填充）
   │     · arguments 是 JSON 字符串时自动二次解析
   │
   └─ ④ 解析失败 → ParseError（带 hint + detail）
         └─ 由 Agent 作为 error 消息回灌上下文 → LLM 修正后重试
```

### 5.3 错误信息质量：优先给「最有用的那句」

解析失败时按**信息价值**排序抛出，保证回灌给模型的是可执行建议而不是笼统的「格式错」：

1. **schema / 协议错误优先**：`工具 weather 参数校验失败: arguments 缺少必填字段 city`
   （比「JSON 格式不合法」有用得多）；
2. 格式错误：`LLM 想输出 JSON 但格式不合法（花括号不配对或缺少引号）`；
3. 纯文本降级：模型没按协议、直接说人话时（宽松模式），把它当作 `answer` 而不是丢弃。

每条 `ParseError` 都带 `hint`，例如：

```
ParseError: 工具 `weather` 参数校验失败: arguments 缺少必填字段 `city`
修复建议: 请严格按照 schema 修正 arguments 后重新调用。
```

---

## 6. Session 设计：窗口级隔离

### 6.1 Session Key

```
session_id = f"{user_id}::{window_id}"

用户A ─┬─ 窗口1 ── "userA::win1" ── messages=[…]   ← 计算器对话
       └─ 窗口2 ── "userA::win2" ── messages=[…]   ← 天气对话
用户B ─── 窗口1 ── "userB::win1" ── messages=[…]   ← 与 userA::win1 完全隔离
```

不同 window_id → 独立 message 列表；切换窗口即切换 session。
窗口之间**完全隔离**：不仅消息列表独立，连 `OfflineMockClient` 的决策状态也按 session 分桶。

### 6.2 消息结构（交付要求第 4 节的 5 类信息）

| role | 内容 | 渲染给 LLM 的形式 |
| --- | --- | --- |
| `user` | 用户消息 | `{"role":"user","content":"上海今天天气"}` |
| `assistant` | LLM 输出（思考 / 最终回答） | `{"role":"assistant","content":"…"}` |
| `tool_call` | 工具调用请求 | `assistant` + `{"type":"tool_call","tool_name":…,"arguments":{…}}` |
| `tool_result` | 工具返回结果 | `user` + `【工具结果 weather】\n上海 今天：多云…` |
| `error` | 解析异常 / 工具执行异常 | `user` + `【错误 weather】\nToolValidationError: …` |

> **为什么不用原生 function calling？** 把 tool_call / tool_result 渲染成纯文本 JSON 协议后，
> 任何最简 chat API（含本地小模型）都能跑，Agent 循环与协议解析完全掌握在自己手里；
> 代价是模型可能输出不规范 JSON —— 这正是 `parser.py` 多级容错要解决的问题（也是本项目的演练重点）。

### 6.3 存储

```python
sessions = {
    "userA::win1": {
        "messages": [ … ],     # 完整上下文消息列表
        "max_turn": 10,        # 最大工具调用轮次
        "created_at": 1758534567890,
        …
    },
}
```

- 默认 `MemorySessionStore`（内存字典，demo 用）；
- 通过 `SessionStore` 抽象可替换：已内置 `JsonFileSessionStore`（`MINIAGENT_PERSIST_SESSIONS=true`，
  原子写 `logs/sessions.json`），生产可换成 Redis（实现 `load/save/delete/list_ids` 四个方法即可）。

### 6.4 追问支持

```python
await agent.run("上海今天天气", user_id="userA", window_id="win1")   # 第 1 轮
await agent.run("明天呢",      user_id="userA", window_id="win1")   # 第 2 轮：新消息 append 进同一 session
```

新消息直接 append 进当前 session 消息列表 → 重新进入 Agent Loop → 上下文自带历史，
因此**纯文本追问**与**继续带工具链的追问**都天然支持（实测第 2 轮沿用「上海」、日期切到「明天」）。

---

## 7. Context 设计：组装、轮次限制、基础压缩

### 7.1 放入上下文的信息

- 用户消息（`user`）
- Agent 思考 / LLM 输出（`assistant`）
- 工具调用请求（`tool_call`）
- 工具返回结果（`tool_result`）
- 解析异常、工具执行异常（`error`）

> **system prompt 不塞进 session.messages**：它每次请求时现场拼接在最前面
> （工具 schema + 会话信息 + 时间 + 轮次上限）。否则历史里会堆几十份重复指令，
> 既浪费 token，也会让模型行为漂移。测试 `test_system_prompt_not_persisted_but_present_in_request` 专门守这条约束。

### 7.2 双阈值裁剪

触发条件（任一满足）：

```
len(session.messages) > max_context_messages                     # 默认 40 条
估算 tokens(history) + tokens(system_prompt) > max_context_tokens # 默认 3000
```

裁剪策略：

1. 丢弃**最早**的消息，保留最近 `keep_recent_messages` 条（默认 12）；
2. token 超限时从尾部往前累加，找到最靠前的可保留下标；
3. **边界清理**：丢掉「孤儿 tool_result / error」（其 tool_call 已被裁掉）和结尾孤立的 tool_call；
4. **当前用户问题永不被裁**（`min_recent_messages` 兜底）；
5. 裁剪后在 system prompt 之后插入一条提示：

```
【上下文已裁剪，前面对话已压缩，只保留最近对话】
```

6. 单条工具结果超过 `tool_result_limit`（默认 1500 字符）时截断，并标注原始长度。

**裁剪只影响本次请求，不改 session 原文** —— 历史完整保留，便于审计与后续升级为「LLM 摘要式压缩」。
（测试 `test_case7_compression_does_not_mutate_session` 守住这条。）

### 7.3 轮次限制

```python
# 接近上限时（剩 1 次）注入：
"你只剩 1 次工具调用机会（上限 10 次），请优先给出最终回答，非必要不要调工具。"

# 到达上限时注入：
"你已达到本次可用工具调用轮次上限：本轮**禁止**再调用工具，必须直接输出最终回答。"
"请基于上文已有的工具结果作答；信息不足时明确说明还缺什么。"
```

### 7.4 实测（用例 7）

```
session 消息数=24 → 本次请求 messages=6（本次丢弃 18 条）
累计裁剪 5 次、丢弃 68 条 | 压缩提示=已插入
```

---

## 8. Memory 召回策略

| 层级 | 本项目实现 | 说明 |
| --- | --- | --- |
| **短期记忆（已实现）** | Session 内的完整消息列表 | 一个窗口一份，随对话增长；由 ContextManager 按阈值裁剪后进入 prompt |
| 上下文压缩（已实现，基础版） | 丢弃最早消息 + 保留最近 N 条 + 插入裁剪提示 | 按交付要求做**基础裁剪**，不做 RAG / LLM 摘要 |
| **长期记忆（不做）** | — | 最小版本不引入向量库 / 知识图谱 / 跨会话用户画像 |

**为什么不召回式记忆**：本项目的目标是「最小可用 Agent Runtime」，
记忆召回会引入 embedding、向量检索、写入策略、遗忘策略等一整套复杂度，
与「把 Agent 主循环写清楚」的目标冲突。当前设计里 session 消息既是最简单的记忆，
也是唯一的事实来源 —— 简单、可调试、行为可预测。

**升级路径**（不改主循环，只替换组件）：

1. `SessionStore` → Redis / Postgres，实现跨进程与多实例共享；
2. `ContextManager.build()` 里把「丢弃」换成「调用 LLM 生成摘要」写入 `session.meta["running_summary"]`；
3. 需要长期记忆时，在 `ContextManager` 中新增一个「检索并注入」步骤
   （召回结果作为 `system` 或 `user` 消息插到 prompt 前部），主循环无需改动。

---

## 9. 异常处理与 Trace 日志

### 9.1 异常捕获点（4 类，全部转成可回灌文本）

| 异常类型 | 抛出位置 | 处理方式 |
| --- | --- | --- |
| `LLMError` | LLM 网络异常 / 超时 / 非 2xx / 空响应 / 响应非 JSON | HTTP 层指数退避重试（可重试状态码），主循环再兜一层重试；最终失败返回可读文案并写 trace |
| `ParseError` | JSON 不合法 / 协议错误 / 工具名不存在 / 参数不符合 schema | 错误文本作为 `error` 消息写回上下文 → LLM 修正后重试；连续失败超限则降级返回 |
| `ToolValidationError` / `ToolNotFoundError` | 工具执行前的校验 | 同上（校验发生在**执行之前**，不会真的调用工具） |
| `ToolExecutionError` / `TimeoutError` | 工具 handler 内部报错 / 超时 | 包装成错误文本回灌 → LLM 换参数重试或告知用户失败 |

所有异常继承 `MiniAgentError`，统一提供 `to_observation()` 生成给 LLM 看的文本（含**修复建议**），
`str(exc)` 同样带上建议，便于日志排查。

### 9.2 Trace 产物

每次运行产出两份文件：

| 文件 | 内容 | 用途 |
| --- | --- | --- |
| `logs/trace.jsonl` | 一行一个 JSON 事件，append-only | grep / 导入 ELK / 写脚本统计 |
| `logs/traces/<run_id>.log` | 人可读时间线 | 单次提问的完整回放 |

每条事件都带 `session_id` / `run_id` / `turn` / `ts` / `duration_ms`：

```
event                记录内容
───────────────────  ─────────────────────────────────────────────────
run_start            session_id, user_input, 历史消息数, 轮次上限
llm_request          组装后的 messages 预览、token 估算、是否压缩、是否强制收敛
llm_response         LLM 原始输出、usage、finish_reason、耗时
tool_call            工具名、入参、是否经过修复
tool_result          工具名、返回值、成功/失败、耗时、错误信息
parse_error          原始输出片段、错误原因、累计失败次数
context_compressed   丢弃消息数、角色分布、token 估算、裁剪原因
max_turns_reached    到达轮次上限的提示
exception            where、异常类型、异常信息
run_end              停止原因、轮次、涉及工具、总耗时、最终答案
```

CLI 实时输出（`--quiet` 可关闭）：

```
  ▶ [run_start] 上海今天天气
  🧠 [llm_response] 240ms {"type":"tool_call","tool_name":"weather","arguments":{"city":"上海","date":"今天"}}
  🔧 [tool_call] weather({"city": "上海", "date": "今天"})
  📦 [tool_result] weather -> 上海 今天：多云，气温 28°C（23°C ~ 32°C），湿度 75%，风力 4 级，AQI 154。
  🧠 [llm_response] 310ms {"type":"answer","content":"上海今天天气：多云，气温 28°C …"}
  ✅ [final_answer] 上海今天天气：多云，气温 28°C（全天 23°C ~ 32°C）…
```

```bash
python -m miniagent trace run_26939c648a41     # 回放某次 run 的时间线
```

---

## 10. 验收用例与实测结果

`python -m miniagent demo` 会依次执行下表全部用例（默认离线 mock 客户端，**不需要 API Key，结果可复现**）。
用例代码见 `scripts/demo_cases.py`，等价断言见 `tests/test_agent_cases.py`。

| # | 用例 | 预期行为 | 实测结果 |
| --- | --- | --- | --- |
| 1 | 用户输入 `123+456*7` | Agent 调用 calculator，返回结果 | ✅ `工具链=['calculator']` → `计算完成：123+456*7 = 3315。` |
| 2 | 用户输入 `上海今天天气` | 调用 weather，返回 mock 天气 | ✅ `工具链=['weather']` → `上海今天天气：多云，气温 28°C（全天 23°C ~ 32°C），湿度 75%，风力 4 级，AQI 154。` |
| 3 | 用户输入 `你好，请用一句话说明你是谁` | 无需工具，直接输出答案，循环终止 | ✅ `工具调用=0 次，stop=answer` |
| 4 | 连续追问：先问天气，再追问「明天呢」 | 复用同一 session 上下文继续对话 | ✅ 第 2 轮沿用「上海」、日期切到「明天」；`session 消息数=8，轮次=2` |
| 5 | 用户 A 开两个窗口，分别问计算器和天气 | 两个 session 隔离，互不影响 | ✅ `win1: 工具=['calculator']`、`win2: 工具=['weather']`，交叉污染=无 |
| 6 | LLM 输出错误工具参数（weather 缺 `city`） | Parser 捕获错误 → 错误放回上下文 → LLM 修正重试 | ✅ 第 1 次被 Parser 拦截并回灌 `缺少必填字段 city`；第 2 次带 `city=上海` 执行成功 |
| 7 | 持续多轮对话，超过消息阈值 | 自动裁剪早期上下文，继续对话 | ✅ `24 条 → 请求 6 条`，累计裁剪 5 次、丢弃 68 条，压缩提示已插入 |
| +8 | 不同用户同名窗口（`userA::win1` / `userB::win1`） | session_id 含 user_id，跨用户也隔离 | ✅ 两个会话各自独立，无交叉污染 |
| +9 | 模型持续调用工具（脚本给 4 次 tool_call，上限 3） | 到上限后停止执行工具，强制总结 | ✅ 实际只执行 3 次，`stop=answer` |
| +10 | LLM 输出「想调工具但写坏了」的文本 | 解析失败回灌 → 第 2 次修正成功 | ✅ `工具链=['calculator']`，context 中 1 条 error |
| +11 | LLM 网络异常 | 异常被捕获、写 trace + 上下文，返回可读文案 | ✅ `stop=llm_error`，用户看到「模型服务暂时不可用…」 |

---

## 11. 测试

```bash
# 方式 A：零依赖内置运行器（不需要安装任何东西）
python tests/run_all.py                 # 全部 8 个模块
python tests/run_all.py test_parser     # 只跑解析器测试

# 方式 B：装了 pytest 的话
pytest -q
```

实测结果：**208 passed / 0 failed**。

| 测试模块 | 用例数 | 覆盖内容 |
| --- | --- | --- |
| `test_agent_cases.py` | 32 | 7 个验收用例 + 护栏（轮次上限 / 重复调用 / 解析失败 / LLM 异常 / 工具超时 / 并发多窗口）+ 自定义工具端到端 |
| `test_tools.py` | 42 | 注册/注销、参数校验、calculator 安全边界（11 条注入用例）、search/weather 确定性 |
| `test_parser.py` | 29 | JSON 提取、脏数据修复（全角/单引号/尾随逗号/截断/无引号 key）、协议与 schema 校验、错误信息质量 |
| `test_llm_client.py` | 34 | 多厂商响应解析、重试与快速失败、**注入假传输层**测试 HTTP 行为、采样参数回退、剧本/离线客户端 |
| `test_session_context.py` | 23 | 窗口隔离、存储落盘、双阈值裁剪、边界清理、渲染协议、轮次规则 |
| `test_edge_cases.py` | 19 | 并发同 session、8 窗口并行隔离、特殊字符往返、超大工具结果、多轮混合追问、多 JSON 对象、多实例共享 Session、极小上下文预算 |
| `test_observability.py` | 17 | trace 落盘、事件过滤、统计、字段截断、Prompt 渲染、Config 校验 |
| `test_examples.py` | 8 | 示例脚本不腐烂 |
| `test_cli_and_demo.py` | 7 | CLI 各子命令、`python -m miniagent demo` 子进程端到端 |
| **合计** | **208** | |

测试设计上值得一提的两点：

- **离线可复现**：`ScriptedLLMClient`（按剧本返回，可注入坏 JSON / 异常）+
  `OfflineMockClient`（规则版假 LLM）让全部用例无需网络与 Key；
- **HTTP 层可测**：`OpenAICompatibleClient(poster=...)` 允许注入假传输层，
  因此重试、超时、401 快速失败、响应结构兼容这些行为都能确定性测试。

---

## 12. 工具扩展方式

新增一个工具只需 **3 步**（完整可运行示例：`examples/custom_tool.py`）：

```python
from miniagent import Tool, build_default_registry

# ① 写 handler（同步 / async 均可，返回值自动字符串化）
def word_count(text: str) -> str:
    import re
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    words = len(re.findall(r"[A-Za-z0-9']+", text))
    return f"字符数={len(text)}，中文字数={cjk}，英文单词数={words}"

# ② 定义参数 schema（会写进 System Prompt 给 LLM 看）
SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string", "description": "要统计的文本", "minLength": 1}},
    "required": ["text"],
    "additionalProperties": False,
}

# ③ 注册
registry = build_default_registry()
registry.register(
    Tool(
        name="word_count",
        description="统计文本的字符数、中文字数和英文单词数。当用户问「这段话多少字」时调用。",
        parameters=SCHEMA,
        handler=word_count,
    )
)

agent = MinimalAgent(llm_client, registry, config=AgentConfig.from_env())
```

运行示例：

```bash
python examples/custom_tool.py
```

输出（实测）：

```
工具链: ['word_count']
工具结果: 字符数=18，中文字数=6，英文单词数=2
最终回答: 统计完成：共 19 个字符，其中中文 6 个，英文单词 2 个。

==========================================================================
示例 3：工具内部抛异常 → 错误回灌 → LLM 换成合法参数重试
==========================================================================
第 1 次调用 ok=False
  错误信息: ToolExecutionError: 工具 `exchange_rate` 执行失败: ValueError: 没有 CNY->KRW 的汇率数据（支持: …）
第 2 次调用 ok=True  参数={'base': 'CNY', 'target': 'JPY'}
```

### 进阶用法

**依赖注入**：handler 声明 `ctx` 参数即可拿到 `ToolContext`（session_id / tracer / deps）：

```python
async def my_tool(query: str, ctx=None) -> str:
    ctx.tracer.log("my_tool_call", session_id=ctx.session_id)
    api = ctx.dep("search_api")           # 从 config.tool_deps 注入
    return await api.search(query)

config = AgentConfig.from_env(tool_deps={"search_api": RealSearchAPI()})
```

**写 description 的经验**：写清「能做什么 + 什么场景该用 + 每个参数怎么填」，
比堆同义词更有效；`required` 与 `enum` 尽量收紧，让 Parser 能替你把模型的错误参数拦下来。

---

## 13. 配置项

全部配置集中在 `miniagent/config.py::AgentConfig`，支持 `.env` 与环境变量（`AgentConfig.from_env()`）：

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MINIAGENT_PROVIDER` | `deepseek` | `deepseek/openai/moonshot/dashscope/siliconflow/zhipu/ollama/vllm/mock` |
| `MINIAGENT_BASE_URL` | 按 provider | OpenAI 兼容地址（不带 `/chat/completions`） |
| `MINIAGENT_API_KEY` | — | 也支持 `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` |
| `MINIAGENT_MODEL` | 按 provider | 模型名 |
| `MINIAGENT_TEMPERATURE` / `MINIAGENT_MAX_TOKENS` | `0.2` / `1024` | 采样参数 |
| `MINIAGENT_TIMEOUT` / `MINIAGENT_MAX_RETRIES` | `60` / `3` | HTTP 超时与重试次数（指数退避） |
| `MINIAGENT_HTTP_BACKEND` | `stdlib` | `stdlib` / `httpx` / `aiohttp` |
| `MINIAGENT_MAX_TOOL_TURNS` | `10` | 单次提问的工具调用轮次上限 |
| `MINIAGENT_MAX_CONTEXT_TOKENS` | `3000` | 上下文 token 预算（启发式估算） |
| `MINIAGENT_MAX_CONTEXT_MESSAGES` | `40` | 上下文消息条数上限 |
| `MINIAGENT_KEEP_RECENT_MESSAGES` | `12` | 裁剪时保留最近 N 条 |
| `MINIAGENT_ENABLE_COMPRESSION` | `true` | 是否启用基础压缩 |
| `MINIAGENT_PERSIST_SESSIONS` | `false` | 会话是否落盘 `logs/sessions.json` |
| `MINIAGENT_LOG_DIR` | `logs` | trace 目录 |
| `MINIAGENT_CONSOLE_TRACE` | `true` | 是否实时打印 trace |

`python -m miniagent config` 会打印生效配置（隐去密钥）并给出缺失项提示。

---

## 14. 问题记录：开发踩坑

> 这一节按「现象 → 原因 → 解法 → 对应测试」记录，是本项目最真实的部分。

### 14.1 JSON 解析鲁棒性（踩坑最多）

| # | 现象 | 原因 | 解法 | 测试 |
| --- | --- | --- | --- | --- |
| 1 | **中文回答里的「，」被替换成「,」** | 早期把「全角→半角」做成了整串 `str.replace`，结果连字符串**内容**一起换了 | 写了个小状态机，只在**字符串外**替换结构符号；字符串内只把全角引号视作定界符 | `test_plain_answer` |
| 2 | 含 `{}` 的答案被切碎 | 花括号计数没跳过字符串内部 | `_brace_candidates` 逐字符扫描并跟踪 `in_string` / `escaped` | `test_json_inside_longer_text_with_braces` |
| 3 | 模型把 JSON 包在 markdown 代码块 / 前后加废话 | 模型习惯 | 先抽代码块，再花括号配对扫描，最后整段兜底；最多 8 个候选逐个尝试 | `test_markdown_fence`、`test_leading_and_trailing_prose` |
| 4 | `max_tokens` 用尽导致 JSON 被截断 | 生成中断 | `_close_truncated()` 补齐引号 / 括号，`"key":` 后补 `null` | `test_truncated_json_is_repaired` |
| 5 | 单引号、尾随逗号、无引号 key、全角括号混用 | Python 风格书写 | 多级修复并记录到 `repaired`（可观测修了什么） | `test_single_quotes` 等 4 条 |
| 6 | `arguments` 被序列化成字符串 | 少数模型的输出习惯 | 检测到字符串时二次 `json.loads` | `test_arguments_as_json_string` |
| 7 | 模型直接说人话（完全没 JSON） | 不守协议 | 宽松模式下降级为 `answer`；但用「花括号不配对 / 出现 JSON 结构字符 / 提到协议关键字」区分「说人话」与「JSON 写坏了」，后者必须报错回灌 | `test_plain_text_falls_back_to_answer`、`test_half_json_is_not_treated_as_answer` |
| 8 | 错误信息太笼统，模型改不对 | 早期无论什么错都报「格式不合法」 | 错误分级：**schema/协议错误优先抛出**，其次才是格式错误 | `test_missing_required_argument_raises`、`用例6` |
| 9 | 模型一次吐出多个 JSON 对象 | 模型「自言自语」式重复输出 | 取**第一个合法**的并按它决策（第一个才是真实决策）；若第一个是 `answer` 就直接结束循环，不执行后面那个 `tool_call` | `test_multiple_json_objects_takes_the_first_valid_one` |
| 10 | 用户直接粘贴一段 JSON 当输入 | 与协议文本长得像 | 用户消息原样进上下文，不参与协议解析（只有 **assistant** 输出才被 Parser 解析） | `test_user_input_with_json_like_text_is_not_confused` |
| 11 | 换实例 / 进程重启后，省略式追问「明天呢」失去指代 | 决策状态只在内存里 | 从上下文历史重建状态（扫描历史中的 tool_call 还原城市等槽位） | `test_session_manager_reuse_across_agents` |

### 14.2 上下文膨胀

| # | 现象 | 原因 | 解法 | 测试 |
| --- | --- | --- | --- | --- |
| 1 | 长对话后 token 暴涨、模型开始"忘事" | 历史无上限累积 | 双阈值（条数 + token 估算）裁剪，保留最近 N 条，并插入裁剪提示 | `test_case7_compression_after_threshold` |
| 2 | 裁剪后模型看到的第一个消息是「孤儿工具结果」 | 直接把 `tool_call` 裁掉、留着 `tool_result` | `_clean_boundary()` 丢弃孤儿 tool_result / error 和结尾孤立 tool_call | `test_boundary_cleanup_drops_orphan_tool_result` |
| 3 | 单条搜索结果吃光整个上下文 | 工具返回超长文本 | 单条结果按 `tool_result_limit` 截断并标注原始长度 | `test_long_tool_result_is_trimmed` |
| 4 | 裁剪把用户当前问题也裁掉了 | 只按条数硬切 | `min_recent_messages` 兜底 + 单测守住 | `test_case7_latest_question_always_kept` |
| 5 | 审计时看不到原始对话 | 一开始直接在 session 上切片删除 | 改成**裁剪只影响本次请求**，session 原文完整保留 | `test_case7_compression_does_not_mutate_session` |
| 6 | 压缩计数虚高（24 条消息却显示丢弃 99 条） | 边界清理的孤儿消息被重复计数 | 分别统计「按阈值丢弃」与「边界清理」，并只统计本轮真实移除量 | `用例7` |

### 14.3 工具参数校验

| # | 现象 | 原因 | 解法 | 测试 |
| --- | --- | --- | --- | --- |
| 1 | `{"expression": 123}` 被静默接受 | 早期 coerce 会把数字转成字符串（"无损修正"做过头了） | 去掉 number→string 的宽松化，让校验失败并把「类型应为 string」回灌给模型 | `test_case6_numbers_instead_of_string_is_rejected` |
| 2 | 模型把参数塞成 `"2"` 而 schema 要 integer | 模型习惯 | 保留 `"2"→2`、`"true"→True`、单值→数组、补 default 这类**真无损**修正 | `test_argument_type_coercion`、`test_default_value_filled` |
| 3 | 校验错误发生在工具执行之后（白白执行一次） | 早期只在 handler 内校验 | 把校验前移到 Parser 阶段（`registry.validate_call`），执行前就拦下 | `用例6`（`tool_calls` 为 1，只有修正后那次执行） |
| 4 | 工具超时把整个循环卡死 | 没有超时控制 | `asyncio.wait_for` + `tool_timeout`（默认 10s），超时转成可回灌文本 | `test_tool_timeout_is_caught` |
| 5 | 大小写/别名不一致（`tool` vs `tool_name`） | 不同模型输出习惯不同 | Parser 兼容多组字段别名 | `test_plain_tool_call` 系列 |

### 14.4 循环护栏

| # | 现象 | 原因 | 解法 | 测试 |
| --- | --- | --- | --- | --- |
| 1 | 模型一直调工具不收敛，烧 token | 没有轮次上限 | `max_tool_turns` + 到顶后禁止执行工具 + 强制总结 | `test_max_tool_turns_forces_convergence` |
| 2 | 模型坚持调工具，连"禁止"都不听 | 模型不服从 | 计入失败次数，超过上限返回兜底文案（不死循环） | `test_max_tool_turns_with_stubborn_model` |
| 3 | 同工具同参数反复调用 | 模型原地打转 | 检测连续重复调用，注入提示 | `test_repeated_identical_call_gets_hint` |
| 4 | 解析一直失败，循环空转 | 没有失败计数 | `max_parse_retries` + 降级返回 | `test_parse_failure_then_give_up_gracefully` |

### 14.5 工程细节

| # | 现象 | 解法 |
| --- | --- | --- |
| 1 | `str(exc)` 丢掉了修复建议，日志里看不到关键信息 | `MiniAgentError.__str__` 拼接 `hint` |
| 2 | LLM 请求体里的 `temperature` 与客户端配置不一致 | `LLMRequest.temperature/max_tokens` 默认 `None` 表示「用客户端默认值」，避免 dataclass 默认值覆盖配置 |
| 3 | Windows 控制台中文乱码 | `ensure_utf8_stdio()` 强制 stdout/stderr UTF-8 |
| 4 | 会话落盘时进程被杀 → 半个 JSON 文件 | 写 `.tmp` 再 `os.replace` 原子替换 |
| 5 | 没有 API Key 就无法验证任何逻辑 | 抽象出 `ScriptedLLMClient`（剧本）与 `OfflineMockClient`（规则版假 LLM），让框架本身可离线测试 |

---

## 15. 项目结构

```
minimal-agent/
├── miniagent/                    # 核心运行时（≈3550 行有效代码）
│   ├── __init__.py               #   对外 API 汇总
│   ├── __main__.py               #   python -m miniagent 入口
│   ├── agent.py                  #   ★ Agent 主循环（Step1~Step6 + 4 道护栏）
│   ├── config.py                 #   运行配置（唯一事实来源）
│   ├── session.py                #   ★ Session Manager：窗口隔离 + 存储后端
│   ├── context.py                #   ★ Context Manager：组装 / 渲染 / 裁剪
│   ├── parser.py                 #   ★ LLM 输出解析：提取 → 修复 → 校验
│   ├── prompts.py                #   System Prompt 加载与渲染
│   ├── schema.py                 #   零依赖 JSON Schema 校验器
│   ├── errors.py                 #   异常体系（都能转成"给 LLM 看的文本"）
│   ├── tracing.py                #   Tracer：结构化 JSONL + 人可读时间线
│   ├── utils.py                  #   工具函数（token 估算、安全截断…）
│   ├── llm/
│   │   ├── base.py               #   LLMClient 接口 / Request / Response
│   │   ├── openai_compatible.py  #   OpenAI 兼容客户端（重试/超时/多厂商兼容）
│   │   └── offline.py            #   剧本客户端 + 规则版离线客户端
│   └── tools/
│       ├── registry.py           #   ★ ToolRegistry：注册 / 注销 / 校验 / 执行
│       ├── calculator.py         #   安全表达式计算（AST 白名单，无 eval）
│       ├── search.py             #   mock 搜索（确定性）
│       └── weather.py            #   mock 天气（确定性）
├── prompts/
│   └── system_prompt.md          # ★ System Prompt（JSON 协议 + 工具列表占位符）
├── tests/                        # 自动化测试（≈1900 行有效代码，208 个用例）
│   ├── run_all.py                #   零依赖测试运行器（无 pytest 也能跑）
│   ├── conftest.py
│   ├── test_agent_cases.py       #   7 个验收用例 + 护栏
│   ├── test_parser.py            #   JSON 鲁棒性
│   ├── test_tools.py             #   工具注册 + 安全边界
│   ├── test_session_context.py   #   隔离 / 裁剪 / 渲染
│   ├── test_llm_client.py        #   HTTP 层 / 剧本客户端
│   ├── test_edge_cases.py        #   并发 / 特殊字符 / 多实例 / 协议边界
│   ├── test_observability.py     #   trace / prompt / config
│   ├── test_cli_and_demo.py      #   CLI / demo 端到端
│   └── test_examples.py          #   示例不腐烂
├── scripts/
│   ├── demo_cases.py             # 11 个验收用例（python -m miniagent demo）
│   └── collect_readme_snippets.py# 采集真实输出，保证 README 数字不手抄
├── examples/
│   └── custom_tool.py            # 新增自定义工具的完整示例
├── logs/                         # 运行产物（trace.jsonl / traces/*.log）
├── .env.example                  # 配置示例
├── requirements.txt              # 运行期无依赖；列出可选增强
└── pyproject.toml                # 打包 / CLI 入口 / pytest 配置
```

---

## 16. 已知限制与后续可做

**当前限制（有意为之的最小实现）**

1. 上下文压缩是**基础裁剪**（丢最早消息），不是 LLM 摘要式压缩；
2. 无长期记忆 / 向量召回（见 [第 8 节](#8-memory-召回策略)）；
3. Session 默认存内存，多进程部署需替换 `SessionStore`；
4. `search` / `weather` 是 mock 实现，未接真实数据源；
5. 每次只执行一个工具调用（不支持一次返回多个 tool_call 并行执行）；
6. token 用启发式估算（中文≈1 token/字，英文≈1 token/4 字符），未接 tiktoken；
7. 无鉴权 / 限流 / 多租户配额——这些属于服务层，不属于 Agent Runtime。

**后续可做（按性价比排序）**

1. 并发工具执行：Parser 返回 `list[ToolCall]`，`asyncio.gather` 并行跑（协议需扩展为数组）；
2. LLM 摘要式压缩：`ContextManager` 里把丢弃的消息交给 LLM 总结成 `running_summary`；
3. 流式输出：`stream=True` + SSE，把 `answer` 增量吐给用户（协议改为边生成边解析）；
4. 真实工具接入：替换 `mock_search` / `mock_weather`，并引入工具级限流与缓存；
5. Redis SessionStore + 分布式 trace（OpenTelemetry）；
6. 评测集：把 11 个用例扩成 100+ 条回归集，接 CI 做协议兼容性看护。

---

## License

MIT
