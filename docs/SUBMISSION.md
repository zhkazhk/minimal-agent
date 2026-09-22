# 提交说明 · minimal-agent

> 从零手写的最小可用 Agent Runtime（不使用任何 Agent 框架）
> 本文件把四份提交材料组织在一起，并给出每份材料的**位置**与**提交方视角的实证**。

---

## 0. 一览表

| 提交要求 | 交付位置 | 一句话说明 |
| --- | --- | --- |
| **① 使用真实的 LLM API** | `scripts/verify_real_llm.py` + `docs/evidence/real_llm_*.md` | DeepSeek `deepseek-flash`（OpenAI 兼容端点）；10 项端到端验证 **10/10 通过** |
| **② 代码链接** | https://github.com/zhkazhk/minimal-agent | 公开仓库，`main` 分支；50 个文件、13 个 commit |
| **③ README** | [`README.md`](../README.md) | 1038 行：运行方式 / 系统设计 / Session / Context / **Memory 召回时机与放置方式** / 工具扩展 / 问题记录 |
| **④ AI Prompt 与问题解决记录** | `prompts/system_prompt.md` + [`README.md` §14](../README.md#14-问题记录开发踩坑) + `docs/evidence/system_prompt.md` | Prompt 模板（受版本控制）+ 运行时渲染实证；问题记录共 5 大类 35 条 + 代码评审 12 条缺陷 |

---

## ① 真实 LLM API

### 接入方式

标准 OpenAI 兼容协议 `POST {base_url}/chat/completions`，因此可对接 DeepSeek / OpenAI / 通义 / Moonshot /
vLLM / Ollama 等任意兼容端点。**没有使用任何厂商 SDK**，HTTP 层用标准库 `urllib` 写成（可选 `httpx`/`aiohttp`）。

```ini
# .env（已被 .gitignore 忽略，仓库里只有 .env.example 模板）
MINIAGENT_PROVIDER=deepseek
MINIAGENT_BASE_URL=https://api.deepseek.com/v1
MINIAGENT_API_KEY=sk-***
MINIAGENT_MODEL=deepseek-flash
```

### 实测环境

| 项 | 值 |
| --- | --- |
| 提供方 / 模型 | `deepseek` / `deepseek-flash` |
| 端点 | `https://api.deepseek.com/v1/chat/completions` |
| 实测日期 | 2026-09-23 |

### 证据 1：一次完整调用的 trace 时间线

`docs/evidence/real_llm_run.md` —— 由 `Tracer` 自动落盘、未经手工修饰。核心片段：

```
[01:26:32.015] run_start      {"user_input": "上海今天天气怎么样？", "session_messages": 1, "max_tool_turns": 10}
[01:26:32.027] llm_request    messages=2  token_estimate=1979
[01:26:32.733] llm_response   (700ms) {"type":"tool_call","tool_name":"weather","arguments":{"city":"上海","date":"今天"}}
                              usage: prompt_tokens=1617 completion_tokens=34
                                     prompt_cache_hit_tokens=1408   ← 稳定前缀命中缓存
[01:26:32.734] tool_call      weather({"city": "上海", "date": "今天", "unit": "celsius"})
[01:26:32.737] tool_result    (1ms) 上海 今天：多云，气温 28°C（23°C ~ 32°C），湿度 75%，风力 4 级，AQI 154。
[01:26:32.750] llm_request    turn=1  messages=4  token_estimate=2117   ← 工具结果已回灌，上下文变长
[01:26:33.621] llm_response   (868ms) {"type":"answer","content":"上海今天多云，气温 28°C …"}
[01:26:33.623] final_answer   turn=1
[01:26:33.623] run_end        (1608ms) {"stopped_reason":"answer","turns_used":1,"tools":["weather"]}
```

这条时间线一次证明了 4 件事：**真实 API 调用**、**模型按 Schema 自主决策调工具**、**Step6 循环（2 次 LLM 调用）**、
**trace 日志**。

### 证据 2：多场景统计

`docs/evidence/real_llm_stats.md`

| 场景 | 工具链 | 停止原因 | 轮次 | 耗时(ms) |
| --- | --- | --- | --- | --- |
| calculator | calculator | answer | 1 | 1433 |
| weather | weather | answer | 1 | 1653 |
| search | search、search | answer | 2 | 9746 |
| direct（明确要求不调工具） | （直接回答） | answer | 0 | 1321 |

工具调用聚合（Tracer 统计）：`calculator` 1 次成功 / `weather` 1 次成功 / `search` 2 次成功，失败 0。

> `search` 场景模型**自主调用了两次**（先搜一次、看结果不够又换关键词搜一次），说明"根据工具结果决定是否继续 loop"是模型真实行为，不是脚本硬编码。

### 证据 3：端到端验证 10/10

```bash
python scripts/verify_real_llm.py
```

| # | 验证项 | 实测 |
| --- | --- | --- |
| 1 | calculator 工具调用 | 输出 `{"expression":"123+456*7"}` → 3315（**没有自己心算**） |
| 2 | weather 工具调用 | 正确选工具，参数 `{city:上海, date:今天, unit:celsius}` |
| 3 | 追问指代消解 | 「明天呢」→ 沿用 `city:上海`、`date` 切到 `明天` |
| 4 | 多窗口隔离 | win1 只出现 weather、win2 只出现 calculator，交叉污染=无 |
| 5 | 该直接回答时不调工具 | 0 次工具调用，`stop=answer` |
| 6 | 参数校验（模型一次填对） | `strict_echo(text, mode)` 参数全对 |
| 6b | 带约束参数抽取（`minLength=6`） | 从自然语言里抽出 `access_code=ABC123XYZ` |
| 6c | **工具执行期报错 → 回灌 → 可读交代** | `10/0` 触发 `ZeroDivisionError` → 模型答「除数不能为零」 |
| 7 | 轮次上限强制收敛 | `max_tool_turns=1` 时只执行 1 次工具，并如实说明没做完的部分 |
| 8 | 多轮触发上下文裁剪 | session 20 条 → 请求侧裁剪 7 次、累计丢弃 63 条 |

**结果：10/10 通过，总耗时 27.9s。**

复现方式：

```bash
git clone https://github.com/zhkazhk/minimal-agent.git && cd minimal-agent
cp .env.example .env      # 填入你自己的 MINIAGENT_API_KEY
python scripts/verify_real_llm.py
```

---

## ② 代码链接

**https://github.com/zhkazhk/minimal-agent**

| 项 | 值 |
| --- | --- |
| 可见性 | 公开 |
| 默认分支 | `main` |
| 文件数 | 50（含源码 23 个模块、测试 10 个模块、文档与脚本） |
| 运行期依赖 | **零**（纯 Python 标准库） |

自带一条命令自检上传完整性（逐条对照交付要求核验远程真实内容，20/20 通过）：

```bash
python scripts/verify_github_upload.py
```

---

## ③ README

**[`README.md`](../README.md)**（1038 行），目录：

| 章节 | 内容 | 交付要求对应 |
| --- | --- | --- |
| §1 快速开始 | 零配置跑通 / 接真实 LLM / 作为库使用 / 真实 LLM 验证结果 | **运行方式** |
| §2 系统架构 | 分层视图 + 一次带工具调用的完整数据流 | **系统设计** |
| §3 核心 Agent Loop | Step1~Step6 与代码位置对照表 + 4 道生产护栏 | **系统设计** |
| §4 工具层 | 注册机制 + 3 个内置工具 + 参数校验策略 | 系统设计 |
| §5 LLM 输出解析 | JSON 协议 + 多级容错 + 错误信息分级 | 系统设计 |
| §6 Session 设计 | 窗口隔离 / 5 类消息 / 存储 / 追问支持 | 系统设计 |
| §7 Context 设计 | 放什么信息 / 双阈值裁剪 / 轮次限制 | 系统设计 |
| **§8 Memory** | **召回时机（When）+ 放置方式（Where）+ 实证 + 为什么不检索 + 升级路径** | **memory 的召回时机与放置方式** |
| §9 异常处理与 Trace | 4 类异常捕获点 + 两类 trace 产物 + 实时输出示例 | 系统设计 |
| §10 验收用例 | 7 个必测 + 4 个补充的实测结果表 | 测试用例 |
| §11 测试 | 253 个用例的分类与覆盖 | 测试用例 |
| §12 工具扩展方式 | 3 步新增自定义工具 + 依赖注入 + 写 description 的经验 | **工具扩展** |
| §13 配置项 | 全部环境变量 | 运行方式 |
| §14 问题记录 | 5 大类 35 条踩坑 + 代码评审 12 条缺陷 | **问题解决记录** |
| §15 项目结构 | 带注释的文件树 | 系统设计 |
| §16 已知限制与后续 | 7 条限制 + 6 条规划 | 系统设计 |

---

## ④ AI Prompt 与问题解决记录

### 4.1 AI Prompt

**模板文件：[`prompts/system_prompt.md`](../prompts/system_prompt.md)**（64 行，受版本控制、可 diff）

设计要点：

- **强制 JSON 协议**：明确两种输出（`tool_call` / `answer`）与 5 条硬性规则（type 取值、工具名精确匹配、
  arguments 必须是对象、不加额外文字、不编造工具结果）；
- **工作方式引导**：什么情况必须用工具（算术、实时事实）、什么情况必须直接回答（常识、概念）、
  一次只调一个工具、追问时复用上一轮参数、工具报错时怎么自修；
- **工具列表用占位符 `{{tools_json}}`**：运行时由 `ToolRegistry.snapshot()` 注入，
  **新增工具无需改 Prompt 文件**；
- **运行时占位符**：`{{session_id}}` / `{{current_time}}` / `{{max_turns}}`。

**运行时渲染实证：`docs/evidence/system_prompt.md`** —— 同一文件里既有模板（64 行）也有
真实请求里发出的渲染结果（145 行），可以逐字对比。

按前缀稳定性排序工具列表（`snapshot()` 按名字排序），使前缀在多次请求间完全一致 ——
实测命中 DeepSeek Prompt Cache（`prompt_cache_hit_tokens: 1408`）。

### 4.2 问题解决记录

**位置：[`README.md` §14](../README.md#14-问题记录开发踩坑)**，五大类共 31 条，每条都是
「现象 → 原因 → 解法 → 对应测试」：

| 类别 | 条数 | 代表性坑 |
| --- | --- | --- |
| **JSON 解析鲁棒性** | 11 | 全角→半角替换把中文回答的「，」也换掉了（内容被污染）；花括号计数没跳过字符串内部；`max_tokens` 用尽导致 JSON 截断；单引号/尾随逗号/无引号 key；模型直接说人话；模型一次吐多个 JSON |
| **上下文膨胀** | 6 | 裁剪后模型看到「孤儿工具结果」；单条搜索结果吃光上下文；压缩计数虚高（24 条消息却显示丢弃 99 条） |
| **工具参数校验** | 5 | 早期 coerce 把 `{"expression": 123}` 静默转成 `"123"`，掩盖了模型的真实错误；校验发生在工具执行之后（白执行一次） |
| **循环护栏** | 4 | 模型一直调工具不收敛；连"禁止调工具"都不听；同工具同参数反复调用 |
| **工程细节** | 5 | `str(exc)` 丢掉修复建议；Windows 中文乱码；会话落盘时进程被杀导致半个 JSON 文件 |

**另加一轮对抗式代码评审**（§14.6），发现并修复 **12 个真实缺陷**（其中 4 个 HIGH），
全部在 `tests/test_review_regressions.py`（41 条用例）里锁死。最值得记录的四条：

| 缺陷 | 为什么危险 |
| --- | --- |
| `prompts/system_prompt.md` **从未被加载** | `prompt_path` 默认空串 → `open("")` 抛 OSError → **静默退化**成 6 行 fallback。真实 LLM 一直拿的是残缺协议说明，而测试"全绿" |
| `ToolContext` 依赖注入是**死代码** | README 与示例都宣称支持 `ctx.dep(...)`，实际 handler 永远收到 `None` |
| **截断的 JSON 被补全后真的执行了工具** | `{"expression":` 被补成 `null` / `{}` 后照常执行，用户拿到莫名结果 |
| 解析器回退**绕过语义校验** | `[{call1},{call2}]` 的"一次只能一个工具调用"校验被切片静默放行 |

> 这轮的收获写在 README 里：**「测试全绿」和「功能正确」是两件事** ——
> 上述缺陷都属于"主流程看起来完全正常"的类型，只有把「请求里到底发了什么 prompt」
> 「handler 到底收到了什么参数」逐字打出来看，才会暴露。

---

## 复现全部结论（可选）

```bash
git clone https://github.com/zhkazhk/minimal-agent.git && cd minimal-agent

# 1) 零依赖跑通：253 个测试 + 11 个验收用例（无需 API Key）
python tests/run_all.py          # → 253 passed / 0 failed
python -m miniagent demo         # → 11/11 通过

# 2) 真实 LLM（需 .env）
cp .env.example .env             # 填 MINIAGENT_API_KEY
python scripts/verify_real_llm.py      # → 10/10 通过
python scripts/collect_evidence.py     # → 重新生成 docs/evidence/ 全部证据

# 3) 核验 GitHub 上传完整性
python scripts/verify_github_upload.py # → 20/20 通过
```
