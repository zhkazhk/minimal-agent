# Memory 证据：上下文如何随追问演化

- 采集时间：2026-09-23 01:26:32　模型：`deepseek-flash`　会话：`evidence::memory`
- 连续三轮追问（第 2、3 轮都是省略式追问，必须依赖 memory 才能理解指代）。
  每轮记录三件事：
  1. **本轮开始前 session 里已有的上下文**（= 会被召回的 memory 总量）
  2. **本轮实际发出的第一条请求**（直接取自 trace 的 `llm_request` 事件，证明 memory 被放在哪里）
  3. 该轮结束后 session 累积了什么

---

## 第 1 轮：用户说 `上海今天天气`

### ① 本轮开始前，session 里已有的上下文

- session 消息数：**0** 条
- 组装后进入 LLM 的消息：**1** 条（1 system + 0 历史），token 估算 1965，压缩=False

| # | role | 内容（截断显示） |
| --- | --- | --- |
| 1 | `system` | 你是 `minimal-agent`——一个用自研 Runtime 驱动的最小可用 Agent。  你的能力边界：你可以直接回答常识问题，也可以通过下方工具列表中的工具获取外部信息或做精确计算。 工具由宿主程序真正执行，你**只需要输出要调用的工具名和参数** |

### ② 本轮实际发出的第一条请求（来自 trace）

共 **2** 条 —— 相比上面多了本轮的 `上海今天天气`：

| # | role | 内容（截断显示） |
| --- | --- | --- |
| 1 | `system` | 你是 `minimal-agent`——一个用自研 Runtime 驱动的最小可用 Agent。  你的能力边界：你可以直接回答常识问题，也可以通过下方工具列表中的工具获取外部信息或做精确计算。 工具由宿主程序真正执行，你**只需要输出要调用的工具名和参数** |
| 2 | `user` | 上海今天天气 |

> **召回时机**：`ContextManager.build()` 在每次 LLM 调用前执行，全量注入该 session 历史，无检索。
> **放置方式**：`[system prompt] → [历史消息按时间升序] → [本轮用户消息]`。

### ③ 本轮工具调用与回答

- 工具链：`['weather']`　停止原因 `answer`　轮次 1

```
上海今天（2026-09-23）多云，气温 28°C，全天 23°C ~ 32°C；湿度 75%，风力 4 级，AQI 154（中度污染，敏感人群建议戴口罩外出）。以上为模拟数据，非真实气象信息。
```

### ④ 该轮结束后 session 累积内容（5 条）

```
  1. [user] 上海今天天气
  2. [tool_call(weather)] {"city": "上海", "date": "今天", "unit": "celsius"}
  3. [tool_result(weather)] 上海 今天：多云，气温 28°C（23°C ~ 32°C），湿度 75%，风力 4 级，AQI 154。
{"city": "上海", "date": "今天", "day_offset": 0, "condition": "多云", "temperature": "28°C", "temp_range": "23°C ~ 32°C", "humidity": "75%", "wind": "4 级", "aqi": 154, "source": "mock-weather-api (demo, 非真实气象数据)"}
  4. [error] ParseError: LLM 想输出 JSON 但格式不合法（花括号不配对或缺少引号）。原因：候选片段不是合法 JSON: {"type":"answer","content":"上海今天（2026-09-23）多云，气温 28°C，全天在 23°C ~ 32°C 之间；湿度 75%，风力 4 级，空气质量指数 AQI 154，属于中度污染，敏感人群外出建议戴口…(截断, 共 144 字符)
修复建议: 必须只输出一个 JSON 对象，且不要包含任何多余文字。调用工具时输出 {"type":"tool_call","tool_name":"<工具名>","arguments":{...}}，给出结论时输出 {"type":"answer","content":"<中文回答>"}。
细节: {'raw_output': '{"type":"answer","content":"上海今天（2026-09-23）多云，气温 28°C，全天在 23°C ~ 32°C 之间；湿度 75%，风力 4 级，空气质量指数 AQI 154，属于中度污染，敏感人群外出建议戴口罩。\n\n（以上为模拟数据，非真实气象信息。）"}'}
  5. [assistant] 上海今天（2026-09-23）多云，气温 28°C，全天 23°C ~ 32°C；湿度 75%，风力 4 级，AQI 154（中度污染，敏感人群建议戴口罩外出）。以上为模拟数据，非真实气象信息。
```

---

## 第 2 轮：用户说 `那明天呢`

### ① 本轮开始前，session 里已有的上下文

- session 消息数：**5** 条
- 组装后进入 LLM 的消息：**5** 条（1 system + 4 历史），token 估算 2474，压缩=False

| # | role | 内容（截断显示） |
| --- | --- | --- |
| 1 | `system` | 你是 `minimal-agent`——一个用自研 Runtime 驱动的最小可用 Agent。  你的能力边界：你可以直接回答常识问题，也可以通过下方工具列表中的工具获取外部信息或做精确计算。 工具由宿主程序真正执行，你**只需要输出要调用的工具名和参数** |
| 2 | `user` | 上海今天天气 |
| 3 | `assistant` | {"type": "tool_call", "tool_name": "weather", "arguments": {"city": "上海", "date": "今天", "unit": "celsius"}} |
| 4 | `user` | 【工具结果 weather】 上海 今天：多云，气温 28°C（23°C ~ 32°C），湿度 75%，风力 4 级，AQI 154。 {"city": "上海", "date": "今天", "day_offset": 0, "condition": "多云 |
| 5 | `assistant` | 上海今天（2026-09-23）多云，气温 28°C，全天 23°C ~ 32°C；湿度 75%，风力 4 级，AQI 154（中度污染，敏感人群建议戴口罩外出）。以上为模拟数据，非真实气象信息。 |

### ② 本轮实际发出的第一条请求（来自 trace）

共 **6** 条 —— 相比上面多了本轮的 `那明天呢`：

| # | role | 内容（截断显示） |
| --- | --- | --- |
| 1 | `system` | 你是 `minimal-agent`——一个用自研 Runtime 驱动的最小可用 Agent。  你的能力边界：你可以直接回答常识问题，也可以通过下方工具列表中的工具获取外部信息或做精确计算。 工具由宿主程序真正执行，你**只需要输出要调用的工具名和参数** |
| 2 | `user` | 上海今天天气 |
| 3 | `assistant` | {"type": "tool_call", "tool_name": "weather", "arguments": {"city": "上海", "date": "今天", "unit": "celsius"}} |
| 4 | `user` | 【工具结果 weather】 上海 今天：多云，气温 28°C（23°C ~ 32°C），湿度 75%，风力 4 级，AQI 154。 {"city": "上海", "date": "今天", "day_offset": 0, "condition": "多云 |
| 5 | `assistant` | 上海今天（2026-09-23）多云，气温 28°C，全天 23°C ~ 32°C；湿度 75%，风力 4 级，AQI 154（中度污染，敏感人群建议戴口罩外出）。以上为模拟数据，非真实气象信息。 |
| 6 | `user` | 那明天呢 |

> **召回时机**：`ContextManager.build()` 在每次 LLM 调用前执行，全量注入该 session 历史，无检索。
> **放置方式**：`[system prompt] → [历史消息按时间升序] → [本轮用户消息]`。

### ③ 本轮工具调用与回答

- 工具链：`['weather']`　停止原因 `answer`　轮次 1

```
上海明天（2026-09-24）阴，气温 23°C，全天 19°C ~ 26°C，比今天明显转凉；湿度 68%，风力 5 级，AQI 151（中度污染）。出门建议带件外套，敏感人群注意防护。以上为模拟数据，非真实气象信息。
```

### ④ 该轮结束后 session 累积内容（9 条）

```
  1. [user] 上海今天天气
  2. [tool_call(weather)] {"city": "上海", "date": "今天", "unit": "celsius"}
  3. [tool_result(weather)] 上海 今天：多云，气温 28°C（23°C ~ 32°C），湿度 75%，风力 4 级，AQI 154。
{"city": "上海", "date": "今天", "day_offset": 0, "condition": "多云", "temperature": "28°C", "temp_range": "23°C ~ 32°C", "humidity": "75%", "wind": "4 级", "aqi": 154, "source": "mock-weather-api (demo, 非真实气象数据)"}
  4. [error] ParseError: LLM 想输出 JSON 但格式不合法（花括号不配对或缺少引号）。原因：候选片段不是合法 JSON: {"type":"answer","content":"上海今天（2026-09-23）多云，气温 28°C，全天在 23°C ~ 32°C 之间；湿度 75%，风力 4 级，空气质量指数 AQI 154，属于中度污染，敏感人群外出建议戴口…(截断, 共 144 字符)
修复建议: 必须只输出一个 JSON 对象，且不要包含任何多余文字。调用工具时输出 {"type":"tool_call","tool_name":"<工具名>","arguments":{...}}，给出结论时输出 {"type":"answer","content":"<中文回答>"}。
细节: {'raw_output': '{"type":"answer","content":"上海今天（2026-09-23）多云，气温 28°C，全天在 23°C ~ 32°C 之间；湿度 75%，风力 4 级，空气质量指数 AQI 154，属于中度污染，敏感人群外出建议戴口罩。\n\n（以上为模拟数据，非真实气象信息。）"}'}
  5. [assistant] 上海今天（2026-09-23）多云，气温 28°C，全天 23°C ~ 32°C；湿度 75%，风力 4 级，AQI 154（中度污染，敏感人群建议戴口罩外出）。以上为模拟数据，非真实气象信息。
  6. [user] 那明天呢
  7. [tool_call(weather)] {"city": "上海", "date": "明天", "unit": "celsius"}
  8. [tool_result(weather)] 上海 明天：阴，气温 23°C（19°C ~ 26°C），湿度 68%，风力 5 级，AQI 151。
{"city": "上海", "date": "明天", "day_offset": 1, "condition": "阴", "temperature": "23°C", "temp_range": "19°C ~ 26°C", "humidity": "68%", "wind": "5 级", "aqi": 151, "source": "mock-weather-api (demo, 非真实气象数据)"}
  9. [assistant] 上海明天（2026-09-24）阴，气温 23°C，全天 19°C ~ 26°C，比今天明显转凉；湿度 68%，风力 5 级，AQI 151（中度污染）。出门建议带件外套，敏感人群注意防护。以上为模拟数据，非真实气象信息。
```

---

## 第 3 轮：用户说 `那北京呢`

### ① 本轮开始前，session 里已有的上下文

- session 消息数：**9** 条
- 组装后进入 LLM 的消息：**9** 条（1 system + 8 历史），token 估算 2703，压缩=False

| # | role | 内容（截断显示） |
| --- | --- | --- |
| 1 | `system` | 你是 `minimal-agent`——一个用自研 Runtime 驱动的最小可用 Agent。  你的能力边界：你可以直接回答常识问题，也可以通过下方工具列表中的工具获取外部信息或做精确计算。 工具由宿主程序真正执行，你**只需要输出要调用的工具名和参数** |
| 2 | `user` | 上海今天天气 |
| 3 | `assistant` | {"type": "tool_call", "tool_name": "weather", "arguments": {"city": "上海", "date": "今天", "unit": "celsius"}} |
| 4 | `user` | 【工具结果 weather】 上海 今天：多云，气温 28°C（23°C ~ 32°C），湿度 75%，风力 4 级，AQI 154。 {"city": "上海", "date": "今天", "day_offset": 0, "condition": "多云 |
| 5 | `assistant` | 上海今天（2026-09-23）多云，气温 28°C，全天 23°C ~ 32°C；湿度 75%，风力 4 级，AQI 154（中度污染，敏感人群建议戴口罩外出）。以上为模拟数据，非真实气象信息。 |
| 6 | `user` | 那明天呢 |
| 7 | `assistant` | {"type": "tool_call", "tool_name": "weather", "arguments": {"city": "上海", "date": "明天", "unit": "celsius"}} |
| 8 | `user` | 【工具结果 weather】 上海 明天：阴，气温 23°C（19°C ~ 26°C），湿度 68%，风力 5 级，AQI 151。 {"city": "上海", "date": "明天", "day_offset": 1, "condition": "阴", |
| 9 | `assistant` | 上海明天（2026-09-24）阴，气温 23°C，全天 19°C ~ 26°C，比今天明显转凉；湿度 68%，风力 5 级，AQI 151（中度污染）。出门建议带件外套，敏感人群注意防护。以上为模拟数据，非真实气象信息。 |

### ② 本轮实际发出的第一条请求（来自 trace）

共 **10** 条 —— 相比上面多了本轮的 `那北京呢`：

| # | role | 内容（截断显示） |
| --- | --- | --- |
| 1 | `system` | 你是 `minimal-agent`——一个用自研 Runtime 驱动的最小可用 Agent。  你的能力边界：你可以直接回答常识问题，也可以通过下方工具列表中的工具获取外部信息或做精确计算。 工具由宿主程序真正执行，你**只需要输出要调用的工具名和参数** |
| 2 | `user` | 上海今天天气 |
| 3 | `assistant` | {"type": "tool_call", "tool_name": "weather", "arguments": {"city": "上海", "date": "今天", "unit": "celsius"}} |
| 4 | `user` | 【工具结果 weather】 上海 今天：多云，气温 28°C（23°C ~ 32°C），湿度 75%，风力 4 级，AQI 154。 {"city": "上海", "date": "今天", "day_offset": 0, "condition": "多云 |
| 5 | `assistant` | 上海今天（2026-09-23）多云，气温 28°C，全天 23°C ~ 32°C；湿度 75%，风力 4 级，AQI 154（中度污染，敏感人群建议戴口罩外出）。以上为模拟数据，非真实气象信息。 |
| 6 | `user` | 那明天呢 |
| 7 | `assistant` | {"type": "tool_call", "tool_name": "weather", "arguments": {"city": "上海", "date": "明天", "unit": "celsius"}} |
| 8 | `user` | 【工具结果 weather】 上海 明天：阴，气温 23°C（19°C ~ 26°C），湿度 68%，风力 5 级，AQI 151。 {"city": "上海", "date": "明天", "day_offset": 1, "condition": "阴", |
| 9 | `assistant` | 上海明天（2026-09-24）阴，气温 23°C，全天 19°C ~ 26°C，比今天明显转凉；湿度 68%，风力 5 级，AQI 151（中度污染）。出门建议带件外套，敏感人群注意防护。以上为模拟数据，非真实气象信息。 |
| 10 | `user` | 那北京呢 |

> **召回时机**：`ContextManager.build()` 在每次 LLM 调用前执行，全量注入该 session 历史，无检索。
> **放置方式**：`[system prompt] → [历史消息按时间升序] → [本轮用户消息]`。

### ③ 本轮工具调用与回答

- 工具链：`['weather']`　停止原因 `answer`　轮次 1

```
北京明天（2026-09-24）霾，气温 25°C，全天 21°C ~ 29°C；湿度 25%（较干燥），风力 2 级，AQI 55（空气质量良）。

对比一下：明天上海 19°C ~ 26°C 比北京更凉，且北京有霾、风力小不利于污染物扩散，外出可考虑戴口罩；两地湿度差异也大（上海 68% vs 北京 25%），北京记得多补水。以上为模拟数据，非真实气象信息。
```

### ④ 该轮结束后 session 累积内容（13 条）

```
  1. [user] 上海今天天气
  2. [tool_call(weather)] {"city": "上海", "date": "今天", "unit": "celsius"}
  3. [tool_result(weather)] 上海 今天：多云，气温 28°C（23°C ~ 32°C），湿度 75%，风力 4 级，AQI 154。
{"city": "上海", "date": "今天", "day_offset": 0, "condition": "多云", "temperature": "28°C", "temp_range": "23°C ~ 32°C", "humidity": "75%", "wind": "4 级", "aqi": 154, "source": "mock-weather-api (demo, 非真实气象数据)"}
  4. [error] ParseError: LLM 想输出 JSON 但格式不合法（花括号不配对或缺少引号）。原因：候选片段不是合法 JSON: {"type":"answer","content":"上海今天（2026-09-23）多云，气温 28°C，全天在 23°C ~ 32°C 之间；湿度 75%，风力 4 级，空气质量指数 AQI 154，属于中度污染，敏感人群外出建议戴口…(截断, 共 144 字符)
修复建议: 必须只输出一个 JSON 对象，且不要包含任何多余文字。调用工具时输出 {"type":"tool_call","tool_name":"<工具名>","arguments":{...}}，给出结论时输出 {"type":"answer","content":"<中文回答>"}。
细节: {'raw_output': '{"type":"answer","content":"上海今天（2026-09-23）多云，气温 28°C，全天在 23°C ~ 32°C 之间；湿度 75%，风力 4 级，空气质量指数 AQI 154，属于中度污染，敏感人群外出建议戴口罩。\n\n（以上为模拟数据，非真实气象信息。）"}'}
  5. [assistant] 上海今天（2026-09-23）多云，气温 28°C，全天 23°C ~ 32°C；湿度 75%，风力 4 级，AQI 154（中度污染，敏感人群建议戴口罩外出）。以上为模拟数据，非真实气象信息。
  6. [user] 那明天呢
  7. [tool_call(weather)] {"city": "上海", "date": "明天", "unit": "celsius"}
  8. [tool_result(weather)] 上海 明天：阴，气温 23°C（19°C ~ 26°C），湿度 68%，风力 5 级，AQI 151。
{"city": "上海", "date": "明天", "day_offset": 1, "condition": "阴", "temperature": "23°C", "temp_range": "19°C ~ 26°C", "humidity": "68%", "wind": "5 级", "aqi": 151, "source": "mock-weather-api (demo, 非真实气象数据)"}
  9. [assistant] 上海明天（2026-09-24）阴，气温 23°C，全天 19°C ~ 26°C，比今天明显转凉；湿度 68%，风力 5 级，AQI 151（中度污染）。出门建议带件外套，敏感人群注意防护。以上为模拟数据，非真实气象信息。
 10. [user] 那北京呢
 11. [tool_call(weather)] {"city": "北京", "date": "明天", "unit": "celsius"}
 12. [tool_result(weather)] 北京 明天：霾，气温 25°C（21°C ~ 29°C），湿度 25%，风力 2 级，AQI 55。
{"city": "北京", "date": "明天", "day_offset": 1, "condition": "霾", "temperature": "25°C", "temp_range": "21°C ~ 29°C", "humidity": "25%", "wind": "2 级", "aqi": 55, "source": "mock-weather-api (demo, 非真实气象数据)"}
 13. [assistant] 北京明天（2026-09-24）霾，气温 25°C，全天 21°C ~ 29°C；湿度 25%（较干燥），风力 2 级，AQI 55（空气质量良）。

对比一下：明天上海 19°C ~ 26°C 比北京更凉，且北京有霾、风力小不利于污染物扩散，外出可考虑戴口罩；两地湿度差异也大（上海 68% vs 北京 25%），北京记得多补水。以上为模拟数据，非真实气象信息。
```

---

## 结论：召回时机与放置方式

| 问题 | 本项目的做法 | 证据 |
| --- | --- | --- |
| **召回时机** | 每次 LLM 调用前全量召回（`ContextManager.build()`）。触发点：① 新一轮用户提问；② 每次工具结果回灌后继续循环（同一轮内的第 2、3 次 LLM 调用）；③ 追问 —— 与①同路径，因为追问就是往同一 session append 一条 user 消息 | 上表第 ② 节；以及第 1 轮 `工具链=['weather']` 说明本轮内至少发生了 2 次 LLM 调用，每次都重新组装了上下文 |
| **放置方式** | 固定顺序 `[system] → [历史升序] → [本轮用户消息]`；`tool_call` 渲染为 assistant 的 JSON 文本，`tool_result`/`error` 渲染为 user 的 `【…】` 文本块 | 上表逐条可见 |
| **压缩时的放置** | 触发裁剪时，在 system 之后、保留历史之前插入一条 `system`：【上下文已裁剪，前面对话已压缩，只保留最近对话】 | `python -m miniagent demo` 用例 7 |
| **不放置什么** | ① system prompt 不重复放（否则历史里堆叠几十份指令）；② 被裁掉的原始消息不放；③ 工具 Schema 只在 system 里出现一次 | `test_system_prompt_not_persisted_but_present_in_request` |
| **追问为何能work** | 第 2 轮「那明天呢」既没提城市也没提"天气"，模型能答出"上海明天阴天"，唯一来源就是被召回的第 1 轮历史 | 第 2 轮回答 vs 第 1 轮历史 |
