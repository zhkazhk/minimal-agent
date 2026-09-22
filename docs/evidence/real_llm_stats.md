# 真实 LLM 调用实证：多场景统计

- 采集时间：2026-09-23 01:26:32
- 模型：`deepseek-flash` @ `https://api.deepseek.com/v1`

## 各场景结果

| 场景 | 工具链 | 停止原因 | 轮次 | 耗时(ms) |
| --- | --- | --- | --- | --- |
| calculator | calculator | answer | 1 | 1433 |
| weather | weather | answer | 1 | 1653 |
| search | search、search | answer | 2 | 9746 |
| direct | （直接回答） | answer | 0 | 1321 |

## 工具调用统计（Tracer 聚合）

| 工具 | 调用次数 | 成功 | 失败 | 平均耗时(ms) |
| --- | --- | --- | --- | --- |
| calculator | 1 | 1 | 0 | 1.0 |
| search | 2 | 2 | 0 | 1.5 |
| weather | 1 | 1 | 0 | 0.0 |

> 完整断言见 `scripts/verify_real_llm.py`（10 项端到端验证）。
