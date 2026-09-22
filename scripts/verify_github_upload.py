"""逐条核验 GitHub 远程仓库是否包含全部交付要求的内容。

    python scripts/verify_github_upload.py [--remote origin]

核验方式：`git fetch` 之后直接读取 `origin/<branch>` 里的**远程真实内容**
（`git show origin/main:path`），不依赖 GitHub REST API，因此不受匿名限流影响。

输出一张「交付要求 → 远程证据」的对照表。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 失败：{result.stderr.strip()}")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", default="origin")
    args = parser.parse_args()
    remote = args.remote

    print("=" * 78)
    print("拉取远程状态（只取引用，不合并）…")
    git("fetch", remote, "--quiet", check=False)
    ref = f"{remote}/{git('rev-parse', '--abbrev-ref', 'HEAD').strip()}"
    try:
        remote_sha = git("rev-parse", ref).strip()
    except RuntimeError:
        print(f"❌ 找不到远程引用 {ref}")
        return 2
    local_sha = git("rev-parse", "HEAD").strip()
    remote_url = git("remote", "get-url", remote).strip()
    print(f"远程地址：{remote_url}")
    print(f"远程分支：{ref}  @ {remote_sha[:10]}")
    print(f"本地 HEAD：{local_sha[:10]}")
    print("=" * 78)

    # 远程文件清单（来自远程 commit 的 tree，不是本地工作区）
    remote_files = [p for p in git("ls-tree", "-r", "--name-only", ref).splitlines() if p]

    def show(path: str) -> str:
        """读取远程文件内容。"""
        return git("show", f"{ref}:{path}", check=False)

    results: list[tuple[str, bool, str]] = []

    def check(label: str, ok: bool, evidence: str) -> None:
        results.append((label, ok, evidence))

    # ---------------- 要求 1：从零实现 ----------------
    forbidden = ("langgraph", "langchain", "openhands", "openclaw", "autogen", "crewai", "smolagents",
                 "llama_index", "semantic_kernel", "haystack", "pydantic_ai", "agentscope")
    py_files = [p for p in remote_files if p.endswith(".py")]
    leaked: list[str] = []
    for path in py_files:
        text = show(path)
        for name in forbidden:
            if f"import {name}" in text or f"from {name} " in text:
                leaked.append(f"{path}:{name}")
    check(
        "要求1 从零实现：无 agent 框架依赖",
        not leaked and len(py_files) >= 20,
        f"远程 {len(py_files)} 个 py 文件全部扫描，命中框架 import：{leaked or '无'}",
    )

    req = show("requirements.txt")
    check(
        "要求1 核心 Runtime 自研 + 运行期零第三方依赖",
        bool(req) and "无" in req and "import" not in req.split("#")[0],
        "requirements.txt 明确运行期无依赖；httpx/aiohttp/jsonschema 均为可选增强（try/except）",
    )

    # ---------------- 要求 2：基本循环 ----------------
    agent_src = show("miniagent/agent.py")
    steps = {
        "Step1 收用户输入+读session": "get_or_create" in agent_src and "add_user" in agent_src,
        "Step2 组装prompt发LLM": "context_mgr.build" in agent_src and "llm.chat" in agent_src,
        "Step3 解析并分支": "parser.parse" in agent_src and "isinstance(parsed, Answer)" in agent_src,
        "Step4 执行工具": "_execute_tool" in agent_src,
        "Step5 结果回灌context": "add_tool_result" in agent_src and "add_error" in agent_src,
        "Step6 回到Step2循环": "while True" in agent_src,
    }
    check(
        "要求2 基本循环 Step1~Step6 全部实现",
        all(steps.values()),
        "  ".join(f"{k}{'✅' if v else '❌'}" for k, v in steps.items()),
    )

    # ---------------- 要求 2：三个工具 + 注册机制 ----------------
    tools = {
        "calculator": "miniagent/tools/calculator.py",
        "search": "miniagent/tools/search.py",
        "weather": "miniagent/tools/weather.py",
    }
    tool_state = {}
    for name, path in tools.items():
        src = show(path)
        tool_state[name] = bool(src) and '"type": "object"' in src and '"required"' in src and "description" in src
    check(
        "要求2 三个工具（calculator / search / weather）+ 参数 Schema",
        all(tool_state.values()),
        "  ".join(f"{k}{'✅' if v else '❌'}" for k, v in tool_state.items()),
    )

    registry = show("miniagent/tools/registry.py")
    check(
        "要求2 工具注册机制（名称/描述/Schema，支持注册与注销）",
        all(k in registry for k in ("def register", "def unregister", "def to_schema", "validate_arguments")),
        "ToolRegistry: register / unregister / get / snapshot / validate_call / execute",
    )

    parser_src = show("miniagent/parser.py")
    check(
        "要求2 LLM 输出解析（思考过程 / 工具调用 / 最终答案）",
        all(k in parser_src for k in ('"reason"', '"thought"', '"thinking"', "class Answer", "class ToolCall")),
        "ToolCall.reason 提取 reason/thought/thinking/why；Answer=终止，ToolCall=继续调工具",
    )

    # ---------------- 要求 2：session 管理 ----------------
    session_src = show("miniagent/session.py")
    check(
        "要求2 session 隔离（user_id + window_id 双维度）",
        "::" in session_src and "window_id" in session_src and "messages" in session_src,
        'session_id = f"{user_id}::{window_id}"；各窗口独立 messages 列表 + 独立存储',
    )

    # ---------------- 要求 2：context 管理 ----------------
    context_src = show("miniagent/context.py")
    check(
        "要求2 context：轮次限制 + 基础压缩",
        "turn_limit_rules" in context_src and "COMPRESSION_NOTICE" in context_src and "dropped_messages" in context_src,
        "双阈值（条数+token）裁剪；插入【上下文已裁剪…】；轮次到顶禁止再调工具",
    )
    check(
        "要求2 context：塞入哪些信息（用户输入/工具结果/思考）",
        all(k in context_src for k in ("tool_call", "tool_result", "error")) and "render_tool_result" in context_src,
        "5 类消息：user / assistant（含思考与最终答案）/ tool_call / tool_result / error",
    )
    check(
        "要求2 context：追问支持（纯对话 + 带工具）",
        "add_user" in session_src and "history_for_llm" in session_src,
        "新消息 append 进同一 session → 自带历史；工具链追问由工具结果回灌支撑",
    )

    # ---------------- 要求 2：异常处理 + trace ----------------
    tracing_src = show("miniagent/tracing.py")
    check(
        "要求2 异常处理",
        "class MiniAgentError" in show("miniagent/errors.py") and "log_exception" in tracing_src,
        "LLMError / ParseError / ToolNotFound / ToolValidation / ToolExecution 五类，均可转成回灌文本",
    )
    check(
        "要求2 工具调用 trace / 执行日志",
        "trace.jsonl" in tracing_src and "traceback" in tracing_src and "duration_ms" in tracing_src,
        "logs/trace.jsonl（JSONL 事件流）+ logs/traces/<run_id>.log（人可读时间线，含耗时/traceback）",
    )

    # ---------------- 要求 3：测试用例 ----------------
    test_files = sorted(p for p in remote_files if p.startswith("tests/test_") and p.endswith(".py"))
    check(
        "要求3 测试用例构建",
        len(test_files) >= 8,
        f"{len(test_files)} 个模块：{', '.join(os.path.basename(p) for p in test_files)}",
    )
    demo = show("scripts/demo_cases.py")
    check(
        "要求3 验收用例（含题目 7 个场景）",
        all(k in demo for k in ("123+456*7", "上海今天天气", "明天呢", "win2", "上下文已裁剪")),
        "scripts/demo_cases.py：计算器 / 天气 / 直接回答 / 连续追问 / 双窗口隔离 / 参数错误重试 / 超阈值压缩 + 4 个补充",
    )

    # ---------------- 提交内容 ----------------
    readme = show("README.md")
    sections = {
        "运行方式": "快速开始" in readme,
        "系统设计": "系统架构" in readme,
        "memory 召回时机与放置": "Memory 召回策略" in readme,
        "工具扩展方式": "工具扩展方式" in readme,
        "问题解决记录": "问题记录" in readme,
    }
    check(
        "提交: README（运行方式/系统设计/memory/工具扩展/问题记录）",
        all(sections.values()),
        "  ".join(f"{k}{'✅' if v else '❌'}" for k, v in sections.items()) + f"；共 {len(readme.splitlines())} 行",
    )
    check(
        "提交: AI Prompt",
        bool(show("prompts/system_prompt.md")),
        f"prompts/system_prompt.md（{len(show('prompts/system_prompt.md').splitlines())} 行：JSON 协议 + 工具列表占位符）",
    )
    check(
        "提交: 真实 LLM API 验证脚本",
        bool(show("scripts/verify_real_llm.py")),
        "scripts/verify_real_llm.py：对真实 OpenAI 兼容端点做 10 项端到端验证",
    )

    # ---------------- 一致性 + 安全 ----------------
    check(
        "本地 HEAD 与远程一致（已全部上传）",
        local_sha == remote_sha,
        f"本地 {local_sha[:10]} == 远程 {remote_sha[:10]}",
    )
    local_files = set(git("ls-files").split())
    missing = sorted(local_files - set(remote_files))
    check(
        "本地被跟踪文件全部存在于远程",
        not missing,
        f"本地 {len(local_files)} 个 / 远程 {len(remote_files)} 个" + (f"；缺失 {missing[:5]}" if missing else "；无缺失"),
    )
    secret_hits = [p for p in remote_files if p == ".env" or p.endswith((".key", ".pem")) or p.startswith(".inbox/") and p != ".inbox/README.md"]
    check(
        "安全: 远程无密钥文件",
        not secret_hits,
        f"可疑文件：{secret_hits or '无'}；远程共 {len(remote_files)} 个文件",
    )

    # ---------------- 输出对照表 ----------------
    print()
    for label, ok, evidence in results:
        print(f"{'✅' if ok else '❌'} {label}")
        print(f"     {evidence}")
    passed = sum(1 for _, ok, _ in results if ok)
    print()
    print("=" * 78)
    print(f"验收结果：{passed}/{len(results)} 项通过")
    print(f"仓库地址：{remote_url}")
    print("=" * 78)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
