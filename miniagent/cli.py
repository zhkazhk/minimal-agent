"""命令行入口：`python -m miniagent <子命令>`。

    ask     单次提问（支持 -u/--user、-w/--window 模拟多用户多窗口）
    repl    交互式多轮对话（同 session 追问；`/help` 看内置命令）
    demo    跑一遍 7 个验收用例
    tools   列出已注册工具及其 schema
    trace   查看某次 run 的 trace 时间线
    config  打印当前生效配置（隐去密钥）
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any, Optional

from .agent import MinimalAgent, build_agent
from .config import AgentConfig
from .tools.registry import build_default_registry
from .utils import ensure_utf8_stdio


def _make_agent(args: argparse.Namespace) -> MinimalAgent:
    overrides: dict[str, Any] = {}
    if getattr(args, "provider", None):
        overrides["provider"] = args.provider
    if getattr(args, "model", None):
        overrides["model"] = args.model
    if getattr(args, "trace_dir", None):
        overrides["log_dir"] = args.trace_dir
    if getattr(args, "quiet", False):
        overrides["console_trace"] = False
    config = AgentConfig.from_env(**overrides)
    return build_agent(config)


async def _ask(args: argparse.Namespace) -> int:
    agent = _make_agent(args)
    problems = agent.config.check()
    if problems and not agent.config.is_offline:
        print("⚠ 配置检查发现问题：", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print("  （提示：加 --provider mock 可用离线规则客户端跑通流程）", file=sys.stderr)

    result = await agent.run(args.question, user_id=args.user, window_id=args.window)
    if args.json:
        import json

        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print("\n" + "=" * 72)
        print(f"会话: {result.session_id}   轮次: {result.turns_used}   工具: {result.used_tools or '无'}")
        print(f"停止原因: {result.stopped_reason}   耗时: {result.latency_ms}ms")
        print("=" * 72)
        print(result.answer)
    await agent.aclose()
    return 0 if result.stopped_reason in ("answer", "max_turns") else 1


async def _repl(args: argparse.Namespace) -> int:
    agent = _make_agent(args)
    user_id, window_id = args.user, args.window
    session_id = f"{user_id}::{window_id}"
    print("=" * 72)
    print(" minimal-agent 交互模式")
    print(f" 用户={user_id}  窗口={window_id}  会话={session_id}  模型={agent.config.model} ({agent.config.provider})")
    print(" 输入 /help 查看命令，/exit 退出")
    print("=" * 72)
    while True:
        try:
            line = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/exit", "/quit", ":q"):
            break
        if line.startswith("/"):
            _handle_slash(agent, line, session_id)
            continue
        result = await agent.run(line, session_id=session_id)
        print(f"\nAgent > {result.answer}")
        print(f"        [工具: {result.used_tools or '无'} | 轮次: {result.turns_used} | {result.latency_ms}ms | stop={result.stopped_reason}]")
    await agent.aclose()
    agent.session_mgr.save_all()
    print("会话已保存。再见！")
    return 0


def _handle_slash(agent: MinimalAgent, line: str, session_id: str) -> None:
    command = line.split()[0]
    if command == "/help":
        print(
            "  /tools          查看已注册工具\n"
            "  /sessions       查看所有会话（按用户/窗口分组）\n"
            "  /history [n]    查看当前会话最近 n 条消息\n"
            "  /reset          清空当前窗口的上下文\n"
            "  /stats          工具调用统计\n"
            "  /exit           退出"
        )
    elif command == "/tools":
        print(agent.tool_registry.describe())
    elif command == "/sessions":
        print(agent.session_mgr.overview())
    elif command == "/history":
        parts = line.split()
        limit = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 20
        session = agent.session_mgr.get_or_create(session_id)
        print(session.transcript(limit=limit))
    elif command == "/reset":
        agent.reset(session_id)
        print("已清空当前窗口上下文。")
    elif command == "/stats":
        stats = agent.tracer.tool_call_stats()
        if not stats:
            print("（还没有工具调用）")
        for name, row in stats.items():
            print(f"  {name}: 调用 {row['calls']} 次, 成功 {row['ok']}, 失败 {row['error']}, 平均 {row['avg_ms']}ms")
    else:
        print(f"未知命令 {command}，输入 /help 查看帮助")


def _tools(args: argparse.Namespace) -> int:
    registry = build_default_registry()
    import json

    print(registry.describe())
    print("\n--- JSON Schema（会写进 System Prompt 给 LLM 看）---")
    print(json.dumps(registry.snapshot(), ensure_ascii=False, indent=2))
    return 0


def _trace(args: argparse.Namespace) -> int:
    path = os.path.join(args.trace_dir, "traces", f"{args.run_id}.log")
    if not os.path.isfile(path):
        print(f"找不到 trace 文件: {path}", file=sys.stderr)
        print("提示：run_id 会打印在每次提问的输出里；也可直接查看 logs/trace.jsonl", file=sys.stderr)
        return 1
    with open(path, "r", encoding="utf-8") as fh:
        print(fh.read())
    return 0


def _config(args: argparse.Namespace) -> int:
    config = AgentConfig.from_env()
    import json

    print(config.describe())
    print("\n完整配置（密钥已隐去）:")
    print(json.dumps(config.redacted(), ensure_ascii=False, indent=2))
    problems = config.check()
    if problems:
        print("\n⚠ 配置问题:")
        for problem in problems:
            print(f"  - {problem}")
    return 0


def _demo(args: argparse.Namespace) -> int:
    from scripts.demo_cases import run_demo

    return asyncio.run(run_demo(provider=args.provider, verbose=not args.quiet))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="miniagent",
        description="minimal-agent：不用任何 Agent 框架，从零手写的最小可用 Agent Runtime",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python -m miniagent demo\n"
               "  python -m miniagent ask \"123+456*7\" --provider mock\n"
               "  python -m miniagent ask \"上海今天天气\" -u userA -w win1\n"
               "  python -m miniagent repl -u userA -w win1\n",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--provider", help="LLM 提供方: deepseek/openai/dashscope/ollama/mock(离线)")
        p.add_argument("--model", help="模型名，覆盖配置")
        p.add_argument("--trace-dir", help="trace 日志目录，默认 logs")
        p.add_argument("-q", "--quiet", action="store_true", help="不打印实时 trace")

    p_ask = sub.add_parser("ask", help="单次提问")
    p_ask.add_argument("question", help="用户问题")
    p_ask.add_argument("-u", "--user", default="userA", help="用户ID，默认 userA")
    p_ask.add_argument("-w", "--window", default="win1", help="窗口ID，默认 win1")
    p_ask.add_argument("--json", action="store_true", help="以 JSON 输出结果（便于脚本断言）")
    add_common(p_ask)

    p_repl = sub.add_parser("repl", help="交互式多轮对话")
    p_repl.add_argument("-u", "--user", default="userA")
    p_repl.add_argument("-w", "--window", default="win1")
    add_common(p_repl)

    p_demo = sub.add_parser("demo", help="跑一遍 7 个验收用例")
    add_common(p_demo)

    sub.add_parser("tools", help="列出已注册工具").add_argument("--trace-dir", default="logs", help=argparse.SUPPRESS)

    p_trace = sub.add_parser("trace", help="查看某次 run 的 trace")
    p_trace.add_argument("run_id", help="run_id，例如 run_1a2b3c4d5e6f")
    p_trace.add_argument("--trace-dir", default="logs")

    sub.add_parser("config", help="打印当前配置")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    ensure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "ask": lambda: asyncio.run(_ask(args)),
        "repl": lambda: asyncio.run(_repl(args)),
        "demo": lambda: _demo(args),
        "tools": lambda: _tools(args),
        "trace": lambda: _trace(args),
        "config": lambda: _config(args),
    }
    return handlers[args.command]()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
