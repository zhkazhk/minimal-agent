"""为 README 采集真实输出片段（避免手抄数字出错）。"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from miniagent.utils import ensure_utf8_stdio  # noqa: E402

OUT = os.path.join(ROOT, "logs", "readme_snippets.txt")


def section(buf: io.StringIO, title: str) -> None:
    buf.write(f"\n{'#' * 78}\n# {title}\n{'#' * 78}\n")


def capture(buffer: io.StringIO, func, *args, **kwargs):
    """把一段代码的 stdout 抓进 buffer（用于把真实输出贴进 README）。"""
    real_stdout = sys.stdout
    sink = io.StringIO()
    sys.stdout = sink
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout = real_stdout
    buffer.write(sink.getvalue())
    return result


def main() -> int:
    ensure_utf8_stdio()
    buf = io.StringIO()

    # 1) 项目行数
    section(buf, "1. 代码规模")
    total = 0
    rows = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in {"logs", "__pycache__", ".tmp-tests", ".testlib", ".git", ".piptmp"}]
        for name in sorted(filenames):
            if not name.endswith((".py", ".md", ".toml")):
                continue
            path = os.path.join(dirpath, name)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    lines = sum(1 for _ in fh)
            except OSError:
                continue
            rel = os.path.relpath(path, ROOT).replace("\\", "/")
            rows.append((rel, lines))
            total += lines
    for rel, lines in sorted(rows):
        buf.write(f"{rel:44} {lines:>5}\n")
    buf.write(f"{'TOTAL':44} {total:>5}\n")

    # 2) 测试结果
    section(buf, "2. 测试结果")
    from tests.run_all import main as run_tests

    try:
        code = capture(buf, run_tests, [])
    except SystemExit as exc:
        code = int(exc.code or 0)
    buf.write(f"\nrun_all exit={code}\n")

    # 3) demo 用例结果
    section(buf, "3. 验收用例（7 必测 + 4 补充）")
    from scripts.demo_cases import run_demo

    capture(buf, lambda: asyncio.run(run_demo(provider="mock", verbose=False)))

    # 4) CLI 输出
    from miniagent.cli import main as cli_main

    section(buf, "4. CLI: tools")
    capture(buf, cli_main, ["tools"])

    section(buf, "5. CLI: ask --json")
    capture(buf, cli_main, ["ask", "123+456*7", "--provider", "mock", "-q", "--json"])

    section(buf, "6. miniagent config")
    capture(buf, cli_main, ["config"])

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(buf.getvalue())
    print(f"written -> {OUT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
