"""跑通 CLI 与 demo 用例（端到端冒烟测试）。"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys

from miniagent.cli import main
from scripts.demo_cases import CASES, run_demo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_all_demo_cases_pass() -> None:
    """11 个验收用例（7 个必测 + 4 个补充护栏）必须全绿。"""
    code = asyncio.run(run_demo(provider="mock", verbose=False))
    assert code == 0
    assert len(CASES) >= 7


def test_cli_tools_command(capsys) -> None:
    assert main(["tools"]) == 0
    out = capsys.readouterr().out
    assert "calculator" in out and "weather" in out
    assert "expression" in out


def test_cli_config_command(capsys) -> None:
    assert main(["config"]) == 0
    out = capsys.readouterr().out
    assert "provider" in out


def test_cli_ask_json_output(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    code = main(["ask", "123+456*7", "--provider", "mock", "-q", "--json", "--trace-dir", str(tmp_path / "logs")])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stopped_reason"] == "answer"
    assert "3315" in payload["answer"]
    assert payload["tool_calls"][0]["name"] == "calculator"


def test_cli_ask_writes_trace_files(tmp_path, capsys) -> None:
    log_dir = tmp_path / "logs"
    code = main(["ask", "上海今天天气", "--provider", "mock", "-q", "--trace-dir", str(log_dir)])
    assert code == 0
    assert (log_dir / "trace.jsonl").is_file()
    assert list((log_dir / "traces").glob("*.log"))


def test_cli_unknown_run_id_trace(tmp_path, capsys) -> None:
    assert main(["trace", "run_not_exist", "--trace-dir", str(tmp_path)]) == 1
    assert "找不到" in capsys.readouterr().err


def test_module_entry_point_runs() -> None:
    """`python -m miniagent demo` 必须能跑（子进程验证真实入口）。"""
    proc = subprocess.run(
        [sys.executable, "-m", "miniagent", "demo"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "通过" in proc.stdout
