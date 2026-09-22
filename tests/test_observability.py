"""Tracer / Prompt / Config 测试。"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

import pytest

from miniagent.config import AgentConfig
from miniagent.prompts import FALLBACK_SYSTEM_PROMPT, PromptLoader, load_system_prompt
from miniagent.tools import build_default_registry
from miniagent.tracing import Timer, Tracer

#: 临时目录放在项目内（平台 TEMP 在部分沙箱环境下不可写）
TMP_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".tmp-tests")
os.makedirs(TMP_ROOT, exist_ok=True)


def make_temp_dir(prefix: str = "obs") -> str:
    """使用普通子目录（而不是 mkdtemp）：部分受限沙箱下 mkdtemp 产物不可再写入。"""
    seq = 0
    while True:
        seq += 1
        path = os.path.join(TMP_ROOT, f"{prefix}-{os.getpid()}-{seq}")
        if not os.path.exists(path):
            os.makedirs(path)
            return path


def remove_dir(path: str) -> None:
    """清理临时目录 —— 默认跳过（部分沙箱下删目录会直接终止进程）。

    需要清理时设置 `MINIAGENT_TEST_CLEANUP=1`。
    """
    if os.getenv("MINIAGENT_TEST_CLEANUP", "").strip().lower() not in ("1", "true", "yes"):
        return
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            try:
                os.remove(os.path.join(root, name))
            except OSError:
                pass
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except OSError:
                pass
    try:
        os.rmdir(path)
    except OSError:
        pass


# --------------------------------------------------------------------- Tracer
def test_tracer_writes_jsonl_and_run_file() -> None:
    tmp = make_temp_dir()
    try:
        tracer = Tracer(tmp, console=False)
        tracer.log("run_start", session_id="userA::win1", run_id="run_test", user_input="你好")
        tracer.log_tool_call("calculator", {"expression": "1+1"}, session_id="userA::win1", run_id="run_test", turn=0)
        tracer.log_tool_result("calculator", "1+1 = 2", duration_ms=3, ok=True, session_id="userA::win1", run_id="run_test", turn=0)
        tracer.flush()

        jsonl = os.path.join(tmp, "trace.jsonl")
        assert os.path.isfile(jsonl)
        with open(jsonl, encoding="utf-8") as fh:
            lines = [json.loads(line) for line in fh.read().strip().splitlines()]
        assert lines[0]["event"] == "run_start"
        assert lines[1]["data"]["tool_name"] == "calculator"
        assert lines[2]["duration_ms"] == 3

        run_file = os.path.join(tmp, "traces", "run_test.log")
        assert os.path.isfile(run_file)
        with open(run_file, encoding="utf-8") as fh:
            content = fh.read()
        assert "calculator" in content and "run_test" in content
        tracer.close()
    finally:
        remove_dir(tmp)


def test_tracer_events_filter() -> None:
    tracer = Tracer.in_memory()
    tracer.log("a", session_id="s1", run_id="r1")
    tracer.log("b", session_id="s2", run_id="r2")
    assert len(tracer.events()) == 2
    assert len(tracer.events(session_id="s1")) == 1
    assert len(tracer.events(run_id="r2")) == 1


def test_tracer_tool_stats() -> None:
    tracer = Tracer.in_memory()
    tracer.log_tool_result("calculator", "ok", duration_ms=10, ok=True)
    tracer.log_tool_result("calculator", "bad", duration_ms=20, ok=False, error="boom")
    stats = tracer.tool_call_stats()
    assert stats["calculator"]["calls"] == 2
    assert stats["calculator"]["ok"] == 1
    assert stats["calculator"]["error"] == 1
    assert stats["calculator"]["avg_ms"] == 15.0


def test_tracer_truncates_long_fields() -> None:
    tracer = Tracer.in_memory()
    tracer.max_field_chars = 50
    event = tracer.log("llm_response", raw_output="x" * 500)
    assert len(event.data["raw_output"]) < 200
    assert "截断" in event.data["raw_output"]


def test_tracer_exception_logging() -> None:
    tracer = Tracer.in_memory()
    event = tracer.log_exception(ValueError("boom"), where="unit-test")
    assert event.level == "ERROR"
    assert event.data["error_type"] == "ValueError"


def test_timer_measures() -> None:
    with Timer() as timer:
        sum(range(1000))
    assert timer.ms >= 0


# --------------------------------------------------------------------- Prompt
def test_prompt_loader_renders_placeholders() -> None:
    prompt = load_system_prompt(
        tools=build_default_registry().snapshot(),
        session_id="userA::win1",
        user_id="userA",
        window_id="win1",
        current_time="2026-09-22 17:00:00",
        max_turns=7,
    )
    assert "userA::win1" in prompt
    assert "2026-09-22 17:00:00" in prompt
    assert "7" in prompt
    assert "calculator" in prompt and "weather" in prompt
    assert "{{" not in prompt          # 占位符必须全部替换掉


def test_prompt_loader_fallback_on_missing_file() -> None:
    loader = PromptLoader("definitely/not/here.md")
    prompt = loader.render(tools=[], session_id="s", current_time="now", max_turns=3)
    assert "minimal-agent" in prompt
    assert FALLBACK_SYSTEM_PROMPT[:20] in prompt


def test_prompt_loader_handles_no_tools() -> None:
    prompt = load_system_prompt(tools=[], session_id="s", current_time="now", max_turns=1)
    assert "没有注册任何工具" in prompt


# --------------------------------------------------------------------- Config
def test_config_defaults_and_model_inference() -> None:
    config = AgentConfig(provider="deepseek")
    assert config.model == "deepseek-chat"
    assert AgentConfig(provider="openai").model == "gpt-4o-mini"
    assert AgentConfig(provider="mock").is_offline is True
    assert AgentConfig(provider="deepseek").is_offline is False


def test_config_check_reports_missing_key() -> None:
    config = AgentConfig(provider="deepseek", base_url="https://api.deepseek.com/v1", api_key="")
    problems = config.check()
    assert any("API Key" in item for item in problems)
    assert AgentConfig(provider="mock").check() == []


def test_config_check_allows_local_endpoint_without_key() -> None:
    config = AgentConfig(provider="ollama", base_url="http://localhost:11434/v1", api_key="")
    assert config.check() == []


def test_config_redacts_api_key() -> None:
    config = AgentConfig(provider="deepseek", api_key="sk-1234567890abcdef")
    redacted = config.redacted()
    assert redacted["api_key"] != config.api_key
    assert redacted["api_key"].startswith("sk-1")


def test_config_from_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("MINIAGENT_PROVIDER", "mock")
    monkeypatch.setenv("MINIAGENT_MAX_TOOL_TURNS", "4")
    monkeypatch.setenv("MINIAGENT_MAX_CONTEXT_MESSAGES", "9")
    config = AgentConfig.from_env(env_file=".env.not-exists")
    assert config.provider == "mock"
    assert config.max_tool_turns == 4
    assert config.max_context_messages == 9

    overridden = AgentConfig.from_env(env_file=".env.not-exists", max_tool_turns=1)
    assert overridden.max_tool_turns == 1


def test_config_rejects_unknown_override() -> None:
    with pytest.raises(KeyError):
        AgentConfig.from_env(env_file=".env.not-exists", not_a_field=1)


def test_config_merged_keeps_other_fields() -> None:
    config = AgentConfig(provider="mock", max_tool_turns=5)
    clone = config.merged(max_tool_turns=2)
    assert clone.max_tool_turns == 2
    assert clone.provider == "mock"
    assert config.max_tool_turns == 5      # 原对象不变


def test_config_describe_is_human_readable() -> None:
    text = AgentConfig(provider="mock").describe()
    assert "provider" in text and "max_tool_turns" in text
