"""Session / Context 测试：窗口隔离、存储持久化、裁剪策略、轮次规则、渲染协议。"""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest

from miniagent.context import COMPRESSION_NOTICE, ContextManager, render_tool_result
from miniagent.session import JsonFileSessionStore, Message, Session, SessionManager
from miniagent.utils import estimate_tokens, make_session_id, split_session_id

#: 临时目录放在项目内（平台 TEMP 在部分沙箱环境下不可写）
TMP_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".tmp-tests")
os.makedirs(TMP_ROOT, exist_ok=True)


def make_temp_dir(prefix: str = "session") -> str:
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


# --------------------------------------------------------------------- 隔离
def test_session_key_is_user_plus_window() -> None:
    assert make_session_id("userA", "win1") == "userA::win1"
    assert split_session_id("userA::win1") == ("userA", "win1")
    assert split_session_id("weird") == ("weird", "default")


def test_two_windows_are_isolated() -> None:
    mgr = SessionManager()
    win1 = mgr.for_window("userA", "win1")
    win2 = mgr.for_window("userA", "win2")
    win1.add_user("计算 1+1")
    win1.add_tool_result("calculator", "1+1 = 2")
    win2.add_user("上海天气")
    win2.add_tool_result("weather", "上海 今天：多云")

    assert win1.session_id != win2.session_id
    assert win1.size == 2 and win2.size == 2
    assert "calculator" in win1.transcript()
    assert "calculator" not in win2.transcript()
    assert "weather" not in win1.transcript()


def test_same_window_accumulates_history() -> None:
    mgr = SessionManager()
    session = mgr.for_window("userA", "win1")
    mgr.append_user(session.session_id, "第一问")
    mgr.append_msg(session.session_id, "assistant", "第一答")
    mgr.append_user(session.session_id, "第二问")
    assert mgr.for_window("userA", "win1").size == 3
    assert mgr.for_window("userA", "win1").turn_count == 2


def test_different_users_same_window_name_isolated() -> None:
    mgr = SessionManager()
    a = mgr.for_window("userA", "win1")
    b = mgr.for_window("userB", "win1")
    a.add_user("A 的问题")
    b.add_user("B 的问题")
    assert "A 的问题" in a.transcript()
    assert "A 的问题" not in b.transcript()
    assert mgr.stats()["session_count"] == 2


def test_overview_groups_by_user() -> None:
    mgr = SessionManager()
    mgr.for_window("userA", "win1").add_user("q1")
    mgr.for_window("userA", "win2").add_user("q2")
    mgr.for_window("userB", "win1").add_user("q3")
    text = mgr.overview()
    assert "用户 userA" in text and "用户 userB" in text
    assert text.count("窗口") == 3


# --------------------------------------------------------------------- 消息
def test_message_role_validation() -> None:
    with pytest.raises(ValueError):
        Message(role="tool", content="x")


def test_append_helpers() -> None:
    session = Session("u::w")
    session.add_user("hi")
    session.add_tool_call("weather", {"city": "上海"})
    session.add_tool_result("weather", "多云")
    session.add_error("boom", name="weather")
    roles = [m.role for m in session.messages]
    assert roles == ["user", "tool_call", "tool_result", "error"]
    assert session.turn_count == 1
    assert session.messages[1].name == "weather"
    assert session.messages[1].content == '{"city": "上海"}'


def test_snapshot_counts() -> None:
    session = Session("u::w")
    session.add_user("hi")
    session.add_tool_result("calculator", "1")
    snap = session.snapshot()
    assert snap["roles"]["user"] == 1
    assert snap["roles"]["tool_result"] == 1
    assert snap["user_id"] == "u" and snap["window_id"] == "w"


# --------------------------------------------------------------------- 存储
def test_memory_store_roundtrip() -> None:
    mgr = SessionManager()
    session = mgr.for_window("userA", "win1")
    session.add_user("持久化测试")
    mgr.save(session)
    # 新建一个 manager 共用 store，应能恢复
    fresh = SessionManager(store=mgr.store)
    restored = fresh.for_window("userA", "win1")
    assert restored.size == 1
    assert restored.messages[0].content == "持久化测试"


def test_json_file_store_roundtrip() -> None:
    tmp = make_temp_dir()
    try:
        path = os.path.join(tmp, "sessions.json")
        store = JsonFileSessionStore(path)
        mgr = SessionManager(store=store)
        session = mgr.for_window("userA", "win1")
        session.add_user("写到文件")
        mgr.save_all()
        assert os.path.isfile(path)

        recovered = SessionManager(store=JsonFileSessionStore(path)).for_window("userA", "win1")
        assert recovered.size == 1
        assert recovered.messages[0].content == "写到文件"
    finally:
        remove_dir(tmp)


def test_reset_and_drop() -> None:
    mgr = SessionManager()
    session = mgr.for_window("userA", "win1")
    session.add_user("x")
    mgr.save(session)
    mgr.reset("userA::win1")
    assert mgr.for_window("userA", "win1").size == 0
    assert mgr.drop("userA::win1") is True
    assert mgr.drop("userA::win1") is False


# --------------------------------------------------------------------- 上下文组装
def _filled_session(turns: int = 4, *, tool: bool = True) -> Session:
    session = Session("u::w")
    for i in range(turns):
        session.add_user(f"第{i}问：上海天气怎么样")
        if tool:
            session.add_tool_call("weather", {"city": "上海"})
            session.add_tool_result("weather", f"上海 第{i}天：多云，气温 2{i}°C")
        session.add_assistant(f"第{i}答：多云，2{i} 度。")
    return session


def test_build_puts_system_prompt_first_and_does_not_persist_it() -> None:
    session = _filled_session(turns=1)
    build = ContextManager().build(session, "SYS-PROMPT")
    assert build.messages[0]['role'] == 'system'
    assert build.messages[0]['content'].startswith("SYS-PROMPT")
    assert all("SYS-PROMPT" not in m.content for m in session.messages)   # 不落 session


def test_tool_call_and_result_rendering() -> None:
    session = Session("u::w")
    session.add_user("上海天气")
    session.add_tool_call("weather", {"city": "上海"})
    session.add_tool_result("weather", "上海 今天：多云")
    session.add_error("工具参数校验失败", name="weather")
    build = ContextManager(compress=False).build(session, "SYS")
    contents = [m["content"] for m in build.messages]
    assert any('"type":"tool_call"' in c or '"type": "tool_call"' in c for c in contents)
    assert any("【工具结果 weather】" in c for c in contents)
    assert any("【错误 weather】" in c for c in contents)
    assert all(m["role"] in ("system", "user", "assistant") for m in build.messages)   # 只出现标准角色


def test_no_compression_below_threshold() -> None:
    session = _filled_session(turns=2)
    build = ContextManager(max_context_messages=40).build(session, "SYS")
    assert build.compressed is False
    assert build.dropped_messages == 0
    assert not any(COMPRESSION_NOTICE in m["content"] for m in build.messages)


def test_compression_by_message_count() -> None:
    session = _filled_session(turns=6)          # 6 轮 × 4 条 = 24 条
    cm = ContextManager(max_context_messages=8, keep_recent_messages=4, max_context_tokens=100000)
    build = cm.build(session, "SYS")
    assert build.compressed is True
    assert 0 < build.dropped_messages < session.size
    assert any(COMPRESSION_NOTICE in m["content"] for m in build.messages)
    # 最近的对话必须保留
    assert any("第5答" in m["content"] for m in build.messages)


def test_compression_by_token_budget() -> None:
    session = _filled_session(turns=8)
    cm = ContextManager(max_context_messages=1000, max_context_tokens=400, keep_recent_messages=100)
    build = cm.build(session, "SYS")
    assert build.compressed is True
    assert build.token_estimate < 1200


def test_compression_always_keeps_recent_turns() -> None:
    session = _filled_session(turns=20)
    cm = ContextManager(max_context_messages=6, keep_recent_messages=3, min_recent_messages=3)
    build = cm.build(session, "SYS")
    assert any("第19答" in m["content"] for m in build.messages)
    assert len(build.messages) >= 3


def test_boundary_cleanup_drops_orphan_tool_result() -> None:
    session = Session("u::w")
    session.add_user("很早的问题")
    session.add_tool_call("calculator", {"expression": "1+1"})
    session.add_tool_result("calculator", "1+1 = 2")
    session.add_user("最近的问题")
    session.add_assistant("最近的回答")
    cm = ContextManager(max_context_messages=2, keep_recent_messages=2)
    build = cm.build(session, "SYS")
    # 被保留下来的片段不应以孤儿 tool_result 开头
    non_system = [m for m in build.messages if m["role"] != "system"]
    assert non_system
    assert not non_system[0]["content"].startswith("【工具结果")


def test_long_tool_result_is_trimmed() -> None:
    session = Session("u::w")
    session.add_user("q")
    session.add_tool_result("search", "超长内容" * 2000)
    build = ContextManager(tool_result_limit=200).build(session, "SYS")
    assert build.trimmed_tool_results >= 1
    assert any("已截断" in m["content"] for m in build.messages)


def test_consecutive_user_messages_are_merged() -> None:
    session = Session("u::w")
    session.add_user("第一个问题")
    session.add_tool_result("calculator", "1+1 = 2")   # 渲染成 user
    build = ContextManager(compress=False).build(session, "SYS")
    users = [m for m in build.messages if m["role"] == "user"]
    assert len(users) == 1
    assert "第一个问题" in users[0]["content"]
    assert "【工具结果 calculator】" in users[0]["content"]


def test_turn_limit_rules() -> None:
    cm = ContextManager()
    session = Session("u::w")
    assert cm.turn_limit_rules(session, 0, max_turns=3) == []
    assert cm.turn_limit_rules(session, 2, max_turns=3)      # 只剩 1 次 → 提醒
    rules = cm.turn_limit_rules(session, 3, max_turns=3)     # 到顶 → 禁止再调工具
    assert any("禁止" in r for r in rules)


def test_render_tool_result_truncates() -> None:
    text = render_tool_result("search", "x" * 5000, limit=100)
    assert "已截断" in text


def test_token_estimate_sanity() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("中文中文") == 4
    assert 0 < estimate_tokens("hello world") < 10
