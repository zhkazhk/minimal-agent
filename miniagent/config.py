"""运行配置：一个 dataclass 管全部参数，支持环境变量与 .env。

设计原则：**所有魔法数字都从这里出**，Agent 主循环里不出现硬编码阈值，
这样 README 里「轮次上限 / 上下文阈值 / 重试次数」都有唯一事实来源。
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

from .utils import load_dotenv

PROVIDER_DEFAULT_MODEL = {
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-chat",
    "moonshot": "moonshot-v1-8k",
    "dashscope": "qwen-plus",
    "siliconflow": "deepseek-ai/DeepSeek-V3",
    "zhipu": "glm-4-flash",
    "ollama": "qwen2.5:7b",
    "vllm": "local-model",
    "mock": "offline-mock",
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default


@dataclass
class AgentConfig:
    """Agent 运行时配置。"""

    # ---------------- LLM ----------------
    provider: str = "deepseek"          # deepseek / openai / dashscope / ollama / mock(离线)
    model: str = ""                     # 留空则按 provider 取默认模型
    base_url: str = ""                  # OpenAI 兼容地址，如 https://api.deepseek.com/v1
    api_key: str = ""
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout: float = 60.0
    max_retries: int = 3                # HTTP 层重试（指数退避）
    llm_call_retries: int = 2           # 主循环层额外重试
    http_backend: str = "stdlib"        # stdlib / httpx / aiohttp
    extra_body: dict[str, Any] = field(default_factory=dict)

    # ---------------- 循环与轮次 ----------------
    max_tool_turns: int = 10            # 单次提问最多允许的工具调用轮次
    max_parse_retries: int = 3          # 连续解析失败上限（超过则降级返回）
    tool_timeout: float = 10.0          # 单个工具执行超时（秒）

    # ---------------- 上下文 ----------------
    max_context_tokens: int = 3000      # 估算 token 预算
    max_context_messages: int = 40      # 消息条数上限
    keep_recent_messages: int = 12      # 裁剪时保留最近 N 条
    tool_result_limit: int = 1500       # 单条工具结果进入上下文的最大字符数
    enable_compression: bool = True
    lenient_plain_text: bool = True     # 解析不出 JSON 时，纯文本是否当作回答

    # ---------------- 会话 ----------------
    persist_sessions: bool = False
    session_path: str = "logs/sessions.json"

    # ---------------- Prompt / 日志 ----------------
    prompt_path: str = ""               # 留空用 prompts/system_prompt.md
    tool_prompt_mode: str = "json"      # json / text
    log_dir: str = "logs"
    console_trace: bool = True

    # ---------------- 文案 ----------------
    llm_error_reply: str = "抱歉，模型服务暂时不可用（多次重试仍失败）。请稍后再试，或检查 API Key / 网络配置。"
    parse_error_reply: str = "抱歉，我连续多次没能输出符合协议的格式，已停止本次尝试。请换一种问法，或重新发送一次。"
    max_turns_reply: str = "抱歉，本次工具调用轮次已达上限，我先停在这里。可以把问题拆小一点再问我。"
    internal_error_reply: str = "抱歉，Agent 内部出现未预期异常，已记录到 trace 日志，请稍后重试。"

    # ---------------- 工具依赖注入 ----------------
    tool_deps: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model:
            self.model = PROVIDER_DEFAULT_MODEL.get(self.provider.lower(), "deepseek-chat")

    # ------------------------------------------------------------------ 构造
    @classmethod
    def from_env(cls, env_file: str = ".env", **overrides: Any) -> "AgentConfig":
        """从 `.env` + 环境变量构造配置，`overrides` 优先级最高。"""
        load_dotenv(env_file)
        provider = os.getenv("MINIAGENT_PROVIDER", "deepseek").strip().lower()
        cfg = cls(
            provider=provider,
            model=os.getenv("MINIAGENT_MODEL", "") or PROVIDER_DEFAULT_MODEL.get(provider, "deepseek-chat"),
            base_url=os.getenv("MINIAGENT_BASE_URL", ""),
            api_key=os.getenv("MINIAGENT_API_KEY", "") or os.getenv("OPENAI_API_KEY", "") or os.getenv("DEEPSEEK_API_KEY", ""),
            temperature=_env_float("MINIAGENT_TEMPERATURE", 0.2),
            max_tokens=_env_int("MINIAGENT_MAX_TOKENS", 1024),
            timeout=_env_float("MINIAGENT_TIMEOUT", 60.0),
            max_retries=_env_int("MINIAGENT_MAX_RETRIES", 3),
            max_tool_turns=_env_int("MINIAGENT_MAX_TOOL_TURNS", 10),
            max_context_tokens=_env_int("MINIAGENT_MAX_CONTEXT_TOKENS", 3000),
            max_context_messages=_env_int("MINIAGENT_MAX_CONTEXT_MESSAGES", 40),
            keep_recent_messages=_env_int("MINIAGENT_KEEP_RECENT_MESSAGES", 12),
            enable_compression=_env_bool("MINIAGENT_ENABLE_COMPRESSION", True),
            persist_sessions=_env_bool("MINIAGENT_PERSIST_SESSIONS", False),
            log_dir=os.getenv("MINIAGENT_LOG_DIR", "logs"),
            console_trace=_env_bool("MINIAGENT_CONSOLE_TRACE", True),
            http_backend=os.getenv("MINIAGENT_HTTP_BACKEND", "stdlib"),
        )
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise KeyError(f"未知配置项: {key}")
            setattr(cfg, key, value)
        if not cfg.model:
            cfg.model = PROVIDER_DEFAULT_MODEL.get(cfg.provider, "deepseek-chat")
        return cfg

    def merged(self, **overrides: Any) -> "AgentConfig":
        data = asdict(self)
        for key, value in overrides.items():
            if key not in data:
                raise KeyError(f"未知配置项: {key}")
            data[key] = value
        return AgentConfig(**data)

    # ------------------------------------------------------------------ 校验
    def check(self) -> list[str]:
        """返回配置层面的问题列表（CLI 启动时打印，避免跑到一半才报错）。"""
        problems: list[str] = []
        if self.provider.lower() in ("mock", "offline", "fake", "scripted"):
            return problems
        if not self.base_url:
            problems.append(
                f"provider={self.provider} 但没有 base_url：请设置 MINIAGENT_BASE_URL（如 https://api.deepseek.com/v1）"
            )
        if not self.api_key and "localhost" not in self.base_url and "127.0.0.1" not in self.base_url:
            problems.append("缺少 API Key：请设置 MINIAGENT_API_KEY（或 OPENAI_API_KEY / DEEPSEEK_API_KEY）")
        return problems

    @property
    def is_offline(self) -> bool:
        return self.provider.lower() in ("mock", "offline", "fake", "scripted")

    def redacted(self) -> dict[str, Any]:
        """给日志用的配置快照（隐去密钥）。"""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        if data.get("api_key"):
            key = str(data["api_key"])
            data["api_key"] = f"{key[:4]}…{key[-4:]}" if len(key) > 8 else "***"
        return data

    def describe(self) -> str:
        lines = [
            f"provider   : {self.provider}",
            f"model      : {self.model}",
            f"base_url   : {self.base_url or '(未设置)'}",
            f"max_tool_turns : {self.max_tool_turns}",
            f"context    : tokens<={self.max_context_tokens}, messages<={self.max_context_messages}, keep_recent={self.keep_recent_messages}",
            f"log_dir    : {self.log_dir}",
        ]
        return "\n".join(lines)


__all__ = ["AgentConfig", "PROVIDER_DEFAULT_MODEL"]
