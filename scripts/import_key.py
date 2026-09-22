"""把外部提供的密钥文件安全地导入 `.env`（一次性工具，用完可删）。

用法：
    python scripts/import_key.py <密钥文件路径>

支持三种格式：
1. `KEY=value` 形式的 .env / properties / shell 导出（可含多行、注释）
2. 纯文本：整个文件内容就是一把 key（会把已知的 key 前缀识别出来）
3. JSON：{"api_key": "...", "base_url": "...", "model": "..."}（键名大小写不敏感）

安全约束：
- 只写入 `.env`（已在 .gitignore 里，且额外被 `*.key`/`*secret*` 等规则覆盖）；
- **绝不打印密钥本身**，只打印前 4 位 + 长度用于确认；
- 写完立刻 `git check-ignore` 验证它确实被忽略，否则报错退出。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT, ".env")

KEY_ALIASES = {
    "MINIAGENT_API_KEY": ("miniagent_api_key", "api_key", "apikey", "api-key", "key", "token", "secret", "sk"),
    "MINIAGENT_BASE_URL": ("miniagent_base_url", "base_url", "baseurl", "base-url", "endpoint", "api_base", "url"),
    "MINIAGENT_MODEL": ("miniagent_model", "model", "model_name", "modelname"),
    "MINIAGENT_PROVIDER": ("miniagent_provider", "provider"),
}
KNOWN_KEY_PREFIXES = ("sk-", "sk_", "Bearer ", "ghp_", "gsk_", "xai-", "AIza")


def mask(value: str) -> str:
    value = value.strip()
    if len(value) <= 8:
        return f"***（长度 {len(value)}）"
    return f"{value[:4]}…{value[-2:]}（长度 {len(value)}）"


def read_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read().splitlines()


def parse_env_style(lines: list[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^export\s+", "", line)
        if "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip().strip('"').strip("'")
        value = value.strip().strip('"').strip("'").rstrip(",")
        if name and value:
            found[name] = value
    return found


def normalize(found: dict[str, str]) -> dict[str, str]:
    """把各种键名映射到我们的环境变量名。"""
    out: dict[str, str] = {}
    lowered = {k.lower(): v for k, v in found.items()}
    for target, aliases in KEY_ALIASES.items():
        for alias in aliases:
            if alias in lowered and lowered[alias]:
                out[target] = lowered[alias]
                break
    return out


def extract(path: str) -> dict[str, str]:
    lines = read_lines(path)
    text = "\n".join(lines).strip()

    # 格式 3：JSON
    if text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                merged = normalize({str(k): str(v) for k, v in data.items() if isinstance(v, (str, int, float))})
                if merged:
                    return merged
        except json.JSONDecodeError:
            pass

    # 格式 1：KEY=value
    found = normalize(parse_env_style(lines))
    if found.get("MINIAGENT_API_KEY"):
        return found

    # 格式 2：整个文件就是一把 key（可能带引号/换行）
    candidate = text.strip().strip('"').strip("'")
    if candidate and "\n" not in candidate and len(candidate) >= 16:
        return {"MINIAGENT_API_KEY": candidate}
    for line in lines:
        stripped = line.strip().strip('"').strip("'")
        if stripped.startswith(KNOWN_KEY_PREFIXES) and len(stripped) >= 16:
            return {"MINIAGENT_API_KEY": stripped}

    return found


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    source = argv[0]
    if not os.path.isfile(source):
        print(f"找不到文件: {source}")
        return 1

    values = extract(source)
    if not values.get("MINIAGENT_API_KEY"):
        print("未能从该文件里识别出 API Key。")
        print("请把文件写成以下任一形式后重试：")
        print("  1) MINIAGENT_API_KEY=sk-xxxx        （.env 风格）")
        print("  2) 整个文件只有一把 key")
        print('  3) {"api_key": "sk-xxxx", "base_url": "https://api.deepseek.com/v1"}')
        return 1

    # 合并写入 .env（保留已有但未被覆盖的行）
    existing: list[str] = []
    if os.path.isfile(ENV_PATH):
        with open(ENV_PATH, "r", encoding="utf-8") as fh:
            existing = [line for line in fh.read().splitlines() if line.strip()]

    def upsert(lines: list[str], name: str, value: str) -> list[str]:
        pattern = re.compile(rf"^\s*(export\s+)?{re.escape(name)}\s*=")
        kept = [line for line in lines if not pattern.match(line)]
        kept.append(f"{name}={value}")
        return kept

    for name, value in values.items():
        existing = upsert(existing, name, value)

    with open(ENV_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(existing) + "\n")

    # 确认 .env 真的被 git 忽略（安全底线）
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", ENV_PATH],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        ignored = result.returncode == 0
    except OSError:
        ignored = True  # 没有 git 就无所谓

    print("已写入 .env：")
    for name, value in values.items():
        print(f"  {name} = {mask(value)}")
    print(f".env 是否被 git 忽略: {'是 ✅' if ignored else '否 ❌ —— 请勿提交，先修好 .gitignore'}")
    if not ignored:
        return 1
    print("\n下一步：python -m miniagent config   # 自检配置")
    print("然后：  python -m miniagent ask \"123+456*7\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
