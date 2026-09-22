"""一键把本仓库推送到 GitHub（幂等，可重复执行）。

    python scripts/push_to_github.py https://github.com/<user>/minimal-agent.git

做五件事，每步都先检查再动手：
1. 校验远程地址格式，并设置/更新 `origin`；
2. 推送前跑一次**安全检查**：确认没有任何密钥文件被 git 跟踪
   （`.env`、`*.key`、`.inbox/*` 等），有则直接中止；
3. 打印将要推送的 commit 列表，确认工作区干净；
4. **探测 GitHub 连通性**，直连不通时自动改用可用代理（见下）；
5. 执行 `git push -u origin <当前分支>`。

代理处理：
国内网络直连 `github.com:443` 经常被 reset。脚本会先测直连，失败则按
`--proxy` 参数 > `MINIAGENT_GIT_PROXY`/`HTTPS_PROXY` 环境变量 > 常见本地代理端口
（7895/7890/7897/10809/1080…）的顺序找一个能通的，然后**只对本次命令**附加
`-c http.proxy=...`（不写入 git 全局配置，避免污染其他仓库）。

用法：
    python scripts/push_to_github.py https://github.com/<user>/<repo>.git
    python scripts/push_to_github.py <url> --proxy http://127.0.0.1:7895
    python scripts/push_to_github.py <url> --no-proxy      # 强制直连

认证说明：推送需要 GitHub 凭据。若本机已登录 Git Credential Manager，
命令会直接成功；否则 git 会提示输入用户名与 Personal Access Token（PAT），
或弹出浏览器登录窗口。
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

URL_PATTERNS = (
    re.compile(r"^https://github\.com/[\w.\-]+/[\w.\-]+(\.git)?$"),
    re.compile(r"^git@github\.com:[\w.\-]+/[\w.\-]+(\.git)?$"),
    re.compile(r"^https://[\w.\-]+@github\.com/[\w.\-]+/[\w.\-]+(\.git)?$"),
)

#: 绝对不允许进入远程仓库的路径特征
FORBIDDEN_TRACKED = (
    re.compile(r"(^|/)\.env$"),
    re.compile(r"(^|/)\.env\.(?!example)"),
    re.compile(r"\.key$"),
    re.compile(r"\.pem$"),
    re.compile(r"^\.inbox/(?!README\.md$)"),
    re.compile(r"(secret|credential)", re.IGNORECASE),
    re.compile(r"^key.*\.(txt|json)$", re.IGNORECASE),
)

#: 常见本地代理端口（Clash / Mihomo / V2Ray / SS 等），按命中概率排序
COMMON_PROXY_PORTS = (7895, 7890, 7897, 10809, 1080)

#: 单次连通性探测超时（秒）。必须短：直连不通时要靠它快速失败、换下一个候选。
PROBE_TIMEOUT = 12

#: git push 的最长等待时间（秒）。凭据缺失时不能让脚本无限挂着。
PUSH_TIMEOUT = int(os.getenv("MINIAGENT_PUSH_TIMEOUT", "120"))


def git(*args: str, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def fail(message: str) -> int:
    print(f"❌ {message}")
    return 1


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def probe(proxy: str | None, timeout: int = PROBE_TIMEOUT) -> bool:
    """用 git ls-remote 探一个公开仓库，判断当前链路是否可用。"""
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    cmd = ["git"]
    if proxy:
        cmd += ["-c", f"http.proxy={proxy}"]
    cmd += ["ls-remote", "https://github.com/cli/cli.git", "HEAD"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env, errors="replace"
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def find_working_proxy(explicit: str | None) -> tuple[str | None, str]:
    """返回 (代理地址或 None, 说明)。按「越明确的候选越先试」的顺序，避免长时间卡住。"""
    if explicit:
        print(f"   使用指定代理 {explicit} 探测…")
        return (explicit, "命令行指定") if probe(explicit) else (None, f"指定代理 {explicit} 不通")

    env_proxy = os.getenv("MINIAGENT_GIT_PROXY") or os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    if env_proxy:
        print(f"   使用环境变量代理 {env_proxy} 探测…")
        if probe(env_proxy):
            return env_proxy, "环境变量"

    # 先看哪些端口真的在监听（瞬时判断，不耗网络超时）
    listening = [port for port in COMMON_PROXY_PORTS if port_open(port)]
    if listening:
        print(f"   发现本地监听端口 {listening}，逐个探测…")
        for port in listening:
            candidate = f"http://127.0.0.1:{port}"
            if probe(candidate):
                return candidate, f"自动发现 127.0.0.1:{port}"
        print("   本机代理端口都不通，改测直连…")

    print("   直连探测中…")
    if probe(None):
        return None, "直连可用"
    return None, "未找到可用代理"


def main(argv: list[str]) -> int:
    explicit_proxy: str | None = None
    no_proxy = False
    positional: list[str] = []
    idx = 0
    while idx < len(argv):
        item = argv[idx]
        if item == "--no-proxy":
            no_proxy = True
        elif item == "--proxy":
            idx += 1
            if idx >= len(argv):
                return fail("--proxy 后面要跟代理地址，例如 --proxy http://127.0.0.1:7895")
            explicit_proxy = argv[idx]
        else:
            positional.append(item)
        idx += 1

    if len(positional) != 1:
        print(__doc__)
        return 2
    url = positional[0].strip()

    if not any(pattern.match(url) for pattern in URL_PATTERNS):
        return fail(
            f"远程地址看起来不是 GitHub 仓库：{url}\n"
            "   期望形如 https://github.com/<用户名>/<仓库名>.git"
        )

    # ---------- 1. 安全检查：绝不能把密钥推上去 ----------
    tracked = git("ls-files").stdout.splitlines()
    leaked = [path for path in tracked if any(pattern.search(path) for pattern in FORBIDDEN_TRACKED)]
    if leaked:
        print("❌ 检测到疑似密钥文件已被 git 跟踪，已中止推送：")
        for path in leaked:
            print(f"     {path}")
        print("\n   处理方式：")
        print("     git rm --cached <文件>       # 从版本库移除但保留本地文件")
        print("     然后确认 .gitignore 已覆盖它，再重新执行本脚本")
        return 1
    print(f"✅ 安全检查通过：{len(tracked)} 个被跟踪文件里没有密钥类文件")

    # ---------- 2. 工作区状态 ----------
    dirty = git("status", "--porcelain").stdout.strip()
    if dirty:
        print("⚠ 工作区有未提交改动（会照常推送已提交内容，未提交的不会上去）：")
        for line in dirty.splitlines()[:10]:
            print(f"     {line}")

    branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    commits = git("log", "--oneline", "-10").stdout.strip().splitlines()
    print(f"\n分支：{branch}    待推送提交（最近 {len(commits)} 条）：")
    for line in commits:
        print(f"     {line}")

    # ---------- 3. 连通性 / 代理 ----------
    print("\n网络连通性检查：")
    if no_proxy:
        proxy, reason = None, "--no-proxy 强制直连"
    else:
        proxy, reason = find_working_proxy(explicit_proxy)
    if proxy is None and not no_proxy and reason != "直连可用":
        print(f"❌ {reason}：直连 GitHub 不通，也没找到可用代理。")
        print("   处理方式（任选其一）：")
        print("     1) 启动你的代理客户端（Clash / V2Ray 等）后重试；")
        print("     2) 显式指定：python scripts/push_to_github.py <url> --proxy http://127.0.0.1:<端口>")
        print("     3) 若你的网络本来就能直连（海外机器），加 --no-proxy 跳过探测")
        return 1
    print(f"   ✅ 链路可用（{reason}）：{proxy or '直连'}")

    # ---------- 4. 设置远程 ----------
    existing = git("remote", "get-url", "origin", check=False).stdout.strip()
    if existing and existing != url:
        git("remote", "set-url", "origin", url)
        print(f"\n已更新 origin: {existing} -> {url}")
    elif not existing:
        git("remote", "add", "origin", url)
        print(f"\n已添加 origin: {url}")
    else:
        print(f"\norigin 已就绪: {url}")

    # ---------- 5. 推送 ----------
    push_cmd = ["git"]
    if proxy:
        push_cmd += ["-c", f"http.proxy={proxy}"]
    push_cmd += ["push", "-u", "origin", branch]

    # 关键：凭据缺失时 git 会**无限等待**输入用户名/PAT。
    # 在自动脚本里必须显式禁止交互，否则表现为"卡住不返回"，很难排查。
    push_env = dict(os.environ)
    push_env["GIT_TERMINAL_PROMPT"] = "0"     # 不询问终端输入
    push_env["GCM_INTERACTIVE"] = "never"     # 不让 Git Credential Manager 弹窗
    push_env.setdefault("GIT_ASKPASS", "")    # 不走 askpass

    print(f"\n执行：{' '.join(push_cmd)}")
    print(f"（已禁止凭据交互；{PUSH_TIMEOUT}s 无响应会判为超时）\n" + "-" * 60)
    try:
        result = subprocess.run(
            push_cmd,
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=push_env,
            timeout=PUSH_TIMEOUT,
        )
        returncode = result.returncode
    except subprocess.TimeoutExpired:
        returncode = -1
        print(f"\n⏳ 超过 {PUSH_TIMEOUT}s 没有返回。")
    print("-" * 60)

    if returncode == 0:
        print(f"✅ 推送成功：{url}")
        print("   下一步：把 README 与 pyproject.toml 里的 your-name 占位符换成你的用户名。")
        return 0

    if returncode == -1:
        print("❌ 推送超时。最可能是**凭据交互被挂起**（网络问题通常表现为 Connection reset，而不是超时）。")
        print("   处理方式：")
        print("     1) 确认已安装并登录 Git Credential Manager：")
        print("        winget install Git.Git        # 自带 GCM")
        print("        git config --global credential.helper manager")
        print("     2) 首次推送会弹一次浏览器登录窗口，登录后即可；")
        print("     3) 或者改用 SSH：把远程换成 git@github.com:<user>/<repo>.git（需已配置 SSH Key）")
        return 1

    print("❌ 推送失败。常见原因与处理：")
    print("   1) 仓库不是空的（建仓库时勾了 README/.gitignore/License）")
    print("      → git pull --rebase origin main --allow-unrelated-histories  然后重试")
    print("   2) 未登录 / 凭据失效")
    print("      → git config --global credential.helper manager  后用浏览器登录一次")
    print("   3) 没有该仓库的写权限 → 确认仓库属于你本人或你有 collaborator 权限")
    print("   4) 连接被 reset → 说明代理没生效，用 --proxy 显式指定端口")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
