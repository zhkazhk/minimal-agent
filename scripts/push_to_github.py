"""一键把本仓库推送到 GitHub（幂等，可重复执行）。

    python scripts/push_to_github.py https://github.com/<user>/minimal-agent.git

做四件事，每步都先检查再动手：
1. 校验远程地址格式，并设置/更新 `origin`；
2. 推送前跑一次**安全检查**：确认没有任何密钥文件被 git 跟踪
   （`.env`、`*.key`、`.inbox/*` 等），有则直接中止；
3. 打印将要推送的 commit 列表，确认工作区干净；
4. 执行 `git push -u origin <当前分支>`。

认证说明：推送需要 GitHub 凭据。若本机已登录 Git Credential Manager，
命令会直接成功；否则 git 会提示输入用户名与 Personal Access Token（PAT），
或弹出浏览器登录窗口。
"""

from __future__ import annotations

import os
import re
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


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    url = argv[0].strip()

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

    # ---------- 3. 设置远程 ----------
    existing = git("remote", "get-url", "origin", check=False).stdout.strip()
    if existing and existing != url:
        git("remote", "set-url", "origin", url)
        print(f"\n已更新 origin: {existing} -> {url}")
    elif not existing:
        git("remote", "add", "origin", url)
        print(f"\n已添加 origin: {url}")
    else:
        print(f"\norigin 已就绪: {url}")

    # ---------- 4. 推送 ----------
    print(f"\n执行：git push -u origin {branch}\n" + "-" * 60)
    result = subprocess.run(
        ["git", "push", "-u", "origin", branch],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    print("-" * 60)
    if result.returncode == 0:
        print(f"✅ 推送成功：{url}")
        print("   下一步：把 README 与 pyproject.toml 里的 your-name 占位符换成你的用户名。")
        return 0

    print("❌ 推送失败。常见原因与处理：")
    print("   1) 仓库不是空的（建仓库时勾了 README/.gitignore/License）")
    print("      → git pull --rebase origin main --allow-unrelated-histories  然后重试")
    print("   2) 未登录 / 凭据失效")
    print("      → 安装 Git Credential Manager 后重试，或改用 SSH 地址 git@github.com:...")
    print("   3) 没有该仓库的写权限 → 确认仓库属于你本人或你有 collaborator 权限")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
