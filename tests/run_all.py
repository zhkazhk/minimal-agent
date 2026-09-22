"""零依赖测试入口：没有 pytest 也能跑全部测试。

    python tests/run_all.py                 # 跑全部模块
    python tests/run_all.py test_parser     # 只跑指定模块

为什么需要它：本项目通过 pip 安装 pytest 需要联网 / 有沙箱写权限，
为了让「clone → 立刻验证」在**任何环境**都成立，这里实现了一个最小测试运行器，
支持 tests/ 里实际用到的 pytest 能力：

- `@pytest.fixture`（含依赖 fixture 的 fixture）
- `@pytest.mark.parametrize`
- `pytest.raises(...)`（含 `str(excinfo.value)`）
- `monkeypatch.setenv/delenv`、`capsys.readouterr()`、`tmp_path`
- 测试函数的 fixture 参数注入

装了 pytest 的环境仍然推荐 `pytest -q`（报告更全）；两者覆盖同一批用例。
"""

from __future__ import annotations

import atexit
import importlib
import inspect
import io
import os
import re
import shutil
import sys
import tempfile
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

#: 统一把临时目录放进项目内，避开平台 TEMP 可能不可写的问题
TMP_ROOT = os.path.join(ROOT, ".tmp-tests")
os.makedirs(TMP_ROOT, exist_ok=True)


# ---------------------------------------------------------------------------
# 伪 pytest（仅在真正 import pytest 失败时启用）
# ---------------------------------------------------------------------------
class _ExcInfo:
    """pytest 的 ExceptionInfo 最小实现。

    `with pytest.raises(Err) as e:` 之后可能写 `e.value`，也可能直接被当成异常对象用
    （`str(e)`），因此这里把 `__str__` / `__repr__` 都代理到内部异常上。
    """

    def __init__(self) -> None:
        self.value: BaseException | None = None
        self.type: type[BaseException] | None = None

    def __str__(self) -> str:
        return str(self.value) if self.value is not None else ""

    def __repr__(self) -> str:
        return repr(self.value) if self.value is not None else "<ExceptionInfo: no exception>"


class _Raises:
    def __init__(self, expected: Any, match: str | None = None) -> None:
        self.expected = expected
        self.match = match
        self.value: BaseException | None = None
        self.type: type[BaseException] | None = None

    def __enter__(self) -> "_Raises":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        names = (
            tuple(getattr(item, "__name__", str(item)) for item in self.expected)
            if isinstance(self.expected, tuple)
            else getattr(self.expected, "__name__", str(self.expected))
        )
        if exc is None:
            raise AssertionError(f"期望抛出 {names}，但没有任何异常")
        if not isinstance(exc, self.expected):
            raise AssertionError(f"期望 {names}，实际抛出 {type(exc).__name__}: {exc}")
        if self.match and not re.search(self.match, str(exc)):
            raise AssertionError(f"异常信息 {str(exc)!r} 不匹配 {self.match!r}")
        self.value = exc
        self.type = type(exc)
        return True

    def __str__(self) -> str:
        return str(self.value) if self.value is not None else ""

    def __repr__(self) -> str:
        return repr(self.value) if self.value is not None else "<Raises: no exception>"


class _MonkeyPatch:
    def __init__(self) -> None:
        self._env: list[tuple[str, str | None]] = []
        self._cwd: str | None = None

    def setenv(self, name: str, value: str) -> None:
        self._env.append((name, os.environ.get(name)))
        os.environ[name] = value

    def delenv(self, name: str, raising: bool = True) -> None:
        self._env.append((name, os.environ.get(name)))
        if name in os.environ:
            del os.environ[name]
        elif raising:
            raise KeyError(name)

    def chdir(self, path: Any) -> None:
        if self._cwd is None:
            self._cwd = os.getcwd()
        try:
            os.chdir(str(path))
        except OSError:
            # 少数受限环境禁止切换工作目录；测试不应因此失败
            pass

    def undo(self) -> None:
        for name, value in reversed(self._env):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self._env.clear()
        if self._cwd is not None:
            os.chdir(self._cwd)
            self._cwd = None


class _Capture:
    def __init__(self, out: str, err: str) -> None:
        self.out = out
        self.err = err


class _CapSys:
    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self.out = io.StringIO()
        self.err = io.StringIO()

    def readouterr(self) -> _Capture:
        capture = _Capture(self.out.getvalue(), self.err.getvalue())
        self._reset()
        return capture


def _install_fake_pytest() -> None:
    if "pytest" in sys.modules:
        return
    module = type(sys)("pytest")

    class _Mark:
        def parametrize(self, argnames: Any, argvalues: Any, **kwargs: Any) -> Callable:
            def decorator(func: Callable) -> Callable:
                setattr(func, "_parametrize", (argnames, list(argvalues)))
                return func

            return decorator

        def skip(self, *args: Any, **kwargs: Any) -> Callable:
            def decorator(func: Callable) -> Callable:
                setattr(func, "_skip", True)
                return func

            return decorator

    def fixture(*args: Any, **kwargs: Any) -> Any:
        """支持 @pytest.fixture 与 @pytest.fixture() 两种写法。"""
        if args and callable(args[0]) and not kwargs:
            setattr(args[0], "_is_fixture", True)
            return args[0]

        def decorator(func: Callable) -> Callable:
            setattr(func, "_is_fixture", True)
            return func

        return decorator

    module.mark = _Mark()
    module.raises = lambda expected, match=None: _Raises(expected, match)
    module.fixture = fixture
    module.skip = lambda *a, **k: None
    module.fail = lambda msg="": (_ for _ in ()).throw(AssertionError(msg))
    module.importorskip = lambda name, *a, **k: importlib.import_module(name)
    sys.modules["pytest"] = module


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------
class _Runner:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.failures: list[str] = []
        self.temp_dirs: list[str] = []
        self._tmp_seq = 0

    # ---------------------------------------------------------------- fixture
    def collect_fixtures(self, module: Any) -> dict[str, Callable]:
        fixtures: dict[str, Callable] = {}
        for name, obj in vars(module).items():
            if callable(obj) and getattr(obj, "_is_fixture", False):
                fixtures[name] = obj
        return fixtures

    def resolve(self, name: str, fixtures: dict[str, Callable], cache: dict[str, Any], stack: tuple[str, ...] = ()) -> Any:
        if name in cache:
            return cache[name]
        if name == "tmp_path":
            # 用普通子目录而不是 mkdtemp：部分受限沙箱下 mkdtemp 产物不可再写入
            self._tmp_seq += 1
            path = Path(os.path.join(TMP_ROOT, f"case-{os.getpid()}-{self._tmp_seq}"))
            path.mkdir(parents=True, exist_ok=True)
            self.temp_dirs.append(str(path))
            cache[name] = path
            return path
        if name == "monkeypatch":
            patch = _MonkeyPatch()
            cache[name] = patch
            return patch
        if name == "capsys":
            cap = _CapSys()
            cache[name] = cap
            return cap
        if name not in fixtures:
            raise TypeError(f"无法解析 fixture `{name}`")
        if name in stack:
            raise TypeError(f"fixture 循环依赖: {' -> '.join(stack + (name,))}")
        func = fixtures[name]
        kwargs = {
            param: self.resolve(param, fixtures, cache, stack + (name,))
            for param in inspect.signature(func).parameters
        }
        value = func(**kwargs)
        if inspect.isgenerator(value):      # 支持 yield fixture
            gen = value
            value = next(gen)
            atexit.register(lambda g=gen: _drain(g))
        cache[name] = value
        return value

    # ---------------------------------------------------------------- 用例
    def cases(self, func: Callable) -> list[tuple[str, tuple]]:
        parametrize = getattr(func, "_parametrize", None)
        if not parametrize:
            return [("", ())]
        argnames, argvalues = parametrize
        names = [item.strip() for item in argnames.split(",")] if isinstance(argnames, str) else list(argnames)
        cases = []
        for values in argvalues:
            if len(names) == 1:
                values = (values,)
            label = " | ".join(str(item).replace("\n", "\\n")[:28] for item in values)
            cases.append((label, tuple(values)))
        return cases

    def param_names(self, func: Callable) -> list[str]:
        """参数化覆盖了哪些形参名（这些不能再当 fixture 解析）。"""
        parametrize = getattr(func, "_parametrize", None)
        if not parametrize:
            return []
        argnames = parametrize[0]
        return [item.strip() for item in argnames.split(",")] if isinstance(argnames, str) else list(argnames)

    def run_module(self, module_name: str) -> None:
        module = importlib.import_module(module_name)
        fixtures = self.collect_fixtures(module)
        tests = [
            obj
            for name, obj in vars(module).items()
            if name.startswith("test_") and inspect.isfunction(obj)
        ]
        tests.sort(key=lambda func: func.__code__.co_firstlineno)

        for func in tests:
            if getattr(func, "_skip", False):
                continue
            for label, values in self.cases(func):
                name = f"{module_name.split('.')[-1]}::{func.__name__}" + (f"[{label}]" if label else "")
                self.run_case(func, values, fixtures, name)

    def run_case(self, func: Callable, values: tuple, fixtures: dict[str, Callable], name: str) -> None:
        cache: dict[str, Any] = {}
        capsys: _CapSys = _CapSys()
        monkeypatch: _MonkeyPatch = _MonkeyPatch()
        arg_map = dict(zip(self.param_names(func), values))
        try:
            kwargs = {}
            for param in inspect.signature(func).parameters:
                if param in arg_map:
                    continue                      # 参数化提供，不用 fixture
                if param == "capsys":
                    kwargs[param] = capsys
                elif param == "monkeypatch":
                    kwargs[param] = monkeypatch
                else:
                    kwargs[param] = self.resolve(param, fixtures, cache)
            with redirect_stdout(capsys.out), redirect_stderr(capsys.err):
                func(**arg_map, **kwargs)
            self.passed += 1
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - 测试失败要全部收集，不中断
            self.failed += 1
            self.failures.append(f"{name}\n{traceback.format_exc(limit=8)}")
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        finally:
            monkeypatch.undo()

    def cleanup(self) -> None:
        """清理临时目录 —— **默认跳过**。

        原因：部分受限沙箱会对「批量删除目录」直接终止进程（表现为整个测试运行被 kill）。
        临时目录本身在 `.tmp-tests/` 下且已被 .gitignore 忽略，留着无害。
        需要清理时设置环境变量 `MINIAGENT_TEST_CLEANUP=1`。
        """
        if os.getenv("MINIAGENT_TEST_CLEANUP", "").strip().lower() not in ("1", "true", "yes"):
            return
        for path in self.temp_dirs:
            try:
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
                os.rmdir(path)
            except OSError:
                pass
        self.temp_dirs.clear()


def _drain(gen: Any) -> None:
    try:
        next(gen)
    except StopIteration:
        pass
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    # 必须在导入任何测试模块**之前**准备好 pytest：
    # 测试文件顶部会 `import pytest`，若此时还没装好 fake 模块，导入会直接失败。
    real_pytest = False
    try:
        import pytest  # noqa: F401

        real_pytest = True
    except ImportError:
        _install_fake_pytest()
    if not real_pytest:
        # 确认 fake 可用（防止前面某处已把 pytest 塞进 sys.modules 但不可用）
        real_pytest = "pytest" not in sys.modules or not hasattr(sys.modules["pytest"], "raises")
        if not hasattr(sys.modules.get("pytest"), "raises"):
            sys.modules.pop("pytest", None)
            _install_fake_pytest()

    if argv:
        modules = [name if name.startswith("tests.") else f"tests.{name}" for name in argv]
    else:
        modules = sorted(
            f"tests.{os.path.splitext(name)[0]}"
            for name in os.listdir(os.path.join(ROOT, "tests"))
            if name.startswith("test_") and name.endswith(".py")
        )

    print("=" * 74)
    print(f" minimal-agent 测试运行器（{'已检测到 pytest' if real_pytest else '零依赖内置运行器'}）")
    print(f" 模块: {', '.join(m.split('.')[-1] for m in modules)}")
    print("=" * 74)

    runner = _Runner()
    for module_name in modules:
        print(f"\n▶ {module_name}")
        try:
            runner.run_module(module_name)
        except Exception as exc:  # noqa: BLE001 - 导入期错误也要报告
            runner.failed += 1
            runner.failures.append(f"{module_name} 导入失败\n{traceback.format_exc(limit=8)}")
            print(f"  FAIL  模块导入失败: {type(exc).__name__}: {exc}")

    runner.cleanup()
    print("\n" + "=" * 74)
    print(f" 结果：{runner.passed} passed, {runner.failed} failed")
    if runner.failures:
        print("-" * 74)
        for item in runner.failures:
            print(item)
    print("=" * 74)
    return 0 if runner.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
