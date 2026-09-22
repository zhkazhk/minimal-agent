"""calculator：安全表达式计算。

安全策略（不使用 eval / exec）：
1. 先用 `ast.parse(expr, mode="eval")` 把表达式解析成 AST；
2. 白名单遍历 AST，只允许数字常量、二元/一元运算符、比较、以及白名单函数名；
3. 命中黑名单关键字（`__`、`import`、`lambda`、`open`、`eval`…）直接拒绝；
4. 在**受限命名空间**里用递归求值器执行 —— 全程不调用 `eval`。

支持：+ - * / // % ** 、括号、比较、内置常量 pi/e、sqrt/abs/round/min/max/log/exp/sin/cos/tan/factorial 等。
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from .registry import Tool, ToolContext

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "expression": {
            "type": "string",
            "description": "要计算的数学表达式，例如 `123+456*7`、`sqrt(16)+2**10`、`(3.5+1.5)*2`。只能包含数字、运算符、括号和受支持的函数。",
            "minLength": 1,
        },
        "precision": {
            "type": "integer",
            "description": "结果保留的小数位数，默认 6 位后去掉多余的 0。",
            "default": 6,
        },
    },
    "required": ["expression"],
    "additionalProperties": False,
}

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_CMP_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

_FUNCS: dict[str, Any] = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow,
    "sqrt": math.sqrt, "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x),
    "log": math.log, "log2": math.log2, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos,
    "floor": math.floor, "ceil": math.ceil, "trunc": math.trunc,
    "factorial": math.factorial, "gcd": math.gcd, "hypot": math.hypot,
    "degrees": math.degrees, "radians": math.radians, "fabs": math.fabs, "fmod": math.fmod,
}
_CONSTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}

#: 严禁出现在表达式里的名字/模式（即使 AST 白名单已挡住，这里做双保险）
FORBIDDEN_NAMES = {
    "__import__", "eval", "exec", "compile", "open", "input", "globals", "locals",
    "vars", "dir", "getattr", "setattr", "delattr", "exit", "quit", "help", "breakpoint",
    "memoryview", "object", "type", "super", "classmethod", "staticmethod", "print",
}
FORBIDDEN_SUBSTRINGS = ("__", "import", "lambda", "os.", "sys.", "subprocess", "shutil", "socket", "pickle", "yaml", "builtins")

MAX_EXPRESSION_LEN = 500
MAX_POW_EXPONENT = 4096


class UnsafeExpression(ValueError):
    """表达式包含不被允许的语法/名字。"""


def _check_safety(expr: str) -> None:
    if len(expr) > MAX_EXPRESSION_LEN:
        raise UnsafeExpression(f"表达式过长（>{MAX_EXPRESSION_LEN} 字符），请拆分计算")
    lowered = expr.lower()
    for bad in FORBIDDEN_SUBSTRINGS:
        if bad in lowered:
            raise UnsafeExpression(f"表达式包含被禁止的内容 `{bad}`")


def _eval_node(node: ast.AST, depth: int = 0) -> Any:
    if depth > 40:
        raise UnsafeExpression("表达式嵌套层级过深")
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, depth + 1)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)) and not isinstance(node.value, bool):
            return node.value
        raise UnsafeExpression(f"不支持的常量类型: {type(node.value).__name__}")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise UnsafeExpression(f"不支持的运算符: {type(node.op).__name__}")
        left, right = _eval_node(node.left, depth + 1), _eval_node(node.right, depth + 1)
        if op is operator.pow and isinstance(right, (int, float)) and abs(right) > MAX_POW_EXPONENT:
            raise UnsafeExpression(f"指数过大（|{right}| > {MAX_POW_EXPONENT}），拒绝计算以防资源耗尽")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise UnsafeExpression(f"不支持的一元运算符: {type(node.op).__name__}")
        return op(_eval_node(node.operand, depth + 1))
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, depth + 1)
        for op_node, comparator in zip(node.ops, node.comparators):
            op = _CMP_OPS.get(type(op_node))
            if op is None:
                raise UnsafeExpression(f"不支持的比较运算符: {type(op_node).__name__}")
            right = _eval_node(comparator, depth + 1)
            if not op(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Name):
        name = node.id
        if name in FORBIDDEN_NAMES:
            raise UnsafeExpression(f"不允许使用名字 `{name}`")
        if name in _CONSTS:
            return _CONSTS[name]
        if name in _FUNCS:
            return _FUNCS[name]
        raise UnsafeExpression(f"未知标识符 `{name}`；可用常量 {sorted(_CONSTS)}，可用函数 {sorted(_FUNCS)}")
    if isinstance(node, ast.Call):
        func = _eval_node(node.func, depth + 1)
        if func not in _FUNCS.values():
            raise UnsafeExpression("只允许直接调用白名单函数")
        if node.keywords:
            raise UnsafeExpression("函数调用不支持关键字参数")
        args = [_eval_node(arg, depth + 1) for arg in node.args]
        return func(*args)
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_node(elt, depth + 1) for elt in node.elts]
    raise UnsafeExpression(f"不支持的语法节点: {type(node).__name__}（本项目只做数学表达式计算）")


def safe_eval(expression: str) -> Any:
    """安全求值：解析 → 白名单校验 → 递归求值。**不使用 eval/exec**。"""
    expr = (expression or "").strip()
    if not expr:
        raise UnsafeExpression("表达式为空")
    # 常见 LLM 习惯：全角符号 / `×` `÷` / 结尾等号
    expr = (
        expr.replace("×", "*").replace("÷", "/").replace("（", "(").replace("）", ")")
        .replace("，", ",").replace("^", "**").rstrip("= ").strip()
    )
    _check_safety(expr)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpression(f"表达式语法错误: {exc.msg}") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.Attribute, ast.Subscript, ast.Lambda, ast.Dict, ast.Set, ast.Await, ast.NamedExpr)):
            raise UnsafeExpression(f"不支持的语法: {type(node).__name__}")
    return _eval_node(tree)


def format_number(value: Any, precision: int = 6) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        text = f"{value:.{max(0, min(int(precision), 15))}f}".rstrip("0").rstrip(".")
        return text or "0"
    return str(value)


def calculator(expression: str, precision: int = 6) -> str:
    """工具 handler：返回给 LLM 的文本结果。"""
    expr = (expression or "").strip()
    value = safe_eval(expr)
    pretty = format_number(value, precision)
    normalized = (
        expr.replace("×", "*").replace("÷", "/").replace("（", "(").replace("）", ")").replace(" ", "")
    )
    return f"{normalized} = {pretty}" if normalized != pretty else pretty


def calculator_tool() -> Tool:
    return Tool(
        name="calculator",
        description=(
            "数学表达式计算器。当用户的问题涉及算术运算、百分比、幂、开方、三角函数、取整等"
            "数值计算时调用。输入一个合法数学表达式字符串，返回精确计算结果。"
        ),
        parameters=SCHEMA,
        handler=calculator,
        tags=("math", "offline"),
    )


async def calculator_async(expression: str, precision: int = 6, ctx: ToolContext | None = None) -> str:
    return calculator(expression, precision)
