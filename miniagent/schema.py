"""零依赖 JSON Schema 校验器（够用即可）。

为什么自己写：
- 交付要求「无重型框架依赖」，`jsonschema` 虽然是轻量库，但为了 `pip install -r requirements.txt`
  之后能**立刻跑通**（甚至完全不装任何依赖），这里内置一个覆盖常用关键字的最小实现。
- 若环境里存在官方 `jsonschema` 库，`validate_tool_arguments()` 会优先调用它（更严格、更全）。

支持关键字：type / properties / required / additionalProperties / items / enum / const /
minimum / maximum / minLength / maxLength / minItems / maxItems / anyOf / oneOf / allOf / default。
"""

from __future__ import annotations

from typing import Any, Iterable

try:  # 可选：装了官方库就用官方库
    import jsonschema as _jsonschema  # type: ignore
except Exception:  # pragma: no cover
    _jsonschema = None

from .errors import ToolValidationError


def has_jsonschema() -> bool:
    return _jsonschema is not None


_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list, tuple),
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "null": (type(None),),
}


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    types = _TYPE_MAP.get(expected)
    if types is None:
        return True
    if expected in ("object", "array", "string"):
        return isinstance(value, types)
    return isinstance(value, types)


def _describe(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return f"string({value[:40]!r})"
    if isinstance(value, dict):
        return f"object(keys={list(value)[:6]})"
    if isinstance(value, (list, tuple)):
        return f"array(len={len(value)})"
    return type(value).__name__


def coerce_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """对 LLM 常见的小毛病做**无损**修正（不改变语义）：

    - `"3"` → `3`（schema 要求 number/integer）
    - `"true"` / `"false"` → bool
    - 单值 → 单元素数组（schema 要求 array）
    - 补上 schema 里声明的 `default`

    刻意**不做** number → string 这类反向转换：string 字段收到数字通常是模型理解错了，
    应该让校验失败并把错误回灌给模型（否则会把 `expression: 123` 静默变成 `"123"`，
    掩盖真实错误）。
    """
    if not isinstance(schema, dict) or not isinstance(arguments, dict):
        return arguments
    props: dict[str, Any] = schema.get("properties") or {}
    out: dict[str, Any] = {}
    for key, value in arguments.items():
        spec = props.get(key)
        if not isinstance(spec, dict):
            out[key] = value
            continue
        expected = spec.get("type")
        if isinstance(expected, list):
            expected = next((t for t in expected if t != "null"), None)
        try:
            if expected in ("number", "integer") and isinstance(value, str):
                text = value.strip()
                if text:
                    number = float(text)
                    out[key] = int(number) if expected == "integer" and number.is_integer() else number
                    continue
            if expected == "boolean" and isinstance(value, str):
                low = value.strip().lower()
                if low in ("true", "yes", "1", "是"):
                    out[key] = True
                    continue
                if low in ("false", "no", "0", "否"):
                    out[key] = False
                    continue
            if expected == "array" and not isinstance(value, (list, tuple, dict)):
                out[key] = [value]
                continue
        except (TypeError, ValueError):
            pass
        out[key] = value
    # 补默认值
    for key, spec in props.items():
        if key not in out and isinstance(spec, dict) and "default" in spec:
            out[key] = spec["default"]
    return out


def validate_arguments(schema: dict[str, Any], arguments: Any, *, tool_name: str) -> dict[str, Any]:
    """校验并以 dict 形式返回参数；不合法时抛 `ToolValidationError`。"""
    if arguments is None:
        arguments = {}
    if isinstance(arguments, str):
        # 少数模型会把 arguments 序列化成字符串
        import json

        try:
            arguments = json.loads(arguments)
        except Exception as exc:
            raise ToolValidationError(
                f"工具 `{tool_name}` 的 arguments 是字符串且不是合法 JSON: {exc}",
                detail={"raw_arguments": arguments[:200]},
            ) from exc
    if not isinstance(arguments, dict):
        raise ToolValidationError(
            f"工具 `{tool_name}` 的 arguments 必须是 JSON 对象(object)，收到 {_describe(arguments)}",
            detail={"arguments": arguments},
        )

    coerced = coerce_arguments(schema, arguments)

    if _jsonschema is not None:
        try:
            _jsonschema.validate(coerced, schema)
        except Exception as exc:  # pragma: no cover - 取决于是否安装
            raise ToolValidationError(f"工具 `{tool_name}` 参数校验失败: {exc.message if hasattr(exc, 'message') else exc}") from exc
        return coerced

    errors = list(_iter_errors(schema, coerced, path="arguments"))
    if errors:
        raise ToolValidationError(
            f"工具 `{tool_name}` 参数校验失败: " + "; ".join(errors),
            detail={"received": coerced, "schema": schema},
        )
    return coerced


def _iter_errors(schema: Any, value: Any, path: str) -> Iterable[str]:
    if not isinstance(schema, dict):
        return
    if "const" in schema and value != schema["const"]:
        yield f"{path} 必须等于 {schema['const']!r}"
    if "enum" in schema and value not in schema["enum"]:
        yield f"{path} 必须是 {schema['enum']} 之一，收到 {value!r}"

    for combinator in ("allOf", "anyOf", "oneOf"):
        subs = schema.get(combinator)
        if not isinstance(subs, list):
            continue
        branch_errors = [list(_iter_errors(sub, value, path)) for sub in subs]
        ok = [not errs for errs in branch_errors]
        if combinator == "allOf":
            for errs in branch_errors:
                yield from errs
        elif combinator == "anyOf" and not any(ok):
            yield f"{path} 不满足 anyOf 任一分支: {branch_errors}"
        elif combinator == "oneOf" and sum(ok) != 1:
            yield f"{path} 必须且只能满足 oneOf 中的一个分支"

    expected = schema.get("type")
    if expected is not None:
        types = expected if isinstance(expected, list) else [expected]
        if not any(_type_ok(value, t) for t in types):
            yield f"{path} 类型应为 {'/'.join(types)}，实际 {_describe(value)}"
            return

    if isinstance(value, dict):
        props: dict[str, Any] = schema.get("properties") or {}
        for key in schema.get("required", []) or []:
            if key not in value:
                yield f"{path} 缺少必填字段 `{key}`"
        if schema.get("additionalProperties") is False:
            extra = [k for k in value if k not in props]
            if extra:
                yield f"{path} 出现未声明字段 {extra}，允许字段: {list(props)}"
        for key, sub in props.items():
            if key in value:
                yield from _iter_errors(sub, value[key], f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        if "minItems" in schema and len(value) < schema["minItems"]:
            yield f"{path} 至少 {schema['minItems']} 个元素"
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            yield f"{path} 最多 {schema['maxItems']} 个元素"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                yield from _iter_errors(item_schema, item, f"{path}[{idx}]")
    elif isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            yield f"{path} 长度至少 {schema['minLength']}"
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            yield f"{path} 长度最多 {schema['maxLength']}"
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            yield f"{path} 不能小于 {schema['minimum']}"
        if "maximum" in schema and value > schema["maximum"]:
            yield f"{path} 不能大于 {schema['maximum']}"
