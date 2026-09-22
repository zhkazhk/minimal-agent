"""weather：mock 天气查询。

- 内置若干城市的气候基线（base 温度、常见天气候选）；
- 未收录的城市根据城市名哈希「确定性造」一份合理数据（同一城市永远同一结果）；
- 支持「今天 / 明天 / 后天」等 date 参数，温度做确定性偏移，方便验证追问场景（用例 4）。

替换成真实数据源：把 `mock_weather` 换成调用和风/OpenWeather API 即可。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .registry import Tool

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "city": {
            "type": "string",
            "description": "城市名，例如 `上海`、`北京`、`Shenzhen`。必填。",
            "minLength": 1,
        },
        "date": {
            "type": "string",
            "description": "查询日期，可填 `今天`、`明天`、`后天` 或 `YYYY-MM-DD`，默认 `今天`。",
            "default": "今天",
        },
        "unit": {
            "type": "string",
            "description": "温度单位：`celsius`（摄氏，默认）或 `fahrenheit`（华氏）。",
            "enum": ["celsius", "fahrenheit"],
            "default": "celsius",
        },
    },
    "required": ["city"],
    "additionalProperties": False,
}

#: 城市 → (基准温度℃, 天气候选, 湿度区间)
CITY_PROFILE: dict[str, tuple[int, list[str], tuple[int, int]]] = {
    "上海": (24, ["多云", "阴", "小雨", "晴"], (60, 85)),
    "北京": (20, ["晴", "多云", "霾", "大风"], (25, 50)),
    "深圳": (29, ["雷阵雨", "多云", "晴"], (70, 95)),
    "广州": (30, ["雷阵雨", "多云", "闷热"], (70, 95)),
    "杭州": (25, ["多云", "小雨", "晴"], (60, 85)),
    "成都": (22, ["阴", "小雨", "多云"], (65, 90)),
    "西安": (21, ["晴", "多云", "浮尘"], (30, 60)),
    "哈尔滨": (12, ["晴", "多云", "阵雨"], (35, 60)),
    "shenzhen": (29, ["雷阵雨", "多云", "晴"], (70, 95)),
    "shanghai": (24, ["多云", "阴", "小雨", "晴"], (60, 85)),
    "beijing": (20, ["晴", "多云", "霾", "大风"], (25, 50)),
    "tokyo": (23, ["多云", "小雨", "晴"], (55, 80)),
    "singapore": (31, ["雷阵雨", "多云"], (75, 95)),
    "london": (16, ["小雨", "阴", "多云"], (65, 90)),
    "new york": (19, ["晴", "多云", "阵雨"], (40, 70)),
}

_DATE_OFFSET = {"今天": 0, "today": 0, "明天": 1, "tomorrow": 1, "后天": 2, "大后天": 3, "昨天": -1, "yesterday": -1}
_WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def _hash_int(*parts: str) -> int:
    return int(hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest(), 16)


def _resolve_offset(date: str) -> tuple[int, str]:
    date = (date or "今天").strip()
    key = date.lower()
    if key in _DATE_OFFSET:
        return _DATE_OFFSET[key], date
    for token, offset in _DATE_OFFSET.items():
        if token and token in key:
            return offset, date
    from datetime import date as _date, timedelta

    try:
        parsed = _date.fromisoformat(date)
        return (parsed - _date.today()).days, date
    except ValueError:
        return (0, date + "（无法识别，按今天处理）")


def mock_weather(city: str, date: str = "今天", unit: str = "celsius", *, ctx: Any = None) -> str:
    """返回给 LLM 的模拟天气文本（确定性）。"""
    city = (city or "").strip()
    if not city:
        return "未提供城市名，无法查询天气。"

    offset, date_label = _resolve_offset(date)
    profile = CITY_PROFILE.get(city) or CITY_PROFILE.get(city.lower())
    seed = _hash_int(city, date_label)
    if profile is None:
        base = 15 + _hash_int(city) % 18          # 未收录城市：15~32℃ 之间确定取值
        conditions = ["晴", "多云", "阴", "小雨", "阵雨", "雷阵雨"]
        humidity_range = (45, 85)
    else:
        base, conditions, humidity_range = profile

    temp_c = base + (seed % 9) - 4 + offset * 2
    condition = conditions[(seed // 7) % len(conditions)]
    humidity = humidity_range[0] + (seed // 13) % max(1, humidity_range[1] - humidity_range[0] + 1)
    wind_level = 1 + (seed // 17) % 5
    aqi = 25 + (seed // 19) % 130

    lo, hi = temp_c - 3 - (seed % 3), temp_c + 3 + (seed % 2)

    def render(value: int) -> str:
        if unit == "fahrenheit":
            return f"{round(value * 9 / 5 + 32)}°F"
        return f"{value}°C"

    payload = {
        "city": city,
        "date": date_label,
        "day_offset": offset,
        "condition": condition,
        "temperature": render(temp_c),
        "temp_range": f"{render(lo)} ~ {render(hi)}",
        "humidity": f"{humidity}%",
        "wind": f"{wind_level} 级",
        "aqi": aqi,
        "source": "mock-weather-api (demo, 非真实气象数据)",
    }
    readable = (
        f"{city} {date_label}：{condition}，气温 {render(temp_c)}（{render(lo)} ~ {render(hi)}），"
        f"湿度 {humidity}%，风力 {wind_level} 级，AQI {aqi}。"
    )
    return readable + "\n" + json.dumps(payload, ensure_ascii=False)


def weather_tool() -> Tool:
    return Tool(
        name="weather",
        description=(
            "天气查询工具（mock 实现）。当用户询问某个城市某天的天气、气温、是否下雨、空气质量时调用。"
            "输入城市名，可选日期（今天/明天/后天/YYYY-MM-DD）。返回温度、天气状况、湿度、风力、AQI。"
        ),
        parameters=SCHEMA,
        handler=mock_weather,
        tags=("weather", "mock"),
    )
