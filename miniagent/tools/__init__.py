"""内置工具集。"""

from .calculator import SCHEMA as CALCULATOR_SCHEMA, calculator, calculator_tool, safe_eval
from .registry import Tool, ToolContext, ToolRegistry, build_default_registry, normalize_result
from .search import KNOWLEDGE_BASE, mock_search, search_tool
from .weather import CITY_PROFILE, mock_weather, weather_tool

__all__ = [
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "build_default_registry",
    "normalize_result",
    "calculator",
    "calculator_tool",
    "safe_eval",
    "CALCULATOR_SCHEMA",
    "mock_search",
    "search_tool",
    "KNOWLEDGE_BASE",
    "mock_weather",
    "weather_tool",
    "CITY_PROFILE",
]
