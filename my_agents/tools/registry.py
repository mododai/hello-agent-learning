"""
工具注册表。
"""

import logging
from typing import Any, Callable, Dict

from my_agents.tools.base import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """管理可供 Agent 调用的工具。"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._functions: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, tool: Tool):
        """注册 Tool 对象。"""
        if tool.name in self._tools:
            logger.warning(f"警告: 工具 '{tool.name}' 已存在，将被覆盖。")
        self._tools[tool.name] = tool
        logger.info(f"工具 '{tool.name}' 注册成功。")

    def register_function(self, name: str, description: str, func: Callable[[str], str]):
        """直接注册普通函数作为工具。"""
        if name in self._functions:
            logger.warning(f"警告: 工具 '{name}' 已存在，将被覆盖。")
        self._functions[name] = {
            "description": description,
            "func": func,
        }
        logger.info(f"工具 '{name}' 注册成功。")

    def get_tools_description(self) -> str:
        """返回所有已注册工具的文本描述。"""
        description = []
        for tool in self._tools.values():
            description.append(f"- {tool.name}: {tool.description}")

        for name, info in self._functions.items():
            description.append(f"- {name}: {info['description']}")

        return "\n".join(description) if description else "无可用工具"

    def get_tool(self, name: str) -> Tool:
        """根据名称获取 Tool 对象。"""
        return self._tools[name]

    def get_all_tools(self):
        """返回所有 Tool 对象。"""
        return self._tools.values()
