"""
TODO: 实现工具注册，现在只是暂位依赖
"""
from my_agents.tools.base import Tool
from typing import Dict, Any, Callable
import logging
logger = logging.getLogger(__name__)
class ToolRegistry:
    """
    工具注册表工具

    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._functions: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, tool: Tool):
        """注册Tool对象"""
        if tool.name in self._tools:
            logger.warning(f"警告:工具 '{tool.name}' 已存在，将被覆盖。")
        self._tools[tool.name] = tool
        logger.info(f"工具 '{tool.name} 注册成功。'")

    def register_function(self, name: str, description: str, func: Callable[[str], str]):
        """
        直接注册函数作为工具
        :param name:
        :param description:
        :param func:
        :return:
        """

        if name in self._functions:
            logger.warning(f"警告:工具 '{name}' 已存在，将被覆盖。")
        self._functions[name] = {
            "description": description,
            "func": func,
        }
        logger.info(f"工具 '{name} 注册成功。'")

    def get_tools_description(self) -> str:
        description = []
        for tool in self._tools.values():
            description.append(f"- {tool.name}: {tool.description}")

        for name, info in self._functions.items():
            description.append(f"- {name}: {info['description']}")

        return "\n".join(description) if description else "无可用工具"

    def get_tool(self, name: str) -> Tool:
        return self._tools[name]

    def get_all_tools(self):
        return self._tools.values()