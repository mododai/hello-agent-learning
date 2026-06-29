from abc import ABC, abstractmethod
from my_agents.core.llm import AgenticLLM
from .message import Message
from typing import Optional, List, Dict, Any
from .config import Config
from ..tools.registry import ToolRegistry
import logging
logger = logging.getLogger(__name__)

class Agent(ABC):
    """Agent 基类"""

    def __init__(self,
                 name: str,
                 llm: AgenticLLM,
                 system_prompt: Optional[str] = None,
                 config: Optional[Config] = None,
                 tool_registry: Optional[ToolRegistry] = None,
                 ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = []
        self.tool_registry = tool_registry
    @abstractmethod
    def run(self, input_text: str, **kwargs):
        pass

    def add_message(self, message: Message):
        self._history.append(message)

    def clear_history(self):
        self._history.clear()

    def get_history(self):
        return self._history.copy()

    def __str__(self):
        return f"Agent(name={self.name}, provider={self.llm.provider})"

    def _build_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        构建工具 JSON Schema
        :return:
        """
        # 没有工具注册器则返回空
        if not self.tool_registry:
            return []

        schemas: List[Dict[str, Any]] = []
        # tool 工具类
        for tool in self.tool_registry.get_all_tools():
            schemas.append(tool.to_openai_schema())

        logger.debug(f"tools schemas: {schemas}")

        # TODO:function 函数性

        return schemas

    def _execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str | None:
        """
        执行工具调用并返回字符串结果
        统一的工具执行逻辑，支持：
        - Tool 对象（带类型转换）
        - TODO: 函数工具（简化调用）
        :param tool_name:
        :param arguments:
        :return:
        """

        if not self.tool_registry:
            logger.error("错误：未配置工具注册表")
            return "错误：未配置工具注册表"

        tool = self.tool_registry.get_tool(tool_name)
        # 解析Tool对象并执行
        if tool:
            try:
                typed_args = self._convert_parameter_types(tool_name, arguments)
                response = tool.run(typed_args)
            except Exception as e:
                logger.error(f"工具调用失败：{e}")
                return f"工具调用失败：{e}"

        # TODO:解析函数工具
        return response


    def _convert_parameter_types(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        根据工具定义转换参数类型
        :param tool_name:
        :param arguments:
        :return:
        """

        if not self.tool_registry:
            return arguments

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return arguments

        try:
            tool_parameters = tool.get_parameters()
        except Exception as e:
            return arguments

        # 映射 参数名: 参数类型
        type_map = {
            param.name: param.type
                for param in tool_parameters
        }

        # 将参数值转为对应的类型
        converted: Dict[str, Any] = {}
        for key, value in arguments.items():
            param_type = type_map.get(key)
            # 不存在类型要求, 直接使用
            if not param_type:
                converted[key] = value
                continue

            # 类型转化, 可能会出现转化异常, 出现则直接使用
            try:
                normalized = param_type.lower()

                # 浮点数
                if normalized in {"number", "float"}:
                    converted[key] = float(value)
                # 布尔值
                elif normalized in {"boolean", "bool"}:
                    if isinstance(value, bool):
                        converted[key] = value
                    elif isinstance(value, str):
                        converted[key] = value.lower() in {"true", "1"}
                    else:
                        converted[key] = bool(value)
                # 整型
                elif normalized in {"integer", "int"}:
                    converted[key] = int(value)

            except (TypeError, ValueError):
                converted[key] = value

        return converted
