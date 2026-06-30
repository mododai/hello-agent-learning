from abc import ABC, abstractmethod
from typing import Dict, Any, List

from pydantic import BaseModel

from .response import ToolResponse


class ToolParameter(BaseModel):
    """工具参数定义"""
    name: str   # 参数名
    type: str   # 参数类型
    description: str    # 描述
    required: bool = True   # 是否必须
    default: Any = None     # 默认值

class Tool(ABC):
    """
    工具基类 - 给大模型调用的

    """

    def __init__(self,
                 name: str,
                 description: str,
                 ):
        self.name = name
        self.description = description


    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        """执行"""
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self) ->  List[ToolParameter]:
        """获取工具参数"""
        raise NotImplementedError

    def to_openai_schema(self) -> Dict[str, Any]:
        """
        转换为 OpenAI function calling schema 格式

        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather of a location, the user should supply a location first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"]
            },
        }

        :return:
        """
        parameters = self.get_parameters()

        #
        properties = {}
        required = []

        for param in parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }

            # 如果有默认值，添加到描述中（OpenAI schema 不支持 default 字段）
            if param.default is not None:
                prop["description"] += f"(默认: {param.default})"

            # 如果是数组类型，添加 items 定义
            if param.type == "array":
                prop["items"] = {
                    "type": "string",
                }

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function", # 固定"function"
            "function": {
                "name": self.name, # 工具名称
                "description": self.description,    # 工具描述
                "parameters": {
                    "type": "object",   # 固定"object"
                    "properties": properties,   # 参数字典
                    "required": required,       # 必填参数
                }
            }
        }
