"""
记忆工具

"""
from typing import List, Dict, Any

from ..base import Tool, ToolParameter
from ..response import ToolResponse
from ...memory.base import MemoryConfig


class MemoryTool(Tool):
    """
    提供记忆功能
    - 添加记忆
    - 检索
    - 管理
    """
    def __init__(self,
                 user_id: str = "default",
                 memory_config: MemoryConfig = None,):

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        pass

    def get_parameters(self) -> List[ToolParameter]:
        pass