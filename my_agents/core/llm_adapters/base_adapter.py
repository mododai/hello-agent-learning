"""
基础 LLM 适配器，子类通过继承进行扩展

"""

from abc import ABC, abstractmethod
from typing import Optional, Any, List, Dict, Iterator

from .llm_response import LLMToolResponse


class BaseLLMAdapter(ABC):

    def __init__(self,
                 api_key: str,
                 base_url: Optional[str],
                 model: str,
                 timeout: int,
                 ):

        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.model = model
        self._client = None
        self._async_client = None
        self.provider = "OpenAI"

    @abstractmethod
    def create_client(self) -> Any:
        """创建客户端实例"""
        pass

    @abstractmethod
    def invoke(self, messages: List[Dict], **kwargs) -> Any:
        """非流式调用"""
        pass

    @abstractmethod
    def stream_invoke(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        """流式调用，放回生成器"""
        pass

    @abstractmethod
    def invoke_with_tools(self, messages: List[Dict], tools: List[Dict], **kwargs) -> LLMToolResponse:
        """工具调用（Function Calling）"""
        pass

    def _is_thinking_model(self, **kwargs) -> bool:
        """
        判断是否为thinking model
        ps: deepseek的思考模式是可控的, 不能直接采用这种方式调用, reasoner将会弃用，参考最新的api文档
        """
        thinking_keywords = ["reasoner", "o1", "o3", "thinking"]
        model_lower = self.model.lower()
        return any(keyword in model_lower for keyword in thinking_keywords)



    # 可控制思考模式
    def enable_thinking_model(self, reasoning_effort, param):
        pass


    def disable_thinking_model(self, param):
        pass
