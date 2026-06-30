import time
from typing import List, Dict, Iterator, Any, Optional

from .llm_response import ToolCall
from .base_adapter import BaseLLMAdapter
import logging

from .llm_response import LLMToolResponse, LLMResponse, StreamStats

logger = logging.getLogger(__name__)

class OpenAIAdapter(BaseLLMAdapter):
    """
    OpenAI 兼容

    """

    def __init__(self,
                 api_key: str,
                 base_url: Optional[str],
                 model: str,
                 timeout: int,
                 ):
        super().__init__(api_key, base_url, model, timeout)
        self.last_stats = None
        self.provider = "OpenAI"

    def create_client(self) -> Any:
        """创建OpenAI客户端"""
        try:
            from openai import OpenAI
            return OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        except ImportError as e:
            raise ImportError("导入openai失败")


    def invoke(self, messages: List[Dict], **kwargs) -> Any:
        # 如果不存在客户端则创建
        if not self._client:
            self._client = self.create_client()

        try:
            # 生成回复
            start = time.time()
            response = self._client.chat.completions.create(
                model = self.model,
                messages = messages,
                **kwargs,
            )
            latency_ms = int((time.time() - start) * 1000)
            # 提出内容
            choice = response.choices[0]
            content = choice.message.content or ""
            # 思维链特殊处理
            reasoning_content = self._get_reasoning_content(response) if self._is_thinking_model(**kwargs) else None

            # 记录使用信息
            usage = {}
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                latency_ms=latency_ms,
                reasoning_content=reasoning_content,
            )

        except Exception as e:
            logger.error(f"OpenAI API调用失败: {str(e)}")
            raise Exception(e)

    def stream_invoke(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        """流水调用"""
        if not self._client:
            self._client = self.create_client()

            try:
                start = time.time()
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    **kwargs
                )
                # stream流
                collected_content = []
                reasoning_content = None
                usage = {}
                for chunk in response:
                    logger.debug(chunk)
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage = {
                            "prompt_tokens": chunk.usage.prompt_tokens,
                            "completion_tokens": chunk.usage.completion_tokens,
                            "total_tokens": chunk.usage.total_tokens,
                        }
                    # 判断是否有内容
                    choice = getattr(chunk, "choices", None)
                    if not choice:
                        continue
                    delta = getattr(choice[0], "delta", None)
                    if not delta:
                        continue
                    content = getattr(delta, "content", None)

                    if content:
                        # 这里不能判断为None直接continue, 因为模型可能在思考
                        collected_content.append(content)
                        yield content

                    if self._is_thinking_model(**kwargs):
                        reasoning_delta = getattr(delta, "reasoning_content", None)
                        if reasoning_delta:
                            reasoning_content += reasoning_delta if reasoning_content else ""

                # 返回统计信息（存储到适配器，供外部获取）
                latency_ms = int((time.time() - start) * 1000)

                self.last_stats = StreamStats(
                    model=self.model,
                    usage=usage,
                    latency_ms=latency_ms,
                    reasoning_content=reasoning_content
                )

            except Exception as e:
                logger.error(f"OpenAI API流式调用失败: {str(e)}")



    def invoke_with_tools(self, messages: List[Dict], tools: List[Dict], **kwargs) -> LLMToolResponse:
        """工具调用(Function Calling)"""
        if not self._client:
            self._client = self.create_client()

        try:
            start = time.time()
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                **kwargs
            )

            latency_ms = int((time.time() - start) * 1000)
            tool_calls = [] #  LLM 调用的工具列表
            message = response.choices[0].message

            content = message.content or ""

            if message:
                for tc in message.tool_calls or []:
                    tool_calls.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        )
                    )
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return LLMToolResponse(
                content=content,
                tool_calls=tool_calls,
                model=response.model,
                usage=usage,
                latency_ms=latency_ms,
            )
        except Exception as e:
            #logger.error(f"OpenAI Function Calling调用失败: {str(e)}")
            raise RuntimeError(f"OpenAI Function Calling调用失败: {e}") from e

    def _is_thinking_model(self,**kwargs) -> True:

        return super()._is_thinking_model(**kwargs)

    def _get_reasoning_content(self, response) -> str:
        return response.choices[0].reasoning_content



