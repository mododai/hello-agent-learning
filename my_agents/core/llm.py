from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List, Dict, Iterator
from typing import Optional
import logging

from my_agents.core.llm_adapters.deepseek_adapter import DeepSeekerAdapter
from .llm_adapters import BaseLLMAdapter, OpenAIAdapter, LLMResponse, LLMToolResponse

logger = logging.getLogger(__name__)
load_dotenv()


def create_adapter(api_key: str, base_url: Optional[str], timeout: int, model: str) -> BaseLLMAdapter:
    """
    创建适配器，
    可以根据需求自行添加其他提供商

    :param api_key: 密钥
    :param base_url: 服务地址
    :param timeout: 超时时间（秒）
    :param model: 模型
    :return: BaseLLMAdapter
    """
    if model.lower().startswith("deepseek"):
        return DeepSeekerAdapter(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            model=model
        )

    # 其他适配器，这里只实现OpenAI的
    return OpenAIAdapter(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        model=model
    )


class AgenticLLM:
    """
    LLM 客户端
    """

    def __init__(self,
                 model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 timeout: Optional[int] = None,
                 **kwargs
                 ):
        """
        初始化LLM客户端

        :param model: 模型名称，默认从 LLM_MODEL_ID 读取
        :param api_key: API密钥，默认从 LLM_API_KEY 读取
        :param base_url: 服务地址，默认从 LLM_BASE_URL 读取
        :param temperature: 温度参数，默认0.7，控制模型输出的 “随机性” 与 “确定性”
        :param max_tokens: 最大token数
        :param timeout: 超时时间（秒），默认从 LLM_TIMEOUT 读取，默认60秒
        :param kwargs:
        """
        # 加载配置
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs

        # 参数校验
        if not self.model:
            raise ValueError("必须提供模型名称（model参数或LLM_MODEL_ID环境变量）")
        if not self.api_key:
            raise ValueError("必须提供API密钥（api_key参数或LLM_API_KEY环境变量）")
        if not self.base_url:
            raise ValueError("必须提供服务地址（base_url参数或LLM_BASE_URL环境变量）")

        self._adapter: BaseLLMAdapter = create_adapter(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            model=self.model,
        )

    @property
    def provider(self):
        return self._adapter.provider

    def think(self,
              message: List[Dict[str, str]],
              temperature: Optional[float] = None,
              enable_thinking: bool = True,
              reasoning_effort: Optional[str] = None,
              **kwargs
              ) -> Iterator[str]:
        """
        调用大语言模型进行思考，并返回其流式响应。

        :param reasoning_effort: 思考强度
        :param enable_thinking: 是否启动思考模式
        :param message: 消息列表
        :param temperature: 模型温度
        :return: yield: 流式响应的文本片段(str)
        """
        logger.info(f"正在调用 {self.model} 模型...")
        # 准备请求参数
        # 使用默认温度如果不指定的话
        if not hasattr(kwargs, "temperature"):
            kwargs["temperature"] = self.temperature if not temperature else temperature
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        # ps: deepseek默认启动思考模式
        if enable_thinking:
            kwargs = self._adapter.enable_thinking_model(reasoning_effort ,**kwargs)
        else:
            kwargs = self._adapter.disable_thinking_model(**kwargs)

        try:
            # 获取流
            response = self._adapter.stream_invoke(message, **kwargs)
            for chunk in response:
                print(chunk, end="", flush=True)
                yield chunk
            print() #换行

        except Exception as e:
            logger.error(f"调用LLM API时发生错误: {e}")
            return None

    def invoke(self,
               messages: List[Dict[str, str]],
               enable_thinking: bool = False,
               reasoning_effort: Optional[str] = None,
               **kwargs
               ) -> LLMResponse:
        """
        非流式调用LLM，返回完整响应对象。
        :param reasoning_effort: 思考强度
        :param enable_thinking: 是否启动思考模式
        :param messages: 消息列表
        :param kwargs:
        :return:
        """
        logger.info(f"正在调用 {self.model} 模型...")
        # 准备请求参数
        # 使用默认温度如果不指定的话
        if enable_thinking:
            kwargs = self._adapter.enable_thinking_model(reasoning_effort, **kwargs)
        else:
            kwargs = self._adapter.disable_thinking_model(**kwargs)

        if not hasattr(kwargs, "temperature"):
            kwargs["temperature"] = self.temperature

        return self._adapter.invoke(messages, **kwargs)

    def invoke_with_tools(self, messages, tools, tool_choice, param) -> LLMToolResponse:
        pass


from serpapi import SerpApiClient

def search(query: str) -> str:
    """
        一个基于SerpApi的实战网页搜索引擎工具。
        它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    logger.info("正在执行 [SerpApi] 网页搜索: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            logger.error("SERPAPI_API_KEY 未在 .env 文件中配置。")

        params = {
            "engine": "google",
            "q": query,
            "gl": "cn",
            "hl": "zh-cn",
            "api_key": api_key,
        }

        # 搜索
        client = SerpApiClient(params)
        results = client.get_dict()

        # 智能解析:优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])

        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]

        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]

        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i + 1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)

        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        er = f"搜索时发生错误: {e}"
        logger.error(er)
        return er


if __name__ == "__main__":
    try:
        llmClient = AgenticLLM()

        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]

        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)