import json
from typing import Optional, TYPE_CHECKING, List, Dict

from my_agents.core.agent import Agent
from my_agents.core.llm import AgenticLLM
from ..core.config import Config
import logging

from ..core.message import Message

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry

class SimpleAgent(Agent):




    def __init__(self,
                 name: str,
                 llm: AgenticLLM,
                 system_prompt: Optional[str] = None,
                 config: Optional[Config] = None,
                 tool_registry: Optional['ToolRegistry'] = None,
                 enable_tool_calling: bool = True,
                 max_tool_iterations: int = 3
                 ):
        """
        初始话 SimpleAgent
        :param name: Agent名称
        :param llm: LLM实例
        :param system_prompt: 系统提示词
        :param config: 配置对象
        :param tool_registry: 工具注册表（可选，如果提供则启用工具调用）
        :param enable_tool_calling: 是否启用工具调用（只有在提供tool_registry时生效）
        :param max_tool_iterations: 最大工具调用迭代次数
        """

        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,

            config=config
        )
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling
        self.max_tool_iterations = max_tool_iterations
        logger.info(f"{self.name} 初始化完成，工具调用: {'启用' if self.enable_tool_calling else '禁用'}")


    def run(self, input_text: str, **kwargs):
        """
        重写的运行方法 - 实现简单对话逻辑，支持可选工具调用 （基于 Function Calling）
        :param input_text: 问题
        :param kwargs: 自定义参数
        """
        logger.info(f"{self.name} 正在处理: {input_text}")

        # 创建消息列表, 包含提示词, 历史消息
        messages = self._create_messages(input_text)

        # 判断是否启动工具调用, 没有则返回 LLM 响应
        if not self.enable_tool_calling or not self.tool_registry:
            llm_response = self.llm.invoke(
                messages=messages,
                enable_thinking=kwargs.pop("enable_thinking", False),
                **kwargs
            )
            response_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response_text, "assistant"))

            return response_text

        tool_schemas = self._build_tool_schemas()

        current_iteration = 0
        final_response = ""

        while current_iteration < self.max_tool_iterations:
            current_iteration += 1

            try:
                #current_tool_choice = "none" if messages and messages[-1].get("role") == "tool" else "auto"
                response = self.llm.invoke_with_tools(
                    messages=messages,
                    tools=tool_schemas,
                    #tool_choice=current_tool_choice,
                    **kwargs,
                )
            except Exception as e:
                logger.exception(f"LLM 调用失败: {e}")
                break

            # 处理工具调用
            tool_calls = response.tool_calls
            if not tool_calls:
                final_response = response.content
                break
            # 保存对话历史
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments,
                            }
                        }
                        for tc in tool_calls
                    ],
                }
            )

            # 执行tool call
            for tool_call in tool_calls:
                tool_name = tool_call.name
                tool_call_id = tool_call.id

                try:
                    arg = json.loads(tool_call.arguments)
                except json.JSONDecodeError as e:
                    logger.error(f"工具参数解析失败: {e}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": f"错误: 参数格式不正确 - {str(e)}"
                    })
                    continue
                # 执行工具（复用基类方法）
                tool_content = self._execute_tool_call(tool_name, arg)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_content
                })

        # 如果达到最大工具调用迭代轮次, llm没有收到最新的工具调用结果, 可以重新调用invoke
        if current_iteration >= self.max_tool_iterations and not final_response:
            llm_response = self.llm.invoke(
                messages=messages,
                **kwargs,
            )
            final_response = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

        # 保存历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))

        return final_response



    def _create_messages(self, input_text) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []

        # 添加系统提示词
        if self.system_prompt is not None:
            messages.append({
                "role": "system",
                "content": self.system_prompt,
            })

        # 添加历史消息, msg: Messages
        for msg in self._history:
            messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        # 用户提问
        messages.append({
            "role": "user",
            "content": input_text,
        })

        return messages



