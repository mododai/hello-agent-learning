"""SimpleAgent 通过 Function Calling 使用完整 MemoryTool 的端到端测试。"""

import json
import os
import tempfile
import unittest
from copy import deepcopy

from my_agents.agents.simple_agent import SimpleAgent
from my_agents.core.llm_adapters.llm_response import (
    LLMResponse,
    LLMToolResponse,
    ToolCall,
)
from my_agents.memory import (
    EpisodicMemory,
    MemoryConfig,
    MemoryManager,
    SemanticMemory,
    WorkingMemory,
)
from my_agents.memory.storage.document_store import SQLiteDocumentStore
from my_agents.tools import MemoryTool, ToolRegistry
from tests.support import FakeEmbedder, FakeVectorStore


class ScriptedMemoryLLM:
    """按用户问题产生确定性工具调用，并记录 Agent 发送的完整消息。"""

    provider = "fake"

    def __init__(self):
        self.tool_requests = []
        self.invoke_requests = []

    @staticmethod
    def _latest_user_text(messages):
        """查找当前轮用户消息，避免被此前对话历史干扰。"""
        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("content", "")
        return ""

    def invoke_with_tools(self, messages, tools, **kwargs):
        """首次返回工具调用，收到工具结果后生成最终自然语言回答。"""
        self.tool_requests.append(
            {
                "messages": deepcopy(messages),
                "tools": deepcopy(tools),
                "kwargs": dict(kwargs),
            }
        )
        if messages and messages[-1].get("role") == "tool":
            tool_messages = [
                message["content"]
                for message in messages
                if message.get("role") == "tool"
            ]
            return LLMToolResponse(
                content="工具执行完成：\n" + "\n".join(tool_messages),
                tool_calls=[],
                model="fake-memory-model",
            )

        user_text = self._latest_user_text(messages)
        if "三类记忆" in user_text:
            calls = [
                ToolCall(
                    id="working-call",
                    name="memory",
                    arguments=json.dumps(
                        {
                            "action": "remember",
                            "memory_type": "working",
                            "content": "当前正在进行 Agent 记忆联调",
                        },
                        ensure_ascii=False,
                    ),
                ),
                ToolCall(
                    id="episodic-call",
                    name="memory",
                    arguments=json.dumps(
                        {
                            "action": "remember",
                            "memory_type": "episodic",
                            "content": "今天完成了 Agent 与 MemoryTool 的联调",
                            "session_id": "agent-memory-test",
                            "outcome": "联调成功",
                        },
                        ensure_ascii=False,
                    ),
                ),
                ToolCall(
                    id="semantic-call",
                    name="memory",
                    arguments=json.dumps(
                        {
                            "action": "remember",
                            "memory_type": "semantic",
                            "content": "用户喜欢喝绿茶",
                            "predicate": "drink_preference",
                            "object_value": "绿茶",
                            "knowledge_type": "preference",
                        },
                        ensure_ascii=False,
                    ),
                ),
            ]
        elif "喜欢喝什么" in user_text:
            calls = [
                ToolCall(
                    id="recall-call",
                    name="memory",
                    arguments=json.dumps(
                        {
                            "action": "recall",
                            "predicate": "drink_preference",
                        },
                        ensure_ascii=False,
                    ),
                )
            ]
        else:
            calls = []

        return LLMToolResponse(
            content=None,
            tool_calls=calls,
            model="fake-memory-model",
        )

    def invoke(self, messages, **kwargs):
        """为 SimpleAgent 达到最大工具轮数时提供确定性兜底响应。"""
        self.invoke_requests.append(
            {"messages": deepcopy(messages), "kwargs": dict(kwargs)}
        )
        return LLMResponse(content="兜底回答", model="fake-memory-model")


class EndlessToolCallingLLM(ScriptedMemoryLLM):
    """持续请求 stats 工具，用于触发 Agent 的最大工具迭代兜底分支。"""

    def invoke_with_tools(self, messages, tools, **kwargs):
        self.tool_requests.append(
            {
                "messages": deepcopy(messages),
                "tools": deepcopy(tools),
                "kwargs": dict(kwargs),
            }
        )
        return LLMToolResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id=f"stats-call-{len(self.tool_requests)}",
                    name="memory",
                    arguments=json.dumps({"action": "stats"}),
                )
            ],
            model="fake-memory-model",
        )

    def invoke(self, messages, **kwargs):
        self.invoke_requests.append(
            {"messages": deepcopy(messages), "kwargs": dict(kwargs)}
        )
        return LLMResponse(
            content="已根据最后一次记忆统计生成兜底回答",
            model="fake-memory-model",
        )


class AgentMemoryToolTest(unittest.TestCase):
    """验证 Agent、工具注册表、MemoryTool 和三类存储的完整调用链。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = MemoryConfig(storage_path=self.temp_dir.name)
        self.embedder = FakeEmbedder()
        self.vector_store = FakeVectorStore()
        self.working = WorkingMemory(self.config)
        self.episodic = EpisodicMemory(
            self.config,
            storage_backend={
                "embedder": self.embedder,
                "vector_store": self.vector_store,
            },
        )
        self.semantic = SemanticMemory(
            self.config,
            storage_backend={
                "embedder": self.embedder,
                "vector_store": self.vector_store,
            },
        )
        self.manager = self.build_manager("u1")
        self.memory_tool = MemoryTool(user_id="u1", memory_manager=self.manager)
        self.registry = ToolRegistry()
        self.registry.register_tool(self.memory_tool)
        self.llm = ScriptedMemoryLLM()
        self.agent = SimpleAgent(
            name="memory-test-agent",
            llm=self.llm,
            system_prompt="需要长期保存或回忆信息时调用 memory 工具。",
            tool_registry=self.registry,
            max_tool_iterations=3,
        )

    def tearDown(self):
        self.semantic.doc_store.close()
        abs_path = os.path.abspath(self.semantic.doc_store.db_path)
        SQLiteDocumentStore._instances.pop(abs_path, None)
        SQLiteDocumentStore._initialized_dbs.discard(abs_path)
        self.temp_dir.cleanup()

    def build_manager(self, user_id: str) -> MemoryManager:
        """让不同用户管理器共享底层实例，以真实验证用户隔离。"""
        return MemoryManager(
            user_id=user_id,
            config=self.config,
            memory_instances={
                "working": self.working,
                "episodic": self.episodic,
                "semantic": self.semantic,
            },
        )

    def test_agent_calls_memory_tool_to_store_and_recall(self):
        """Agent 应完成 LLM→工具→LLM 循环，并在下一轮召回已存语义事实。"""
        store_answer = self.agent.run("请把测试内容分别保存到三类记忆中")
        recall_answer = self.agent.run("我喜欢喝什么？")

        stats = self.manager.get_stats()
        self.assertIn("已保存 working 记忆", store_answer)
        self.assertIn("已保存 episodic 记忆", store_answer)
        self.assertIn("已保存 semantic 记忆", store_answer)
        self.assertIn("用户喜欢喝绿茶", recall_answer)
        self.assertEqual(stats["total_count"], 3)
        self.assertEqual(stats["by_type"]["working"]["count"], 1)
        self.assertEqual(stats["by_type"]["episodic"]["count"], 1)
        self.assertEqual(stats["by_type"]["semantic"]["count"], 1)

        # 两轮对话都应保存 user/assistant 历史，但工具中间消息只属于当轮调用上下文。
        history = self.agent.get_history()
        self.assertEqual([message.role for message in history], [
            "user",
            "assistant",
            "user",
            "assistant",
        ])

    def test_agent_sends_valid_schema_and_tool_results_back_to_llm(self):
        """验证 Agent 提供 memory schema，并用匹配的 tool_call_id 回传每个结果。"""
        self.agent.run("请把测试内容分别保存到三类记忆中")

        first_request = self.llm.tool_requests[0]
        second_request = self.llm.tool_requests[1]
        schema_names = [tool["function"]["name"] for tool in first_request["tools"]]
        self.assertEqual(schema_names, ["memory"])

        tool_messages = [
            message
            for message in second_request["messages"]
            if message.get("role") == "tool"
        ]
        self.assertEqual(
            [message["tool_call_id"] for message in tool_messages],
            ["working-call", "episodic-call", "semantic-call"],
        )
        self.assertTrue(all(message["content"] for message in tool_messages))

    def test_agent_memory_tool_keeps_users_isolated(self):
        """另一个 Agent 即使提出相同查询，也不能读取用户一保存的语义事实。"""
        self.agent.run("请把测试内容分别保存到三类记忆中")

        other_llm = ScriptedMemoryLLM()
        other_registry = ToolRegistry()
        other_registry.register_tool(
            MemoryTool(user_id="u2", memory_manager=self.build_manager("u2"))
        )
        other_agent = SimpleAgent(
            name="other-user-agent",
            llm=other_llm,
            tool_registry=other_registry,
        )

        answer = other_agent.run("我喜欢喝什么？")

        self.assertIn("没有找到相关记忆", answer)
        self.assertNotIn("绿茶", answer)

    def test_agent_uses_fallback_after_max_tool_iterations(self):
        """达到工具轮数上限后，最后一次工具结果仍应进入兜底 LLM 请求。"""
        endless_llm = EndlessToolCallingLLM()
        agent = SimpleAgent(
            name="bounded-memory-agent",
            llm=endless_llm,
            tool_registry=self.registry,
            max_tool_iterations=1,
        )

        answer = agent.run("查询记忆统计")

        self.assertEqual(answer, "已根据最后一次记忆统计生成兜底回答")
        self.assertEqual(len(endless_llm.tool_requests), 1)
        self.assertEqual(len(endless_llm.invoke_requests), 1)
        fallback_messages = endless_llm.invoke_requests[0]["messages"]
        self.assertEqual(fallback_messages[-1]["role"], "tool")
        self.assertIn("当前用户共有", fallback_messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
