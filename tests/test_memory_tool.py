"""MemoryTool 的参数边界、用户隔离和语义事实生命周期测试。"""

import os
import tempfile
import unittest

from my_agents.memory.base import MemoryConfig
from my_agents.memory.manager import MemoryManager
from my_agents.memory.storage.document_store import SQLiteDocumentStore
from my_agents.memory.types.episodic import EpisodicMemory
from my_agents.memory.types.semantic import SemanticMemory
from my_agents.memory.types.working import WorkingMemory
from my_agents.tools.builtin.memory_tool import MemoryTool
from my_agents.tools.registry import ToolRegistry
from my_agents.tools.response import ToolStatus
from tests.support import FakeEmbedder, FakeVectorStore


class MemoryToolTest(unittest.TestCase):
    """使用真实 SQLite 与测试向量存储验证完整工具调用链。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = MemoryConfig(storage_path=self.temp_dir.name)
        self.embedder = FakeEmbedder()
        self.vector_store = FakeVectorStore()
        self.working_memory = WorkingMemory(self.config)
        self.episodic_memory = EpisodicMemory(
            self.config,
            storage_backend={
                "embedder": self.embedder,
                "vector_store": self.vector_store,
            },
        )
        self.semantic_memory = SemanticMemory(
            self.config,
            storage_backend={
                "embedder": self.embedder,
                "vector_store": self.vector_store,
            },
        )
        self.manager = MemoryManager(
            user_id="u1",
            config=self.config,
            memory_instances={
                "working": self.working_memory,
                "episodic": self.episodic_memory,
                "semantic": self.semantic_memory,
            },
        )
        self.tool = MemoryTool(user_id="u1", memory_manager=self.manager)

    def tearDown(self):
        self.semantic_memory.doc_store.close()
        abs_path = os.path.abspath(self.semantic_memory.doc_store.db_path)
        SQLiteDocumentStore._instances.pop(abs_path, None)
        SQLiteDocumentStore._initialized_dbs.discard(abs_path)
        self.temp_dir.cleanup()

    def remember(self, content: str, predicate: str, object_value: str):
        """通过正式工具接口保存一条结构化事实。"""
        return self.tool.run(
            {
                "action": "remember",
                "content": content,
                "memory_type": "semantic",
                "predicate": predicate,
                "object_value": object_value,
            }
        )

    def test_schema_only_requires_action(self):
        """Function Calling schema 只把分派动作声明为全局必填参数。"""
        schema = self.tool.to_openai_schema()["function"]["parameters"]

        self.assertEqual(schema["required"], ["action"])
        self.assertIn("predicate", schema["properties"])
        self.assertIn("retrieval_mode", schema["properties"])

        registry = ToolRegistry()
        registry.register_tool(self.tool)
        self.assertIs(registry.get_tool("memory"), self.tool)

    def test_remember_recall_and_deduplicate_structured_fact(self):
        """结构化事实可保存、按谓词召回，并复用重复事实的 ID。"""
        first = self.remember("用户喜欢喝拿铁", "drink_preference", "拿铁")
        duplicate = self.remember("用户喜欢喝拿铁咖啡", "drink_preference", "拿铁")

        recalled = self.tool.run(
            {"action": "recall", "predicate": "drink_preference"}
        )

        self.assertEqual(first.status, ToolStatus.SUCCESS)
        self.assertTrue(first.data["created"])
        self.assertEqual(duplicate.status, ToolStatus.SUCCESS)
        self.assertFalse(duplicate.data["created"])
        self.assertEqual(first.data["memory_id"], duplicate.data["memory_id"])
        self.assertEqual(recalled.data["count"], 1)
        self.assertEqual(recalled.data["memories"][0]["fact"]["object"], "拿铁")

    def test_single_value_replacement_and_natural_language_timeline(self):
        """单值城市事实自动替代，历史问题能返回当前值和旧值。"""
        hangzhou = self.remember("用户当前居住在杭州", "current_city", "杭州")
        shanghai = self.remember("用户已经搬到上海", "current_city", "上海")

        result = self.tool.run(
            {
                "action": "recall",
                "query": "用户现在和以前住在哪里？",
                "limit": 10,
            }
        )

        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(
            [item["id"] for item in result.data["memories"]],
            [shanghai.data["memory_id"], hangzhou.data["memory_id"]],
        )
        self.assertEqual(
            [item["fact"]["status"] for item in result.data["memories"]],
            ["active", "superseded"],
        )

    def test_retract_preserves_audit_record(self):
        """撤回操作不物理删除事实，审计模式仍可读取该记录。"""
        saved = self.remember("用户会游泳", "skill", "游泳")
        memory_id = saved.data["memory_id"]

        retracted = self.tool.run(
            {
                "action": "retract",
                "memory_id": memory_id,
                "reason": "用户否认该信息",
            }
        )
        current = self.tool.run({"action": "recall", "predicate": "skill"})
        audit = self.tool.run(
            {
                "action": "recall",
                "predicate": "skill",
                "retrieval_mode": "audit",
            }
        )

        self.assertEqual(retracted.status, ToolStatus.SUCCESS)
        self.assertEqual(current.data["count"], 0)
        self.assertEqual(audit.data["count"], 1)
        self.assertEqual(audit.data["memories"][0]["fact"]["status"], "retracted")

    def test_user_isolation_applies_to_recall_retract_delete_and_stats(self):
        """另一个用户的工具实例不能读取或修改当前用户的记忆。"""
        saved = self.remember("用户居住在上海", "current_city", "上海")
        memory_id = saved.data["memory_id"]
        other_manager = MemoryManager(
            user_id="u2",
            config=self.config,
            memory_instances={
                "working": self.working_memory,
                "episodic": self.episodic_memory,
                "semantic": self.semantic_memory,
            },
        )
        other_tool = MemoryTool(user_id="u2", memory_manager=other_manager)

        other_recall = other_tool.run(
            {"action": "recall", "predicate": "current_city"}
        )
        other_retract = other_tool.run(
            {"action": "retract", "memory_id": memory_id}
        )
        other_delete = other_tool.run({"action": "delete", "memory_id": memory_id})
        own_stats = self.tool.run({"action": "stats"})
        other_stats = other_tool.run({"action": "stats"})

        self.assertEqual(other_recall.data["count"], 0)
        self.assertEqual(other_retract.status, ToolStatus.ERROR)
        self.assertEqual(other_delete.status, ToolStatus.ERROR)
        self.assertEqual(own_stats.data["total_count"], 1)
        self.assertEqual(
            own_stats.data["by_type"]["semantic"]["fact_statuses"]["active"],
            1,
        )
        self.assertEqual(other_stats.data["total_count"], 0)
        self.assertTrue(self.semantic_memory.has_memory(memory_id, user_id="u1"))

    def test_delete_removes_current_users_memory(self):
        """delete 明确执行物理删除，并返回结构化结果。"""
        saved = self.remember("一条待删除的普通事实", "skill", "测试技能")
        memory_id = saved.data["memory_id"]

        deleted = self.tool.run({"action": "delete", "memory_id": memory_id})

        self.assertEqual(deleted.status, ToolStatus.SUCCESS)
        self.assertTrue(deleted.data["deleted"])
        self.assertFalse(self.semantic_memory.has_memory(memory_id, user_id="u1"))

    def test_complete_memory_system_routes_all_three_types(self):
        """工具通过管理器保存、检索和统计工作、情景、语义三类记忆。"""
        working = self.tool.run(
            {
                "action": "remember",
                "memory_type": "working",
                "content": "当前正在排查登录问题",
                "importance": 0.6,
            }
        )
        episodic = self.tool.run(
            {
                "action": "remember",
                "memory_type": "episodic",
                "content": "昨天完成了记忆系统测试",
                "session_id": "session-1",
                "outcome": "测试通过",
                "importance": 0.7,
            }
        )
        semantic = self.remember("用户喜欢绿茶", "drink_preference", "绿茶")

        working_result = self.tool.run(
            {"action": "recall", "memory_type": "working", "query": "登录问题"}
        )
        episodic_result = self.tool.run(
            {
                "action": "recall",
                "memory_type": "episodic",
                "query": "记忆系统测试",
                "session_id": "session-1",
            }
        )
        semantic_result = self.tool.run(
            {"action": "recall", "predicate": "drink_preference"}
        )
        stats = self.tool.run({"action": "stats"})

        self.assertEqual(working.status, ToolStatus.SUCCESS)
        self.assertEqual(episodic.status, ToolStatus.SUCCESS)
        self.assertEqual(semantic.status, ToolStatus.SUCCESS)
        self.assertEqual(working_result.data["memories"][0]["memory_type"], "working")
        self.assertEqual(episodic_result.data["memories"][0]["memory_type"], "episodic")
        self.assertEqual(semantic_result.data["memories"][0]["memory_type"], "semantic")
        self.assertEqual(stats.data["total_count"], 3)
        self.assertEqual(stats.data["by_type"]["working"]["count"], 1)
        self.assertEqual(stats.data["by_type"]["episodic"]["count"], 1)
        self.assertEqual(stats.data["by_type"]["semantic"]["count"], 1)

        aggregate = self.tool.run(
            {"action": "recall", "query": "记忆系统", "limit": 10}
        )
        self.assertEqual(aggregate.status, ToolStatus.SUCCESS)
        self.assertGreaterEqual(aggregate.data["count"], 1)

    def test_update_routes_to_selected_memory_type(self):
        """更新动作必须明确类型，并保持用户权限检查。"""
        saved = self.tool.run(
            {
                "action": "remember",
                "memory_type": "working",
                "content": "当前任务尚未开始",
            }
        )

        updated = self.tool.run(
            {
                "action": "update",
                "memory_type": "working",
                "memory_id": saved.data["memory_id"],
                "content": "当前任务正在进行",
                "importance": 0.9,
            }
        )
        recalled = self.tool.run(
            {"action": "recall", "memory_type": "working", "query": "当前任务"}
        )

        self.assertEqual(updated.status, ToolStatus.SUCCESS)
        self.assertEqual(recalled.data["memories"][0]["content"], "当前任务正在进行")
        self.assertEqual(recalled.data["memories"][0]["importance"], 0.9)

    def test_working_and_episodic_mutations_are_isolated_by_user(self):
        """统一管理器必须对短期和情景记忆执行与语义记忆相同的用户隔离。"""
        working = self.tool.run(
            {
                "action": "remember",
                "memory_type": "working",
                "content": "用户一的临时任务",
            }
        )
        episodic = self.tool.run(
            {
                "action": "remember",
                "memory_type": "episodic",
                "content": "用户一参加了项目复盘",
                "session_id": "private-session",
            }
        )
        other_manager = MemoryManager(
            user_id="u2",
            config=self.config,
            memory_instances={
                "working": self.working_memory,
                "episodic": self.episodic_memory,
                "semantic": self.semantic_memory,
            },
        )
        other_tool = MemoryTool(user_id="u2", memory_manager=other_manager)

        working_delete = other_tool.run(
            {
                "action": "delete",
                "memory_type": "working",
                "memory_id": working.data["memory_id"],
            }
        )
        episodic_update = other_tool.run(
            {
                "action": "update",
                "memory_type": "episodic",
                "memory_id": episodic.data["memory_id"],
                "content": "越权修改",
            }
        )
        episodic_recall = other_tool.run(
            {
                "action": "recall",
                "memory_type": "episodic",
                "query": "项目复盘",
            }
        )

        self.assertEqual(working_delete.status, ToolStatus.ERROR)
        self.assertEqual(episodic_update.status, ToolStatus.ERROR)
        self.assertEqual(episodic_recall.data["count"], 0)
        self.assertTrue(
            self.working_memory.has_memory(working.data["memory_id"], user_id="u1")
        )
        self.assertTrue(
            self.episodic_memory.has_memory(episodic.data["memory_id"], user_id="u1")
        )

    def test_invalid_parameters_return_errors_instead_of_raising(self):
        """无效模型参数应成为工具错误，不能中断 Agent 的工具调用循环。"""
        cases = [
            {},
            {"action": "unknown"},
            {
                "action": "remember",
                "content": "用户喜欢绿茶",
                "predicate": "drink_preference",
            },
            {"action": "remember", "content": "用户喜欢绿茶", "importance": 2},
            {
                "action": "remember",
                "memory_type": "working",
                "content": "错误使用谓词",
                "predicate": "skill",
                "object_value": "Python",
            },
            {"action": "recall"},
            {"action": "recall", "query": "测试", "limit": 0},
        ]

        for parameters in cases:
            with self.subTest(parameters=parameters):
                response = self.tool.run(parameters)
                self.assertEqual(response.status, ToolStatus.ERROR)
                self.assertEqual(response.error_info["code"], "INVALID_PARAM")


if __name__ == "__main__":
    unittest.main()
