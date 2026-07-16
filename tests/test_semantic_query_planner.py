"""FactQuery 与规则式 SemanticQueryPlanner 的验收测试。"""

import os
import tempfile
import unittest
from datetime import datetime

from pydantic import ValidationError

from my_agents.memory.base import MemoryConfig, MemoryItem
from my_agents.memory.semantic_query_planner import SemanticQueryPlanner
from my_agents.memory.storage.document_store import SQLiteDocumentStore
from my_agents.memory.types.semantic import SemanticMemory
from my_agents.memory.types.semantic_fact import SemanticFact
from my_agents.memory.types.semantic_query import FactQuery
from tests.support import FakeEmbedder, FakeVectorStore


class FactQueryTest(unittest.TestCase):
    """验证结构化查询模型的边界和规范化行为。"""

    def test_fact_query_normalizes_text_and_rejects_invalid_values(self):
        query = FactQuery(
            user_id=" u1 ",
            subject=" user ",
            predicate=" current_city ",
            retrieval_mode="timeline",
            source_query=" 以前住在哪里 ",
        )

        self.assertEqual(query.user_id, "u1")
        self.assertEqual(query.predicate, "current_city")
        self.assertEqual(query.source_query, "以前住在哪里")

        with self.assertRaises(ValidationError):
            FactQuery(user_id=" ", predicate="current_city")
        with self.assertRaises(ValidationError):
            FactQuery(user_id="u1", predicate="current_city", retrieval_mode="wrong")
        with self.assertRaises(ValidationError):
            FactQuery(user_id="u1", predicate="current_city", limit=0)


class SemanticQueryPlannerTest(unittest.TestCase):
    """验证谓词、生命周期意图和安全回退规划。"""

    def setUp(self):
        self.planner = SemanticQueryPlanner()

    def test_plans_current_timeline_and_audit_queries(self):
        current = self.planner.plan("用户住在哪里？", "u1")
        timeline = self.planner.plan("用户现在和以前住在哪里？", "u1")
        audit = self.planner.plan("审计用户撤回的技能记录", "u1")

        self.assertEqual((current.predicate, current.retrieval_mode), ("current_city", "current"))
        self.assertEqual(
            (timeline.predicate, timeline.retrieval_mode),
            ("current_city", "timeline"),
        )
        self.assertEqual((audit.predicate, audit.retrieval_mode), ("skill", "audit"))

    def test_unknown_or_ambiguous_predicate_returns_none(self):
        self.assertIsNone(self.planner.plan("请回忆一件相关的事情", "u1"))

        self.planner.register_aliases("custom_one", "共同别名")
        self.planner.register_aliases("custom_two", "共同别名")
        self.assertIsNone(self.planner.plan("查询共同别名", "u1"))

    def test_custom_alias_can_be_registered(self):
        self.planner.register_aliases("favorite_color", ("喜欢什么颜色", "颜色偏好"))
        plan = self.planner.plan("用户喜欢什么颜色？", "u1")

        self.assertEqual(plan.predicate, "favorite_color")
        self.assertEqual(plan.retrieval_mode, "current")


class PlannedSemanticRetrievalTest(unittest.TestCase):
    """验证自然语言规划成功走 SQLite，失败时回退 Qdrant。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.embedder = FakeEmbedder()
        self.vector_store = FakeVectorStore()
        self.memory = SemanticMemory(
            MemoryConfig(storage_path=self.temp_dir.name),
            storage_backend={
                "embedder": self.embedder,
                "vector_store": self.vector_store,
            },
        )

    def tearDown(self):
        self.memory.doc_store.close()
        abs_path = os.path.abspath(self.memory.doc_store.db_path)
        SQLiteDocumentStore._instances.pop(abs_path, None)
        SQLiteDocumentStore._initialized_dbs.discard(abs_path)
        self.temp_dir.cleanup()

    @staticmethod
    def item(memory_id: str, content: str, predicate: str = None, value: str = None):
        metadata = {}
        if predicate is not None:
            metadata["fact"] = SemanticFact(
                subject="user",
                predicate=predicate,
                object=value,
            )
        return MemoryItem(
            id=memory_id,
            content=content,
            memory_type="semantic",
            user_id="u1",
            timestamp=datetime.now(),
            importance=0.8,
            metadata=metadata,
        )

    def test_natural_language_timeline_uses_sqlite_without_new_embedding(self):
        self.memory.add(self.item("latte", "用户喜欢拿铁", "drink_preference", "拿铁"))
        self.memory.add(self.item("hangzhou", "用户住在杭州", "current_city", "杭州"))
        self.memory.add(self.item("shanghai", "用户搬到上海", "current_city", "上海"))
        calls_before = len(self.embedder.calls)

        result = self.memory.retrieve_natural_language(
            "用户现在和以前住在哪里？",
            user_id="u1",
            limit=10,
        )

        self.assertEqual([item.id for item in result], ["shanghai", "hangzhou"])
        self.assertEqual(len(self.embedder.calls), calls_before)

    def test_unknown_query_falls_back_to_vector_retrieval(self):
        self.memory.add(self.item("note", "这是一条无法结构化的项目笔记"))
        calls_before = len(self.embedder.calls)

        result = self.memory.retrieve_natural_language(
            "请回忆一件相关的事情",
            user_id="u1",
        )

        self.assertEqual([item.id for item in result], ["note"])
        self.assertEqual(len(self.embedder.calls), calls_before + 1)

    def test_explicit_predicate_query_allows_empty_query(self):
        self.memory.add(self.item("shanghai", "用户住在上海", "current_city", "上海"))

        result = self.memory.retrieve(
            None,
            user_id="u1",
            predicate="current_city",
            retrieval_mode="current",
        )
        self.assertEqual([item.id for item in result], ["shanghai"])

        with self.assertRaises(ValueError):
            self.memory.retrieve(None, user_id="u1")


if __name__ == "__main__":
    unittest.main()
