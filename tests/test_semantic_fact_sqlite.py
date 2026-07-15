"""结构化语义事实的 SQLite 持久化验收测试。

本文件刻意检查真实表、索引和事务，而不仅检查 ``SemanticMemory`` 的返回值。
这样可以防止业务代码看似支持多值事实，但数据库仍然只能扫描 JSON 的回归。
"""

import os
import sqlite3
import tempfile
import unittest
from datetime import datetime

from my_agents.memory.base import MemoryConfig, MemoryItem
from my_agents.memory.storage.document_store import SQLiteDocumentStore
from my_agents.memory.types.semantic import SemanticMemory
from my_agents.memory.types.semantic_fact import SemanticFact
from tests.support import FakeEmbedder, FakeVectorStore


class SemanticFactSQLiteTest(unittest.TestCase):
    """验证规范化事实表、多值查询、唯一约束和事务回滚。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "memory.db")
        self.vector_store = FakeVectorStore()
        self.memory = SemanticMemory(
            MemoryConfig(storage_path=self.temp_dir.name),
            storage_backend={
                "embedder": FakeEmbedder(),
                "vector_store": self.vector_store,
            },
        )

    def tearDown(self):
        self.memory.doc_store.close()
        abs_path = os.path.abspath(self.db_path)
        SQLiteDocumentStore._instances.pop(abs_path, None)
        SQLiteDocumentStore._initialized_dbs.discard(abs_path)
        self.temp_dir.cleanup()

    @staticmethod
    def fact(predicate: str, value: str, knowledge_type: str = "fact") -> SemanticFact:
        """创建测试所需的当前事实。"""
        return SemanticFact(
            subject="user",
            predicate=predicate,
            object=value,
            knowledge_type=knowledge_type,
        )

    @classmethod
    def item(cls, memory_id: str, predicate: str, value: str) -> MemoryItem:
        """构造一条属于同一测试用户的语义记忆。"""
        return MemoryItem(
            id=memory_id,
            content=f"用户的 {predicate} 是 {value}",
            memory_type="semantic",
            user_id="u1",
            timestamp=datetime.now(),
            importance=0.8,
            metadata={"fact": cls.fact(predicate, value)},
        )

    def test_schema_contains_fact_table_and_partial_unique_indexes(self):
        """数据库必须直接创建事实表和索引，不再创建迁移版本表。"""
        conn = self.memory.doc_store._get_connection()
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(semantic_facts)").fetchall()
        }
        indexes = {
            row["name"]
            for row in conn.execute("PRAGMA index_list(semantic_facts)").fetchall()
        }

        self.assertTrue(
            {"memory_id", "user_id", "subject", "predicate", "object", "status"}
            <= columns
        )
        self.assertNotIn("schema_migrations", tables)
        self.assertIn("uq_semantic_facts_active_value", indexes)
        self.assertIn("uq_semantic_facts_active_single", indexes)

    def test_multiple_values_are_returned_and_second_value_is_deduplicated(self):
        """多值查询返回每个独立值，重复添加第二个值时不能产生第三条记录。"""
        self.assertEqual(
            self.memory.add(self.item("latte", "drink_preference", "拿铁")),
            "latte",
        )
        self.assertEqual(
            self.memory.add(self.item("tea", "drink_preference", "绿茶")),
            "tea",
        )

        # 这是旧实现会失败的关键场景：同谓词已有两个值，再次添加第二个值。
        duplicate_id = self.memory.add(
            self.item("tea-duplicate", "drink_preference", "绿茶")
        )
        matches = self.memory.find_active_facts(
            user_id="u1",
            subject="user",
            predicate="drink_preference",
        )

        self.assertEqual(duplicate_id, "tea")
        self.assertEqual({item.id for item in matches}, {"latte", "tea"})
        self.assertFalse(self.memory.has_memory("tea-duplicate"))
        self.assertEqual(
            self.memory.doc_store.get_database_stats()["semantic_facts_count"],
            2,
        )

    def test_database_rejects_duplicate_active_value_and_rolls_back_memory(self):
        """即使绕过 SemanticMemory，数据库唯一索引也必须阻止完全重复的 active 值。"""
        self.memory.add(self.item("tea", "drink_preference", "绿茶"))
        duplicate_fact = self.fact("drink_preference", "绿茶").model_dump(mode="json")

        with self.assertRaises(sqlite3.IntegrityError):
            self.memory.doc_store.add_memory(
                memory_id="tea-direct-duplicate",
                user_id="u1",
                content="直接绕过业务层写入重复绿茶",
                memory_type="semantic",
                timestamp=int(datetime.now().timestamp()),
                importance=0.8,
                properties={"fact": duplicate_fact},
                semantic_fact=duplicate_fact,
                fact_cardinality="multiple",
            )

        # memories 和 semantic_facts 属于同一事务，约束失败后不能留下半条记忆。
        self.assertIsNone(self.memory.doc_store.get_memory("tea-direct-duplicate"))

    def test_single_value_replacement_updates_both_tables_atomically(self):
        """单值替换后，JSON 镜像和规范化事实表必须具有一致状态。"""
        self.memory.add(self.item("hangzhou", "current_city", "杭州"))
        self.memory.add(self.item("shanghai", "current_city", "上海"))

        old_row = self.memory.doc_store.get_semantic_fact("hangzhou")
        new_row = self.memory.doc_store.get_semantic_fact("shanghai")
        old_json = self.memory.doc_store.get_memory("hangzhou")["properties"]["fact"]
        new_json = self.memory.doc_store.get_memory("shanghai")["properties"]["fact"]

        self.assertEqual(old_row["status"], "superseded")
        self.assertEqual(old_json["status"], "superseded")
        self.assertEqual(new_row["status"], "active")
        self.assertEqual(new_json["status"], "active")
        self.assertEqual(new_row["supersedes"], "hangzhou")

if __name__ == "__main__":
    unittest.main()
