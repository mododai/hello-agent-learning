import tempfile
import unittest
import os
from datetime import datetime

from my_agents.memory.base import MemoryConfig, MemoryItem
from my_agents.memory.storage.document_store import SQLiteDocumentStore
from my_agents.memory.types.semantic import SemanticMemory
from my_agents.memory.types.semantic_fact import SemanticFact


class FakeEmbedder:
    dimension = 3

    def __init__(self):
        self.calls = []

    def encode(self, text):
        self.calls.append(text)
        # 简单且确定的向量，测试只关注调用契约而非模型质量。
        return [[float(len(text)), float(text.count("咖啡")), 1.0]]


class FakeVectorStore:
    def __init__(self):
        self.points = {}
        self.fail_next_add = False

    def add_vectors(self, vectors, metadata, ids=None):
        if self.fail_next_add:
            self.fail_next_add = False
            return False
        for vector, payload, memory_id in zip(vectors, metadata, ids):
            self.points[memory_id] = {"vector": vector, "metadata": dict(payload)}
        return True

    def search_similar(self, query_vector, limit=10, score_threshold=None, where=None):
        hits = []
        for memory_id, point in self.points.items():
            payload = point["metadata"]
            if where and any(payload.get(key) != value for key, value in where.items()):
                continue
            score = 0.9 if point["vector"][1] == query_vector[1] else 0.4
            if score_threshold is None or score >= score_threshold:
                hits.append({"id": memory_id, "score": score, "metadata": payload})
        return sorted(hits, key=lambda hit: hit["score"], reverse=True)[:limit]

    def delete_memories(self, memory_ids):
        for memory_id in memory_ids:
            self.points.pop(memory_id, None)
        return True

    def get_collection_info(self):
        return {"store_type": "fake", "points_count": len(self.points)}


class SemanticMemoryTest(unittest.TestCase):
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
        connection = getattr(self.memory.doc_store.local, "connection", None)
        if connection is not None:
            connection.close()
        abs_path = os.path.abspath(self.memory.doc_store.db_path)
        SQLiteDocumentStore._instances.pop(abs_path, None)
        SQLiteDocumentStore._initialized_dbs.discard(abs_path)
        self.temp_dir.cleanup()

    @staticmethod
    def item(memory_id, user_id, content, importance=0.5, metadata=None):
        return MemoryItem(
            id=memory_id,
            user_id=user_id,
            content=content,
            memory_type="semantic",
            timestamp=datetime(2026, 7, 11, 12, 0, 0),
            importance=importance,
            metadata=metadata or {},
        )

    def test_complete_lifecycle(self):
        self.assertEqual(
            self.memory.add(self.item("m1", "u1", "用户喜欢咖啡", 0.8, {"source": "chat"})),
            "m1",
        )
        self.assertTrue(self.memory.has_memory("m1", user_id="u1"))
        self.assertFalse(self.memory.has_memory("m1", user_id="u2"))

        results = self.memory.retrieve("咖啡偏好", user_id="u1")
        self.assertEqual([item.id for item in results], ["m1"])
        self.assertEqual(results[0].metadata["source"], "chat")
        self.assertIn("retrieval_score", results[0].metadata)

        calls_before_metadata_update = len(self.embedder.calls)
        self.assertTrue(self.memory.update("m1", importance=0.9, metadata={"verified": True}))
        self.assertEqual(len(self.embedder.calls), calls_before_metadata_update)

        self.assertTrue(self.memory.update("m1", content="用户现在喜欢茶"))
        self.assertEqual(self.vector_store.points["m1"]["metadata"]["content"], "用户现在喜欢茶")

        self.assertTrue(self.memory.remove("m1", user_id="u1"))
        self.assertFalse(self.memory.has_memory("m1"))
        self.assertNotIn("m1", self.vector_store.points)

    def test_user_and_type_isolation(self):
        self.memory.add(self.item("u1-memory", "u1", "用户喜欢咖啡"))
        self.memory.add(self.item("u2-memory", "u2", "用户也喜欢咖啡"))

        results = self.memory.retrieve("咖啡", user_id="u1")
        self.assertEqual([item.id for item in results], ["u1-memory"])

        # 即使伪造一个 Qdrant 候选，SQLite 类型校验仍会阻止串库。
        self.memory.doc_store.add_memory(
            memory_id="episodic-memory",
            user_id="u1",
            content="一次喝咖啡的经历",
            memory_type="episodic",
            timestamp=1,
            importance=1.0,
            properties={},
        )
        self.vector_store.points["episodic-memory"] = {
            "vector": [1.0, 1.0, 1.0],
            "metadata": {
                "memory_id": "episodic-memory",
                "user_id": "u1",
                "memory_type": "semantic",
            },
        }
        results = self.memory.retrieve("咖啡", user_id="u1")
        self.assertNotIn("episodic-memory", [item.id for item in results])

    def test_vector_failure_does_not_create_document(self):
        self.vector_store.fail_next_add = True
        with self.assertRaises(RuntimeError):
            self.memory.add(self.item("failed", "u1", "不能成为半条记忆"))
        self.assertFalse(self.memory.has_memory("failed"))

    def test_clear_only_removes_requested_user(self):
        self.memory.add(self.item("m1", "u1", "事实一"))
        self.memory.add(self.item("m2", "u2", "事实二"))
        self.memory.clear(user_id="u1")
        self.assertFalse(self.memory.has_memory("m1"))
        self.assertTrue(self.memory.has_memory("m2"))
        self.assertEqual(self.memory.get_stats()["count"], 1)

    def test_structured_fact_round_trip_through_sqlite(self):
        """SemanticFact 写入 SQLite 后应能无损恢复成模型对象。"""
        fact = SemanticFact(
            subject="user",
            predicate="drink_preference",
            object="无糖拿铁",
            knowledge_type="preference",
            confidence=0.9,
        )
        item = self.item(
            "fact-memory",
            "u1",
            "用户喜欢喝无糖拿铁",
            metadata={"fact": fact, "source_message_id": "message-1"},
        )

        self.memory.add(item)

        # SQLite 权威存储中必须是普通字典和字符串，证明数据可以被 JSON 序列化。
        stored = self.memory.doc_store.get_memory("fact-memory")
        self.assertIsInstance(stored["properties"]["fact"], dict)
        self.assertIsInstance(stored["properties"]["fact"]["valid_from"], str)

        # 业务检索结果应恢复为 SemanticFact，调用方不需要手动解析字典。
        results = self.memory.retrieve("用户喜欢喝什么", user_id="u1")
        restored = next(result for result in results if result.id == "fact-memory")
        restored_fact = self.memory.get_fact(restored)
        self.assertIsInstance(restored_fact, SemanticFact)
        self.assertEqual(restored_fact.object, "无糖拿铁")
        self.assertEqual(restored.metadata["source_message_id"], "message-1")

    def test_fact_dictionary_is_validated_and_normalized(self):
        """调用方传入字典时也必须执行 SemanticFact 的完整字段校验。"""
        item = self.item(
            "dict-fact",
            "u1",
            "用户正在学习 Python",
            metadata={
                "fact": {
                    "subject": " user ",
                    "predicate": " learning ",
                    "object": " Python ",
                    "knowledge_type": "skill",
                }
            },
        )

        self.memory.add(item)
        stored = self.memory.doc_store.get_memory("dict-fact")
        self.assertEqual(stored["properties"]["fact"]["subject"], "user")
        self.assertEqual(stored["properties"]["fact"]["predicate"], "learning")
        self.assertEqual(stored["properties"]["fact"]["object"], "Python")

    def test_invalid_fact_is_rejected_before_vector_write(self):
        """非法事实不能进入 Qdrant 或 SQLite，避免出现跨存储的半条数据。"""
        item = self.item(
            "invalid-fact",
            "u1",
            "一条非法事实",
            metadata={
                "fact": {
                    "subject": "user",
                    "predicate": "likes",
                    "object": "茶",
                    "confidence": 2.0,
                }
            },
        )

        with self.assertRaises(ValueError):
            self.memory.add(item)

        self.assertFalse(self.memory.has_memory("invalid-fact"))
        self.assertNotIn("invalid-fact", self.vector_store.points)

    def test_update_structured_fact(self):
        """更新 metadata 中的 fact 时应校验并保存新的结构化事实。"""
        self.memory.add(self.item("updated-fact", "u1", "用户喜欢拿铁"))
        replacement = SemanticFact(
            subject="user",
            predicate="drink_preference",
            object="无糖绿茶",
            knowledge_type="preference",
        )

        self.assertTrue(
            self.memory.update("updated-fact", metadata={"fact": replacement})
        )
        stored = self.memory.doc_store.get_memory("updated-fact")
        self.assertEqual(stored["properties"]["fact"]["object"], "无糖绿茶")

    def test_find_active_fact_by_key_and_optional_value(self):
        """active 事实既可以按事实键查找，也可以进一步限定 object。"""
        fact = SemanticFact(
            subject="user",
            predicate="drink_preference",
            object="无糖拿铁",
            knowledge_type="preference",
        )
        self.memory.add(
            self.item(
                "active-fact",
                "u1",
                "用户喜欢无糖拿铁",
                metadata={"fact": fact},
            )
        )

        by_key = self.memory.find_active_fact(
            user_id="u1",
            subject=" user ",
            predicate=" drink_preference ",
        )
        by_value = self.memory.find_active_fact(
            user_id="u1",
            subject="user",
            predicate="drink_preference",
            object_value=" 无糖拿铁 ",
        )
        missing_value = self.memory.find_active_fact(
            user_id="u1",
            subject="user",
            predicate="drink_preference",
            object_value="无糖绿茶",
        )

        self.assertEqual(by_key.id, "active-fact")
        self.assertEqual(by_value.id, "active-fact")
        self.assertIsInstance(self.memory.get_fact(by_key), SemanticFact)
        self.assertIsNone(missing_value)

    def test_same_active_fact_reuses_existing_memory(self):
        """完全相同的 active 事实不得重复调用 embedding 或写入两个存储。"""
        first = self.item(
            "first-fact",
            "u1",
            "用户喜欢喝无糖拿铁",
            metadata={
                "fact": SemanticFact(
                    subject="user",
                    predicate="drink_preference",
                    object="无糖拿铁",
                    knowledge_type="preference",
                )
            },
        )
        duplicate = self.item(
            "duplicate-fact",
            "u1",
            "用户很喜欢不加糖的拿铁",
            metadata={
                "fact": {
                    "subject": "user",
                    "predicate": "drink_preference",
                    "object": "无糖拿铁",
                    "knowledge_type": "preference",
                }
            },
        )

        self.assertEqual(self.memory.add(first), "first-fact")
        embedding_calls_after_first_add = len(self.embedder.calls)
        self.assertEqual(self.memory.add(duplicate), "first-fact")

        # 第二次添加应在生成向量前返回，所以 embedding 调用次数不变。
        self.assertEqual(len(self.embedder.calls), embedding_calls_after_first_add)
        self.assertFalse(self.memory.has_memory("duplicate-fact"))
        self.assertNotIn("duplicate-fact", self.vector_store.points)
        documents = self.memory.doc_store.search_memories(
            user_id="u1", memory_type="semantic", limit=10
        )
        self.assertEqual([doc["memory_id"] for doc in documents], ["first-fact"])

    def test_fact_deduplication_is_isolated_by_user(self):
        """相同事实属于不同用户时必须分别保存，不能跨用户去重。"""
        def preference(memory_id, user_id):
            return self.item(
                memory_id,
                user_id,
                "用户喜欢无糖拿铁",
                metadata={
                    "fact": SemanticFact(
                        subject="user",
                        predicate="drink_preference",
                        object="无糖拿铁",
                        knowledge_type="preference",
                    )
                },
            )

        self.assertEqual(self.memory.add(preference("u1-fact", "u1")), "u1-fact")
        self.assertEqual(self.memory.add(preference("u2-fact", "u2")), "u2-fact")
        self.assertTrue(self.memory.has_memory("u1-fact", user_id="u1"))
        self.assertTrue(self.memory.has_memory("u2-fact", user_id="u2"))

    def test_multiple_value_fact_key_keeps_different_values_active(self):
        """多值 drink_preference 的不同 object 应作为独立事实同时保留。"""
        latte = SemanticFact(
            subject="user",
            predicate="drink_preference",
            object="无糖拿铁",
            knowledge_type="preference",
        )
        tea = SemanticFact(
            subject="user",
            predicate="drink_preference",
            object="无糖绿茶",
            knowledge_type="preference",
        )

        self.assertEqual(
            self.memory.add(
                self.item("latte", "u1", "用户喜欢拿铁", metadata={"fact": latte})
            ),
            "latte",
        )
        self.assertEqual(
            self.memory.add(
                self.item("tea", "u1", "用户喜欢绿茶", metadata={"fact": tea})
            ),
            "tea",
        )
        self.assertTrue(self.memory.has_memory("latte"))
        self.assertTrue(self.memory.has_memory("tea"))


if __name__ == "__main__":
    unittest.main()
