import tempfile
import unittest
import os
from datetime import datetime

from my_agents.memory.base import MemoryConfig, MemoryItem
from my_agents.memory.storage.document_store import SQLiteDocumentStore
from my_agents.memory.types.semantic import SemanticMemory


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


if __name__ == "__main__":
    unittest.main()
