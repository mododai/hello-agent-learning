"""语义记忆测试使用的可控替身。

这些替身不会访问 Ollama 或 Qdrant Cloud，使生命周期测试可以快速、稳定地重复运行。
真实外部服务的连通性仍由根目录中的集成脚本单独验证。
"""


class FakeEmbedder:
    """返回固定维度向量，并记录调用次数。"""

    dimension = 3

    def __init__(self):
        self.calls = []

    def encode(self, text):
        self.calls.append(text)
        return [[float(len(text)), float(text.count("喜欢")), 1.0]]


class FakeVectorStore:
    """在内存中模拟 Qdrant 的 upsert、过滤、搜索和删除行为。"""

    def __init__(self):
        self.points = {}
        self.fail_next_add = False

    def add_vectors(self, vectors, metadata, ids=None):
        if self.fail_next_add:
            self.fail_next_add = False
            return False
        for vector, payload, memory_id in zip(vectors, metadata, ids):
            self.points[memory_id] = {
                "vector": list(vector),
                "metadata": dict(payload),
            }
        return True

    def search_similar(self, query_vector, limit=10, score_threshold=None, where=None):
        results = []
        for memory_id, point in self.points.items():
            payload = point["metadata"]
            if where and any(payload.get(key) != value for key, value in where.items()):
                continue
            score = 0.9
            if score_threshold is None or score >= score_threshold:
                results.append(
                    {"id": memory_id, "score": score, "metadata": payload}
                )
        return results[:limit]

    def delete_memories(self, memory_ids):
        for memory_id in memory_ids:
            self.points.pop(memory_id, None)
        return True

    def get_collection_info(self):
        return {"store_type": "fake", "points_count": len(self.points)}

