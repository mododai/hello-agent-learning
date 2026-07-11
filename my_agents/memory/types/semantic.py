"""语义记忆：保存去时间化、可复用的事实、概念和规律。"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..base import BaseMemory, MemoryConfig, MemoryItem
from ..embedding import get_dimension, get_embedding_model
from ..storage.document_store import SQLiteDocumentStore
from ..storage.qdrant_store import QdrantConnectionManager


logger = logging.getLogger(__name__)


class SemanticMemory(BaseMemory):
    """以 SQLite 为权威存储、Qdrant 为召回索引的语义记忆。"""

    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)

        # storage_backend 可用于测试或替换基础设施；正常运行时无需传入。
        backends = storage_backend if isinstance(storage_backend, dict) else {}
        self.embedder = backends.get("embedder") or get_embedding_model()

        db_dir = self.config.storage_path
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, "memory.db")
        self.doc_store = backends.get("doc_store") or SQLiteDocumentStore(db_path=db_path)

        self.vector_store = backends.get("vector_store")
        if self.vector_store is None:
            self.vector_store = QdrantConnectionManager.get_instance(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                collection_name=os.getenv("QDRANT_COLLECTION", "agents_vectors"),
                vector_size=get_dimension(getattr(self.embedder, "dimension", 384)),
                distance=os.getenv("QDRANT_DISTANCE", "cosine"),
            )

    @staticmethod
    def _single_vector(value: Any) -> List[float]:
        """统一不同 embedding 实现对单条文本的返回格式。"""
        if hasattr(value, "tolist"):
            value = value.tolist()
        if value and isinstance(value[0], (list, tuple)):
            value = value[0]
        if not isinstance(value, list) or not value:
            raise ValueError("embedding 模型返回了空向量或不支持的向量格式")
        return value

    def _encode(self, content: str) -> List[List[float]]:
        return [self._single_vector(self.embedder.encode(content))]

    def _payload(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "memory_id": doc["memory_id"],
            "user_id": doc["user_id"],
            "content": doc["content"],
            "memory_type": self.memory_type,
            "timestamp": int(doc["timestamp"]),
            "importance": float(doc.get("importance", 0.5)),
        }

    @staticmethod
    def _validate_content(content: str) -> str:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("语义记忆 content 不能为空")
        return content.strip()

    @staticmethod
    def _validate_importance(importance: float) -> float:
        value = float(importance)
        if not 0.0 <= value <= 1.0:
            raise ValueError("importance 必须在 0.0 到 1.0 之间")
        return value

    def add(self, memory_item: MemoryItem) -> Optional[str]:
        if memory_item.memory_type != self.memory_type:
            raise ValueError(
                f"SemanticMemory 不能保存 {memory_item.memory_type!r} 类型的记忆"
            )
        if not memory_item.user_id.strip():
            raise ValueError("user_id 不能为空")

        content = self._validate_content(memory_item.content)
        importance = self._validate_importance(memory_item.importance)
        timestamp = int(memory_item.timestamp.timestamp())
        doc = {
            "memory_id": memory_item.id,
            "user_id": memory_item.user_id,
            "content": content,
            "timestamp": timestamp,
            "importance": importance,
        }

        # 先确保向量可生成、可写入，避免产生无法召回的 SQLite 记录。
        vectors = self._encode(content)
        if not self.vector_store.add_vectors(
            vectors=vectors,
            metadata=[self._payload(doc)],
            ids=[memory_item.id],
        ):
            raise RuntimeError("语义记忆向量写入失败")

        try:
            self.doc_store.add_memory(
                memory_id=memory_item.id,
                user_id=memory_item.user_id,
                content=content,
                memory_type=self.memory_type,
                timestamp=timestamp,
                importance=importance,
                properties=dict(memory_item.metadata),
            )
        except Exception:
            # SQLite 写入失败时尽力回滚刚写入的向量。
            self.vector_store.delete_memories([memory_item.id])
            raise

        return memory_item.id

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> List[MemoryItem]:
        query = self._validate_content(query)
        if limit <= 0:
            return []

        min_importance = kwargs.get(
            "min_importance", kwargs.get("importance_threshold")
        )
        if min_importance is not None:
            min_importance = self._validate_importance(min_importance)
        score_threshold = kwargs.get("score_threshold")

        where = {"memory_type": self.memory_type}
        if user_id:
            where["user_id"] = user_id

        query_vector = self._single_vector(self.embedder.encode(query))
        hits = self.vector_store.search_similar(
            query_vector=query_vector,
            limit=max(limit * 3, 20),
            score_threshold=score_threshold,
            where=where,
        )

        results: List[Tuple[float, MemoryItem]] = []
        seen_ids = set()
        for hit in hits:
            payload = hit.get("metadata") or {}
            memory_id = payload.get("memory_id")
            if not memory_id or memory_id in seen_ids:
                continue

            # Qdrant 只负责候选召回；SQLite 再次执行类型及用户权限校验。
            doc = self.doc_store.get_memory(memory_id)
            if not doc or doc.get("memory_type") != self.memory_type:
                continue
            if user_id and doc.get("user_id") != user_id:
                continue

            importance = float(doc.get("importance", 0.5))
            if min_importance is not None and importance < min_importance:
                continue

            vector_score = float(hit.get("score", 0.0))
            final_score = 0.85 * vector_score + 0.15 * importance
            metadata = dict(doc.get("properties") or {})
            metadata.update(
                {"vector_score": vector_score, "retrieval_score": final_score}
            )
            results.append(
                (
                    final_score,
                    MemoryItem(
                        id=doc["memory_id"],
                        content=doc["content"],
                        memory_type=self.memory_type,
                        user_id=doc["user_id"],
                        timestamp=datetime.fromtimestamp(doc["timestamp"]),
                        importance=importance,
                        metadata=metadata,
                    ),
                )
            )
            seen_ids.add(memory_id)

        results.sort(key=lambda item: item[0], reverse=True)
        return [memory for _, memory in results[:limit]]

    def update(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        metadata: Dict[str, Any] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        doc = self.doc_store.get_memory(memory_id)
        if not doc or doc.get("memory_type") != self.memory_type:
            return False
        if user_id and doc.get("user_id") != user_id:
            return False
        if content is None and importance is None and metadata is None:
            return False

        new_content = self._validate_content(content) if content is not None else doc["content"]
        new_importance = (
            self._validate_importance(importance)
            if importance is not None
            else float(doc.get("importance", 0.5))
        )
        new_properties = dict(doc.get("properties") or {})
        if metadata is not None:
            new_properties.update(metadata)

        # 内容变化时先更新向量；若 SQLite 更新失败则尽力恢复旧向量。
        if content is not None:
            new_doc = dict(doc)
            new_doc.update(content=new_content, importance=new_importance)
            if not self.vector_store.add_vectors(
                vectors=self._encode(new_content),
                metadata=[self._payload(new_doc)],
                ids=[memory_id],
            ):
                return False

        try:
            updated = self.doc_store.update_memory(
                memory_id=memory_id,
                content=new_content if content is not None else None,
                importance=new_importance if importance is not None else None,
                properties=new_properties if metadata is not None else None,
            )
        except Exception:
            if content is not None:
                try:
                    self.vector_store.add_vectors(
                        vectors=self._encode(doc["content"]),
                        metadata=[self._payload(doc)],
                        ids=[memory_id],
                    )
                except Exception:
                    logger.exception("恢复记忆 %s 的旧向量失败", memory_id)
            raise
        return updated

    def remove(self, memory_id: str, user_id: Optional[str] = None) -> bool:
        doc = self.doc_store.get_memory(memory_id)
        if not doc or doc.get("memory_type") != self.memory_type:
            return False
        if user_id and doc.get("user_id") != user_id:
            return False

        deleted = self.doc_store.delete_memory(memory_id)
        if deleted and not self.vector_store.delete_memories([memory_id]):
            # 孤立向量不会被 retrieve 返回，因为 SQLite 权威记录已不存在。
            logger.warning("记忆 %s 已从 SQLite 删除，但向量清理失败", memory_id)
        return deleted

    def has_memory(self, memory_id: str, user_id: Optional[str] = None) -> bool:
        doc = self.doc_store.get_memory(memory_id)
        return bool(
            doc
            and doc.get("memory_type") == self.memory_type
            and (user_id is None or doc.get("user_id") == user_id)
        )

    def clear(self, user_id: Optional[str] = None) -> None:
        docs = self.doc_store.search_memories(
            user_id=user_id,
            memory_type=self.memory_type,
            limit=1_000_000,
        )
        ids = [doc["memory_id"] for doc in docs]
        for memory_id in ids:
            self.doc_store.delete_memory(memory_id)
        if ids and not self.vector_store.delete_memories(ids):
            logger.warning("SQLite 已清空，但部分语义记忆向量可能未清理")

    def get_stats(self) -> Dict[str, Any]:
        document_stats = self.doc_store.get_database_stats()
        try:
            vector_stats = self.vector_store.get_collection_info()
        except Exception:
            vector_stats = {"store_type": "qdrant", "available": False}
        return {
            "memory_type": self.memory_type,
            "count": document_stats.get("memory_types", {}).get(self.memory_type, 0),
            "document_store": document_stats,
            "vector_store": vector_stats,
        }
