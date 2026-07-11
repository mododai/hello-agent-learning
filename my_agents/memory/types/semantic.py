"""语义记忆：保存去时间化、可复用的事实、概念和规律。"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..base import BaseMemory, MemoryConfig, MemoryItem
from ..embedding import get_dimension, get_embedding_model
from ..storage.document_store import SQLiteDocumentStore
from ..storage.qdrant_store import QdrantConnectionManager
from .semantic_fact import SemanticFact


logger = logging.getLogger(__name__)


class SemanticMemory(BaseMemory):
    """以 SQLite 为权威存储、Qdrant 为召回索引的语义记忆。"""

    # 结构化事实仍然存放在通用 metadata 中，避免当前阶段修改 SQLite 表结构。
    # 使用固定键名可以让后续的事实查重、替代和状态过滤拥有统一入口。
    FACT_METADATA_KEY = "fact"

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

    @classmethod
    def _normalize_metadata(cls, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """校验 metadata 中的结构化事实，并转换为可安全 JSON 序列化的数据。

        SQLiteDocumentStore 最终使用 ``json.dumps`` 保存 properties，因此不能直接
        写入 ``SemanticFact`` 对象或 ``datetime``。这里是写入边界：普通 metadata
        原样复制；若存在 ``metadata["fact"]``，则先通过 Pydantic 完整校验，再使用
        JSON 模式导出，使 datetime 自动转换为 ISO 8601 字符串。

        Args:
            metadata: 调用方提供的元数据，可为空，也可包含 SemanticFact 或字典。

        Returns:
            一个新的字典。该字典可以直接交给 SQLiteDocumentStore 序列化保存。

        Raises:
            TypeError: fact 既不是 SemanticFact，也不是字典。
            pydantic.ValidationError: fact 字典不满足 SemanticFact 的字段约束。
        """
        normalized = dict(metadata or {})
        raw_fact = normalized.get(cls.FACT_METADATA_KEY)

        # metadata 中没有结构化事实时保持原有语义记忆行为，不强迫所有文本都结构化。
        if raw_fact is None:
            return normalized

        if isinstance(raw_fact, SemanticFact):
            fact = raw_fact
        elif isinstance(raw_fact, dict):
            # 字典也必须经过模型校验，不能让非法 confidence/status 悄悄进入数据库。
            fact = SemanticFact.model_validate(raw_fact)
        else:
            raise TypeError(
                "metadata['fact'] 必须是 SemanticFact 或可构造 SemanticFact 的字典"
            )

        # mode="json" 会把 datetime 等 Python 对象转换成 JSON 兼容值。
        normalized[cls.FACT_METADATA_KEY] = fact.model_dump(mode="json")
        return normalized

    @classmethod
    def _restore_metadata(cls, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """把 SQLite 读取出的 fact 字典还原为 SemanticFact 对象。

        文档存储层只负责 JSON，不应该依赖具体业务模型。因此恢复动作放在
        SemanticMemory 的读取边界完成。这样上层代码可以直接访问
        ``memory.metadata["fact"].subject``，而不需要重复解析字典。
        """
        restored = dict(metadata or {})
        raw_fact = restored.get(cls.FACT_METADATA_KEY)
        if raw_fact is not None and not isinstance(raw_fact, SemanticFact):
            restored[cls.FACT_METADATA_KEY] = SemanticFact.model_validate(raw_fact)
        return restored

    @classmethod
    def get_fact(cls, memory_item: MemoryItem) -> Optional[SemanticFact]:
        """从 MemoryItem 中读取结构化事实，没有 fact 时返回 None。

        该方法同时兼容刚由调用方创建的 MemoryItem 和从 SQLite 恢复的 MemoryItem，
        因此允许 fact 当前是模型对象或普通字典。
        """
        raw_fact = memory_item.metadata.get(cls.FACT_METADATA_KEY)
        if raw_fact is None:
            return None
        if isinstance(raw_fact, SemanticFact):
            return raw_fact
        return SemanticFact.model_validate(raw_fact)

    @classmethod
    def _memory_item_from_document(cls, doc: Dict[str, Any]) -> MemoryItem:
        """把 SQLite 文档记录转换成统一的 MemoryItem。

        ``find_active_fact`` 和向量检索都会从 SQLite 读取权威记录。把转换逻辑集中
        在这里，可以确保 timestamp、importance 和 SemanticFact 的恢复规则一致，
        避免不同读取路径返回形态不同的数据。
        """
        return MemoryItem(
            id=doc["memory_id"],
            content=doc["content"],
            memory_type=doc["memory_type"],
            user_id=doc["user_id"],
            timestamp=datetime.fromtimestamp(doc["timestamp"]),
            importance=float(doc.get("importance", 0.5)),
            metadata=cls._restore_metadata(doc.get("properties")),
        )

    def find_active_fact(
        self,
        user_id: str,
        subject: str,
        predicate: str,
        object_value: Optional[str] = None,
    ) -> Optional[MemoryItem]:
        """查找用户当前生效的结构化事实。

        当前 SQLite 表还没有把 subject、predicate、status 拆成独立列，因此本阶段
        先读取该用户的语义记忆并检查 properties 中的 fact。这个实现清晰、可靠，
        适合学习阶段和小数据量；当事实规模增大后，再迁移为独立字段和数据库索引。

        Args:
            user_id: 事实所属用户。查询必须限定用户，防止跨用户去重和数据泄漏。
            subject: 事实主体，例如 ``user``。
            predicate: 主体属性或关系，例如 ``drink_preference``。
            object_value: 可选的事实值。传入时要求 object 也相同；不传时只匹配
                ``subject + predicate``，为下一步事实替代功能提供复用入口。

        Returns:
            第一条匹配的 active 事实；没有匹配时返回 None。
        """
        # 使用 SemanticFact 的字段校验规则完成去空格，避免 " user " 与 "user"
        # 因格式差异绕过去重。临时 object 仅用于构造合法模型，不参与无值查询。
        lookup_fact = SemanticFact(
            subject=subject,
            predicate=predicate,
            object=object_value if object_value is not None else "__lookup__",
        )

        documents = self.doc_store.search_memories(
            user_id=user_id,
            memory_type=self.memory_type,
            # 当前存储层尚不支持按 JSON 字段过滤，因此给出足够大的上限完成扫描。
            # 后续拆分事实表时应使用数据库索引替代该扫描。
            limit=1_000_000,
        )
        for doc in documents:
            raw_fact = (doc.get("properties") or {}).get(self.FACT_METADATA_KEY)
            if raw_fact is None:
                # 普通文本语义记忆不参与结构化事实去重。
                continue

            try:
                fact = SemanticFact.model_validate(raw_fact)
            except Exception:
                # 历史数据可能来自旧版本。单条损坏记录不应阻断其他事实的查询，
                # 但记录警告，便于后续数据修复或索引重建。
                logger.warning("跳过无法解析的语义事实: memory_id=%s", doc.get("memory_id"))
                continue

            if fact.status != "active" or fact.key != lookup_fact.key:
                continue
            if object_value is not None and fact.object != lookup_fact.object:
                continue
            return self._memory_item_from_document(doc)

        return None

    def add(self, memory_item: MemoryItem) -> Optional[str]:
        if memory_item.memory_type != self.memory_type:
            raise ValueError(
                f"SemanticMemory 不能保存 {memory_item.memory_type!r} 类型的记忆"
            )
        if not memory_item.user_id.strip():
            raise ValueError("user_id 不能为空")

        content = self._validate_content(memory_item.content)
        importance = self._validate_importance(memory_item.importance)
        # 在生成向量和写入任一存储之前校验事实，避免非法数据造成半条记忆。
        metadata = self._normalize_metadata(memory_item.metadata)

        # 只有结构化事实才执行确定性去重。普通文本没有稳定的 subject/predicate，
        # 此时强行依靠文本或向量相似度去重容易把否定句、近义但不同的事实误合并。
        raw_fact = metadata.get(self.FACT_METADATA_KEY)
        if raw_fact is not None:
            incoming_fact = SemanticFact.model_validate(raw_fact)
            if incoming_fact.status == "active":
                duplicate = self.find_active_fact(
                    user_id=memory_item.user_id,
                    subject=incoming_fact.subject,
                    predicate=incoming_fact.predicate,
                    object_value=incoming_fact.object,
                )
                if duplicate is not None:
                    logger.info(
                        "检测到重复语义事实，复用已有记忆: incoming_id=%s existing_id=%s",
                        memory_item.id,
                        duplicate.id,
                    )
                    return duplicate.id

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
                properties=metadata,
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
            # SQLite 中保存的是 JSON 字典；返回业务层前恢复成 SemanticFact 对象。
            metadata = self._restore_metadata(doc.get("properties"))
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
            # 更新时也走同一校验/序列化边界，保证 add 与 update 契约一致。
            new_properties = self._normalize_metadata(new_properties)

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
