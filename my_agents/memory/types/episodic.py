"""
情景记忆:
    -负责存储具体的事件和经历，它的设计重点在于保持事件的完整性和时间序列关系
"""
import logging
from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import os
logger = logging.getLogger(__name__)


from ..base import MemoryConfig, BaseMemory, MemoryItem
from ..storage.document_store import SQLiteDocumentStore
from ..embedding import get_embedding_model, get_dimension

class Episode(BaseModel):
    episode_id: str
    user_id: str
    session_id: str
    timestamp: datetime
    content: str
    context: Dict[str, Any]
    outcome: Optional[str] = None
    importance: float = 0.5

class EpisodicMemory(BaseMemory):
    """
    情景记忆实现
    """

    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)

        # 本地缓存（内存）
        self.episodes: List[Episode] = []
        self.sessions: Dict[str, List[str]] = {}  # session_id -> episode_ids

        # 模式识别缓存
        self.patterns_cache = {}
        self.last_pattern_analysis = None

        # 权威文档存储（SQLite）
        db_dir = self.config.storage_path if hasattr(self.config, 'storage_path') else "./memory_data"
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, "memory.db")
        self.doc_store = SQLiteDocumentStore(db_path=db_path)

        # 统一嵌入模型（多语言，默认384维）
        self.embedder = get_embedding_model()

        # 向量存储（Qdrant - 使用连接管理器避免重复连接）
        from ..storage.qdrant_store import QdrantConnectionManager
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.vector_store = QdrantConnectionManager.get_instance(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=os.getenv("QDRANT_COLLECTION", "agents_vectors"),
            vector_size=get_dimension(getattr(self.embedder, 'dimension', 384)),
            distance=os.getenv("QDRANT_DISTANCE", "cosine")
        )

    def add(self, memory_item: MemoryItem) -> str | None:
        """添加情景记忆"""
        if memory_item.memory_type != self.memory_type:
            return None

        # 从元数据中提取情景信息
        session_id = memory_item.metadata.get("session_id", "default_session")
        context = memory_item.metadata.get("context", {})
        outcome = memory_item.metadata.get("outcome")
        participants = memory_item.metadata.get("participants", [])
        tags = memory_item.metadata.get("tags", [])

        # 创建情景（内存缓存）
        episode = Episode(
            episode_id=memory_item.id,
            user_id=memory_item.user_id,
            session_id=session_id,
            timestamp=memory_item.timestamp,
            content=memory_item.content,
            context=context,
            outcome=outcome,
            importance=memory_item.importance
        )
        self.episodes.append(episode)
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(episode.episode_id)

        # 1. SQlite
        ts_int = int(memory_item.timestamp.timestamp())
        self.doc_store.add_memory(
            memory_id=memory_item.id,
            user_id=memory_item.user_id,
            content=memory_item.content,
            memory_type=self.memory_type,
            timestamp=ts_int,
            importance=memory_item.importance,
            properties={
                "session_id": session_id,
                "context": context,
                "outcome": outcome,
                "participants": participants,
                "tags": tags
            }
        )

        # 2. 向量Qdrant
        try:
            vectors = self.embedder.encode(memory_item.content)
            self.vector_store.add_vectors(
                vectors=vectors,
                metadata=[{
                    "memory_id": memory_item.id,
                    "user_id": memory_item.user_id,
                    "memory_type": self.memory_type,
                    "importance": memory_item.importance,
                    "session_id": session_id,
                    "content": memory_item.content
                }],
                ids=[memory_item.id]
            )
        except Exception:
            # 向量入库失败不影响权威存储
            pass

        return memory_item.id


    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索情景记忆：Qdrant 负责召回，SQLite 负责补全和兜底。"""
        user_id = kwargs.get("user_id")
        session_id = kwargs.get("session_id")
        time_range: Optional[Tuple[datetime, datetime]] = kwargs.get("time_range")
        importance_threshold: Optional[float] = kwargs.get("importance_threshold", kwargs.get("min_importance"))

        # 默认路径不扫描 SQLite 候选集：先从向量库召回少量相关 ID。
        try:
            query_vector = self.embedder.encode(query)
            # 当前 Ollama embedder 即使传入单个字符串也返回 [[...]]。
            if hasattr(query_vector, "tolist"):
                query_vector = query_vector.tolist()
            if query_vector and isinstance(query_vector[0], (list, tuple)):
                query_vector = query_vector[0]

            where = {"memory_type": self.memory_type}
            if user_id:
                where["user_id"] = user_id
            # 命中的记忆
            hits = self.vector_store.search_similar(
                query_vector=query_vector,
                limit=max(limit * 3, 20),
                where=where,
            )
        except Exception:
            # Qdrant 或嵌入服务不可用时，下面改由 SQLite 关键词兜底。
            hits = []

        now = datetime.now()
        results: List[Tuple[float, MemoryItem]] = []
        seen_ids = set()

        for hit in hits:
            metadata = hit.get("metadata", {})
            memory_id = metadata.get("memory_id")
            if not memory_id or memory_id in seen_ids:
                continue

            # 用命中的 ID 读取权威记录，避免把 Qdrant payload 当作完整数据源。
            doc = self.doc_store.get_memory(memory_id)
            if not doc or doc.get("memory_type") != self.memory_type:
                continue
            if user_id and doc.get("user_id") != user_id:
                continue

            timestamp = datetime.fromtimestamp(doc["timestamp"])
            # 不在时间范围的记忆
            if time_range and not (time_range[0] <= timestamp <= time_range[1]):
                continue
            if importance_threshold is not None and doc.get("importance", 0.0) < importance_threshold:
                continue

            properties = doc.get("properties", {})
            if session_id and properties.get("session_id") != session_id:
                continue

            # 相似度为主，近因性和重要度用于同类结果的重排。
            semantic_score = float(hit.get("score", 0.0))
            age_days = max(0.0, (now - timestamp).total_seconds() / 86400)
            recency_score = 1.0 / (1.0 + age_days)
            importance = float(doc.get("importance", 0.5))
            relevance_score = (
                0.70 * semantic_score
                + 0.15 * recency_score
                + 0.15 * importance
            )

            results.append(
                (
                    relevance_score,
                    MemoryItem(
                        id=doc["memory_id"],
                        content=doc["content"],
                        memory_type=self.memory_type,
                        user_id=doc["user_id"],
                        timestamp=timestamp,
                        importance=importance,
                        metadata={
                            **properties,
                            "relevance_score": relevance_score,
                            "vector_score": semantic_score,
                            "recency_score": recency_score,
                        },
                    )
                 )
            )
            seen_ids.add(memory_id)

        # TODO: 当嵌入模型或 Qdrant 不可用时，增加 SQLite FTS5/BM25 降级检索。
        # 当前学习重点是 Agent 的主检索链路，因此这里不实现回退策略。

        results.sort(key=lambda result: result[0], reverse=True)


        return [memory for _, memory in results[:limit]]



    def update(self,
               memory_id: str,
               content: str = None,
               importance: float = None,
               metadata: Dict[str, Any] = None
               ) -> bool:
        """更新情景记忆；SQLite 为权威源，内容变更时同步重建向量索引。"""
        if content is None and importance is None and metadata is None:
            return False

        # 先读取权威记录：进程重启后，内存缓存可能尚未回填。
        doc = self.doc_store.get_memory(memory_id)
        if not doc or doc.get("memory_type") != self.memory_type:
            return False

        # 元数据采用合并而不是覆盖，避免仅更新 outcome 时丢失 session_id 等字段。
        properties = dict(doc.get("properties", {}))
        if metadata is not None:
            for key, value in metadata.items():
                if key == "context" and isinstance(value, dict):
                    old_context = properties.get("context", {})
                    properties["context"] = {
                        **(old_context if isinstance(old_context, dict) else {}),
                        **value,
                    }
                else:
                    properties[key] = value

        updated = self.doc_store.update_memory(
            memory_id=memory_id,
            content=content,
            importance=importance,
            properties=properties if metadata is not None else None,
        )
        if not updated:
            return False

        # 同步当前进程的缓存；没有缓存条目并不表示更新失败。
        for episode in self.episodes:
            if episode.episode_id != memory_id:
                continue
            if content is not None:
                episode.content = content
            if importance is not None:
                episode.importance = importance
            if metadata is not None:
                if "context" in metadata and isinstance(metadata["context"], dict):
                    episode.context.update(metadata["context"])
                if "outcome" in metadata:
                    episode.outcome = metadata["outcome"]
                if "session_id" in metadata:
                    old_session_id = episode.session_id
                    new_session_id = metadata["session_id"]

                    if old_session_id != new_session_id:
                        # 情景记忆的会话更变
                        if old_session_id in self.sessions:
                            self.sessions[old_session_id].remove(memory_id)
                            if not self.sessions[old_session_id]:
                                del self.sessions[old_session_id]
                        self.sessions.setdefault(new_session_id, []).append(memory_id)
                        episode.session_id = new_session_id
            break

        # 向量只取决于 content；重要度和 metadata 在检索时从 SQLite 读取。
        if content is not None:
            try:
                vectors = self.embedder.encode(content)
                latest_doc = self.doc_store.get_memory(memory_id) or doc
                latest_properties = latest_doc.get("properties", {})
                self.vector_store.add_vectors(
                    vectors=vectors,
                    metadata=[{
                        "memory_id": memory_id,
                        "user_id": latest_doc["user_id"],
                        "memory_type": self.memory_type,
                        "importance": latest_doc.get("importance", 0.5),
                        "session_id": latest_properties.get("session_id", "default_session"),
                        "content": latest_doc["content"],
                    }],
                    ids=[memory_id],
                )
            except Exception as error:
                # SQLite 已更新；向量索引可在后续通过重建任务恢复。
                logger.warning("情景记忆向量索引更新失败: %s", error)

        return True

    def remove(self, memory_id: str) -> bool:

        # SQLite
        doc_deleted = self.doc_store.delete_memory(memory_id)
        # Qdrant
        try:
            self.vector_store.delete_memories([memory_id])
        except Exception:
            pass

        # 内存
        for i, episode in enumerate(self.episodes):
            if episode.episode_id == memory_id:
                removed_episode = self.episodes.pop(i)
                session_id = removed_episode.session_id
                if session_id in self.sessions:
                    self.sessions[session_id].remove(memory_id)
                    if not self.sessions[session_id]:
                        del self.sessions[session_id]
                break

        return doc_deleted


    def has_memory(self, memory_id: str) -> bool:
        """检查内存中是否存在情景记忆"""
        return any(episode.episode_id == memory_id for episode in self.episodes)

    def clear(self):
        """清空所有情景记忆（仅清理episodic，不影响其他类型）"""
        self.episodes.clear()
        self.sessions.clear()
        self.patterns_cache.clear()

        docs = self.doc_store.search_memories(memory_type=self.memory_type, limit=10000)
        ids = [d["memory_id"] for d in docs]
        for mid in ids:
            self.doc_store.delete_memory(mid)

        try:
            if ids:
                self.vector_store.delete_memories(ids)
        except Exception:
            pass


    def get_stats(self) -> Dict[str, Any]:
        """获取情景记忆统计信息（合并SQLite与Qdrant）"""

        db_stats = self.doc_store.get_database_stats()
        try:
            vs_stats = self.vector_store.get_collection_info()
        except Exception:
            vs_stats = {"store_type": "qdrant"}

        return {
            "memory_type": self.memory_type,
            "vector_store": vs_stats,
            "document_store": {
                k: v
                for k, v
                in db_stats.items()
                if k.endswith("_count") or k in ["store_type", "db_path"]
            }
        }