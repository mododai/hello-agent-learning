"""语义记忆：保存去时间化、可复用的事实、概念和规律。"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..base import BaseMemory, MemoryConfig, MemoryItem
from ..embedding import get_dimension, get_embedding_model
from ..storage.document_store import SQLiteDocumentStore
from ..storage.qdrant_store import QdrantConnectionManager
from ..semantic_query_planner import SemanticQueryPlanner
from .semantic_change import FactChange
from .semantic_fact import SemanticFact
from .semantic_policy import PredicatePolicy, PredicatePolicyRegistry
from .semantic_query import FactQuery, FactRetrievalMode


logger = logging.getLogger(__name__)


class SemanticMemory(BaseMemory):
    """以 SQLite 为权威存储、Qdrant 为召回索引的语义记忆。"""

    # 事实同时保存在 semantic_facts 规范化表与 metadata 中。前者负责精确查询和
    # 唯一约束，后者作为 MemoryItem 的兼容镜像，避免上层读取接口发生破坏性变化。
    FACT_METADATA_KEY = "fact"
    FACT_RETRIEVAL_STATUSES = {
        "current": ("active",),
        "timeline": ("active", "superseded"),
        "audit": ("active", "superseded", "retracted"),
    }

    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)

        # storage_backend 可用于测试或替换基础设施；正常运行时无需传入。
        backends = storage_backend if isinstance(storage_backend, dict) else {}
        self.embedder = backends.get("embedder") or get_embedding_model()
        # 策略注册表允许调用方按业务扩展谓词，同时保持各实例之间相互隔离。
        self.predicate_policies = (
            backends.get("predicate_registry") or PredicatePolicyRegistry()
        )
        # 查询规划器只负责把自然语言转换成 FactQuery；执行仍由 SemanticMemory 控制。
        # 通过 storage_backend 注入可在未来替换成 LLM 规划器或测试替身。
        self.query_planner = backends.get("query_planner") or SemanticQueryPlanner()

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

    def find_active_facts(
        self,
        user_id: str,
        subject: str,
        predicate: str,
        object_value: Optional[str] = None,
    ) -> List[MemoryItem]:
        """查找用户当前生效的全部结构化事实。

        正常 SQLite 路径直接查询 ``semantic_facts`` 索引列。扫描回退仅用于兼容
        教学或测试中注入的简化 DocumentStore 实现。

        Args:
            user_id: 事实所属用户。查询必须限定用户，防止跨用户去重和数据泄漏。
            subject: 事实主体，例如 ``user``。
            predicate: 主体属性或关系，例如 ``drink_preference``。
            object_value: 可选的事实值。传入时要求 object 也相同；不传时只匹配
                ``subject + predicate``，为下一步事实替代功能提供复用入口。

        Returns:
            所有匹配的 active 事实；没有匹配时返回空列表。
        """
        # 使用 SemanticFact 的字段校验规则完成去空格，避免 " user " 与 "user"
        # 因格式差异绕过去重。临时 object 仅用于构造合法模型，不参与无值查询。
        lookup_fact = SemanticFact(
            subject=subject,
            predicate=predicate,
            object=object_value if object_value is not None else "__lookup__",
        )

        if hasattr(self.doc_store, "find_semantic_facts"):
            documents = self.doc_store.find_semantic_facts(
                user_id=user_id,
                subject=lookup_fact.subject,
                predicate=lookup_fact.predicate,
                object_value=lookup_fact.object if object_value is not None else None,
                status="active",
            )
            return [self._memory_item_from_document(doc) for doc in documents]

        documents = self.doc_store.search_memories(
            user_id=user_id,
            memory_type=self.memory_type,
            limit=1_000_000,
        )
        matches: List[MemoryItem] = []
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
            matches.append(self._memory_item_from_document(doc))

        return matches

    def find_active_fact(
        self,
        user_id: str,
        subject: str,
        predicate: str,
        object_value: Optional[str] = None,
    ) -> Optional[MemoryItem]:
        """返回一条 active 事实，仅供单值谓词或指定 object 的查询使用。"""
        matches = self.find_active_facts(
            user_id=user_id,
            subject=subject,
            predicate=predicate,
            object_value=object_value,
        )
        return matches[0] if matches else None

    def register_predicate_policy(
        self,
        predicate: str,
        policy: PredicatePolicy,
    ) -> None:
        """为当前实例注册谓词策略。

        策略决定后续写入事实的 cardinality，不会隐式扫描或修改已经保存的事实。
        因此业务初始化阶段应先注册策略，再开始写入该谓词的数据。
        """
        self.predicate_policies.register(predicate, policy)

    def get_predicate_policy(self, predicate: str) -> PredicatePolicy:
        """获取谓词策略；未知谓词使用默认的多值、显式替代策略。"""
        return self.predicate_policies.get(predicate)

    def _insert(
        self,
        memory_item: MemoryItem,
        normalized_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """执行单条语义记忆的底层双写，不包含去重或生命周期决策。

        业务层 ``add``、``supersede_fact`` 都需要写入新记录。将真实写入集中在这里，
        可以避免替代逻辑递归调用 add，也保证所有入口共享相同的校验和回滚行为。
        Qdrant 先写入用于确认向量可用；SQLite 写入失败时删除刚写入的向量。
        """
        if memory_item.memory_type != self.memory_type:
            raise ValueError(
                f"SemanticMemory 不能保存 {memory_item.memory_type!r} 类型的记忆"
            )
        if not memory_item.user_id.strip():
            raise ValueError("user_id 不能为空")

        content = self._validate_content(memory_item.content)
        importance = self._validate_importance(memory_item.importance)
        metadata = (
            normalized_metadata
            if normalized_metadata is not None
            else self._normalize_metadata(memory_item.metadata)
        )
        timestamp = int(memory_item.timestamp.timestamp())
        doc = {
            "memory_id": memory_item.id,
            "user_id": memory_item.user_id,
            "content": content,
            "timestamp": timestamp,
            "importance": importance,
        }

        vectors = self._encode(content)
        if not self.vector_store.add_vectors(
            vectors=vectors,
            metadata=[self._payload(doc)],
            ids=[memory_item.id],
        ):
            raise RuntimeError("语义记忆向量写入失败")

        try:
            raw_fact = metadata.get(self.FACT_METADATA_KEY)
            fact = SemanticFact.model_validate(raw_fact) if raw_fact is not None else None
            policy = self.get_predicate_policy(fact.predicate) if fact else None
            self.doc_store.add_memory(
                memory_id=memory_item.id,
                user_id=memory_item.user_id,
                content=content,
                memory_type=self.memory_type,
                timestamp=timestamp,
                importance=importance,
                properties=metadata,
                semantic_fact=fact.model_dump(mode="json") if fact else None,
                fact_cardinality=policy.cardinality if policy else None,
            )
        except Exception:
            self.vector_store.delete_memories([memory_item.id])
            raise
        return memory_item.id

    def add(self, memory_item: MemoryItem) -> Optional[str]:
        """添加语义记忆，并按谓词策略执行去重或单值事实替代。"""
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
        if raw_fact is None:
            return self._insert(memory_item, metadata)

        incoming_fact = SemanticFact.model_validate(raw_fact)
        if incoming_fact.status != "active":
            # 历史导入可能直接写入非 active 事实；它不参与当前事实决策。
            return self._insert(memory_item, metadata)

        # 先按完整四元组精确查重。多值谓词不能只检查任意一条同 key 事实，
        # 否则已有“拿铁、绿茶”时再次添加“绿茶”可能因先看到拿铁而重复写入。
        exact_matches = self.find_active_facts(
            user_id=memory_item.user_id,
            subject=incoming_fact.subject,
            predicate=incoming_fact.predicate,
            object_value=incoming_fact.object,
        )
        if exact_matches:
            existing = exact_matches[0]
            logger.info(
                "检测到重复语义事实，复用已有记忆: incoming_id=%s existing_id=%s",
                memory_item.id,
                existing.id,
            )
            return existing.id

        policy = self.get_predicate_policy(incoming_fact.predicate)
        if policy.cardinality == "single" and policy.replacement_mode == "automatic":
            current_matches = self.find_active_facts(
                user_id=memory_item.user_id,
                subject=incoming_fact.subject,
                predicate=incoming_fact.predicate,
            )
            # 如果已经存在单值事实，则替换
            if current_matches:
                return self.supersede_fact(current_matches[0].id, memory_item)

        # 多值或 explicit_only 谓词允许不同 object 同时保持 active。
        return self._insert(memory_item, metadata)

    def supersede_fact(
        self,
        old_memory_id: str,
        new_memory_item: MemoryItem,
        reason: Optional[str] = None,
    ) -> str:
        """使用新事实替代指定旧事实，并提供失败补偿。

        新事实先写入，成功后才把旧事实标记为 superseded。若旧事实状态更新失败，
        会删除新事实作为补偿，使旧事实继续保持 active，避免出现“当前事实消失”。
        """
        old_doc = self.doc_store.get_memory(old_memory_id)
        if not old_doc or old_doc.get("memory_type") != self.memory_type:
            raise ValueError(f"未找到待替代的语义记忆: {old_memory_id}")
        if old_doc.get("user_id") != new_memory_item.user_id:
            raise PermissionError("不能跨用户替代语义事实")

        old_memory = self._memory_item_from_document(old_doc)
        old_fact = self.get_fact(old_memory)
        new_metadata = self._normalize_metadata(new_memory_item.metadata)
        raw_new_fact = new_metadata.get(self.FACT_METADATA_KEY)
        if old_fact is None or raw_new_fact is None:
            raise ValueError("事实替代要求新旧记忆都包含 SemanticFact")

        new_fact = SemanticFact.model_validate(raw_new_fact)
        if old_fact.status != "active":
            raise ValueError("只有 active 事实可以被替代")
        if old_fact.key != new_fact.key:
            raise ValueError("替代事实的 subject 和 predicate 必须与旧事实一致")
        if old_fact.object == new_fact.object:
            return old_memory_id

        changed_at = new_memory_item.timestamp
        new_fact = new_fact.model_copy(
            update={
                "status": "active",
                "valid_from": changed_at,
                "valid_to": None,
                "supersedes": old_memory_id,
            }
        )
        new_metadata[self.FACT_METADATA_KEY] = new_fact.model_dump(mode="json")

        old_fact = old_fact.model_copy(
            update={"status": "superseded", "valid_to": changed_at}
        )
        old_metadata = dict(old_memory.metadata)
        old_metadata[self.FACT_METADATA_KEY] = old_fact
        if reason:
            # 原因属于旧事实为何失效的审计信息，保存在旧记录上最容易追溯。
            old_metadata["replacement_reason"] = reason

        # SQLiteDocumentStore 提供原子替代接口：旧事实失效与新事实插入在同一个事务
        # 中完成。Qdrant 无法参与 SQLite 事务，因此先写向量，数据库失败时删除向量。
        if hasattr(self.doc_store, "replace_memory_fact"):
            content = self._validate_content(new_memory_item.content)
            importance = self._validate_importance(new_memory_item.importance)
            timestamp = int(new_memory_item.timestamp.timestamp())
            new_doc = {
                "memory_id": new_memory_item.id,
                "user_id": new_memory_item.user_id,
                "content": content,
                "timestamp": timestamp,
                "importance": importance,
            }
            if not self.vector_store.add_vectors(
                vectors=self._encode(content),
                metadata=[self._payload(new_doc)],
                ids=[new_memory_item.id],
            ):
                raise RuntimeError("语义记忆向量写入失败")
            policy = self.get_predicate_policy(new_fact.predicate)
            try:
                return self.doc_store.replace_memory_fact(
                    old_memory_id=old_memory_id,
                    old_properties=self._normalize_metadata(old_metadata),
                    old_fact=old_fact.model_dump(mode="json"),
                    new_memory_id=new_memory_item.id,
                    user_id=new_memory_item.user_id,
                    content=content,
                    memory_type=self.memory_type,
                    timestamp=timestamp,
                    importance=importance,
                    new_properties=new_metadata,
                    new_fact=new_fact.model_dump(mode="json"),
                    fact_cardinality=policy.cardinality,
                )
            except Exception:
                self.vector_store.delete_memories([new_memory_item.id])
                raise

        # 简化的自定义 DocumentStore 没有事务接口时，保留原有补偿式实现。
        new_id = self._insert(new_memory_item, new_metadata)
        try:
            if not self.update(
                old_memory_id,
                metadata=old_metadata,
                user_id=new_memory_item.user_id,
            ):
                raise RuntimeError("旧事实状态更新失败")
        except Exception:
            self.remove(new_id, user_id=new_memory_item.user_id)
            raise
        return new_id

    def retract_fact(
        self,
        memory_id: str,
        user_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """显式撤回一条 active 事实，但保留其历史记录和向量。"""
        doc = self.doc_store.get_memory(memory_id)
        if not doc or doc.get("memory_type") != self.memory_type:
            return False
        if doc.get("user_id") != user_id:
            return False

        memory_item = self._memory_item_from_document(doc)
        fact = self.get_fact(memory_item)
        if fact is None or fact.status != "active":
            return False

        retracted_fact = fact.model_copy(
            update={"status": "retracted", "valid_to": datetime.now()}
        )
        metadata = dict(memory_item.metadata)
        metadata[self.FACT_METADATA_KEY] = retracted_fact
        if reason:
            metadata["retraction_reason"] = reason
        return self.update(memory_id, metadata=metadata, user_id=user_id)

    def apply_fact_change(self, change: FactChange, memory_item: MemoryItem) -> Optional[str]:
        """执行事实提取器产生的 assert/retract/replace 变更意图。"""
        # 以 FactChange 中的事实为准，避免调用方 memory_item metadata 与操作意图不一致。
        prepared_metadata = dict(memory_item.metadata)
        prepared_metadata[self.FACT_METADATA_KEY] = change.fact
        prepared_item = memory_item.model_copy(update={"metadata": prepared_metadata})

        if change.operation == "assert":
            return self.add(prepared_item)
        if change.operation == "retract":
            target = change.target_memory_id
            if target is None:
                existing = self.find_active_fact(
                    user_id=memory_item.user_id,
                    subject=change.fact.subject,
                    predicate=change.fact.predicate,
                    object_value=change.fact.object,
                )
                target = existing.id if existing else None
            if target is None:
                return None
            return target if self.retract_fact(target, memory_item.user_id, change.reason) else None

        target = change.target_memory_id
        if target is None:
            policy = self.get_predicate_policy(change.fact.predicate)
            if policy.cardinality == "multiple":
                # 多值谓词可能同时存在多个 active object。如果没有明确目标 ID，
                # 任意选择一条替代会造成不可预测的数据丢失，因此要求调用方消除歧义。
                raise ValueError("替换多值谓词事实时必须提供 target_memory_id")
            existing = self.find_active_fact(
                user_id=memory_item.user_id,
                subject=change.fact.subject,
                predicate=change.fact.predicate,
            )
            target = existing.id if existing else None
        return (
            self.supersede_fact(target, prepared_item, reason=change.reason)
            if target
            else self.add(prepared_item)
        )

    @classmethod
    def _validate_retrieval_mode(cls, mode: str) -> FactRetrievalMode:
        """校验事实检索模式，避免未知模式被静默当成普通向量检索。"""
        if mode not in cls.FACT_RETRIEVAL_STATUSES:
            allowed = ", ".join(cls.FACT_RETRIEVAL_STATUSES)
            raise ValueError(f"retrieval_mode 必须是以下值之一: {allowed}")
        return mode

    def retrieve_facts(
        self,
        user_id: str,
        subject: str,
        predicate: str,
        retrieval_mode: FactRetrievalMode = "current",
        object_value: Optional[str] = None,
        limit: int = 100,
    ) -> List[MemoryItem]:
        """按结构化事实键和生命周期意图进行确定性检索。

        ``current`` 只返回 active；``timeline`` 返回 active 与 superseded，用于回答
        “现在和以前”；``audit`` 额外返回 retracted，用于诊断或人工审计。该入口不调用
        Embedding/Qdrant，因此不会让其他高相似度谓词混入同一条事实时间线。
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("结构化事实检索必须提供 user_id")
        if limit <= 0:
            return []
        mode = self._validate_retrieval_mode(retrieval_mode)

        # 借助 SemanticFact 的字段校验统一去除首尾空格，并拒绝空主体、谓词或 object。
        lookup = SemanticFact(
            subject=subject,
            predicate=predicate,
            object=object_value if object_value is not None else "__lookup__",
        )
        statuses = list(self.FACT_RETRIEVAL_STATUSES[mode])

        if hasattr(self.doc_store, "find_semantic_facts"):
            documents = self.doc_store.find_semantic_facts(
                user_id=user_id.strip(),
                subject=lookup.subject,
                predicate=lookup.predicate,
                object_value=lookup.object if object_value is not None else None,
                status=None,
                statuses=statuses,
                limit=limit,
            )
            return [self._memory_item_from_document(doc) for doc in documents]

        # 兼容只实现通用文档查询的教学测试替身；正常 SQLite 路径不会执行全量扫描。
        documents = self.doc_store.search_memories(
            user_id=user_id.strip(),
            memory_type=self.memory_type,
            limit=1_000_000,
        )
        results: List[MemoryItem] = []
        for doc in documents:
            item = self._memory_item_from_document(doc)
            fact = self.get_fact(item)
            if fact is None or fact.key != lookup.key or fact.status not in statuses:
                continue
            if object_value is not None and fact.object != lookup.object:
                continue
            results.append(item)

        status_order = {status: index for index, status in enumerate(statuses)}
        results.sort(
            key=lambda item: (
                status_order.get(self.get_fact(item).status, len(statuses)),
                -item.timestamp.timestamp(),
            )
        )
        return results[:limit]

    def execute_fact_query(self, fact_query: FactQuery) -> List[MemoryItem]:
        """执行经过 Pydantic 校验的 FactQuery，作为规划器与存储层的稳定边界。"""
        query = FactQuery.model_validate(fact_query)
        return self.retrieve_facts(
            user_id=query.user_id,
            subject=query.subject,
            predicate=query.predicate,
            retrieval_mode=query.retrieval_mode,
            object_value=query.object_value,
            limit=query.limit,
        )

    def plan_query(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        subject: str = "user",
    ) -> Optional[FactQuery]:
        """把自然语言问题规划为 FactQuery；无法确定谓词时返回 None。"""
        normalized_query = self._validate_content(query)
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("自然语言记忆检索必须提供 user_id")
        return self.query_planner.plan(
            query=normalized_query,
            user_id=user_id.strip(),
            limit=limit,
            subject=subject,
        )

    def retrieve_natural_language(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        subject: str = "user",
        **vector_kwargs,
    ) -> List[MemoryItem]:
        """自动规划自然语言问题，无法结构化时安全回退到 Qdrant。

        该方法不改变原有 ``retrieve`` 的默认行为，便于调用方逐步接入查询规划器。
        规划成功时执行 SQLite 精确查询；失败时保留完整 query 走向量召回。
        """
        plan = self.plan_query(query, user_id, limit=limit, subject=subject)
        if plan is not None:
            return self.execute_fact_query(plan)
        return self.retrieve(
            query=query,
            limit=limit,
            user_id=user_id,
            **vector_kwargs,
        )

    def retrieve(
        self,
        query: Optional[str] = None,
        limit: int = 5,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> List[MemoryItem]:

        if limit <= 0:
            return []

        min_importance = kwargs.get(
            "min_importance", kwargs.get("importance_threshold")
        )
        if min_importance is not None:
            min_importance = self._validate_importance(min_importance)
        score_threshold = kwargs.get("score_threshold")
        include_inactive = bool(kwargs.get("include_inactive", False))
        retrieval_mode = kwargs.get(
            "retrieval_mode",
            "audit" if include_inactive else "current",
        )
        retrieval_mode = self._validate_retrieval_mode(retrieval_mode)

        # 调用方已经从问题中识别出谓词时，优先走确定性事实检索。query 仍保留在统一
        # retrieve 签名中，但不会再生成向量，也不会混入其他谓词的 active 事实。
        predicate = kwargs.get("predicate")
        if predicate is not None:
            if not user_id:
                raise ValueError("按 predicate 检索结构化事实时必须提供 user_id")
            return self.retrieve_facts(
                user_id=user_id,
                subject=kwargs.get("subject", "user"),
                predicate=predicate,
                retrieval_mode=retrieval_mode,
                object_value=kwargs.get("object_value"),
                limit=limit,
            )

        query = self._validate_content(query)

        allowed_statuses = self.FACT_RETRIEVAL_STATUSES[retrieval_mode]

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

            metadata = self._restore_metadata(doc.get("properties"))
            fact = metadata.get(self.FACT_METADATA_KEY)
            if isinstance(fact, SemanticFact) and fact.status not in allowed_statuses:
                # Qdrant 只召回候选；事实是否符合当前、时间线或审计意图由 SQLite
                # 中恢复出的生命周期状态决定。
                continue

            vector_score = float(hit.get("score", 0.0))
            status_weights = (
                {"active": 1.0, "superseded": 1.0, "retracted": 0.0}
                if retrieval_mode == "timeline"
                else {"active": 1.0, "superseded": 0.8, "retracted": 0.6}
            )
            status_weight = status_weights.get(
                fact.status if isinstance(fact, SemanticFact) else "active",
                1.0,
            )
            final_score = (0.85 * vector_score + 0.15 * importance) * status_weight
            metadata.update(
                {
                    "vector_score": vector_score,
                    "retrieval_score": final_score,
                    "status_weight": status_weight,
                }
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

        def result_sort_key(scored_item: Tuple[float, MemoryItem]) -> Tuple[int, float]:
            """历史检索时先按状态分组，再在组内按相关性排序。"""
            score, item = scored_item
            item_fact = self.get_fact(item)
            is_current = item_fact is None or item_fact.status == "active"
            return (1 if is_current else 0, score)

        # 即使某条历史事实的向量分更高，当前 active 事实也应整体排在历史之前；
        # 默认检索已过滤历史，使用同一排序规则不会改变原有行为。
        results.sort(key=result_sort_key, reverse=True)
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
            raw_fact = new_properties.get(self.FACT_METADATA_KEY)
            fact = SemanticFact.model_validate(raw_fact) if raw_fact is not None else None
            policy = self.get_predicate_policy(fact.predicate) if fact else None
            updated = self.doc_store.update_memory(
                memory_id=memory_id,
                content=new_content if content is not None else None,
                importance=new_importance if importance is not None else None,
                properties=new_properties if metadata is not None else None,
                semantic_fact=(
                    fact.model_dump(mode="json")
                    if fact is not None and metadata is not None
                    else None
                ),
                fact_cardinality=policy.cardinality if policy else None,
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

    def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """返回语义记忆统计；提供 ``user_id`` 时只统计该用户的数据。

        工具调用属于用户可见边界，不能返回整个数据库的用户汇总或共享 Qdrant 集合
        信息。因此用户级统计只提供当前用户的记忆数量和事实状态分布；不提供
        ``user_id`` 时保留原有的系统级诊断信息。
        """
        if user_id is not None:
            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError("统计语义记忆时 user_id 不能为空")
            documents = self.doc_store.search_memories(
                user_id=user_id.strip(),
                memory_type=self.memory_type,
                limit=1_000_000,
            )
            fact_statuses = {"active": 0, "superseded": 0, "retracted": 0}
            structured_fact_count = 0
            for document in documents:
                item = self._memory_item_from_document(document)
                fact = self.get_fact(item)
                if fact is None:
                    continue
                structured_fact_count += 1
                fact_statuses[fact.status] += 1
            return {
                "memory_type": self.memory_type,
                "scope": "user",
                "count": len(documents),
                "structured_fact_count": structured_fact_count,
                "fact_statuses": fact_statuses,
            }

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
