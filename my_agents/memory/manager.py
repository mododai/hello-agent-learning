"""工作记忆、情景记忆和语义记忆的统一管理入口。"""

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from .base import BaseMemory, MemoryConfig, MemoryItem
from .types.episodic import EpisodicMemory
from .types.semantic import SemanticMemory
from .types.working import WorkingMemory


class MemoryManager:
    """为单个用户路由和聚合项目中现有的三类记忆。

    管理器绑定 ``user_id``，所有读取、更新、删除和统计都会自动携带该用户标识。
    Agent 与工具层不应直接选择数据库连接或自行拼接用户过滤条件。

    Args:
        user_id: 当前管理器所属的用户。
        config: 三类记忆共享的基础配置。
        enabled_types: 要启用的记忆类型，默认启用项目当前实现的全部三类记忆。
        memory_instances: 可选的已构造实例，用于共享连接和自动化测试。
        storage_backends: 按记忆类型提供的底层依赖，例如测试向量存储。
    """

    SUPPORTED_TYPES = ("working", "episodic", "semantic")

    def __init__(
        self,
        user_id: str,
        config: Optional[MemoryConfig] = None,
        enabled_types: Optional[Iterable[str]] = None,
        memory_instances: Optional[Dict[str, BaseMemory]] = None,
        storage_backends: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("MemoryManager 的 user_id 不能为空")
        self.user_id = user_id.strip()
        self.config = config or MemoryConfig()

        selected = tuple(
            self.SUPPORTED_TYPES if enabled_types is None else enabled_types
        )
        invalid = [name for name in selected if name not in self.SUPPORTED_TYPES]
        if invalid:
            raise ValueError(f"不支持的记忆类型: {', '.join(invalid)}")
        if not selected:
            raise ValueError("至少需要启用一种记忆类型")

        supplied = memory_instances or {}
        backends = storage_backends or {}
        factories = {
            "working": lambda: WorkingMemory(
                self.config,
                storage_backend=backends.get("working"),
            ),
            "episodic": lambda: EpisodicMemory(
                self.config,
                storage_backend=backends.get("episodic"),
            ),
            "semantic": lambda: SemanticMemory(
                self.config,
                storage_backend=backends.get("semantic"),
            ),
        }
        self.memories: Dict[str, BaseMemory] = {}
        for memory_type in dict.fromkeys(selected):
            instance = supplied.get(memory_type) or factories[memory_type]()
            if instance.memory_type != memory_type:
                raise ValueError(
                    f"{memory_type} 注入实例的实际类型是 {instance.memory_type}"
                )
            self.memories[memory_type] = instance

    @property
    def enabled_types(self) -> List[str]:
        """返回当前管理器已启用的记忆类型。"""
        return list(self.memories)

    def get_memory(self, memory_type: str) -> BaseMemory:
        """获取指定类型实例，不允许静默回退到其他记忆类型。"""
        if not isinstance(memory_type, str) or not memory_type.strip():
            raise ValueError("memory_type 不能为空")
        normalized = memory_type.strip().casefold()
        if normalized not in self.memories:
            enabled = ", ".join(self.memories)
            raise ValueError(f"记忆类型 {normalized!r} 未启用，可用类型: {enabled}")
        return self.memories[normalized]

    def add_memory(
        self,
        content: str,
        memory_type: str = "working",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> str:
        """构造统一 ``MemoryItem`` 并路由到指定记忆类型。"""
        if not isinstance(content, str) or not content.strip():
            raise ValueError("记忆内容不能为空")
        if isinstance(importance, bool):
            raise ValueError("importance 必须是数字")
        importance = float(importance)
        if not 0.0 <= importance <= 1.0:
            raise ValueError("importance 必须在 0.0 到 1.0 之间")

        target = self.get_memory(memory_type)
        item = MemoryItem(
            id=memory_id or str(uuid4()),
            content=content.strip(),
            memory_type=target.memory_type,
            user_id=self.user_id,
            timestamp=timestamp or datetime.now(),
            importance=importance,
            metadata=dict(metadata or {}),
        )
        stored_id = target.add(item)
        if not stored_id:
            raise RuntimeError(f"{target.memory_type} 记忆写入失败")
        return stored_id

    def retrieve_memories(
        self,
        query: str,
        limit: int = 5,
        memory_types: Optional[Iterable[str]] = None,
        min_importance: Optional[float] = None,
        **kwargs,
    ) -> List[MemoryItem]:
        """检索一种或多种记忆，并把各类型结果合并为统一列表。"""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("记忆检索 query 不能为空")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("limit 必须是整数")
        if limit <= 0:
            return []

        selected = (
            [memory_types]
            if isinstance(memory_types, str)
            else list(memory_types) if memory_types is not None else self.enabled_types
        )
        if not selected:
            return []

        candidates: List[MemoryItem] = []
        for memory_type in selected:
            memory = self.get_memory(memory_type)
            if memory_type == "semantic":
                results = memory.retrieve_natural_language(
                    query=query.strip(),
                    user_id=self.user_id,
                    limit=limit,
                    min_importance=min_importance,
                    **kwargs,
                )
            else:
                results = memory.retrieve(
                    query=query.strip(),
                    user_id=self.user_id,
                    limit=limit,
                    min_importance=min_importance,
                    **kwargs,
                )
            candidates.extend(results)

        # 不同记忆实现的相关性分数字段不同。优先读取实现提供的分数，没有分数时使用
        # 重要性作为保守回退，并以时间作为稳定的次级排序条件。
        def sort_key(item: MemoryItem):
            score = item.metadata.get(
                "retrieval_score",
                item.metadata.get("relevance_score", item.importance),
            )
            return float(score), item.timestamp.timestamp()

        candidates.sort(key=sort_key, reverse=True)
        return candidates[:limit]

    def update_memory(
        self,
        memory_id: str,
        memory_type: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """在用户权限边界内更新指定类型的记忆。"""
        memory = self.get_memory(memory_type)
        return memory.update(
            memory_id,
            content=content,
            importance=importance,
            metadata=metadata,
            user_id=self.user_id,
        )

    def remove_memory(
        self,
        memory_id: str,
        memory_type: Optional[str] = None,
    ) -> bool:
        """删除当前用户的记忆；未给类型时在所有已启用类型中安全定位。"""
        if memory_type is not None:
            return self.get_memory(memory_type).remove(
                memory_id,
                user_id=self.user_id,
            )

        for memory in self.memories.values():
            if memory.has_memory(memory_id, user_id=self.user_id):
                return memory.remove(memory_id, user_id=self.user_id)
        return False

    def retract_semantic_fact(
        self,
        memory_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """撤回当前用户的一条语义事实，并保留生命周期历史。"""
        semantic = self.get_memory("semantic")
        return semantic.retract_fact(memory_id, self.user_id, reason)

    def get_stats(self) -> Dict[str, Any]:
        """聚合当前用户在全部已启用记忆类型中的统计信息。"""
        by_type = {
            name: memory.get_stats(user_id=self.user_id)
            for name, memory in self.memories.items()
        }
        return {
            "user_id": self.user_id,
            "enabled_types": self.enabled_types,
            "total_count": sum(stats.get("count", 0) for stats in by_type.values()),
            "by_type": by_type,
        }

    def clear(self, memory_types: Optional[Iterable[str]] = None) -> None:
        """只清空当前用户在指定类型中的记忆。"""
        selected = (
            [memory_types]
            if isinstance(memory_types, str)
            else list(memory_types) if memory_types is not None else self.enabled_types
        )
        for memory_type in selected:
            self.get_memory(memory_type).clear(user_id=self.user_id)
