"""面向 Agent Function Calling 的统一记忆工具。"""

import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ...memory import MemoryConfig, MemoryManager, MemoryItem
from ...memory.types.semantic import SemanticMemory
from ...memory.types.semantic_fact import SemanticFact
from ..base import Tool, ToolParameter
from ..errors import ToolErrorCode
from ..response import ToolResponse


logger = logging.getLogger(__name__)


class MemoryTool(Tool):
    """通过一个用户级 ``MemoryManager`` 管理完整记忆系统。

    当前项目已经实现工作记忆、情景记忆和语义记忆，工具默认同时启用三者。调用方
    可以用 ``memory_type`` 明确路由；检索时不提供类型则聚合全部已启用类型。

    ``user_id`` 在构造时绑定，模型不能通过工具参数访问其他用户的数据。
    """

    ACTIONS = {"remember", "recall", "update", "retract", "delete", "stats"}

    def __init__(
        self,
        user_id: str = "default",
        memory_config: Optional[MemoryConfig] = None,
        memory_types: Optional[List[str]] = None,
        memory_manager: Optional[MemoryManager] = None,
    ):
        super().__init__(
            name="memory",
            description=(
                "管理当前用户的完整记忆系统。支持 working 工作记忆、episodic 情景记忆"
                "和 semantic 语义记忆，以及 remember、recall、update、retract、"
                "delete、stats 操作。"
            ),
        )
        normalized_user_id = self._required_text(user_id, "user_id")
        if memory_manager is not None and memory_manager.user_id != normalized_user_id:
            raise ValueError("MemoryTool 与 MemoryManager 的 user_id 必须一致")
        self.manager = memory_manager or MemoryManager(
            user_id=normalized_user_id,
            config=memory_config,
            enabled_types=memory_types,
        )
        self.user_id = self.manager.user_id

    @staticmethod
    def _required_text(value: Any, field_name: str) -> str:
        """校验必填文本，并统一去除首尾空白。"""
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} 必须是非空字符串")
        return value.strip()

    @staticmethod
    def _optional_text(value: Any, field_name: str) -> Optional[str]:
        """校验可选文本；空字符串按未提供处理。"""
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{field_name} 必须是字符串")
        return value.strip() or None

    @staticmethod
    def _limit(value: Any) -> int:
        """把工具参数转换为受控的结果数量。"""
        if isinstance(value, bool):
            raise ValueError("limit 必须是整数")
        try:
            limit = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("limit 必须是整数") from exc
        if not 1 <= limit <= 100:
            raise ValueError("limit 必须在 1 到 100 之间")
        return limit

    @staticmethod
    def _importance(value: Any) -> float:
        """校验记忆重要性分数。"""
        if isinstance(value, bool):
            raise ValueError("importance 必须是数字")
        try:
            importance = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("importance 必须是数字") from exc
        if not 0.0 <= importance <= 1.0:
            raise ValueError("importance 必须在 0.0 到 1.0 之间")
        return importance

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        """验证动作并将调用分派给统一记忆管理器。"""
        if not isinstance(parameters, dict):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="记忆工具参数必须是对象",
            )

        try:
            action = self._required_text(parameters.get("action"), "action").casefold()
            if action not in self.ACTIONS:
                allowed = ", ".join(sorted(self.ACTIONS))
                raise ValueError(f"不支持的记忆操作 {action!r}，可选值: {allowed}")
            handlers = {
                "remember": self._remember,
                "recall": self._recall,
                "update": self._update,
                "retract": self._retract,
                "delete": self._delete,
                "stats": self._stats,
            }
            return handlers[action](parameters)
        except ValueError as exc:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=str(exc),
                context={"user_id": self.user_id},
            )
        except Exception as exc:
            # 外部存储异常必须转换为工具响应，不能中断 Agent 的工具调用循环。
            logger.exception("记忆工具执行失败: user_id=%s", self.user_id)
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"记忆操作执行失败: {exc}",
                context={"user_id": self.user_id},
            )

    def _remember(self, parameters: Dict[str, Any]) -> ToolResponse:
        """根据 ``memory_type`` 保存工作、情景或语义记忆。"""
        content = self._required_text(parameters.get("content"), "content")
        memory_type = self._optional_text(
            parameters.get("memory_type"), "memory_type"
        ) or "working"
        memory_type = memory_type.casefold()
        importance = self._importance(parameters.get("importance", 0.5))
        metadata = self._build_metadata(parameters, memory_type)

        requested_id = str(uuid4())
        stored_id = self.manager.add_memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata,
            memory_id=requested_id,
        )
        created = stored_id == requested_id
        text = (
            f"已保存 {memory_type} 记忆，ID: {stored_id}"
            if created
            else f"该语义事实已经存在，复用记忆 ID: {stored_id}"
        )
        return ToolResponse.success(
            text=text,
            data={
                "memory_id": stored_id,
                "memory_type": memory_type,
                "created": created,
                "structured_fact": "fact" in metadata,
            },
            context={"action": "remember", "user_id": self.user_id},
        )

    def _build_metadata(
        self,
        parameters: Dict[str, Any],
        memory_type: str,
    ) -> Dict[str, Any]:
        """构造不同记忆类型需要的元数据，并拒绝跨类型字段误用。"""
        metadata: Dict[str, Any] = {}
        source_message_id = self._optional_text(
            parameters.get("source_message_id"), "source_message_id"
        )
        if source_message_id:
            metadata["source_message_id"] = source_message_id

        if memory_type == "episodic":
            metadata.update(
                {
                    "session_id": self._optional_text(
                        parameters.get("session_id"), "session_id"
                    )
                    or "default_session",
                    "context": parameters.get("context") or {},
                    "outcome": self._optional_text(
                        parameters.get("outcome"), "outcome"
                    ),
                    "participants": parameters.get("participants") or [],
                    "tags": parameters.get("tags") or [],
                }
            )

        predicate = self._optional_text(parameters.get("predicate"), "predicate")
        object_value = self._optional_text(
            parameters.get("object_value"), "object_value"
        )
        if (predicate is None) != (object_value is None):
            raise ValueError("predicate 和 object_value 必须同时提供或同时省略")
        if predicate and memory_type != "semantic":
            raise ValueError("结构化 predicate 只能用于 semantic 记忆")
        if predicate and object_value:
            metadata["fact"] = SemanticFact(
                subject=self._optional_text(parameters.get("subject"), "subject")
                or "user",
                predicate=predicate,
                object=object_value,
                knowledge_type=parameters.get("knowledge_type", "fact"),
                confidence=parameters.get("confidence", 0.8),
                source="memory_tool",
            )
        return metadata

    def _recall(self, parameters: Dict[str, Any]) -> ToolResponse:
        """检索指定类型或聚合检索全部已启用记忆。"""
        query = self._optional_text(parameters.get("query"), "query")
        predicate = self._optional_text(parameters.get("predicate"), "predicate")
        memory_type = self._optional_text(
            parameters.get("memory_type"), "memory_type"
        )
        memory_type = memory_type.casefold() if memory_type else None
        limit = self._limit(parameters.get("limit", 5))

        if predicate:
            if memory_type not in (None, "semantic"):
                raise ValueError("predicate 检索只能用于 semantic 记忆")
            semantic = self.manager.get_memory("semantic")
            memories = semantic.retrieve(
                query=None,
                user_id=self.user_id,
                predicate=predicate,
                subject=self._optional_text(parameters.get("subject"), "subject")
                or "user",
                object_value=self._optional_text(
                    parameters.get("object_value"), "object_value"
                ),
                retrieval_mode=parameters.get("retrieval_mode", "current"),
                limit=limit,
            )
        else:
            if query is None:
                raise ValueError("recall 操作必须提供 query 或 predicate")
            extra: Dict[str, Any] = {}
            session_id = self._optional_text(
                parameters.get("session_id"), "session_id"
            )
            if session_id:
                if memory_type not in (None, "episodic"):
                    raise ValueError("session_id 过滤只适用于 episodic 记忆")
                extra["session_id"] = session_id
                # session_id 是情景记忆专用条件；没有显式类型时也只查询情景记忆。
                memory_type = "episodic"
            memories = self.manager.retrieve_memories(
                query=query,
                limit=limit,
                memory_types=[memory_type] if memory_type else None,
                **extra,
            )

        return self._recall_response(memories)

    def _recall_response(self, memories: List[MemoryItem]) -> ToolResponse:
        """把多种 ``MemoryItem`` 统一转换为工具文本和结构化数据。"""
        serialized = [self._serialize_memory(item) for item in memories]
        if not serialized:
            return ToolResponse.success(
                text="没有找到相关记忆",
                data={"memories": [], "count": 0},
                context={"action": "recall", "user_id": self.user_id},
            )

        lines = [f"找到 {len(serialized)} 条记忆："]
        for index, item in enumerate(serialized, 1):
            fact = item.get("fact")
            lifecycle = f"，状态: {fact['status']}" if fact else ""
            lines.append(
                f"{index}. [{item['memory_type']}, ID: {item['id']}] "
                f"{item['content']}{lifecycle}"
            )
        return ToolResponse.success(
            text="\n".join(lines),
            data={"memories": serialized, "count": len(serialized)},
            context={"action": "recall", "user_id": self.user_id},
        )

    def _update(self, parameters: Dict[str, Any]) -> ToolResponse:
        """更新当前用户的一条指定类型记忆。"""
        memory_id = self._required_text(parameters.get("memory_id"), "memory_id")
        memory_type = self._required_text(
            parameters.get("memory_type"), "memory_type"
        ).casefold()
        content = self._optional_text(parameters.get("content"), "content")
        importance = (
            self._importance(parameters["importance"])
            if parameters.get("importance") is not None
            else None
        )
        if content is None and importance is None:
            raise ValueError("update 至少需要提供 content 或 importance")
        if not self.manager.update_memory(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            importance=importance,
        ):
            return self._not_found("update", "未找到属于当前用户的目标记忆")
        return ToolResponse.success(
            text=f"已更新 {memory_type} 记忆，ID: {memory_id}",
            data={"memory_id": memory_id, "memory_type": memory_type, "updated": True},
            context={"action": "update", "user_id": self.user_id},
        )

    def _retract(self, parameters: Dict[str, Any]) -> ToolResponse:
        """撤回 active 语义事实并保留历史；其他记忆类型不适用。"""
        memory_id = self._required_text(parameters.get("memory_id"), "memory_id")
        reason = self._optional_text(parameters.get("reason"), "reason")
        if not self.manager.retract_semantic_fact(memory_id, reason):
            return self._not_found(
                "retract",
                "未找到属于当前用户的有效语义事实，或该事实已不是 active 状态",
            )
        return ToolResponse.success(
            text=f"已撤回语义事实，ID: {memory_id}",
            data={"memory_id": memory_id, "memory_type": "semantic", "status": "retracted"},
            context={"action": "retract", "user_id": self.user_id},
        )

    def _delete(self, parameters: Dict[str, Any]) -> ToolResponse:
        """永久删除当前用户的一条记忆；事实纠错应优先使用 ``retract``。"""
        memory_id = self._required_text(parameters.get("memory_id"), "memory_id")
        memory_type = self._optional_text(
            parameters.get("memory_type"), "memory_type"
        )
        if not self.manager.remove_memory(memory_id, memory_type=memory_type):
            return self._not_found("delete", "未找到属于当前用户的记忆")
        return ToolResponse.success(
            text=f"已永久删除记忆，ID: {memory_id}",
            data={"memory_id": memory_id, "deleted": True},
            context={"action": "delete", "user_id": self.user_id},
        )

    def _stats(self, parameters: Dict[str, Any]) -> ToolResponse:
        """返回当前用户全部已启用记忆类型的聚合统计。"""
        del parameters
        stats = self.manager.get_stats()
        return ToolResponse.success(
            text=(
                f"当前用户共有 {stats['total_count']} 条记忆，启用类型: "
                f"{', '.join(stats['enabled_types'])}"
            ),
            data=stats,
            context={"action": "stats", "user_id": self.user_id},
        )

    def _not_found(self, action: str, message: str) -> ToolResponse:
        """生成带用户作用域的统一未找到响应。"""
        return ToolResponse.error(
            code=ToolErrorCode.NOT_FOUND,
            message=message,
            context={"action": action, "user_id": self.user_id},
        )

    @classmethod
    def _serialize_memory(cls, item: MemoryItem) -> Dict[str, Any]:
        """把不同类型的记忆转换为可 JSON 序列化的数据。"""
        fact = SemanticMemory.get_fact(item) if item.memory_type == "semantic" else None
        return {
            "id": item.id,
            "content": item.content,
            "memory_type": item.memory_type,
            "timestamp": item.timestamp.isoformat(),
            "importance": item.importance,
            "metadata": {
                key: value
                for key, value in item.metadata.items()
                if key not in {"fact"}
            },
            "fact": fact.model_dump(mode="json") if fact else None,
        }

    def get_parameters(self) -> List[ToolParameter]:
        """返回 Function Calling 使用的完整参数说明。"""
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作：remember、recall、update、retract、delete 或 stats",
                required=True,
            ),
            ToolParameter(
                name="memory_type",
                type="string",
                description="记忆类型：working、episodic 或 semantic；remember 默认 working",
                required=False,
            ),
            ToolParameter(
                name="content",
                type="string",
                description="保存或更新的记忆文本",
                required=False
            ),
            ToolParameter(
                name="query",
                type="string",
                description="recall 的自然语言查询",
                required=False
            ),
            ToolParameter(
                name="memory_id",
                type="string",
                description="update、retract 或 delete 的目标 ID",
                required=False
            ),
            ToolParameter(
                name="predicate",
                type="string",
                description="semantic 结构化事实谓词",
                required=False
            ),
            ToolParameter(
                name="object_value",
                type="string",
                description="semantic 事实对象值",
                required=False
            ),
            ToolParameter(
                name="subject",
                type="string",
                description="semantic 事实主体",
                required=False,
                default="user"
            ),
            ToolParameter(
                name="retrieval_mode",
                type="string",
                description="semantic 谓词检索模式：current、timeline、audit",
                required=False, default="current"
            ),
            ToolParameter(
                name="knowledge_type",
                type="string",
                description="semantic 事实类型：fact、preference、constraint、goal、skill、rule",
                required=False, default="fact"
            ),
            ToolParameter(
                name="importance",
                type="number",
                description="重要性，范围 0.0 到 1.0",
                required=False,
                default=0.5
            ),
            ToolParameter(
                name="confidence",
                type="number",
                description="semantic 事实置信度，范围 0.0 到 1.0",
                required=False,
                default=0.8
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="recall 返回数量，范围 1 到 100",
                required=False, default=5
            ),
            ToolParameter(
                name="session_id",
                type="string",
                description="episodic 会话标识或检索过滤条件",
                required=False
            ),
            ToolParameter(
                name="context",
                type="object",
                description="episodic 事件上下文字典",
                required=False
            ),
            ToolParameter(
                name="outcome",
                type="string",
                description="episodic 事件结果",
                required=False
            ),
            ToolParameter(
                name="participants",
                type="array",
                description="episodic 参与者列表",
                required=False
            ),
            ToolParameter(
                name="tags",
                type="array",
                description="episodic 标签列表",
                required=False),
            ToolParameter(
                name="reason",
                type="string",
                description="撤回语义事实的原因",
                required=False
            ),
            ToolParameter(
                name="source_message_id",
                type="string",
                description="记忆来源消息 ID",
                required=False
            ),
        ]
