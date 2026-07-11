"""语义事实的数据模型。"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


KnowledgeType = Literal[
    "fact",
    "preference",
    "constraint",
    "goal",
    "skill",
    "rule",
]

FactStatus = Literal["active", "superseded", "retracted"]


class SemanticFact(BaseModel):
    """从对话或事件中提炼出的、可长期复用的结构化事实。"""

    model_config = ConfigDict(extra="forbid")

    subject: str
    predicate: str
    object: str
    knowledge_type: KnowledgeType = "fact"
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    source: str = "conversation"
    status: FactStatus = "active"
    valid_from: datetime = Field(default_factory=datetime.now)
    valid_to: Optional[datetime] = None
    supersedes: Optional[str] = None

    @field_validator("subject", "predicate", "object", "source")

    @classmethod
    def fields_must_not_be_blank(cls, value: str) -> str:
        """去掉首尾空白，并拒绝没有实际内容的事实字段。"""
        normalized = value.strip()
        if not normalized:
            raise ValueError("语义事实字段不能为空")
        return normalized

    @property
    def key(self) -> tuple[str, str]:
        """同一主体的同一属性共享事实键，用于查重和替代。"""
        return self.subject, self.predicate

    def has_same_value(self, other: "SemanticFact") -> bool:
        """判断另一条事实是否表达相同的主体、属性和值。"""
        return self.key == other.key and self.object == other.object
