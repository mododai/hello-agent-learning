"""结构化事实的变更意图模型。"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .semantic_fact import SemanticFact


FactOperation = Literal["assert", "retract", "replace"]


class FactChange(BaseModel):
    """表示事实提取器希望记忆层执行的操作，而不是直接操作数据库。

    ``assert`` 表示确认事实成立；``retract`` 表示撤回完整事实；``replace`` 表示
    显式替换旧事实。target_memory_id 对 replace 很有用，但保持可选，以便存储层
    可以按 ``subject + predicate`` 查找单值旧事实。
    """

    model_config = ConfigDict(extra="forbid")

    operation: FactOperation = "assert"
    fact: SemanticFact
    target_memory_id: Optional[str] = None
    reason: Optional[str] = None

    @field_validator("target_memory_id", "reason")
    @classmethod
    def normalize_optional_text(cls, value: Optional[str]) -> Optional[str]:
        """把可选文本统一去空格，空字符串按未提供处理。"""
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def asserted_or_replacement_fact_must_be_active(self) -> "FactChange":
        """assert/replace 描述的是新当前事实，不能携带历史状态。"""
        if self.operation in {"assert", "replace"} and self.fact.status != "active":
            raise ValueError("assert 或 replace 的新事实必须是 active 状态")
        return self
