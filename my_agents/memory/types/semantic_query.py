"""结构化语义事实查询的数据契约。"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


FactRetrievalMode = Literal["current", "timeline", "audit"]


class FactQuery(BaseModel):
    """描述一次不依赖向量相似度的结构化事实查询。

    查询规划器只负责生成该模型，存储层只负责执行该模型。把两者解耦后，规则规划器
    可以在未来替换成 LLM 规划器，而不需要再次修改 SQLite 查询接口。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str
    subject: str = "user"
    predicate: str
    object_value: Optional[str] = None
    retrieval_mode: FactRetrievalMode = "current"
    limit: int = Field(default=5, ge=1, le=1000)
    source_query: Optional[str] = None

    @field_validator("user_id", "subject", "predicate")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        """统一去除查询关键字段的首尾空白，并拒绝空值。"""
        normalized = value.strip()
        if not normalized:
            raise ValueError("事实查询字段不能为空")
        return normalized

    @field_validator("object_value", "source_query")
    @classmethod
    def normalize_optional_text(cls, value: Optional[str]) -> Optional[str]:
        """可选文本的空字符串按未提供处理。"""
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None
