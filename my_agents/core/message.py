
from typing import Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field

MessageRole = Literal["user", "assistant", "system", "tool"]

class Message(BaseModel):
    """
    消息类
    """
    content: str
    role: MessageRole
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None

    def __init__(self, content: str, role: MessageRole, **kwargs):
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get("timestamp", datetime.now()),
            metadata=kwargs.get("metadata"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
        }

    def __str__(self):
        return f"[{self.role}]: {self.content}"
