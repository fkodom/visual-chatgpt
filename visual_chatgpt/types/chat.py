from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, validator


class ChatRole(str, Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"


class ChatArtifactType(str, Enum):
    IMAGE = "image"


class ChatArtifact(BaseModel):
    type: ChatArtifactType
    value: Any

    class Config:
        allow_arbitrary_types = True
        use_enum_values = True


class ChatMessage(BaseModel):
    role: ChatRole
    content: str
    artifacts: Dict[str, ChatArtifact] = {}

    class Config:
        allow_arbitrary_types = True
        use_enum_values = True

    @validator("role")
    def validate_chat_role(cls, value, **kwargs) -> ChatRole:
        for role in ChatRole:
            if value == role.value:
                return role

        roles = list(ChatRole.__members__.keys())
        raise ValueError(f"Invalid role: {value}. Must be one of {roles}")


class ChatHistory(BaseModel):
    messages: List[ChatMessage] = []

    def add(self, message: ChatMessage):
        self.messages.append(message)

    def __str__(self):
        return "\n".join([f"{m.role}: {m.content}" for m in self.messages])
