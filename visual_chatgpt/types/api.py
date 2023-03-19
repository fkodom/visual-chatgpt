from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class ServiceStatus(BaseModel):
    """Data Model For Health Check Response"""

    ok: bool


class MessageRequest(BaseModel):
    text: str
    image: Optional[str] = None


class ResponseType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    # TODO: Need some kind of 'array' type for multiple images, etc.
    # IMAGE_LIST = "image_list"


class ChatInfoResponse(BaseModel):
    chat_id: str
    # TODO: Integrate with the storage connector to get the actual number of messages,
    # created_at, updated_at, etc.
    description: Optional[str] = None
    num_messages: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ListChatsResponse(BaseModel):
    chat_ids: List[str]


class MessageResponse(BaseModel):
    type: ResponseType
    value: str
    description: Optional[str] = None

    class Config:
        use_enum_values = True


class TextResponse(MessageResponse):
    type = ResponseType.TEXT
    description: Optional[str] = "Raw text string"


class ImageResponse(MessageResponse):
    type = ResponseType.IMAGE
    description: Optional[str] = "Base64 encoded image"
