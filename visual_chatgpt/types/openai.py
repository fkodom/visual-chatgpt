from typing import List, Optional

from pydantic import BaseModel, validator


class Usage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


# === Conversation ===
#
# Types used for the conversation API. (e.g. ChatGPT and any other conversation models)
#   See: https://beta.openai.com/docs/api-reference/conversations


class OpenAIChatMessage(BaseModel):
    # NOTE: This is the same as 'ChatMessage' but without the 'artifacts' field
    # which is not allowed in the conversation API.
    role: str
    content: str

    @validator("role")
    def validate_chat_role(cls, value, **kwargs) -> str:
        # When SYSTEM responds to the BOT, we want to use the "user" role.  This
        # is for interfacing with OpenAI -- the SYSTEM role is not allowed, so we
        # pretend as if the USER is responding to the BOT.  (In effect, the user *is*
        # responding, since we have pre-defined the tools that the BOT can use.)
        if value == "system":
            return "user"
        return value


class OpenAIChatChoice(BaseModel):
    message: OpenAIChatMessage
    index: int
    finish_reason: Optional[str]


class OpenAIChatRequest(BaseModel, extra="allow"):
    # TODO: Add explicit typing for other, known parameters
    model: str
    messages: List[OpenAIChatMessage]


class OpenAIChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChatChoice]
    usage: Usage
