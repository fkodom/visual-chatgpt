from __future__ import annotations

import io
import re
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import openai
import tiktoken
import yaml
from PIL import Image
from pydantic import BaseModel, parse_obj_as

from visual_chatgpt.storage.base import BaseStorageConnector
from visual_chatgpt.tools.base import BaseTool
from visual_chatgpt.types.chat import (
    ChatArtifact,
    ChatArtifactType,
    ChatHistory,
    ChatMessage,
    ChatRole,
)
from visual_chatgpt.types.openai import OpenAIChatResponse
from visual_chatgpt.utils.openai import num_tokens

THIS_FILE = Path(__file__).resolve()
with open(THIS_FILE.parent / "instructions.txt") as f:
    INSTRUCTIONS_TEMPLATE = f.read()


class ModelInfo(BaseModel):
    name: str
    model: str
    max_tokens: int
    price_per_token: float


MODEL_INFO = ModelInfo(
    name="ChatGPT",
    model="gpt-3.5-turbo",
    max_tokens=4096,
    # GPT 3.5 Turbo is $0.002 per 1K tokens
    price_per_token=0.002 / 1000,
)
TOKENIZER = tiktoken.encoding_for_model(MODEL_INFO.model)
INSTRUCTIONS_TOKENS = num_tokens(MODEL_INFO.model, INSTRUCTIONS_TEMPLATE)
SYSTEM_TEMPLATE = """
SYSTEM:```yaml
type: {type}
value: {image}
```
BOT:
""".strip()


def trim_chat_history(
    history: ChatHistory,
    max_length: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> ChatHistory:
    """Trim the chat history to a maximum number of messages or tokens.

    Args:
        history (ChatHistory): The chat history to trim.
        max_length (Optional[int], optional): The maximum number of messages to keep. Defaults to None.
        max_tokens (Optional[int], optional): The maximum number of tokens to keep. Defaults to None.

    Returns:
        ChatHistory: The trimmed chat history.
    """
    history = history.copy()
    if max_length is not None:
        history.messages = history.messages[-max_length:]

    if max_tokens is not None:
        total_tokens = 0
        _messages: List[ChatMessage] = []
        for message in reversed(history.messages):
            total_tokens += num_tokens(MODEL_INFO.model, message.content)
            if total_tokens <= max_tokens:
                _messages.append(message)
            else:
                break
        history.messages = list(reversed(_messages))

    return history


def openai_predict(
    instructions: str, history: ChatHistory, num_response_tokens: int = 256
) -> OpenAIChatResponse:
    max_tokens = MODEL_INFO.max_tokens - INSTRUCTIONS_TOKENS - num_response_tokens
    history = trim_chat_history(history, max_tokens=max_tokens)
    history.messages.insert(0, ChatMessage(role=ChatRole.SYSTEM, content=instructions))
    # Parse as OpenAIChatMessage to remove artifacts, and to automatically cast
    # the role from SYSTEM to USER
    return parse_obj_as(
        OpenAIChatResponse,
        openai.ChatCompletion.create(
            model=MODEL_INFO.model,
            messages=[m.dict(exclude={"artifacts"}) for m in history.messages],
        ),
    )


class ResponseType(str, Enum):
    TOOL = "tool"
    IMAGE = "image"
    TEXT = "text"


class BotResponse(BaseModel):
    type: ResponseType
    value: str
    parameters: Optional[Dict[str, str]] = {}

    class Config:
        use_enum_values = True

    @classmethod
    def from_text(cls, text: str) -> BotResponse:
        # Parse with regex the string enclosed by ```yaml and ```
        match = re.search(r"```yaml(.*)```", text, re.DOTALL)
        if match is None:
            raise ValueError("Could not parse the string")
        yaml_string = match.group(1).strip()
        # Parse the yaml into BotResponse
        yaml_dict = yaml.safe_load(yaml_string)
        return BotResponse.parse_obj(yaml_dict)


class ChatGPTAgent:
    def __init__(
        self,
        storage_connector: BaseStorageConnector,
        tools: Dict[str, BaseTool],
        chat_id: Optional[str] = None,
        openai_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if openai_kwargs is None:
            openai_kwargs = {}

        self.storage_connector = storage_connector
        self.tools = tools
        self.openai_kwargs = openai_kwargs
        # Cached properties
        self._chat_id = chat_id
        self._instructions: Optional[str] = None

    @property
    def chat_id(self) -> str:
        if self._chat_id is None:
            self._chat_id = str(uuid.uuid4())
        return self._chat_id

    @property
    def instructions(self) -> str:
        if self._instructions is None:
            tool_configs = {
                name: tool.get_config().dict(by_alias=True)
                for name, tool in self.tools.items()
            }
            buffer = io.StringIO()
            yaml.dump(tool_configs, buffer)
            self._instructions = INSTRUCTIONS_TEMPLATE.format(tools=buffer.getvalue())

        return self._instructions

    def use_tool(
        self, tool_name: str, parameters: Optional[Dict[str, str]]
    ) -> ChatMessage:
        """Use a tool.

        Args:
            tool_name (str): The name of the tool to use.
            parameters (Dict[str, str]): The parameters for the tool.

        Returns:
            ChatMessage: The message to send to the user.
        """
        if parameters is None:
            parameters = {}
        tool = self.tools[tool_name]
        parsed_parameters = tool.parse_parameters(
            parameters, chat_id=self.chat_id, storage_connector=self.storage_connector
        )
        result = tool.predict(parsed_parameters)

        # Need to intelligently parse the tool result into a ChatMessage
        # object, so we can store images, etc in the storage connector.
        # TODO: Split this out into a separate function.
        if isinstance(result, str):
            message = ChatMessage(role=ChatRole.SYSTEM, content=result)
        elif isinstance(result, Image.Image):
            name = self.chat_id + "_" + str(uuid.uuid4())[:8]
            message = ChatMessage(
                role=ChatRole.SYSTEM,
                content=SYSTEM_TEMPLATE.format(type="image", image=name),
                artifacts={
                    name: ChatArtifact(type=ChatArtifactType.IMAGE, value=result),
                },
            )
        else:
            raise ValueError(f"Unexpected result type: {type(result)}")

        return message

    def __call__(
        self, message: Union[str, ChatMessage], num_response_tokens: int = 256
    ) -> ChatMessage:
        # TODO: Place a limit on the number of turns (bot responses) that can be
        # generated in a single call?  This would prevent infinite loops, but it
        # would also prevent the bot from generating a long response.  It should
        # be rare in practice, since the bot can respond with text when it doesn't
        # know how to interpret the input message.
        if isinstance(message, str):
            message = ChatMessage(role=ChatRole.USER, content=f"USER: {message}\nBOT:")
        self.storage_connector.add_message(self.chat_id, message)
        history = self.storage_connector.get_chat_history(self.chat_id)

        response_type: ResponseType = ResponseType.TOOL
        while response_type == ResponseType.TOOL:
            # Get raw prediction from OpenAI model
            openai_response = openai_predict(
                instructions=self.instructions,
                history=history,
                num_response_tokens=num_response_tokens,
            )
            # Add the response to the chat history and storage
            # (Adding to the chat history is less resource intensive, since we
            # don't have to re-query the storage for chat history afterwards.)
            message = parse_obj_as(ChatMessage, openai_response.choices[0].message)
            history.add(message)
            self.storage_connector.add_message(self.chat_id, message)

            # Parse the response, so we know whether to use a tool, or to return
            # an image or text response.
            bot_response = BotResponse.from_text(message.content)
            response_type = bot_response.type

            if response_type == ResponseType.TOOL:
                message = self.use_tool(bot_response.value, bot_response.parameters)
                history.add(message)
                self.storage_connector.add_message(self.chat_id, message)
                print(message)

        return message


ASSISTANT_1 = """
```yaml
type: tool
name: text_to_image
parameters:
  prompt: A dog playing poker
```
""".strip()
SYSTEM_1 = """
SYSTEM:```yaml
type: image
value: my-image-1
```
BOT:
""".strip()
ASSISTANT_2 = """
```yaml
type: tool
value: image_caption
parameters:
  image: my-image-1
```
""".strip()
SYSTEM_2 = """
SYSTEM:```yaml
type: text
value: A dog is playing poker.
```
BOT:
""".strip()


if __name__ == "__main__":
    from visual_chatgpt.storage.in_memory import InMemoryConnector
    from visual_chatgpt.tools.image_caption import ImageCaption
    from visual_chatgpt.tools.image_generation import ImageGeneration
    from visual_chatgpt.tools.image_question_answer import ImageQuestionAnswer

    storage = InMemoryConnector()
    agent = ChatGPTAgent(
        storage_connector=storage,
        tools={
            "image_caption": ImageCaption(),
            "image_generation": ImageGeneration(),
            # "image_question_answer": ImageQuestionAnswer(),
        },
    )
    response = agent("Generate an image of a dog playing poker.")
    print(response)
    response = agent("Create a caption for that image.")
    print(response)

    # history = ChatHistory(
    #     messages=[
    #         ChatMessage(
    #             role=ChatRole.USER,
    #             content="USER: Generate an image of a dog playing poker. BOT:",
    #         ),
    #         ChatMessage(role=ChatRole.ASSISTANT, content=ASSISTANT_1),
    #         ChatMessage(role=ChatRole.SYSTEM, content=SYSTEM_1),
    #         ChatMessage(
    #             role=ChatRole.USER,
    #             content="USER: Create a caption for 'my-image-1'. BOT:",
    #         ),
    #         ChatMessage(role=ChatRole.ASSISTANT, content=ASSISTANT_2),
    #         ChatMessage(role=ChatRole.SYSTEM, content=SYSTEM_2),
    #         ChatMessage(
    #             role=ChatRole.USER,
    #             content="USER: What else can I do with this image? BOT:",
    #         ),
    #     ]
    # )
    # response = openai_predict(history, num_response_tokens=256)
    # total_tokens = response.usage.total_tokens
    # print(f"Total tokens: {total_tokens}")
    # print(f"Total price: ${total_tokens * MODEL_INFO.price_per_token:.4f}")
    # print("Answer:")
    # print(response.choices[0].message.content)
