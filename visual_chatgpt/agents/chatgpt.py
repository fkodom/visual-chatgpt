from typing import Dict, List, Optional

import openai
import tiktoken
from pydantic import BaseModel, parse_obj_as

INSTRUCTIONS = """You are a chatbot that can perform vision-language tasks (image
generation, image captioning, image retrieval, etc).  To complete those tasks, you
can either use a set of pre-defined tools, or use your own NLP abilities.

Given a request from USER, you will determine:
1. Whether or not you need a tool to perform the task. (I.e. whether you already
    have enough information to answer.)
2. If needed, which tool and what parameters to use
3. Else, how to respond to USER

TOOLS
-----
```yaml
- name: text_to_image
  description: Generate an image given a prompt
  parameters: 
    - name: prompt
      description: Text prompt for image generation
      type: string
      required: true

- name: image_caption
  description: Generate a caption given an image
  parameters:
    - name: image
      description: Image for caption generation.
      type: image
      required: true

- name: image_question_answer
  description: Answer a question about the contents of an image
  parameters:
    - name: image
      description: Image for question answering.
      type: image
      required: true
    - name: question
      description: The question to answer.
      type: string
      required: true
```

RESPONSE
--------
Your response must be a single, YAML-formatted string.  Do not include any additional
text. Use the following format:

```yaml
type: <type>  # one of: "tool", "text", "image"
value: <value>  # name of the tool to use, or the resulting image / text value
parameters:  # (optional) list of parameters to use with the tool
  - name: <name>
    value: <value>
  - name: <name>
    value: <value>
```

Images are specified by name.  If USER or SYSTEM provides an image named "my_image-1",
you must refer to it as "my-image-1".

When you use a tool, SYSTEM will respond with the result.  You can request multiple
tools to perform a single task, if needed.  When the task is complete, you must respond
to USER with the result.  If no tools are needed, you can respond directly to USER.
Do not attempt to predict the response of SYSTEM.

If USER asks for help or clarification, give a response of type "text" to provide
that information.

EXAMPLE
-------
USER: "Generate an image of a cat."
BOT:```yaml
type: tool
value: text_to_image
parameters:
  - name: prompt
    value: "A cat"
```
SYSTEM:```yaml
type: image
value: "my-image-1"
```
BOT:```yaml
type: image
value: "my-image-1"
```
"""


def num_tokens(text: str) -> int:
    """This is the official way to calculate the number of tokens, according to OpenAI.
    See:
        - https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        - https://github.com/openai/tiktoken
        - https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    return len(TOKENIZER.encode(text))


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
INSTRUCTIONS_TOKENS = num_tokens(INSTRUCTIONS)
# Leave some room for ChatGPT response
TRIM_MAX_TOKENS = MODEL_INFO.max_tokens - INSTRUCTIONS_TOKENS - 100


class Message(BaseModel):
    role: str
    content: str
    # TODO: Add artifacts for images, etc
    # artifacts: List[Artifact] = []


class Choice(BaseModel):
    message: Message
    index: int
    finish_reason: Optional[str]


class Usage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class ConversationResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class Answer(BaseModel):
    text: str
    total_tokens: int

    @property
    def total_price(self) -> float:
        return self.total_tokens * MODEL_INFO.price_per_token


class ChatHistory(BaseModel):
    messages: List[Message] = []

    def add(self, message: Message):
        self.messages.append(message)

    def trim(self, max_length: Optional[int] = None, max_tokens: Optional[int] = None):
        if max_length is not None:
            self.messages = self.messages[-max_length:]

        if max_tokens is not None:
            total_tokens = 0
            _messages: List[Message] = []
            for message in reversed(self.messages):
                total_tokens += num_tokens(message.content)
                if total_tokens <= max_tokens:
                    _messages.append(message)
                else:
                    break
            self.messages = list(reversed(_messages))

    def __str__(self):
        return "".join([f"{m.role}: {m.content}" for m in self.messages])


class ChatGPT:
    def __init__(
        self, chat_history: Optional[ChatHistory] = None, temperature: float = 0.0
    ):
        if chat_history is None:
            chat_history = ChatHistory()
        self.chat_history = chat_history
        self.temperature = temperature

    def __call__(self, message: Message) -> Answer:
        self.chat_history.add(message)
        # Leave room for the instructions and a reaonsbly long response
        self.chat_history.trim(
            max_tokens=MODEL_INFO.max_tokens - INSTRUCTIONS_TOKENS - 256
        )
        messages: List[Dict] = self.chat_history.dict(by_alias=True)["messages"]
        messages.insert(0, {"role": "user", "content": INSTRUCTIONS})
        response = parse_obj_as(
            ConversationResponse,
            openai.ChatCompletion.create(model=MODEL_INFO.model, messages=messages),
        )

        return Answer(
            text=response.choices[0].message.content,
            total_tokens=response.usage.total_tokens,
        )


# class ChatGPTAgent:
#     def __init__(self):
#         super().__init__()

#     def __call__(self, message: Message):
#         pass


ASSISTANT_1 = """
```yaml
type: tool
name: text_to_image
parameters:
  - name: prompt
    value: A dog playing poker
```
""".strip()
SYSTEM_1 = """
SYSTEM:```yaml
type: image
name: my-image-1
```
BOT:
""".strip()
ASSISTANT_2 = """
```yaml
type: tool
value: image_caption
parameters:
  - name: image
    value: my-image-1
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

    def answer_question(content: str, max_tokens: int = 4096) -> Answer:
        # openai.api_key = get_openai_api_key()
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": INSTRUCTIONS},
                {"role": "user", "content": content},
                {"role": "assistant", "content": ASSISTANT_1},
                {"role": "user", "content": SYSTEM_1},
                {
                    "role": "user",
                    "content": "USER: Create a caption for 'my-image-1'. BOT:",
                },
                {"role": "assistant", "content": ASSISTANT_2},
                {"role": "user", "content": SYSTEM_2},
                {
                    "role": "user",
                    "content": "USER: What else can I do with this image? BOT:",
                },
            ],
        )
        _response = parse_obj_as(ConversationResponse, response)

        return Answer(
            text=_response.choices[0].message.content,
            total_tokens=_response.usage.total_tokens,
        )

    answer = answer_question("USER: Generate an image of a dog playing poker. BOT:")
    print(f"Total tokens: {answer.total_tokens}")
    print(f"Total price: {answer.total_price}")
    print("Answer:")
    print(answer.text)
