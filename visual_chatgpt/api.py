import logging
import logging.config
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Dict

import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from visual_chatgpt.agents.chatgpt import BotResponse, ChatGPTAgent
from visual_chatgpt.settings import Settings
from visual_chatgpt.storage import storage_from_settings
from visual_chatgpt.tools import (
    ImageCaption,
    ImageGeneration,
    ImageQuestionAnswer,
)
from visual_chatgpt.types.api import (
    ChatInfoResponse,
    ListChatsResponse,
    MessageRequest,
    MessageResponse,
    ResponseType,
    ServiceStatus,
)
from visual_chatgpt.types.chat import (
    ChatArtifact,
    ChatArtifactType,
    ChatHistory,
    ChatMessage,
    ChatRole,
)
from visual_chatgpt.utils.image_ops import base64_to_image, image_to_base64

# Setup logging using the log_config.yaml file
LOG_CONFIG = Path(__file__).parent / "log_config.yaml"
if not LOG_CONFIG.exists():
    raise FileNotFoundError(f"File {LOG_CONFIG} does not exist.")
with open(LOG_CONFIG) as f:
    log_config = yaml.safe_load(f)
logging.config.dictConfig(log_config)

settings = Settings()
app = FastAPI(
    title=settings.service_title,
    version=settings.service_version,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
instrumentator = Instrumentator(
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="inprogress",
    inprogress_labels=True,
)

storage = storage_from_settings(settings)
# TODO: Dynamically load the 'tools' from settings
tools = {
    "image_caption": ImageCaption(),
    "image_generation": ImageGeneration(),
    "image_question_answer": ImageQuestionAnswer(),
}


@app.get("/readyz", response_model=ServiceStatus, status_code=200)
async def readyz():
    """Used to determine if the service can receive traffic. If service is
    orchestrated using kubernetes the traffic will not be routed to this api until
    this endpoint returns.
    """
    return ServiceStatus(ok=True)


@app.get("/livez", response_model=ServiceStatus, status_code=200)
async def livez():
    """Use to determine if the service is alive. If service is orchestrated using
    kubernetes this api will be restarted if this check fails.
    """
    return ServiceStatus(ok=True)


def _init_agent(chat_id: str) -> ChatGPTAgent:
    """Initializes the agent for the given conversation"""
    # TODO: In the future, generalize this to load other agent types (e.g. GPT4)
    return ChatGPTAgent(chat_id=chat_id, tools=tools, storage_connector=storage)


@app.get(
    "/v1/chats/list",
    response_model=ListChatsResponse,
    status_code=200,
)
async def list_chat_ids() -> ListChatsResponse:
    """Returns a list of all conversation IDs"""
    return ListChatsResponse(chat_ids=storage.list_chat_ids())


@app.post("/v1/chats/create", response_model=ChatInfoResponse, status_code=200)
async def create_chat() -> ChatInfoResponse:
    """Creates a new conversation and returns the chat ID"""
    chat_id = str(uuid.uuid4())[:8]
    storage.create_chat(chat_id=chat_id)
    return ChatInfoResponse(chat_id=chat_id)


@app.get(
    "/v1/chats/{chat_id}/history",
    response_model=ChatHistory,
    status_code=200,
)
async def get_chat_history(chat_id: str) -> ChatHistory:
    return storage.get_chat_history(chat_id)


# TODO: Add endpoints:
# - GET /v1/chats/{chat_id}/info
# - GET /v1/chats/{chat_id}/messages/{message_id}
# - GET /v1/chats/{chat_id}/messages


@app.post(
    "/v1/chats/{chat_id}/messages/create",
    response_model=MessageResponse,
    status_code=200,
)
async def new_message(chat_id: str, request: MessageRequest):
    """Sends a new message to the agent and returns the response"""
    agent = _init_agent(chat_id)
    artifacts: Dict[str, ChatArtifact] = {}
    if request.image is not None:
        image_name = chat_id + "_" + str(uuid.uuid4())[:8]
        artifacts = {
            image_name: ChatArtifact(
                type=ChatArtifactType.IMAGE,
                value=base64_to_image(request.image),
            )
        }

    message = ChatMessage(role=ChatRole.USER, content=request.text, artifacts=artifacts)
    raw_response = agent(message)
    parsed = BotResponse.from_text(raw_response.content)

    if parsed.type == ResponseType.TEXT:
        return MessageResponse(type=ResponseType.TEXT, value=parsed.value)
    elif parsed.type == ResponseType.IMAGE:
        image = storage.get_artifact(chat_id=chat_id, artifact_id=parsed.value)
        return MessageResponse(type=ResponseType.IMAGE, value=image_to_base64(image))
    else:
        raise ValueError(f"Unknown response type: {parsed.type}")


@app.on_event("shutdown")
def shutdown_event():
    logging.info("server shutting down")


@app.on_event("startup")
async def startup():
    instrumentator.instrument(app).expose(app)


def main():
    logging.info("starting service")
    uvicorn.run(
        app,
        host=settings.service_host,
        port=settings.service_port,
        reload=settings.service_reload,
        root_path=settings.service_root_path,
        workers=settings.service_workers,
        log_config=str(LOG_CONFIG),
        log_level=settings.service_log_level,
    )


if __name__ == "__main__":
    main()
