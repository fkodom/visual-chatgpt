"""
In-memory storage for the chatbot.

NOTE: Mostly used for testing purposes.  Not recommended for production.
Production systems should use a database or other persistent storage.
"""

from typing import Dict, List

from visual_chatgpt.storage.base import BaseStorageConnector
from visual_chatgpt.types.chat import ChatArtifact, ChatHistory, ChatMessage


class InMemoryConnector(BaseStorageConnector):
    def __init__(self):
        self._chat_histories: Dict[str, ChatHistory] = {}

    def create_chat(self, chat_id: str) -> None:
        if chat_id in self._chat_histories:
            raise KeyError(f"Chat {chat_id} already exists.")
        self._chat_histories[chat_id] = ChatHistory()

    def add_message(self, chat_id: str, message: ChatMessage) -> None:
        if chat_id not in self._chat_histories:
            self._chat_histories[chat_id] = ChatHistory()
        self._chat_histories[chat_id].add(message)

    def list_artifact_ids(self, chat_id: str) -> List[str]:
        history = self.get_chat_history(chat_id)
        return [
            artifact_id
            for message in history.messages
            for artifact_id in message.artifacts.keys()
        ]

    def list_chat_ids(self) -> List[str]:
        return list(self._chat_histories.keys())

    def get_chat_history(self, chat_id: str) -> ChatHistory:
        if chat_id not in self._chat_histories:
            raise KeyError(f"Chat {chat_id} not found.")
        return self._chat_histories[chat_id]

    def get_artifact(self, chat_id: str, artifact_id: str) -> ChatArtifact:
        history = self.get_chat_history(chat_id)
        for message in history.messages:
            if artifact_id in message.artifacts:
                return message.artifacts[artifact_id]
        raise KeyError(f"Artifact {artifact_id} not found.")


if __name__ == "__main__":
    import asyncio

    from visual_chatgpt.types.chat import ChatRole

    storage = InMemoryConnector()
    asyncio.run(
        storage.add_message_async(
            chat_id="1",
            message=ChatMessage(role=ChatRole.USER, content="hello"),
        )
    )
    print(storage.get_chat_history(chat_id="1"))
