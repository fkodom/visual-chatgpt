import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List

from visual_chatgpt.types.chat import ChatArtifact, ChatHistory, ChatMessage


class BaseStorageConnector(ABC):
    @abstractmethod
    def create_chat(self, chat_id: str) -> None:
        """Create a new conversation."""

    async def create_chat_async(self, chat_id: str) -> None:
        """Create a new conversation asynchronously."""
        with ThreadPoolExecutor() as executor:
            return await asyncio.get_event_loop().run_in_executor(
                executor, lambda: self.create_chat(chat_id)
            )

    @abstractmethod
    def add_message(self, chat_id: str, message: ChatMessage) -> None:
        """Add a message to the storage."""

    async def add_message_async(self, chat_id: str, message: ChatMessage) -> None:
        """Add a message to the storage asynchronously."""
        with ThreadPoolExecutor() as executor:
            return await asyncio.get_event_loop().run_in_executor(
                executor, lambda: self.add_message(chat_id, message)
            )

    @abstractmethod
    def list_artifact_ids(self, chat_id: str) -> List[str]:
        """List the names of all artifacts for a conversation."""

    async def list_artifact_ids_async(self, chat_id: str) -> List[str]:
        """List the names of all artifacts for a conversation asynchronously."""
        with ThreadPoolExecutor() as executor:
            return await asyncio.get_event_loop().run_in_executor(
                executor, lambda: self.list_artifact_ids(chat_id)
            )

    @abstractmethod
    def list_chat_ids(self) -> List[str]:
        """List all conversation ids."""

    async def list_chat_ids_async(self) -> List[str]:
        """Get all conversation ids asynchronously."""
        with ThreadPoolExecutor() as executor:
            return await asyncio.get_event_loop().run_in_executor(
                executor, lambda: self.list_chat_ids()
            )

    @abstractmethod
    def get_chat_history(self, chat_id: str) -> ChatHistory:
        """Get the chat history for a conversation."""

    async def get_chat_history_async(self, chat_id: str) -> ChatHistory:
        """Get the chat history for a conversation asynchronously."""
        with ThreadPoolExecutor() as executor:
            return await asyncio.get_event_loop().run_in_executor(
                executor, lambda: self.get_chat_history(chat_id)
            )

    @abstractmethod
    def get_artifact(self, chat_id: str, artifact_id: str) -> ChatArtifact:
        """Get an artifact by name."""

    async def get_artifact_async(self, chat_id: str, artifact_id: str) -> ChatArtifact:
        """Get an artifact by name asynchronously."""
        with ThreadPoolExecutor() as executor:
            return await asyncio.get_event_loop().run_in_executor(
                executor, lambda: self.get_artifact(chat_id, artifact_id)
            )
