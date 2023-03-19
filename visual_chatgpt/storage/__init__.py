from visual_chatgpt.settings import Settings
from visual_chatgpt.storage.base import BaseStorageConnector
from visual_chatgpt.storage.in_memory import InMemoryConnector


def storage_from_settings(settings: Settings) -> BaseStorageConnector:
    """Returns the storage connector based on the settings"""
    if settings.storage_type == "in_memory":
        return InMemoryConnector()
    raise ValueError(f"Unknown storage type {settings.storage_type}")
