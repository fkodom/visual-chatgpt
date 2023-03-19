import logging
import os

from pydantic import BaseSettings

VERSION_FILE = os.path.join(os.path.dirname(__file__), "..", "VERSION")
with open(VERSION_FILE, "r") as f:
    SERVICE_VERSION = f.read().strip()


class Settings(BaseSettings):
    service_host: str = "0.0.0.0"
    service_port: int = 8080

    service_workers: int = 1
    service_root_path: str = "/"
    service_log_level: int = logging.INFO
    service_title: str = "visual-chatgpt"
    service_version: str = SERVICE_VERSION
    service_reload: bool = False
    metrics_port: int = 3000

    # TODO: add more settings for the chat agent here
    agent_type: str = "chatgpt"

    # TODO: add more settings for storage backends here
    storage_type: str = "in_memory"
