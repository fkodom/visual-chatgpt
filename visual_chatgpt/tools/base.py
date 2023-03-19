from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Union

from PIL import Image
from pydantic import BaseModel

from visual_chatgpt.storage.base import BaseStorageConnector


class ParameterType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    FLOAT = "float"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    IMAGE = "image"
    ARRAY = "array"
    OBJECT = "object"

    def parse(self, value: Any) -> Any:
        # TODO: Expand this to support more types -- ideally, a superset of JSON Schema
        # so that we can use Images, Datasets, etc. as parameters.
        if self == ParameterType.STRING:
            return str(value)
        elif self == ParameterType.NUMBER:
            return float(value)
        elif self == ParameterType.FLOAT:
            return float(value)
        elif self == ParameterType.INTEGER:
            return int(value)
        elif self == ParameterType.BOOLEAN:
            return bool(value)
        elif self == ParameterType.IMAGE:
            assert isinstance(value, Image.Image)
            return value
        elif self == ParameterType.ARRAY:
            return list(value)
        elif self == ParameterType.OBJECT:
            return dict(value)
        else:
            raise ValueError(f"Unknown parameter type: {self}")


class ParameterConfig(BaseModel):
    type: ParameterType
    description: str
    default: Optional[Any] = None


class ToolConfig(BaseModel):
    """Base configuration for a tool."""

    description: str
    parameters: Dict[str, ParameterConfig] = {}


class BaseTool(ABC):
    @abstractmethod
    def get_config(self) -> ToolConfig:
        """Get the configuration for the tool."""

    @abstractmethod
    def parse_parameters(
        self,
        parameters: Union[Dict[str, Any], BaseModel],
        chat_id: Optional[str] = None,
        storage_connector: Optional[BaseStorageConnector] = None,
    ) -> Any:
        """Parse the parameters for the tool."""

    @abstractmethod
    def predict(self, parameters: Any) -> Any:
        """Execute the tool."""
