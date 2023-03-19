from enum import Enum
from typing import Any, Dict, Optional, Union

import torch
from PIL import Image
from pydantic import BaseModel, parse_obj_as
from transformers import AutoProcessor, BlipForConditionalGeneration

from visual_chatgpt.storage.base import BaseStorageConnector
from visual_chatgpt.tools.base import (
    BaseTool,
    ParameterConfig,
    ParameterType,
    ToolConfig,
)
from visual_chatgpt.types.chat import ChatArtifactType

DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class ModelName(str, Enum):
    BLIP_IMAGE_CAPTIONING_BASE = "Salesforce/blip-image-captioning-base"
    # OPT_2_7B = "Salesforce/blip2-opt-2.7b"


def build_model(
    name: Union[str, ModelName], device: Union[str, torch.device] = DEFAULT_DEVICE
):
    if isinstance(name, ModelName):
        name = name.value
    processor = AutoProcessor.from_pretrained(name)
    if name == ModelName.BLIP_IMAGE_CAPTIONING_BASE.value:
        model = (
            BlipForConditionalGeneration.from_pretrained(
                name, torch_dtype=torch.float16
            )
            .eval()
            .to(device)
        )
    else:
        raise ValueError(f"Unknown model name {name}")

    return model, processor


class ImageCaptionConfig(ToolConfig):
    name: str = "image_caption"
    description: str = "Generate a caption for an image."
    parameters: Dict[str, ParameterConfig] = {
        "image": ParameterConfig(
            type=ParameterType.IMAGE, description="Image to caption"
        ),
    }


class ImageCaptionParameters(BaseModel):
    image: Image.Image
    device: Optional[Union[str, torch.device]] = None

    class Config:
        arbitrary_types_allowed = True


class ImageCaption(BaseTool):
    def __init__(
        self,
        model_name: Union[str, ModelName] = ModelName.BLIP_IMAGE_CAPTIONING_BASE,
        device: Union[str, torch.device] = DEFAULT_DEVICE,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.model, self.processor = build_model(name=self.model_name, device=device)

    def get_config(self) -> ImageCaptionConfig:
        return ImageCaptionConfig()

    def parse_parameters(
        self,
        parameters: Union[Dict[str, Any], BaseModel],
        chat_id: Optional[str] = None,
        storage_connector: Optional[BaseStorageConnector] = None,
    ) -> ImageCaptionParameters:
        if chat_id is None:
            raise ValueError("keyword arg 'chat_id' must be provided")
        elif storage_connector is None:
            raise ValueError("keyword arg 'storage_connector' must be provided")
        # Cast to dict, so we can fetch/replace artifacts from storage before
        # parsing into a Pydantic model.
        if isinstance(parameters, BaseModel):
            parameters = parameters.dict(by_alias=True)

        image_artifact = storage_connector.get_artifact(
            chat_id=chat_id, artifact_id=parameters["image"]
        )
        if image_artifact.type != ChatArtifactType.IMAGE:
            raise ValueError(
                f"Expected artifact type 'image', got '{image_artifact.type}'"
            )
        parameters["image"] = image_artifact.value

        return parse_obj_as(ImageCaptionParameters, parameters)

    def predict(self, parameters: ImageCaptionParameters) -> str:
        if parameters.device is None:
            device = self.device
        else:
            device = parameters.device

        inputs = self.processor(images=parameters.image, return_tensors="pt")
        inputs = inputs.to(device, torch.float16)

        gen_ids = self.model.generate(**inputs)
        gen_texts = self.processor.batch_decode(gen_ids, skip_special_tokens=True)

        return gen_texts[0].strip()


# Backwards compatibility


class ImageCaptionBC(ImageCaption):
    def __init__(self, device: Union[str, torch.device] = DEFAULT_DEVICE):
        super().__init__(device=device)

    def inference(self, image_path: str):
        image = Image.open(image_path)
        return self(image=image)
