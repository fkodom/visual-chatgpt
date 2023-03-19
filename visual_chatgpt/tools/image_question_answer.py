from enum import Enum
from typing import Any, Dict, Optional, Union

import torch
from PIL import Image
from pydantic import BaseModel, parse_obj_as
from transformers import AutoProcessor, BlipForQuestionAnswering

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
    BLIP_VQA_BASE = "Salesforce/blip-vqa-base"
    # FLAN_T5_XXL = "Salesforce/blip2-flan-t5-xxl"


def build_model(
    name: Union[str, ModelName], device: Union[str, torch.device] = DEFAULT_DEVICE
):
    if isinstance(name, ModelName):
        name = name.value
    processor = AutoProcessor.from_pretrained(name)
    if name == ModelName.BLIP_VQA_BASE.value:
        model = (
            BlipForQuestionAnswering.from_pretrained(name, torch_dtype=torch.float16)
            .eval()
            .to(device)
        )
    else:
        raise ValueError(f"Unknown model name {name}")

    return model, processor


class ImageQuestionAnswerConfig(ToolConfig):
    name: str = "image_question_answer"
    description: str = "Answer a question or query about an image"
    parameters: Dict[str, ParameterConfig] = {
        "image": ParameterConfig(
            type=ParameterType.IMAGE, description="Image to caption"
        ),
        "question": ParameterConfig(
            type=ParameterType.STRING, description="Question to answer"
        ),
    }


class ImageQuestionAnswerParameters(BaseModel):
    image: Image.Image
    question: str
    device: Optional[Union[str, torch.device]] = None

    class Config:
        arbitrary_types_allowed = True


class ImageQuestionAnswer(BaseTool):
    def __init__(
        self,
        model_name: Union[str, ModelName] = ModelName.BLIP_VQA_BASE,
        device: Union[str, torch.device] = DEFAULT_DEVICE,
    ):
        self.model_name = model_name
        self.device = device
        self.model, self.processor = build_model(name=self.model_name, device=device)

    def get_config(self) -> ImageQuestionAnswerConfig:
        return ImageQuestionAnswerConfig()

    def parse_parameters(
        self,
        parameters: Union[Dict[str, Any], BaseModel],
        chat_id: Optional[str] = None,
        storage_connector: Optional[BaseStorageConnector] = None,
    ) -> ImageQuestionAnswerParameters:
        if chat_id is None:
            raise ValueError("keyword arg 'conversation_id' must be provided")
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

        return parse_obj_as(ImageQuestionAnswerParameters, parameters)

    def predict(self, parameters: ImageQuestionAnswerParameters) -> str:
        if parameters.device is None:
            device = self.device
        else:
            device = parameters.device

            inputs = self.processor(
                images=parameters.image, text=parameters.question, return_tensors="pt"
            )
        inputs = inputs.to(device, torch.float16)

        gen_ids = self.model.generate(**inputs)
        gen_texts = self.processor.batch_decode(gen_ids, skip_special_tokens=True)

        return gen_texts[0].strip()


# Backwards compatibility


class ImageQuestionAnswerBC(ImageQuestionAnswer):
    def __init__(self, device: Union[str, torch.device] = DEFAULT_DEVICE):
        super().__init__(device=device)

    def get_answer_from_question_and_image(self, inputs: str):
        image_path, question = inputs.split(",")
        image = Image.open(image_path)
        return self(image=image, question=question)
