import os
import uuid
from typing import Any, Dict, Optional, Union

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from pydantic import BaseModel, parse_obj_as
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline, pipeline

from visual_chatgpt.storage.base import BaseStorageConnector
from visual_chatgpt.tools.base import (
    BaseTool,
    ParameterConfig,
    ParameterType,
    ToolConfig,
)

DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def build_text_refiner(device: Union[str, torch.device] = DEFAULT_DEVICE) -> Pipeline:
    text_refine_tokenizer = AutoTokenizer.from_pretrained(
        "Gustavosta/MagicPrompt-Stable-Diffusion"
    )
    text_refine_model = AutoModelForCausalLM.from_pretrained(
        "Gustavosta/MagicPrompt-Stable-Diffusion"
    )
    return pipeline(
        "text-generation",
        model=text_refine_model,
        tokenizer=text_refine_tokenizer,
        device=device,
    )


class ImageGenerationConfig(ToolConfig):
    description: str = "Generate an image given a prompt"
    parameters: Dict[str, ParameterConfig] = {
        "prompt": ParameterConfig(
            description="Text prompt for image generation",
            type=ParameterType.STRING,
        )
    }


class ImageGenerationParameters(BaseModel):
    prompt: str
    device: Optional[Union[str, torch.device]] = None

    class Config:
        arbitrary_types_allowed = True


class ImageGeneration(BaseTool):
    def __init__(self, device: Union[str, torch.device] = DEFAULT_DEVICE):
        self.device = device

    def get_config(self) -> ImageGenerationConfig:
        return ImageGenerationConfig()

    def parse_parameters(
        self,
        parameters: Union[Dict[str, Any], BaseModel],
        chat_id: Optional[str] = None,
        storage_connector: Optional[BaseStorageConnector] = None,
    ) -> ImageGenerationParameters:
        return parse_obj_as(ImageGenerationParameters, parameters)

    def predict(self, parameters: ImageGenerationParameters) -> Image.Image:
        if parameters.device is not None:
            device = parameters.device
        else:
            device = self.device

        # Build pipelines and place on the correct device
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32
        ).to(device)
        text_refiner = build_text_refiner(device=device)

        # Refine the text prompt and generate the image
        refined_text = text_refiner(parameters.prompt)[0]["generated_text"]
        print(f"{parameters.prompt} refined to {refined_text}")
        image = pipe(refined_text).images[0]

        # Move back to CPU and clear the GPU cache
        pipe.to("cpu")
        torch.cuda.empty_cache()

        return image


class ImageGenerationBC(ImageGeneration):
    """Backwards-compaitble version of TextToImage.  Saves the image to disk instead of"""

    def __init__(self, device: Union[str, torch.device] = DEFAULT_DEVICE):
        super().__init__(device=device)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-base", torch_dtype=torch.float32
        )

    def inference(self, text: str, device: Optional[Union[str, torch.device]] = None):
        image = super().__call__(prompt=text, device=device)

        filename = os.path.join("image", f"{uuid.uuid4().hex[:8]}.png")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        image.save(filename)
        return filename


if __name__ == "__main__":
    model = ImageGeneration()
    image = model(prompt="a photo of an astronaut riding a horse on mars")
    breakpoint()
    pass
    # image.save("image.png")
