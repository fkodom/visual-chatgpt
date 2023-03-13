import os
import uuid
from typing import Optional, Union

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline, pipeline

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


class TextToImage:
    def __init__(self, device: Union[str, torch.device] = DEFAULT_DEVICE):
        self.device = device

    def __call__(
        self, text: str, device: Optional[Union[str, torch.device]] = None
    ) -> Image.Image:
        if device is None:
            device = self.device

        # Build pipelines and place on the correct device
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32
        ).to(device)
        text_refiner = build_text_refiner(device=device)

        # Refine the text prompt and generate the image
        refined_text = text_refiner(text)[0]["generated_text"]
        print(f"{text} refined to {refined_text}")
        image = pipe(refined_text).images[0]

        # Move back to CPU and clear the GPU cache
        pipe.to("cpu")
        torch.cuda.empty_cache()

        return image


class TextToImageBC(TextToImage):
    """Backwards-compaitble version of TextToImage.  Saves the image to disk instead of"""

    def __init__(self, device: Union[str, torch.device] = DEFAULT_DEVICE):
        super().__init__(device=device)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-base", torch_dtype=torch.float32
        )

    def inference(self, text: str, device: Optional[Union[str, torch.device]] = None):
        image = super().__call__(text, device)

        filename = os.path.join("image", f"{uuid.uuid4().hex[:8]}.png")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        image.save(filename)
        return filename


if __name__ == "__main__":
    model = TextToImage()
    image = model("a photo of an astronaut riding a horse on mars")
    breakpoint()
    pass
    # image.save("image.png")
