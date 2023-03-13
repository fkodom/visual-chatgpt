from enum import Enum
from typing import Optional, Union

import torch
from PIL import Image
from transformers import (
    # AutoModelForQuestionAnswering,
    # AutoModelForSeq2SeqLM,
    AutoProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
)

DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class ModelName(str, Enum):
    BLIP_IMAGE_CAPTIONING_BASE = "Salesforce/blip-image-captioning-base"
    BLIP_VQA_BASE = "Salesforce/blip-vqa-base"

    # Image Captioning
    # OPT_2_7B = "Salesforce/blip2-opt-2.7b"

    # Visual Question Answering
    # FLAN_T5_XXL = "Salesforce/blip2-flan-t5-xxl"


def build_blip2_model(
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
    elif name == ModelName.BLIP_VQA_BASE.value:
        model = (
            BlipForQuestionAnswering.from_pretrained(name, torch_dtype=torch.float16)
            .eval()
            .to(device)
        )
    else:
        raise ValueError(f"Unknown model name {name}")

    return model, processor


class ImageToText:
    def __init__(
        self,
        model_name: str,
        device: Union[str, torch.device] = DEFAULT_DEVICE,
    ):
        self.blip2_model_name = model_name
        self.device = device
        self.model, self.processor = build_blip2_model(
            name=self.blip2_model_name, device=device
        )

    def __call__(
        self,
        image: Image.Image,
        prompt: Optional[str],
        device: Optional[Union[str, torch.device]] = None,
    ) -> str:
        if device is None:
            device = self.device

        if prompt is not None:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        else:
            inputs = self.processor(images=image, return_tensors="pt")
        inputs = inputs.to(device, torch.float16)

        gen_ids = self.model.generate(**inputs)
        gen_texts = self.processor.batch_decode(gen_ids, skip_special_tokens=True)

        return gen_texts[0].strip()


class ImageCaption(ImageToText):
    def __init__(self, device: Union[str, torch.device] = DEFAULT_DEVICE):
        super().__init__(model_name=ModelName.BLIP_IMAGE_CAPTIONING_BASE, device=device)

    def __call__(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if prompt is not None:
            raise ValueError("prompt must be None for ImageCaption")
        return super().__call__(image=image, prompt=None, device=device)


class ImageQuestionAnswer(ImageToText):
    def __init__(self, device: Union[str, torch.device] = DEFAULT_DEVICE):
        super().__init__(model_name=ModelName.BLIP_VQA_BASE, device=device)

    def __call__(
        self,
        image: Image.Image,
        prompt: Optional[str],
        device: Optional[Union[str, torch.device]] = None,
    ):
        if prompt is None:
            raise ValueError("prompt must be specified for ImageQuestionAnswering")
        prompt = f"Question: {prompt} Answer: "
        return super().__call__(image=image, prompt=prompt, device=device)


# Backwards compatibility


class ImageCaptionBC(ImageCaption):
    def __init__(self, device: Union[str, torch.device] = DEFAULT_DEVICE):
        super().__init__(device=device)

    def inference(self, image_path: str):
        image = Image.open(image_path)
        return self(image=image)


class ImageQuestionAnswerBC(ImageQuestionAnswer):
    def __init__(self, device: Union[str, torch.device] = DEFAULT_DEVICE):
        super().__init__(device=device)

    def get_answer_from_question_and_image(self, inputs: str):
        image_path, prompt = inputs.split(",")
        image = Image.open(image_path)
        return self(image=image, prompt=prompt)


if __name__ == "__main__":
    model = ImageQuestionAnswer()
    image = Image.open("temp.png")

    text = model(image, prompt="What is the color of the sky?")
    breakpoint()
    pass
