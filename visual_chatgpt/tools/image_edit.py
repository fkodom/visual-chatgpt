class MaskFormer:
    def __init__(self, device):
        self.device = device
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        ).to(device)

    def inference(self, image_path, text):
        threshold = 0.5
        min_area = 0.02
        padding = 20
        original_image = Image.open(image_path)
        image = original_image.resize((512, 512))
        inputs = self.processor(
            text=text,
            images=image,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        mask = torch.sigmoid(outputs[0]).squeeze().cpu().numpy() > threshold
        area_ratio = len(np.argwhere(mask)) / (mask.shape[0] * mask.shape[1])
        if area_ratio < min_area:
            return None
        true_indices = np.argwhere(mask)
        mask_array = np.zeros_like(mask, dtype=bool)
        for idx in true_indices:
            padded_slice = tuple(
                slice(max(0, i - padding), i + padding + 1) for i in idx
            )
            mask_array[padded_slice] = True
        visual_mask = (mask_array * 255).astype(np.uint8)
        image_mask = Image.fromarray(visual_mask)
        return image_mask.resize(image.size)


class ImageEditing:
    def __init__(self, device):
        print("Initializing StableDiffusionInpaint to %s" % device)
        self.device = device
        self.mask_former = MaskFormer(device=self.device)
        self.inpainting = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
        ).to(device)

    def remove_part_of_image(self, input):
        image_path, to_be_removed_txt = input.split(",")
        print(f"remove_part_of_image: to_be_removed {to_be_removed_txt}")
        return self.replace_part_of_image(
            f"{image_path},{to_be_removed_txt},background"
        )

    def replace_part_of_image(self, input):
        image_path, to_be_replaced_txt, replace_with_txt = input.split(",")
        print(f"replace_part_of_image: replace_with_txt {replace_with_txt}")
        original_image = Image.open(image_path)
        mask_image = self.mask_former.inference(image_path, to_be_replaced_txt)
        updated_image = self.inpainting(
            prompt=replace_with_txt, image=original_image, mask_image=mask_image
        ).images[0]
        updated_image_path = get_new_image_name(
            image_path, func_name="replace-something"
        )
        updated_image.save(updated_image_path)
        return updated_image_path
