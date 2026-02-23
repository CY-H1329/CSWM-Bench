"""
Sa2VA-specific data collator.
Sa2VAChatModel.forward(data) expects: pixel_values, input_ids, position_ids, attention_mask, labels.
Uses model's template, preprocessing, and IMG_CONTEXT token format.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_ar)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Image.Image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    """Split image by aspect ratio for Sa2VA (from modeling_sa2va_chat)."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((int(target_width), int(target_height)))
    processed_images = []
    tw, th = int(target_width // image_size), int(target_height // image_size)
    for i in range(blocks):
        box = (
            (i % tw) * image_size,
            (i // tw) * image_size,
            ((i % tw) + 1) * image_size,
            ((i // tw) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


@dataclass
class Sa2VASFTDataCollator:
    """Collator for Sa2VAChatModel.forward(data). Requires model with preparing_for_generation called."""

    model: Any
    tokenizer: Any
    max_length: int = 2048
    image_size: int = 448
    patch_size: int = 14
    downsample_ratio: float = 0.5
    use_thumbnail: bool = True
    min_dynamic_patch: int = 1
    max_dynamic_patch: int = 12

    def __post_init__(self):
        if hasattr(self.model, "preparing_for_generation") and not getattr(self.model, "init_prediction_config", False):
            try:
                self.model.preparing_for_generation(self.tokenizer)
            except Exception:
                pass
        cfg = getattr(self.model, "config", None)
        if cfg:
            self.image_size = getattr(cfg, "force_image_size", None) or getattr(
                getattr(cfg, "vision_config", None), "image_size", self.image_size
            )
            vc = getattr(cfg, "vision_config", None)
            if vc:
                self.patch_size = getattr(vc, "patch_size", self.patch_size)
            self.downsample_ratio = getattr(cfg, "downsample_ratio", self.downsample_ratio)
            self.use_thumbnail = getattr(cfg, "use_thumbnail", self.use_thumbnail)
            self.min_dynamic_patch = getattr(cfg, "min_dynamic_patch", self.min_dynamic_patch)
            self.max_dynamic_patch = getattr(cfg, "max_dynamic_patch", self.max_dynamic_patch)
        self.patch_token = int(
            (self.image_size // self.patch_size) ** 2 * (self.downsample_ratio ** 2)
        )
        self.IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        self.IMG_START_TOKEN = "<img>"
        self.IMG_END_TOKEN = "</img>"
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.transformer = Compose(
            [
                lambda img: img.convert("RGB") if img.mode != "RGB" else img,
                Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                ToTensor(),
                Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            ]
        )

    def _process_image(self, image: Image.Image) -> torch.Tensor:
        images = dynamic_preprocess(
            image,
            min_num=self.min_dynamic_patch,
            max_num=self.max_dynamic_patch,
            image_size=self.image_size,
            use_thumbnail=self.use_thumbnail,
        )
        pixel_values = torch.stack([self.transformer(img) for img in images])
        return pixel_values

    def _build_input_text(self, prompt: str, answer: str) -> str:
        num_image_tokens = self.patch_token * 1  # single image
        image_token_str = (
            f"{self.IMG_START_TOKEN}{self.IMG_CONTEXT_TOKEN * num_image_tokens}{self.IMG_END_TOKEN}"
        )
        text = f"{image_token_str}\n{prompt}"
        template = getattr(self.model, "template", {})
        if isinstance(template, dict) and "INSTRUCTION" in template:
            bot_name = getattr(self.model, "bot_name", "BOT")
            input_text = template["INSTRUCTION"].format(input=text, round=1, bot_name=bot_name)
        else:
            input_text = f"USER: {text}\nASSISTANT: "
        input_text += answer
        return input_text

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        all_input_ids = []
        all_labels = []
        all_attention_mask = []
        all_pixel_values = []
        all_position_ids = []

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0

        for item in batch:
            messages = item["messages"]
            img = item["image"]
            if hasattr(img, "convert"):
                img = img.convert("RGB")

            # Extract prompt and answer from messages
            user_content = messages[0].get("content", [])
            asst_content = messages[1].get("content", [])
            prompt = ""
            for c in user_content:
                if isinstance(c, dict) and c.get("type") == "text":
                    prompt = c.get("text", "")
                    break
            answer = ""
            for c in asst_content:
                if isinstance(c, dict) and c.get("type") == "text":
                    answer = c.get("text", "")
                    break

            pixel_values = self._process_image(img)
            input_text = self._build_input_text(prompt, answer)
            ids = self.tokenizer.encode(input_text, add_special_tokens=False)
            if self.tokenizer.bos_token_id is not None:
                ids = [self.tokenizer.bos_token_id] + ids

            input_ids = torch.tensor(ids, dtype=torch.long)
            user_text = self._build_input_text(prompt, "")
            user_ids = self.tokenizer.encode(user_text, add_special_tokens=False)
            if self.tokenizer.bos_token_id is not None:
                user_ids = [self.tokenizer.bos_token_id] + user_ids
            user_len = len(user_ids)

            labels = input_ids.clone()
            labels[labels == pad_id] = -100
            labels[:user_len] = -100

            if input_ids.shape[0] > self.max_length:
                input_ids = input_ids[: self.max_length]
                labels = labels[: self.max_length]

            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            position_ids = attention_mask.cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attention_mask.append(attention_mask)
            all_pixel_values.append(pixel_values)
            all_position_ids.append(position_ids)

        result = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                all_input_ids, batch_first=True, padding_value=pad_id
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(
                all_labels, batch_first=True, padding_value=-100
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                all_attention_mask, batch_first=True, padding_value=0
            ),
            "position_ids": torch.nn.utils.rnn.pad_sequence(
                all_position_ids, batch_first=True, padding_value=0
            ),
            "pixel_values": torch.cat(all_pixel_values, dim=0),
        }
        return result
