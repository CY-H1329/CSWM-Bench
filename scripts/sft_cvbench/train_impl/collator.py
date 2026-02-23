"""
Data collator for VLM SFT (Qwen3-VL, LLaVA-NeXT, Qwen2.5-VL/SpatialReasoner).
Applies chat template with images, masks labels for user tokens.
LLaVA and Qwen2.5-VL use apply_chat_template similarly to Qwen3.
Sa2VA uses collator_sa2va.Sa2VASFTDataCollator.
"""
from dataclasses import dataclass
from typing import Any, Dict, List
import torch


def _get_pad_id(processor: Any) -> int:
    pad_id = getattr(processor, "pad_token_id", None)
    if pad_id is not None:
        return pad_id
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        return getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", 0) or 0
    return 0


@dataclass
class CVBenchSFTDataCollator:
    """Collate batch: apply chat template (with images), tokenize, mask user tokens in labels.
    Works for Qwen3-VL, LLaVA-NeXT, Qwen2.5-VL/SpatialReasoner (apply_chat_template with images).
    """

    processor: Any
    max_length: int = 2048
    model_type: str = "auto"  # "auto" | "llava" for LLaVA processor(image, prompt) fallback

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        all_input_ids = []
        all_labels = []
        all_attention_mask = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_image_sizes = []

        pad_id = _get_pad_id(self.processor)
        for item in batch:
            messages = item["messages"]
            image = item.get("image")
            # Full conversation (user + assistant)
            user_len = None
            try:
                out = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_tensors="pt",
                )
            except Exception as e:
                if self.model_type == "llava" and image is not None:
                    # LLaVA: processor(images, prompt) pattern
                    prompt = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    user_prompt = self.processor.apply_chat_template(
                        [messages[0]], tokenize=False, add_generation_prompt=True
                    )
                    try:
                        out = self.processor(
                            images=[image], text=[prompt], padding=True, return_tensors="pt"
                        )
                        user_out = self.processor(
                            images=[image], text=[user_prompt], padding=True, return_tensors="pt"
                        )
                    except TypeError:
                        out = self.processor(image, prompt, return_tensors="pt")
                        user_out = self.processor(image, user_prompt, return_tensors="pt")
                    user_len = user_out["input_ids"].shape[1]
                else:
                    raise RuntimeError(
                        f"Processor failed on messages. Ensure image in PIL format. {e}"
                    )

            out.pop("token_type_ids", None)
            input_ids = out["input_ids"].squeeze(0)
            pixel_values = out.get("pixel_values")
            image_grid_thw = out.get("image_grid_thw")
            image_sizes = out.get("image_sizes")

            # User part only (to find assistant start)
            if user_len is None:
                user_messages = [messages[0]]
                user_out = self.processor.apply_chat_template(
                    user_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                user_out.pop("token_type_ids", None)
                user_len = user_out["input_ids"].shape[1]

            # Labels: -100 for user + padding, keep assistant
            labels = input_ids.clone()
            labels[labels == pad_id] = -100
            labels[:user_len] = -100

            # Truncate to max_length
            if input_ids.shape[0] > self.max_length:
                input_ids = input_ids[: self.max_length]
                labels = labels[: self.max_length]

            attention_mask = out.get("attention_mask", torch.ones_like(input_ids, dtype=torch.long)).squeeze(0)
            attention_mask = attention_mask[: input_ids.shape[0]]

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attention_mask.append(attention_mask)
            if pixel_values is not None:
                all_pixel_values.append(pixel_values)
            if image_grid_thw is not None:
                all_image_grid_thw.append(image_grid_thw)
            if image_sizes is not None:
                sz = image_sizes
                if not isinstance(sz, torch.Tensor):
                    sz = torch.tensor(sz, dtype=torch.long)
                if sz.dim() == 1:
                    sz = sz.unsqueeze(0)
                all_image_sizes.append(sz)

        result = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True, padding_value=pad_id),
            "labels": torch.nn.utils.rnn.pad_sequence(all_labels, batch_first=True, padding_value=-100),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(all_attention_mask, batch_first=True, padding_value=0),
        }
        if all_pixel_values:
            result["pixel_values"] = torch.cat(all_pixel_values, dim=0)
        if all_image_grid_thw:
            result["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)
        if all_image_sizes:
            result["image_sizes"] = torch.cat(all_image_sizes, dim=0)
        return result
