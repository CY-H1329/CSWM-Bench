"""
Data collator for Qwen3-VL SFT.
Applies chat template with images, masks labels for user tokens.
"""
from dataclasses import dataclass
from typing import Any, Dict, List
import torch


@dataclass
class CVBenchSFTDataCollator:
    """Collate batch: apply chat template (with images), tokenize, mask user tokens in labels."""

    processor: Any
    max_length: int = 2048

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        all_input_ids = []
        all_labels = []
        all_attention_mask = []
        all_pixel_values = []
        all_image_grid_thw = []

        pad_id = getattr(self.processor, "pad_token_id", 0) or 0
        for item in batch:
            messages = item["messages"]
            # Full conversation (user + assistant)
            # Qwen3 processor handles image in messages via apply_chat_template
            try:
                out = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_tensors="pt",
                )
            except Exception as e:
                raise RuntimeError(f"Processor failed on messages. Ensure image in PIL format. {e}")
            out.pop("token_type_ids", None)
            input_ids = out["input_ids"].squeeze(0)
            pixel_values = out.get("pixel_values")
            image_grid_thw = out.get("image_grid_thw")

            # User part only (to find assistant start)
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

        result = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True, padding_value=pad_id),
            "labels": torch.nn.utils.rnn.pad_sequence(all_labels, batch_first=True, padding_value=-100),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(all_attention_mask, batch_first=True, padding_value=0),
        }
        if all_pixel_values:
            result["pixel_values"] = torch.cat(all_pixel_values, dim=0)
        if all_image_grid_thw:
            result["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)
        return result
