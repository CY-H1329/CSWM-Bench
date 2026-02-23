"""
Qwen3-VL SFT training on CV-Bench.
Uses LoRA for efficient fine-tuning on single GPU.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

from .dataset import CVBenchSFTDataset
from .collator import CVBenchSFTDataCollator


def train_qwen3(
    train_indices: list,
    output_dir: str,
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 2048,
    use_spatial_prompt: bool = False,
):
    """Run SFT training for Qwen3-VL on CV-Bench."""
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA for efficient training
    for p in model.parameters():
        p.requires_grad = False
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def _make_inputs_require_grad(module, inp, out):
            out.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)

    dataset = CVBenchSFTDataset(train_indices, processor, use_spatial_prompt=use_spatial_prompt)
    collator = CVBenchSFTDataCollator(processor, max_length=max_length)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=min(100, len(dataset) // batch_size),
        weight_decay=0.01,
        logging_steps=10,
        save_steps=min(500, len(dataset) // batch_size * 2),
        bf16=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
