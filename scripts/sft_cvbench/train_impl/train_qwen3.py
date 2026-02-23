"""
Qwen3-VL SFT training on CV-Bench.
Uses LoRA + custom training loop (avoids Trainer/sklearn for llava env compatibility).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration
    def _load_qwen3_model(model_id, **kwargs):
        return Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
except ImportError:
    import warnings
    from transformers import AutoModelForCausalLM
    warnings.warn(
        "Qwen3VLForConditionalGeneration not found (transformers>=4.57 required). "
        "Using AutoModelForCausalLM with trust_remote_code."
    )
    def _load_qwen3_model(model_id, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

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
    """Run SFT training for Qwen3-VL on CV-Bench (custom loop, no Trainer/sklearn)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    try:
        import accelerate
        load_kwargs["device_map"] = "auto"
    except ImportError:
        pass
    model = _load_qwen3_model(model_id, **load_kwargs)
    if "device_map" not in load_kwargs:
        model = model.to(device)

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

    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    dataset = CVBenchSFTDataset(train_indices, processor, use_spatial_prompt=use_spatial_prompt)
    collator = CVBenchSFTDataCollator(processor, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )

    total_steps = len(loader) * epochs
    warmup_steps = min(100, total_steps // 10)

    device = next(model.parameters()).device
    model.train()
    step = 0
    for epoch in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            lr = learning_rate * min(1.0, step / warmup_steps) if warmup_steps else learning_rate
            for g in optimizer.param_groups:
                g["lr"] = lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            if step % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} step {step} loss={loss.item():.4f}")

            if step % min(500, len(loader) * 2) == 0 and step > 0:
                ckpt_dir = output_path / f"checkpoint-{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                processor.save_pretrained(ckpt_dir)
                print(f"  Saved {ckpt_dir}")

    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    print(f"Done. Saved to {output_path}")
