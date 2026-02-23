"""
Sa2VA (ByteDance/Sa2VA-4B) SFT training on CV-Bench.
Sa2VAChatModel has forward(data) returning loss - uses standard CausalLM-style training.
Requires custom collator for model's data format (pixel_values, input_ids, position_ids, labels).
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from peft import LoraConfig, get_peft_model, TaskType

from .dataset import CVBenchSFTDataset
from .collator_sa2va import Sa2VASFTDataCollator


def _patch_tied_weights_for_sa2va():
    """Sa2VA uses _tied_weights_keys; newer transformers expect all_tied_weights_keys."""
    if not hasattr(PreTrainedModel, "mark_tied_weights_as_initialized"):
        return
    _orig = PreTrainedModel.mark_tied_weights_as_initialized

    def _patched(self):
        if not hasattr(self, "all_tied_weights_keys"):
            old = getattr(self, "_tied_weights_keys", None)
            if old is not None and hasattr(old, "keys"):
                self.all_tied_weights_keys = old
            elif isinstance(old, (list, tuple)):
                keys = []
                for x in old:
                    keys.extend(x if isinstance(x, (list, tuple)) else [x])
                self.all_tied_weights_keys = {k: None for k in keys}
            else:
                self.all_tied_weights_keys = {}
        _orig(self)

    PreTrainedModel.mark_tied_weights_as_initialized = _patched


def _patch_torch_linspace_for_sa2va():
    """InternVisionModel uses torch.linspace().item() which fails on meta tensors."""
    _orig = torch.linspace

    def _patched(*args, **kwargs):
        kwargs.setdefault("device", torch.device("cpu"))
        return _orig(*args, **kwargs)

    torch.linspace = _patched
    return _orig


def train_sa2va(
    train_indices: list,
    output_dir: str,
    model_id: str = "ByteDance/Sa2VA-4B",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 2048,
    use_spatial_prompt: bool = False,
):
    """Run SFT training for Sa2VA on CV-Bench. Uses model.forward(data) which returns loss."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _patch_tied_weights_for_sa2va()
    _orig_linspace = _patch_torch_linspace_for_sa2va()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            use_flash_attn=False,
            device_map=None,
        )
    finally:
        torch.linspace = _orig_linspace

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # LoRA on language model
    llm_arch = getattr(
        model.config.llm_config, "architectures", ["Qwen2ForCausalLM"]
    )[0]
    if "Qwen2" in llm_arch or "Llama" in llm_arch:
        target_modules = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
    elif "InternLM2" in llm_arch:
        target_modules = ["attention.wqkv", "attention.wo", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"]
    else:
        target_modules = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]

    for p in model.parameters():
        p.requires_grad = False
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=target_modules,
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

    dataset = CVBenchSFTDataset(train_indices, tokenizer, use_spatial_prompt=use_spatial_prompt)
    collator = Sa2VASFTDataCollator(model, tokenizer, max_length=max_length)
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
            # Sa2VAChatModel.forward expects data dict
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outputs = model.forward(batch, mode="loss")
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
                tokenizer.save_pretrained(ckpt_dir)
                print(f"  Saved {ckpt_dir}")

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Done. Saved to {output_path}")
