"""
Sa2VA inference (ByteDance/Sa2VA-4B).
Uses model.predict_forward() for image chat.
Requires: transformers, trust_remote_code=True

Note: Sa2VA model loading triggers PEFT -> bitsandbytes. If you get
"CUDA Setup failed despite GPU being available", set LD_LIBRARY_PATH
before importing (e.g. in Jupyter first cell):
  import os
  os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
  # Then restart kernel and run your code
"""
import os
import warnings
from typing import Optional
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel


def _setup_cuda_library_path():
    """Try to fix bitsandbytes CUDA error by adding common CUDA lib paths to LD_LIBRARY_PATH.
    Sa2VA loading triggers PEFT->bitsandbytes; bitsandbytes needs libcudart.so in path."""
    current = os.environ.get("LD_LIBRARY_PATH", "")
    candidates = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/opt/conda/lib",
        os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib") if os.environ.get("CONDA_PREFIX") else None,
        os.path.join(os.path.dirname(torch.__file__), "lib") if hasattr(torch, "__file__") else None,
    ]
    for path in candidates:
        if path and os.path.isdir(path):
            lib = os.path.join(path, "libcudart.so")
            if os.path.exists(lib) and path not in current:
                os.environ["LD_LIBRARY_PATH"] = (current + ":" + path) if current else path
                return


def _patch_tied_weights_for_sa2va():
    """Sa2VA uses _tied_weights_keys; newer transformers expect all_tied_weights_keys."""
    if not hasattr(PreTrainedModel, "mark_tied_weights_as_initialized"):
        return  # Newer transformers: method removed, no patch needed
    _orig = PreTrainedModel.mark_tied_weights_as_initialized

    def _patched(self):
        if not hasattr(self, "all_tied_weights_keys"):
            old = getattr(self, "_tied_weights_keys", None)
            if old is not None and hasattr(old, "keys"):
                self.all_tied_weights_keys = old
            elif isinstance(old, (list, tuple)):
                self.all_tied_weights_keys = {k: None for x in old for k in (x if isinstance(x, (list, tuple)) else [x])}
            else:
                self.all_tied_weights_keys = {}
        _orig(self)

    PreTrainedModel.mark_tied_weights_as_initialized = _patched


def _patch_torch_linspace_for_sa2va():
    """InternVisionModel uses torch.linspace().item() which fails on meta tensors.
    Force CPU device to avoid meta device from transformers/accelerate.
    Returns the original to restore later."""
    _orig = torch.linspace

    def _patched(*args, **kwargs):
        kwargs.setdefault("device", torch.device("cpu"))
        return _orig(*args, **kwargs)

    torch.linspace = _patched
    return _orig


class Sa2VARunner:
    """Runner for Sa2VA (e.g. ByteDance/Sa2VA-4B)."""

    def __init__(
        self,
        model_id: str = "ByteDance/Sa2VA-4B",
        device: Optional[str] = None,
        use_flash_attn: bool = False,
        **kwargs,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id

        _setup_cuda_library_path()

        load_kwargs = dict(
            **kwargs,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            use_flash_attn=use_flash_attn,
            device_map=None,
        )
        _patch_tied_weights_for_sa2va()
        _orig_linspace = _patch_torch_linspace_for_sa2va()
        try:
            self.model = AutoModel.from_pretrained(model_id, **load_kwargs).eval()
        finally:
            torch.linspace = _orig_linspace
        if device == "cuda":
            self.model = self.model.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False
        )
        self.device = device

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        top_k: int = 0,
        top_p: float = 0.0,
        **kwargs,
    ) -> str:
        # Sa2VA format: <image> + text
        text_prompts = f"<image>{prompt}"
        image_rgb = image.convert("RGB") if image.mode != "RGB" else image

        input_dict = {
            "image": image_rgb,
            "text": text_prompts,
            "past_text": "",
            "mask_prompts": None,
            "tokenizer": self.tokenizer,
        }
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Passing `generation_config` together with generation-related arguments",
            )
            return_dict = self.model.predict_forward(**input_dict)
        return (return_dict.get("prediction") or "").strip()
