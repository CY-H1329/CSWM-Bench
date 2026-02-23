"""
CV-Bench SFT Dataset for Qwen3-VL.
Formats (image, question, answer) as chat for supervised fine-tuning.
"""
from typing import List
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset


def build_prompt_simple(question: str, choices: list) -> str:
    """Short instruction + question + options."""
    lines = [question]
    if choices:
        lines.append("Options:")
        for i, c in enumerate(choices):
            lines.append(f"({chr(65 + i)}) {c}")
    return "\n".join(lines)


def format_answer(ans: str) -> str:
    """Normalize answer to (A)/(B)/(C)/(D)."""
    ans = (ans or "").strip().upper()
    for c in "ABCD":
        if f"({c})" in ans or ans == c:
            return f"({c})"
    return ans


class CVBenchSFTDataset(Dataset):
    """Dataset for CV-Bench SFT. Yields (image, messages) for Qwen3 processor."""

    def __init__(self, indices: List[int], processor, use_spatial_prompt: bool = False):
        self.ds = load_dataset("nyu-visionx/CV-Bench", split="test")
        self.indices = indices
        self.processor = processor
        self.use_spatial_prompt = use_spatial_prompt
        if use_spatial_prompt:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
            from scripts.evals.cvbench.common import build_spatial_prompt
            self._build_prompt = build_spatial_prompt
        else:
            self._build_prompt = lambda q: q

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        ex = self.ds[idx]
        img = ex.get("image") or ex.get("images")
        if img is None:
            raise ValueError(f"No image at index {idx}")
        if hasattr(img, "convert"):
            img = img.convert("RGB")
        question = ex.get("question", "")
        choices = ex.get("choices", []) or []
        if choices:
            prompt = build_prompt_simple(question, choices)
        else:
            prompt = question
        if self.use_spatial_prompt:
            prompt = self._build_prompt(prompt)
        ans = format_answer(ex.get("answer", ""))
        assistant_text = f"Final Answer: {ans}"

        messages = [
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": assistant_text},
        ]
        return {"messages": messages, "image": img}
