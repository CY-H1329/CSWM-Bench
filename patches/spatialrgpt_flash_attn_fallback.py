"""
FlashAttention fallback when flash_attn is not installed.
Replaces SpatialRGPT/llava/model/multimodal_encoder/intern/flash_attention.py
"""
import torch
import torch.nn as nn
from einops import rearrange


class FlashAttention(nn.Module):
    """Fallback: use PyTorch scaled_dot_product_attention when flash_attn is not available."""

    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None, max_s=None, need_weights=False):
        # qkv: (B, S, 3, H, D)
        B, S, _, H, D = qkv.shape
        scale = self.softmax_scale or (D ** -0.5)
        q, k, v = qkv.unbind(2)  # each (B, S, H, D)
        # (B, H, S, D) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if causal:
            attn_mask = torch.triu(
                torch.ones(S, S, device=qkv.device, dtype=torch.bool), diagonal=1
            )
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
        else:
            attn_mask = None
        if key_padding_mask is not None:
            # (B, S) -> (B, 1, 1, S)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            if attn_mask is None:
                attn_mask = key_padding_mask.expand(-1, H, S, -1)
            else:
                attn_mask = attn_mask | key_padding_mask.expand(-1, H, S, -1)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p if self.training else 0.0, scale=scale
        )
        out = out.transpose(1, 2)  # (B, S, H, D)
        return out, None
