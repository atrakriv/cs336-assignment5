"""LoRA implemented from scratch (no peft).

W x  ->  W x + (alpha / r) * B A x,   A: (r, in), B: (out, r), B initialised to 0.
Supports torch.nn.Linear and HF GPT-2's Conv1D (weight stored as (in, out)).
"""
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16, dropout=0.0):
        super().__init__()
        assert r > 0
        if isinstance(base, nn.Linear):
            in_features, out_features = base.in_features, base.out_features
        else:  # transformers.pytorch_utils.Conv1D: weight is (in, out)
            in_features, out_features = base.weight.shape
        self.base = base
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        w = base.weight
        self.lora_A = nn.Parameter(torch.empty(r, in_features, dtype=w.dtype, device=w.device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=w.dtype, device=w.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.merged = False

    def forward(self, x):
        out = self.base(x)
        if self.merged:
            return out
        return out + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def _delta(self):  # (out, in)
        return (self.lora_B @ self.lora_A) * self.scaling

    @torch.no_grad()
    def merge(self):
        if self.merged: return
        d = self._delta()
        self.base.weight += d if isinstance(self.base, nn.Linear) else d.T
        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if not self.merged: return
        d = self._delta()
        self.base.weight -= d if isinstance(self.base, nn.Linear) else d.T
        self.merged = False


def apply_lora(model, target_modules=("c_attn", "c_proj"), r=8, alpha=16, dropout=0.0):
    """Freeze all base params and wrap every submodule whose name ends in one of
    `target_modules` with LoRALinear. Only LoRA params remain trainable."""
    for p in model.parameters():
        p.requires_grad = False
    targets = [
        (parent, name, child)
        for parent in model.modules()
        for name, child in parent.named_children()
        if name in target_modules and not isinstance(child, LoRALinear)
        and (isinstance(child, nn.Linear) or child.__class__.__name__ == "Conv1D")
    ]
    for parent, name, child in targets:
        setattr(parent, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
    return model


def lora_parameters(model):
    return [p for n, p in model.named_parameters() if "lora_" in n]


def lora_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def merge_lora(model):
    for m in model.modules():
        if isinstance(m, LoRALinear): m.merge()
    return model
