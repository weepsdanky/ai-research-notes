"""Unfused depth attention from AttnRes §§3.1–3.2; no training integration yet.

Sources and the separate Qwen Table5 mHC ambiguity: RESIDUAL_SOURCE_AUDIT.md.
Block sizes count sublayers, not Transformer layers. Histories are per forward
pass and retain autograd edges; they are never a cache across token batches.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from optim import piece


@dataclass(frozen=True)
class DepthHistory:
    completed: tuple[torch.Tensor, ...]
    partial: torch.Tensor | None = None
    outputs: int = 0

    @classmethod
    def start(cls, embedding):
        return cls((embedding,))

    def sources(self):
        return self.completed + (() if self.partial is None else (self.partial,))

    def append(self, output, block_size):
        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError('block_size must be a positive sublayer count')
        if output.shape != self.completed[0].shape:
            raise ValueError('Every depth source must have the embedding shape')
        partial = output if self.partial is None else self.partial + output
        count = self.outputs + 1
        if count % block_size == 0:
            return DepthHistory(self.completed + (partial,), None, count)
        return DepthHistory(self.completed, partial, count)


class AttentionResidual(nn.Module):
    """One pseudo-query and key RMSNorm per sublayer/readout.

    Normalize keys, retain raw values, softmax over depth, and apply no sqrt(d)
    factor. The zero pseudo-query follows the authors' §5 initialization.
    """
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(width))
        self.key_norm = nn.RMSNorm(width, eps=eps)

    def forward(self, history):
        values = torch.stack(history.sources(), dim=0)
        keys = self.key_norm(values)
        logits = (keys * self.query).sum(-1)
        weights = logits.softmax(dim=0)
        return (weights.unsqueeze(-1) * values).sum(dim=0)


class GatedNorm(nn.Module):
    """Qwen Eq29; rank d/8 is explicit and does not wrap AttnRes key norm."""
    def __init__(self, width, rank=None, eps=1e-6):
        super().__init__()
        if rank is None:
            if width % 8:
                raise ValueError('Default GatedNorm rank requires width divisible by8')
            rank = width // 8
        if rank < 1:
            raise ValueError('GatedNorm rank must be positive')
        self.norm = nn.RMSNorm(width, eps=eps)
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        # Same declared local AdamW policy as other residual gates. Do not let
        # generic matrix ownership silently send these gate matrices to Muon.
        for parameter in self.parameters():
            parameter._recipe_pieces = [piece('adamw', semantic='gated_norm')]

    def forward(self, x):
        normalized = self.norm(x)
        return normalized * self.up(F.silu(self.down(normalized))).sigmoid()
