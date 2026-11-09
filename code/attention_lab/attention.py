"""A readable causal MHA and a tiny decoder-only language model."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class CausalMHA(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        if heads < 1 or width < 1 or width % heads:
            raise ValueError("width must be positive and divisible by heads")
        self.heads = heads
        self.head_dim = width // heads
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, x, return_weights=False):
        b, t, width = x.shape
        q, k, v = [linear(x).reshape(b, t, self.heads, self.head_dim).transpose(1, 2)
                   for linear in (self.q, self.k, self.v)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        future = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        weights = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
        heads = weights @ v
        y = self.out(heads.transpose(1, 2).contiguous().reshape(b, t, width))
        return (y, weights) if return_weights else y


class TinyLM(nn.Module):
    """One pre-norm decoder block, learned positions and character-token logits."""
    def __init__(self, vocab_size, context=32, width=32, heads=4):
        super().__init__()
        self.context = context
        self.tokens = nn.Embedding(vocab_size, width)
        self.positions = nn.Embedding(context, width)
        self.norm1 = nn.LayerNorm(width)
        self.attention = CausalMHA(width, heads)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Linear(4 * width, width))
        self.norm_final = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, bias=False)

    def forward(self, ids, targets=None):
        _, t = ids.shape
        if t > self.context:
            raise ValueError("sequence exceeds context")
        x = self.tokens(ids) + self.positions(torch.arange(t, device=ids.device))
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        logits = self.head(self.norm_final(x))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
