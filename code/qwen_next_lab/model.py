"""Equations first. B=batch, T=time, H=heads, D=head width, R=residual branches."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    width: int = 256
    layers: int = 4
    heads: int = 4
    hidden: int = 384
    experts: int = 4
    top_k: int = 2
    branches: int = 4
    vocab: int = 257  # UTF-8 bytes + end-of-document; no tokenizer download.
    block: int = 4
    budget: int = 32

    def __post_init__(self):
        assert self.width >= 32 and self.width % (4 * self.heads) == 0
        assert self.layers > 0 and self.layers % 4 == 0
        assert 1 <= self.top_k <= self.experts
        assert self.hidden > 0 and self.branches > 0
        assert self.budget >= self.block > 0 and self.budget % self.block == 0


class RMSNorm(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        # Zero-centered *gain parameter*, not subtraction of the activation mean.
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * (1 + self.delta)


def partial_rope(x, positions):
    """Rotate half the channels. x: [B,H,T,D], positions: [T]."""
    rotary = x.shape[-1] // 2
    freq = 10000.0 ** (-torch.arange(0, rotary, 2, device=x.device).float() / rotary)
    angle = positions.float()[:, None] * freq[None, :]
    a, b = x[..., :rotary:2], x[..., 1:rotary:2]
    rotated = torch.stack((a * angle.cos() - b * angle.sin(),
                           a * angle.sin() + b * angle.cos()), dim=-1).flatten(-2)
    return torch.cat((rotated, x[..., rotary:]), dim=-1)


def delta_scan(q, k, v, alpha, beta):
    """Eq. 1–4; q/k/v: [B,T,H,D], gates: [B,T,H]."""
    state = q.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
    outputs = []
    # shortcut: explicit scan is readable but slow; use a verified scan kernel for long contexts.
    for t in range(q.shape[1]):
        state = alpha[:, t, :, None, None] * state
        predicted = (state * k[:, t, :, :, None]).sum(-2)
        error = v[:, t] - predicted
        state = state + beta[:, t, :, None, None] * k[:, t, :, :, None] * error[..., None, :]
        outputs.append((state * q[:, t, :, :, None]).sum(-2))
    return torch.stack(outputs, dim=1)


class GDN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.heads
        self.q = nn.Linear(c.width, c.width, bias=False)
        self.k = nn.Linear(c.width, c.width, bias=False)
        self.v = nn.Linear(c.width, c.width, bias=False)
        self.convs = nn.ModuleList([nn.Conv1d(c.width, c.width, 4, groups=c.width,
                                            bias=False) for _ in range(3)])
        self.alpha = nn.Linear(c.width, c.heads)
        self.beta = nn.Linear(c.width, c.heads, bias=False)
        self.log_rate = nn.Parameter(torch.zeros(c.heads))
        self.norm = RMSNorm(c.width // c.heads)
        self.gate = nn.Linear(c.width, c.width, bias=False)
        self.out = nn.Linear(c.width, c.width, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        streams = [F.silu(conv(F.pad(proj(x).transpose(1, 2), (3, 0))))
                   .transpose(1, 2).reshape(b, t, self.heads, d // self.heads)
                   for proj, conv in zip((self.q, self.k, self.v), self.convs)]
        q, k, v = streams
        alpha = torch.exp(-self.log_rate.exp() * F.softplus(self.alpha(x)))
        beta = self.beta(x).sigmoid()
        y = delta_scan(F.normalize(q, dim=-1), F.normalize(k, dim=-1), v, alpha, beta)
        y = self.norm(y).reshape(b, t, d) * self.gate(x).sigmoid()
        return self.out(y)


class Indexer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.dim = 4, c.width // c.heads
        self.q = nn.Linear(c.width, self.heads * self.dim, bias=False)
        self.k = nn.Linear(c.width, self.dim, bias=False)
        self.qnorm, self.knorm = RMSNorm(self.dim), RMSNorm(self.dim)
        self.block, self.budget = c.block, c.budget

    def forward(self, x):
        batch, t, _ = x.shape
        r, blocks = self.block, t // self.block
        pos = torch.arange(t, device=x.device)
        q = self.qnorm(self.q(x).reshape(batch, t, self.heads, self.dim)).transpose(1, 2)
        k = self.k(x)[:, :blocks * r].reshape(batch, blocks, r, self.dim).mean(2)
        k = self.knorm(k)[:, None]
        q, k = partial_rope(q, pos), partial_rope(k, torch.arange(blocks, device=x.device) * r)
        scores = (q @ k.transpose(-1, -2)).relu().sum(1)  # [B,T,blocks]
        valid = (torch.arange(blocks, device=x.device) * r + r - 1)[None, :] <= pos[:, None]
        scores = scores.masked_fill(~valid, -1e9)
        selected = torch.zeros_like(scores, dtype=torch.bool)
        count = min(self.budget // r, blocks)
        if count:
            selected.scatter_(-1, scores.topk(count, dim=-1).indices, True)
        selected = selected & valid  # Early queries have fewer than count complete blocks.
        expanded = F.pad(selected.repeat_interleave(r, dim=-1), (0, t - blocks * r))
        tail_start = ((pos + 1) // r) * r
        tail = (pos[None, :] >= tail_start[:, None]) & (pos[None, :] <= pos[:, None])
        return scores, valid, selected, expanded | tail


def indexer_kl(attention, scores, eligible, block):
    """Teacher is detached core attention; normalize ONLY over eligible complete blocks."""
    blocks = scores.shape[-1]
    if not blocks:
        return scores.sum() * 0
    batch, _, t, _ = attention.shape
    teacher = attention.detach().mean(1)[..., :blocks * block]
    teacher = teacher.reshape(batch, t, blocks, block).amax(-1) * eligible
    teacher = teacher / teacher.sum(-1, keepdim=True).clamp_min(1e-12)
    log_student = scores.masked_fill(~eligible, -1e9).log_softmax(-1)
    losses = (teacher * (teacher.clamp_min(1e-12).log() - log_student)).sum(-1)
    usable = eligible.any(-1).expand(batch, t)
    return (losses * usable).sum() / usable.sum().clamp_min(1)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.block = c.heads, c.block
        self.q = nn.Linear(c.width, c.width, bias=False)
        self.k = nn.Linear(c.width, c.width, bias=False)
        self.v = nn.Linear(c.width, c.width, bias=False)
        self.qnorm, self.knorm = RMSNorm(c.width // c.heads), RMSNorm(c.width // c.heads)
        self.gate = nn.Linear(c.width, c.width, bias=False)
        self.out = nn.Linear(c.width, c.width, bias=False)
        self.indexer = Indexer(c)

    def forward(self, x, stage):
        b, t, d = x.shape
        q, k, v = [p(x).reshape(b, t, self.heads, d // self.heads).transpose(1, 2)
                   for p in (self.q, self.k, self.v)]
        pos = torch.arange(t, device=x.device)
        q, k = partial_rope(self.qnorm(q), pos), partial_rope(self.knorm(k), pos)
        mask = (pos[None, :] <= pos[:, None])[None, None]
        kl = x.new_zeros(())
        if stage != "pretrain":
            scores, valid, selected, sparse_mask = self.indexer(x.detach())
            if stage == "sparse":
                mask = sparse_mask[:, None]
        # shortcut: dense masked reference, NOT a fast sparse kernel; keep T <= 256 on Mac.
        weights = ((q @ k.transpose(-1, -2)) / math.sqrt(d // self.heads))
        weights = weights.masked_fill(~mask, -1e9).softmax(-1)
        if stage != "pretrain":
            eligible = selected if stage == "sparse" else valid
            kl = indexer_kl(weights, scores, eligible, self.block)
        y = (weights @ v).transpose(1, 2).reshape(b, t, d)
        return self.out(y * self.gate(x).sigmoid()), kl


class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.up = nn.Linear(c.width, c.hidden, bias=False)
        self.gate = nn.Linear(c.width, c.hidden, bias=False)
        self.down = nn.Linear(c.hidden, c.width, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.top_k = c.top_k
        self.router = nn.Linear(c.width, c.experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(c) for _ in range(c.experts)])
        self.shared = SwiGLU(c)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        probs = self.router(x).softmax(-1)
        weights, indices = probs.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        y = self.shared(x)
        for e, expert in enumerate(self.experts):
            tokens, slots = torch.where(indices == e)
            if tokens.numel():
                y = y.index_add(0, tokens, expert(x[tokens]) * weights[tokens, slots, None])
        frequency = F.one_hot(indices, len(self.experts)).float().mean((0, 1)).detach()
        balance = len(self.experts) * (frequency * probs.mean(0)).sum()
        return y.reshape(shape), balance


class GatedResidual(nn.Module):
    def __init__(self, c):
        super().__init__()
        r, d = c.branches, c.width
        self.branches = r
        self.norm = RMSNorm((r, d))
        self.read_down = nn.Linear(r * d, d // 8, bias=False)
        self.read_up = nn.Linear(d // 8, r * d, bias=False)
        self.write = nn.Linear(r * d, r, bias=False)

    def read(self, residual):
        normalized = self.norm(residual)
        flat = normalized.flatten(-2) / self.branches
        gate = self.read_up(F.silu(self.read_down(flat))).sigmoid().reshape_as(residual)
        x = (normalized * gate).mean(-2)
        write = 2 * self.write(flat).sigmoid()
        return x, write

    def forward(self, residual, y, write):
        return residual + write[..., None] * y[..., None, :]


class Layer(nn.Module):
    def __init__(self, c, global_attention):
        super().__init__()
        self.mix = Attention(c) if global_attention else GDN(c)
        self.moe = MoE(c)
        self.mix_residual, self.moe_residual = GatedResidual(c), GatedResidual(c)

    def forward(self, residual, stage):
        x, write = self.mix_residual.read(residual)
        if isinstance(self.mix, Attention):
            y, kl = self.mix(x, stage)
        else:
            y, kl = self.mix(x), x.new_zeros(())
        residual = self.mix_residual(residual, y, write)
        x, write = self.moe_residual.read(residual)
        y, balance = self.moe(x)
        return self.moe_residual(residual, y, write), kl, balance


class Model(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.embedding = nn.Embedding(c.vocab, c.width)
        self.layers = nn.ModuleList([Layer(c, i % 4 == 3) for i in range(c.layers)])
        self.norm = RMSNorm(c.width)
        self.head = nn.Linear(c.width, c.vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids, stage="pretrain"):
        assert stage in {"pretrain", "distill", "sparse"}
        residual = self.embedding(ids)[..., None, :].expand(-1, -1, self.config.branches, -1)
        kl, balance = residual.new_zeros(()), residual.new_zeros(())
        for layer in self.layers:
            residual, local_kl, local_balance = layer(residual, stage)
            kl, balance = kl + local_kl, balance + local_balance
        # Teaching choice: average branches at the final readout.
        logits = self.head(self.norm(residual.mean(-2)))
        return logits, kl / (len(self.layers) // 4), balance / len(self.layers)
