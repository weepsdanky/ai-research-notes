"""Independent nine-arm Mac Table6 backbone; not the private28-layer model.

Qwen3.8 GDN sigmoid gates/MoE and bounded grouped full attention are shared.
Only residual reads/writes change. All arms omit PLE/Ngram, QSA and MTP;
the undisclosed Table6 backbone/final-GN details are not filled in as facts.
No training budget, cached decoder, quality scores or speed claim is implied.
"""
from dataclasses import dataclass
import hashlib

import torch
from torch import nn
from model import Config, GDN, GatedResidual, MoE, RMSNorm
from qwen35_controls import PlainAttention
from attention_residual import AttentionResidual, DepthHistory, GatedNorm


VARIANTS = {
    'prenorm': ('plain', False, 0),
    'prenorm_gn': ('plain', True, 0),
    'attnres_s4': ('depth', False, 4),
    'attnres_s4_gn': ('depth', True, 4),
    'attnres_s2': ('depth', False, 2),
    'attnres_s2_gn': ('depth', True, 2),
    'attnres_full': ('depth', False, 1),
    'attnres_full_gn': ('depth', True, 1),
    'gr': ('gr', True, 0),
}


@dataclass
class ResidualConfig(Config):
    query_chunk: int = 128
    activation_checkpoint: bool = True


class ResidualLayer(nn.Module):
    def __init__(self, c, index, variant):
        super().__init__()
        self.mode, gated, self.block_size = VARIANTS[variant]
        self.mix = PlainAttention(c) if (index + 1) % 4 == 0 else GDN(c)
        self.moe = MoE(c)
        if self.mode == 'gr':
            self.mix_residual = GatedResidual(c)
            self.moe_residual = GatedResidual(c)
        else:
            self.mix_input = GatedNorm(c.width, c.rank) if gated else RMSNorm(c.width)
            self.moe_input = GatedNorm(c.width, c.rank) if gated else RMSNorm(c.width)
            if self.mode == 'depth':
                self.mix_read = AttentionResidual(c.width)
                self.moe_read = AttentionResidual(c.width)

    def forward(self, state):
        if self.mode == 'gr':
            x, write = self.mix_residual(state)
            state = GatedResidual.add(state, self.mix(x), write)
            x, write = self.moe_residual(state)
            output, router = self.moe(x)
            return GatedResidual.add(state, output, write), router
        if self.mode == 'plain':
            state = state + self.mix(self.mix_input(state))
            output, router = self.moe(self.moe_input(state))
            return state + output, router
        state = state.append(self.mix(self.mix_input(self.mix_read(state))), self.block_size)
        output, router = self.moe(self.moe_input(self.moe_read(state)))
        return state.append(output, self.block_size), router


class ResidualControl(nn.Module):
    def __init__(self, c, variant, seed=42):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError('Unknown residual control')
        if c.rank < 1 or c.rank != c.width // 8 or c.gr_mode != 'gated' or c.query_chunk < 1:
            raise ValueError('Controls require rank d/8, positive tiling and gated GR')
        self.config, self.variant = c, variant
        self.mode, gated, _ = VARIANTS[variant]
        self.embedding = nn.Embedding(c.vocab_size, c.width)
        self.layers = nn.ModuleList([ResidualLayer(c, i, variant) for i in range(c.layers)])
        if self.mode == 'gr':
            self.readout = GatedResidual(c, write=False)
        else:
            self.readout = GatedNorm(c.width, c.rank) if gated else RMSNorm(c.width)
            if self.mode == 'depth':
                self.depth_read = AttentionResidual(c.width)
        self.head = nn.Linear(c.width, c.vocab_size, bias=False)
        self._initialize(seed)

    @torch.no_grad()
    def _initialize(self, seed):
        # Names of every shared backbone tensor are identical across all arms.
        # Keep direct-gain nn.RMSNorm at1, custom offset-gain RMSNorm at0, and
        # AttnRes pseudo-queries at0; blanket1D-zero initialization is incorrect.
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
                generator = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(f'{seed}:{name}.weight'.encode()).digest()[:8], 'little') % (2**63))
                value = torch.empty(module.weight.shape, device='cpu').normal_(std=.02, generator=generator)
                module.weight.copy_(value.to(module.weight.device))
            elif isinstance(module, GDN):
                generator = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(f'{seed}:{name}.A_log'.encode()).digest()[:8], 'little') % (2**63))
                value = torch.empty(module.A_log.shape, device='cpu').uniform_(.01, 16, generator=generator).log_()
                module.A_log.copy_(value.to(module.A_log.device))

    def hidden(self, ids):
        state = self.embedding(ids)
        if self.mode == 'gr':
            state = state.repeat(1, 1, self.config.branches)
        elif self.mode == 'depth':
            state = DepthHistory.start(state)
        routers = []
        for layer in self.layers:
            state, router = layer(state)
            routers.append(router)
        if self.mode == 'depth':
            state = self.depth_read(state)
        return self.readout(state), routers

    def forward(self, ids):
        hidden, routers = self.hidden(ids)
        return self.head(hidden), routers
