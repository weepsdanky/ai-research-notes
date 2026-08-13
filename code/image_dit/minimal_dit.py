"""Minimal Diffusion Transformer: the core idea without DiT training tricks.

This module deliberately contains only:
  image patches -> tokens -> time-conditioned vanilla Transformer -> pixels.

It omits class conditioning, classifier-free guidance, adaLN-Zero, learned
variance, EMA, and specialized initialization so those ideas can be introduced
one at a time in the full ``dit.py`` implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from dit import get_2d_sincos_pos_embed, timestep_embedding


@dataclass(frozen=True)
class MinimalDiTConfig:
    image_size: int = 28
    patch_size: int = 4
    in_channels: int = 1
    hidden_size: int = 64
    depth: int = 2
    num_heads: int = 4
    mlp_ratio: float = 4.0

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.hidden_size % 4:
            raise ValueError("hidden_size must be divisible by 4")

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size


class MinimalDiT(nn.Module):
    """A vanilla Transformer denoiser with additive timestep conditioning."""

    def __init__(self, config: MinimalDiTConfig = MinimalDiTConfig()):
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.register_buffer(
            "pos_embed",
            get_2d_sincos_pos_embed(config.hidden_size, config.grid_size),
            persistent=False,
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=int(config.hidden_size * config.mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config.depth, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.output = nn.Linear(
            config.hidden_size,
            config.patch_size * config.patch_size * config.in_channels,
        )

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        return self.patch_embed(images).flatten(2).transpose(1, 2)

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, count, features = tokens.shape
        grid = self.config.grid_size
        patch = self.config.patch_size
        channels = self.config.in_channels
        if count != grid * grid or features != patch * patch * channels:
            raise ValueError("tokens do not match the configured patch grid")
        images = tokens.reshape(batch, grid, grid, patch, patch, channels)
        images = torch.einsum("bhwpqc->bchpwq", images)
        return images.reshape(batch, channels, grid * patch, grid * patch)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        expected = (
            x.shape[0], self.config.in_channels,
            self.config.image_size, self.config.image_size,
        )
        if tuple(x.shape) != expected or t.shape != (x.shape[0],):
            raise ValueError(f"expected x={expected} and t=[batch]")
        tokens = self.patchify(x) + self.pos_embed
        time = self.time_mlp(timestep_embedding(t, self.config.hidden_size))
        tokens = self.transformer(tokens + time.unsqueeze(1))
        return self.unpatchify(self.output(self.output_norm(tokens)))


def minimal_diffusion_loss(
    model: MinimalDiT,
    x0: torch.Tensor,
    t: torch.Tensor,
    alpha_bar: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """The DDPM simple objective, written inline for conceptual transparency."""

    noise = torch.randn_like(x0) if noise is None else noise
    coefficient = alpha_bar[t].reshape(-1, 1, 1, 1)
    xt = coefficient.sqrt() * x0 + (1 - coefficient).sqrt() * noise
    return torch.nn.functional.mse_loss(model(xt, t), noise)
