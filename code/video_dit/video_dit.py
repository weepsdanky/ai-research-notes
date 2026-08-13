"""A laptop-sized, paper-grounded Video Diffusion Transformer.

The model follows Latte's best-performing Variant 1: each pair contains one
complete spatial Transformer block followed by one complete temporal block.
Frames are patchified independently; time is never folded into the channel
dimension, so temporal attention remains an explicit and inspectable operation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import nn


@dataclass(frozen=True)
class VideoDiTConfig:
    frames: int = 8
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    hidden_size: int = 128
    num_heads: int = 4
    num_pairs: int = 4
    mlp_ratio: float = 4.0
    condition_sizes: tuple[int, ...] = (3, 5, 4, 4)
    condition_mode: Literal["structured", "text"] = "structured"
    text_embedding_size: int | None = None
    condition_dropout_prob: float = 0.1
    learn_sigma: bool = False

    def __post_init__(self) -> None:
        if self.frames < 2:
            raise ValueError("frames must be at least 2")
        if self.image_size <= 0 or self.patch_size <= 0:
            raise ValueError("image_size and patch_size must be positive")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.hidden_size <= 0 or self.num_heads <= 0 or self.num_pairs <= 0:
            raise ValueError("hidden_size, num_heads, and num_pairs must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.hidden_size % 4:
            raise ValueError("hidden_size must be divisible by 4 for 2-D positions")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if not self.condition_sizes or any(size <= 0 for size in self.condition_sizes):
            raise ValueError("condition_sizes must contain positive cardinalities")
        if self.condition_mode not in ("structured", "text"):
            raise ValueError("condition_mode must be structured or text")
        if self.condition_mode == "text" and (
            self.text_embedding_size is None or self.text_embedding_size <= 0
        ):
            raise ValueError("text mode requires a positive text_embedding_size")
        if not 0.0 <= self.condition_dropout_prob <= 1.0:
            raise ValueError("condition_dropout_prob must be in [0, 1]")

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_size**2

    @property
    def out_channels(self) -> int:
        return self.in_channels * (2 if self.learn_sigma else 1)


def timestep_embedding(
    timesteps: torch.Tensor, dimension: int, max_period: int = 10_000
) -> torch.Tensor:
    """Sinusoidal diffusion-timestep embeddings used by DiT."""

    if timesteps.ndim != 1:
        raise ValueError("timesteps must have shape [batch]")
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)
    if dimension % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
    return embedding


def get_1d_sincos_pos_embed(dimension: int, length: int) -> torch.Tensor:
    if dimension % 2:
        raise ValueError("dimension must be even for 1-D sin/cos positions")
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = 1.0 / (
        10_000
        ** (torch.arange(dimension // 2, dtype=torch.float32) / (dimension // 2))
    )
    angles = positions * frequencies.unsqueeze(0)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1).unsqueeze(0)


def get_2d_sincos_pos_embed(dimension: int, grid_size: int) -> torch.Tensor:
    if dimension % 4:
        raise ValueError("dimension must be divisible by 4 for 2-D positions")
    coordinates = torch.arange(grid_size, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    frequencies = 1.0 / (
        10_000
        ** (torch.arange(dimension // 4, dtype=torch.float32) / (dimension // 4))
    )

    def embed(position: torch.Tensor) -> torch.Tensor:
        angles = position.reshape(-1, 1) * frequencies.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

    return torch.cat([embed(grid_y), embed(grid_x)], dim=1).unsqueeze(0)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_size: int = 256):
        super().__init__()
        self.frequency_size = frequency_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(timesteps, self.frequency_size))


class StructuredConditionEmbedder(nn.Module):
    """Embed prompt factors and replace the whole prompt with null tokens for CFG."""

    def __init__(
        self,
        condition_sizes: tuple[int, ...],
        hidden_size: int,
        dropout_prob: float,
    ) -> None:
        super().__init__()
        self.condition_sizes = condition_sizes
        self.dropout_prob = dropout_prob
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality + 1, hidden_size)
            for cardinality in condition_sizes
        )

    def forward(
        self,
        condition_ids: torch.Tensor,
        train: bool,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected = (condition_ids.shape[0], len(self.condition_sizes))
        if tuple(condition_ids.shape) != expected:
            raise ValueError(f"condition_ids must have shape {expected}")
        if force_drop_ids is not None and force_drop_ids.shape != (condition_ids.shape[0],):
            raise ValueError("force_drop_ids must have shape [batch]")
        if force_drop_ids is None:
            drop = (
                torch.rand(condition_ids.shape[0], device=condition_ids.device)
                < self.dropout_prob
                if train and self.dropout_prob > 0
                else torch.zeros(
                    condition_ids.shape[0], device=condition_ids.device, dtype=torch.bool
                )
            )
        else:
            drop = force_drop_ids.bool()

        result = torch.zeros(
            condition_ids.shape[0],
            self.embeddings[0].embedding_dim,
            device=condition_ids.device,
            dtype=self.embeddings[0].weight.dtype,
        )
        for index, (embedding, cardinality) in enumerate(
            zip(self.embeddings, self.condition_sizes, strict=True)
        ):
            ids = condition_ids[:, index]
            if torch.any(ids < 0) or torch.any(ids >= cardinality):
                raise ValueError(
                    f"condition factor {index} must be in [0, {cardinality - 1}]"
                )
            ids = torch.where(drop, cardinality, ids)
            result = result + embedding(ids)
        return result


class TextConditionEmbedder(nn.Module):
    """Project cached embeddings from a frozen text encoder into DiT width."""

    def __init__(self, input_size: int, hidden_size: int, dropout_prob: float) -> None:
        super().__init__()
        self.input_size = input_size
        self.dropout_prob = dropout_prob
        self.projection = nn.Linear(input_size, hidden_size)
        self.null_embedding = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self,
        embeddings: torch.Tensor,
        train: bool,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.input_size:
            raise ValueError(
                f"text embeddings must have shape [batch, {self.input_size}]"
            )
        batch = embeddings.shape[0]
        if force_drop_ids is not None and force_drop_ids.shape != (batch,):
            raise ValueError("force_drop_ids must have shape [batch]")
        if force_drop_ids is None:
            drop = (
                torch.rand(batch, device=embeddings.device) < self.dropout_prob
                if train and self.dropout_prob > 0
                else torch.zeros(batch, device=embeddings.device, dtype=torch.bool)
            )
        else:
            drop = force_drop_ids.bool()
        projected = self.projection(embeddings.to(self.projection.weight.dtype))
        return torch.where(drop[:, None], self.null_embedding[None], projected)


def modulate(
    tokens: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return tokens * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AxisBlock(nn.Module):
    """One complete adaLN-Zero Transformer block assigned to one attention axis."""

    def __init__(
        self,
        axis: Literal["spatial", "temporal"],
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.axis = axis
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size)
        )

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(condition).chunk(6, dim=1)
        )
        normalized = modulate(self.norm1(tokens), shift_attn, scale_attn)
        attended = self.attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        tokens = tokens + gate_attn.unsqueeze(1) * attended
        normalized = modulate(self.norm2(tokens), shift_mlp, scale_mlp)
        return tokens + gate_mlp.unsqueeze(1) * self.mlp(normalized)


class FinalLayer(nn.Module):
    def __init__(self, config: VideoDiTConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(
            config.hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(config.hidden_size, 2 * config.hidden_size)
        )
        self.linear = nn.Linear(
            config.hidden_size,
            config.patch_size**2 * config.out_channels,
        )

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
        return self.linear(modulate(self.norm(tokens), shift, scale))


class VideoDiT(nn.Module):
    """Latte Variant-1 Video DiT accepting ``[B,C,T,H,W]`` videos."""

    def __init__(self, config: VideoDiTConfig = VideoDiTConfig()) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.register_buffer(
            "spatial_pos_embed",
            get_2d_sincos_pos_embed(config.hidden_size, config.grid_size),
            persistent=False,
        )
        self.register_buffer(
            "temporal_pos_embed",
            get_1d_sincos_pos_embed(config.hidden_size, config.frames),
            persistent=False,
        )
        self.timestep_embedder = TimestepEmbedder(config.hidden_size)
        if config.condition_mode == "structured":
            self.condition_embedder = StructuredConditionEmbedder(
                config.condition_sizes,
                config.hidden_size,
                config.condition_dropout_prob,
            )
        else:
            self.condition_embedder = TextConditionEmbedder(
                config.text_embedding_size or 0,
                config.hidden_size,
                config.condition_dropout_prob,
            )
        blocks: list[AxisBlock] = []
        for _ in range(config.num_pairs):
            blocks.append(
                AxisBlock(
                    "spatial", config.hidden_size, config.num_heads, config.mlp_ratio
                )
            )
            blocks.append(
                AxisBlock(
                    "temporal", config.hidden_size, config.num_heads, config.mlp_ratio
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.final_layer = FinalLayer(config)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(basic_init)
        nn.init.xavier_uniform_(self.patch_embed.weight.view(self.patch_embed.weight.shape[0], -1))
        nn.init.zeros_(self.patch_embed.bias)
        if isinstance(self.condition_embedder, StructuredConditionEmbedder):
            for embedding in self.condition_embedder.embeddings:
                nn.init.normal_(embedding.weight, std=0.02)
        else:
            nn.init.normal_(self.condition_embedder.projection.weight, std=0.02)
            nn.init.zeros_(self.condition_embedder.projection.bias)
        nn.init.normal_(self.timestep_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[2].weight, std=0.02)

        # adaLN-Zero makes every residual block an identity map at initialization.
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def _patchify(self, video: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = video.shape
        images = video.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        return tokens.reshape(batch, frames, self.config.num_patches, self.config.hidden_size)

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, frames, patches, token_size = tokens.shape
        if patches != self.config.num_patches:
            raise ValueError("token count does not match configured patch grid")
        patch = self.config.patch_size
        channels = self.config.out_channels
        if token_size != patch * patch * channels:
            raise ValueError("token size does not match output patch dimensions")
        grid = self.config.grid_size
        video = tokens.reshape(batch, frames, grid, grid, patch, patch, channels)
        video = torch.einsum("btijpqk->bktipjq", video)
        return video.reshape(
            batch,
            channels,
            frames,
            self.config.image_size,
            self.config.image_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition_ids: torch.Tensor,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = x.shape[0]
        expected = (
            batch,
            self.config.in_channels,
            self.config.frames,
            self.config.image_size,
            self.config.image_size,
        )
        if tuple(x.shape) != expected:
            raise ValueError(f"x must have shape {expected}")
        if timesteps.shape != (batch,):
            raise ValueError("timesteps must have shape [batch]")
        if self.config.condition_mode == "structured":
            if condition_ids.shape != (batch, len(self.config.condition_sizes)):
                raise ValueError(
                    f"condition_ids must have shape {(batch, len(self.config.condition_sizes))}"
                )
        elif condition_ids.shape != (batch, self.config.text_embedding_size):
            raise ValueError(
                f"text embeddings must have shape {(batch, self.config.text_embedding_size)}"
            )

        condition = self.timestep_embedder(timesteps) + self.condition_embedder(
            condition_ids, self.training, force_drop_ids
        )
        tokens = self._patchify(x) + self.spatial_pos_embed[:, None]
        temporal_position_added = False
        for block in self.blocks:
            if block.axis == "spatial":
                spatial = tokens.reshape(
                    batch * self.config.frames,
                    self.config.num_patches,
                    self.config.hidden_size,
                )
                spatial_condition = condition.repeat_interleave(
                    self.config.frames, dim=0
                )
                tokens = block(spatial, spatial_condition).reshape(
                    batch,
                    self.config.frames,
                    self.config.num_patches,
                    self.config.hidden_size,
                )
            else:
                if not temporal_position_added:
                    tokens = tokens + self.temporal_pos_embed[:, :, None, :]
                    temporal_position_added = True
                temporal = tokens.permute(0, 2, 1, 3).reshape(
                    batch * self.config.num_patches,
                    self.config.frames,
                    self.config.hidden_size,
                )
                temporal_condition = condition.repeat_interleave(
                    self.config.num_patches, dim=0
                )
                tokens = block(temporal, temporal_condition).reshape(
                    batch,
                    self.config.num_patches,
                    self.config.frames,
                    self.config.hidden_size,
                ).permute(0, 2, 1, 3)

        flat_tokens = tokens.reshape(
            batch * self.config.frames,
            self.config.num_patches,
            self.config.hidden_size,
        )
        flat_condition = condition.repeat_interleave(self.config.frames, dim=0)
        output = self.final_layer(flat_tokens, flat_condition).reshape(
            batch,
            self.config.frames,
            self.config.num_patches,
            -1,
        )
        return self._unpatchify(output)

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition_ids: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """Guide epsilon only, keeping conditional variance like official DiT."""

        if x.shape[0] % 2:
            raise ValueError("classifier-free guidance requires an even batch size")
        half = x.shape[0] // 2
        paired_x = torch.cat([x[:half], x[:half]], dim=0)
        paired_timesteps = torch.cat([timesteps[:half], timesteps[:half]], dim=0)
        force_drop = torch.cat(
            [
                torch.zeros(half, device=x.device, dtype=torch.bool),
                torch.ones(half, device=x.device, dtype=torch.bool),
            ]
        )
        prediction = self.forward(
            paired_x, paired_timesteps, condition_ids, force_drop
        )
        conditional, unconditional = prediction.chunk(2, dim=0)
        channels = self.config.in_channels
        conditional_epsilon = conditional[:, :channels]
        unconditional_epsilon = unconditional[:, :channels]
        guided_epsilon = unconditional_epsilon + cfg_scale * (
            conditional_epsilon - unconditional_epsilon
        )
        guided = (
            torch.cat([guided_epsilon, conditional[:, channels:]], dim=1)
            if self.config.learn_sigma
            else guided_epsilon
        )
        return torch.cat([guided, guided], dim=0)


def video_dit_tiny(**overrides: object) -> VideoDiT:
    return VideoDiT(VideoDiTConfig(**overrides))
