"""Small Diffusion Transformer building blocks for study and experiments.

The implementation follows the original DiT design:

image -> non-overlapping patch tokens -> Transformer blocks with adaLN-Zero
      -> per-token pixel prediction -> image

It is intentionally compact, but keeps the important architectural seams visible
so each one can be tested independently.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class DiTConfig:
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    hidden_size: int = 128
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    num_classes: int = 10
    class_dropout_prob: float = 0.1
    learn_sigma: bool = False

    def __post_init__(self) -> None:
        if self.image_size <= 0 or self.patch_size <= 0:
            raise ValueError("image_size and patch_size must be positive")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.hidden_size <= 0 or self.depth <= 0 or self.num_heads <= 0:
            raise ValueError("hidden_size, depth, and num_heads must be positive")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not 0.0 <= self.class_dropout_prob <= 1.0:
            raise ValueError("class_dropout_prob must be in [0, 1]")

    @property
    def out_channels(self) -> int:
        return self.in_channels * (2 if self.learn_sigma else 1)

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_size**2


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive LayerNorm modulation to token features."""

    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings, including support for odd dimensions."""

    if t.ndim != 1:
        raise ValueError("t must have shape [batch]")
    half = dim // 2
    if half == 0:
        return torch.zeros((t.shape[0], dim), device=t.device, dtype=torch.float32)
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    """Class embedding with a learned null token for classifier-free guidance."""

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.embedding = nn.Embedding(num_classes + 1, hidden_size)

    def token_drop(
        self,
        labels: torch.Tensor,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if force_drop_ids is None:
            drop = torch.rand(labels.shape, device=labels.device) < self.dropout_prob
        else:
            if force_drop_ids.shape != labels.shape:
                raise ValueError("force_drop_ids must have the same shape as labels")
            drop = force_drop_ids.bool()
        return torch.where(drop, self.num_classes, labels)

    def forward(
        self,
        labels: torch.Tensor,
        train: bool,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        should_drop = (train and self.dropout_prob > 0.0) or force_drop_ids is not None
        if should_drop:
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding(labels)


class PatchEmbed(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.projection = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x).flatten(2).transpose(1, 2)


class DiTBlock(nn.Module):
    """Transformer block conditioned by adaLN-Zero."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
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
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(condition).chunk(6, dim=1)
        )
        attention_input = modulate(self.norm1(x), shift_msa, scale_msa)
        attention_output = self.attention(
            attention_input, attention_input, attention_input, need_weights=False
        )[0]
        x = x + gate_msa.unsqueeze(1) * attention_output
        mlp_input = modulate(self.norm2(x), shift_mlp, scale_mlp)
        return x + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return a fixed 2-D sine/cosine position embedding of shape [1, N, D]."""

    if embed_dim % 4 != 0:
        raise ValueError("hidden_size must be divisible by 4 for 2-D sin/cos positions")
    coordinates = torch.arange(grid_size, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    omega = torch.arange(embed_dim // 4, dtype=torch.float32, device=device)
    omega = 1.0 / (10_000 ** (omega / (embed_dim // 4)))

    def embed(position: torch.Tensor) -> torch.Tensor:
        angles = position.reshape(-1, 1) * omega.reshape(1, -1)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

    return torch.cat([embed(grid_y), embed(grid_x)], dim=1).unsqueeze(0)


class DiT(nn.Module):
    def __init__(self, config: DiTConfig = DiTConfig()):
        super().__init__()
        if config.hidden_size % 4 != 0:
            raise ValueError("hidden_size must be divisible by 4")
        self.config = config
        self.patch_embed = PatchEmbed(config)
        self.register_buffer(
            "pos_embed",
            get_2d_sincos_pos_embed(config.hidden_size, config.grid_size),
            persistent=False,
        )
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.y_embedder = LabelEmbedder(
            config.num_classes, config.hidden_size, config.class_dropout_prob
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(config.hidden_size, config.num_heads, config.mlp_ratio)
                for _ in range(config.depth)
            ]
        )
        self.final_layer = FinalLayer(
            config.hidden_size, config.patch_size, config.out_channels
        )
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(basic_init)
        nn.init.xavier_uniform_(self.patch_embed.projection.weight.view(
            self.patch_embed.projection.weight.shape[0], -1
        ))
        nn.init.constant_(self.patch_embed.projection.bias, 0)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert [B, N, patch^2*C] predictions back to [B, C, H, W]."""

        batch, num_tokens, token_dim = tokens.shape
        grid = int(math.sqrt(num_tokens))
        if grid * grid != num_tokens or grid != self.config.grid_size:
            raise ValueError("token count does not match the configured square patch grid")
        patch = self.config.patch_size
        channels = self.config.out_channels
        if token_dim != patch * patch * channels:
            raise ValueError("token feature size does not match patch_size and channels")
        x = tokens.reshape(batch, grid, grid, patch, patch, channels)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(batch, channels, grid * patch, grid * patch)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected = (
            x.shape[0],
            self.config.in_channels,
            self.config.image_size,
            self.config.image_size,
        )
        if tuple(x.shape) != expected:
            raise ValueError(f"x must have shape {expected}")
        if t.shape != (x.shape[0],) or labels.shape != (x.shape[0],):
            raise ValueError("t and labels must have shape [batch]")
        tokens = self.patch_embed(x) + self.pos_embed
        condition = self.t_embedder(t) + self.y_embedder(
            labels, self.training, force_drop_ids
        )
        for block in self.blocks:
            tokens = block(tokens, condition)
        return self.unpatchify(self.final_layer(tokens, condition))

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """Evaluate conditional/unconditional halves and combine their predictions.

        The first half supplies the conditional labels. The same latent half is used
        for both branches, so their difference measures only the effect of conditioning.
        """

        if x.shape[0] % 2:
            raise ValueError("classifier-free guidance requires an even batch size")
        half = x.shape[0] // 2
        paired_x = torch.cat([x[:half], x[:half]], dim=0)
        paired_t = torch.cat([t[:half], t[:half]], dim=0)
        force_drop = torch.cat(
            [
                torch.zeros(half, device=x.device, dtype=torch.bool),
                torch.ones(half, device=x.device, dtype=torch.bool),
            ]
        )
        prediction = self.forward(paired_x, paired_t, labels, force_drop)
        conditional, unconditional = prediction.chunk(2, dim=0)
        guided = unconditional + cfg_scale * (conditional - unconditional)
        return torch.cat([guided, guided], dim=0)


def dit_tiny(**overrides: object) -> DiT:
    """A laptop-friendly model used by the MNIST learning experiment."""

    return DiT(DiTConfig(**overrides))
