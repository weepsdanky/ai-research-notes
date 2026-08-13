"""Frozen frame-VAE adapter and auditable latent cache for real videos."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import math
from pathlib import Path
from typing import Protocol

import torch

from real_data import decode_video_clip, load_manifest, write_manifest


class VAEEncoder(Protocol):
    vae_id: str
    scaling_factor: float
    latent_size: int
    latent_channels: int
    def encode(self, frames: torch.Tensor) -> torch.Tensor: ...


class VAEDecoder(Protocol):
    vae_id: str
    scaling_factor: float
    def decode(self, latents: torch.Tensor) -> torch.Tensor: ...


class DiffusersFrameVAE:
    """Pinned Diffusers AutoencoderKL with explicit scaling semantics."""

    def __init__(self, model, *, vae_id: str, device: torch.device) -> None:
        self.model = model.eval().requires_grad_(False).to(device)
        self.vae_id = vae_id
        self.device = device
        self.scaling_factor = float(model.config.scaling_factor)
        self.latent_channels = int(model.config.latent_channels)
        block_count = len(model.config.block_out_channels)
        self.downsample_factor = 2 ** (block_count - 1)
        if self.downsample_factor != 8 or self.latent_channels != 4:
            raise ValueError("video latent pipeline requires a 4-channel KL-f8 VAE")
        self.latent_size = 0

    @classmethod
    def from_pretrained(
        cls, model: str, *, revision: str, device: str | torch.device = "cpu",
    ) -> "DiffusersFrameVAE":
        if not revision:
            raise ValueError("VAE revision must be pinned")
        from diffusers import AutoencoderKL
        loaded = AutoencoderKL.from_pretrained(
            model, revision=revision, use_safetensors=True,
        )
        return cls(loaded, vae_id=f"{model}@{revision}", device=torch.device(device))

    @torch.inference_mode()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames.to(self.device)
        # Posterior mode is deterministic and makes cached latents reproducible.
        latents = self.model.encode(frames).latent_dist.mode()
        return latents.cpu()

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.model.decode(latents.to(self.device)).sample.cpu()


def cache_manifest_latents(
    source: Path,
    output: Path,
    *,
    encoder: VAEEncoder,
    frames: int,
    size: int,
) -> Path:
    if (
        not math.isfinite(encoder.scaling_factor)
        or encoder.scaling_factor <= 0
    ):
        raise ValueError("VAE scaling factor must be positive and finite")
    downsample = getattr(encoder, "downsample_factor", 8)
    if downsample != 8 or encoder.latent_channels != 4:
        raise ValueError("video latent pipeline requires a 4-channel KL-f8 VAE")
    if size % downsample:
        raise ValueError("frame size must be divisible by the KL-f8 VAE factor")
    records = load_manifest(source)
    latent_dir = output.parent / f"{output.stem}_latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    updated = []
    for record in records:
        clip = Path(record.clip_path)
        if not clip.is_absolute():
            clip = source.parent / clip
        actual_sha256 = hashlib.sha256(clip.read_bytes()).hexdigest()
        if actual_sha256.lower() != record.sha256.lower():
            raise ValueError(f"sha256 mismatch for {record.sample_id}")
        video = decode_video_clip(clip, frames=frames, size=size)
        frame_batch = video.permute(1, 0, 2, 3).contiguous()
        raw = encoder.encode(frame_batch)
        expected = (
            frames, encoder.latent_channels,
            size // downsample, size // downsample,
        )
        if tuple(raw.shape) != expected or not torch.isfinite(raw).all():
            raise ValueError(f"VAE encoder must return finite latents {expected}")
        latent = (raw * encoder.scaling_factor).permute(1, 0, 2, 3).contiguous()
        safe_id = hashlib.sha256(record.sample_id.encode()).hexdigest()
        path = latent_dir / f"{safe_id}.pt"
        torch.save(latent.float(), path)
        updated.append(replace(
            record,
            latent_path=str(path.relative_to(output.parent)),
            vae_id=encoder.vae_id,
            vae_scaling=encoder.scaling_factor,
        ))
    return write_manifest(output, updated)


def decode_video_latents(latent: torch.Tensor, decoder: VAEDecoder) -> torch.Tensor:
    if latent.ndim != 4 or latent.shape[0] != 4:
        raise ValueError("latent video must have shape [4, T, H, W]")
    if not math.isfinite(decoder.scaling_factor) or decoder.scaling_factor <= 0:
        raise ValueError("VAE scaling factor must be positive and finite")
    frame_batch = latent.permute(1, 0, 2, 3).contiguous() / decoder.scaling_factor
    decoded = decoder.decode(frame_batch).clamp(-1, 1)
    expected = (latent.shape[1], 3, latent.shape[2] * 8, latent.shape[3] * 8)
    if tuple(decoded.shape) != expected or not torch.isfinite(decoded).all():
        raise ValueError(f"VAE decoder must return finite frames {expected}")
    return decoded.permute(1, 0, 2, 3).contiguous()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=96)
    args = parser.parse_args(argv)
    vae = DiffusersFrameVAE.from_pretrained(
        args.model, revision=args.revision, device=args.device,
    )
    cache_manifest_latents(
        args.source, args.output, encoder=vae, frames=args.frames, size=args.size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
