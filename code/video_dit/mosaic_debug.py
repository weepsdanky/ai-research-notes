"""Fast, checkpoint-bound probes for the Video DiT moving-mosaic failure."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Iterable

import torch

from diffusion import DiffusionSchedule, ddim_sample_loop
from real_data import RealVideoDataset
from video_dit import VideoDiT, VideoDiTConfig


@torch.inference_mode()
def direct_x0_probe(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    clean: torch.Tensor,
    conditions: torch.Tensor,
    *,
    timesteps: Iterable[int],
    prediction_type: str,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Measure target and reconstructed-x0 error without a sampling trajectory."""

    if prediction_type not in {"epsilon", "velocity"}:
        raise ValueError("prediction_type must be epsilon or velocity")
    rows: dict[str, dict[str, float]] = {}
    for timestep in timesteps:
        if not 0 <= timestep < schedule.timesteps:
            raise ValueError("probe timestep must be inside the diffusion schedule")
        generator = torch.Generator().manual_seed(seed + timestep)
        noise = torch.randn(clean.shape, generator=generator).to(clean.device)
        batch_timesteps = torch.full(
            (clean.shape[0],), timestep, device=clean.device, dtype=torch.long,
        )
        noisy, _ = schedule.q_sample(clean, batch_timesteps, noise)
        prediction = model(noisy, batch_timesteps, conditions)[:, : clean.shape[1]]
        if prediction_type == "epsilon":
            target = noise
            predicted_clean = schedule.predict_x0_from_epsilon(
                noisy, batch_timesteps, prediction,
            )
        else:
            target = schedule.velocity_target(clean, noise, batch_timesteps)
            predicted_clean, _ = schedule.clean_and_epsilon_from_velocity(
                noisy, prediction, batch_timesteps,
            )
        if not torch.isfinite(prediction).all() or not torch.isfinite(predicted_clean).all():
            raise ValueError("checkpoint produced non-finite diagnostic values")
        rows[str(timestep)] = {
            "target_mse": float((prediction - target).square().mean()),
            "x0_mse": float((predicted_clean - clean).square().mean()),
            "x0_std": float(predicted_clean.std()),
            "noisy_mse": float((noisy - clean).square().mean()),
        }
    return rows


@torch.inference_mode()
def partial_denoising_probe(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    clean: torch.Tensor,
    condition: torch.Tensor,
    *,
    timesteps: Iterable[int],
    prediction_type: str,
    sampling_steps: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Start from q(x_t|x0), run DDIM to zero, and compare with the known x0."""

    if clean.shape[0] != 1 or condition.shape[0] != 1:
        raise ValueError("partial denoising probe requires exactly one sample")
    rows: dict[str, dict[str, float]] = {}
    for timestep in timesteps:
        if not 0 <= timestep < schedule.timesteps:
            raise ValueError("probe timestep must be inside the diffusion schedule")
        generator = torch.Generator().manual_seed(seed + timestep)
        noise = torch.randn(clean.shape, generator=generator).to(clean.device)
        batch_timestep = torch.tensor([timestep], device=clean.device)
        noisy, _ = schedule.q_sample(clean, batch_timestep, noise)
        recovered = ddim_sample_loop(
            model,
            schedule,
            shape=tuple(clean.shape),
            condition_ids=condition,
            device=clean.device,
            cfg_scale=1.0,
            initial_noise=noisy,
            sampling_steps=min(sampling_steps, timestep + 1),
            prediction_type=prediction_type,
            clip_denoised=False,
            start_timestep=timestep,
        )
        rows[str(timestep)] = {
            "reconstruction_mse": float((recovered - clean).square().mean()),
            "recovered_std": float(recovered.std()),
        }
    return rows


def _parse_ints(value: str) -> list[int]:
    parsed = [int(item) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("at least one integer is required")
    return parsed


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--timesteps", default="0,100,500,900,999")
    parser.add_argument("--partial-timesteps", default="100,500,900,999")
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.samples < 1 or args.sampling_steps < 1 or args.seed < 0:
        raise ValueError("samples/sampling-steps must be positive and seed non-negative")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path in {checkpoint_path, manifest_path}:
        raise ValueError("output must not overwrite the checkpoint or manifest")
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint = torch.load(
        io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True,
    )
    manifest_bytes = manifest_path.read_bytes()
    data_contract = checkpoint.get("data_contract", {})
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if data_contract.get("manifest_sha256") != manifest_sha256:
        raise ValueError("manifest does not match checkpoint data contract")
    config = VideoDiTConfig(**checkpoint["config"])
    device = torch.device(args.device)
    schedule = DiffusionSchedule(**checkpoint["diffusion"]).to(device)
    dataset = RealVideoDataset(
        manifest_path,
        split="train",
        frames=config.frames,
        size=config.image_size,
        space=str(data_contract.get("space", "pixel")),
    )
    count = min(args.samples, len(dataset))
    items = [dataset[index] for index in range(count)]
    clean = torch.stack([item["video"] for item in items]).to(device)
    conditions = torch.stack([item["condition"] for item in items]).to(device)
    prediction_type = str(checkpoint.get("prediction_type", "epsilon"))
    report: dict[str, object] = {
        "protocol": "Video DiT moving-mosaic checkpoint probe v1",
        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "manifest_sha256": manifest_sha256,
        "prediction_type": prediction_type,
        "samples": count,
        "clean_mean": float(clean.mean()),
        "clean_std": float(clean.std()),
        "token_grid": config.grid_size,
        "tokens_per_frame": config.num_patches,
        "sampling_steps": args.sampling_steps,
        "weights": {},
    }
    for weights in ("model", "ema"):
        if weights not in checkpoint:
            continue
        model = VideoDiT(config)
        model.load_state_dict(checkpoint[weights])
        model.to(device).eval().requires_grad_(False)
        report["weights"][weights] = {
            "direct_x0": direct_x0_probe(
                model,
                schedule,
                clean,
                conditions,
                timesteps=_parse_ints(args.timesteps),
                prediction_type=prediction_type,
                seed=args.seed,
            ),
            "partial_denoising": partial_denoising_probe(
                model,
                schedule,
                clean[:1],
                conditions[:1],
                timesteps=_parse_ints(args.partial_timesteps),
                prediction_type=prediction_type,
                sampling_steps=args.sampling_steps,
                seed=args.seed,
            ),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return output_path


if __name__ == "__main__":
    print(main())
