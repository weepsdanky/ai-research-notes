"""Checkpoint-driven inference for the educational Video DiT.

The MP4 returned by this module advances along *video time*.  Its companion GIF
advances along *diffusion time* and is deliberately labelled as a diagnostic;
the two axes should never be presented as the same kind of animation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any
import uuid

import torch

from dataset import PromptSpec, parse_prompt, render_video
from diffusion import DiffusionSchedule, ddim_sample_loop
from evaluate_video import save_metrics
from video_dit import VideoDiT, VideoDiTConfig
from video_io import save_denoising_gif, save_video_mp4


@dataclass(frozen=True)
class GenerationResult:
    """Artifacts and raw tensor produced by one model sampling request."""

    video: torch.Tensor
    video_path: Path
    denoising_path: Path | None
    metadata_path: Path


def _require_checkpoint_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint {name!r} must be a mapping")
    return value


def _slug(spec: PromptSpec) -> str:
    return re.sub(
        r"[^a-z0-9_]+",
        "-",
        f"{spec.subject}_{spec.color}_{spec.motion}_{spec.scene}",
    ).strip("-")


def _atomic_write_json(destination: Path, payload: dict[str, object]) -> None:
    """Commit metadata with one rename so readers never observe a partial file."""

    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def render_dataset_preview(
    prompt: str,
    output_dir: str | Path,
    *,
    seed: int = 0,
    frames: int = 8,
    size: int = 32,
    fps: float = 8,
) -> GenerationResult:
    """Render a ground-truth curriculum example, never a model prediction."""

    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    spec = parse_prompt(prompt)
    video = render_video(spec, frames=frames, size=size, seed=seed)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    request_identity = json.dumps(
        {
            "prompt": asdict(spec),
            "seed": seed,
            "frames": frames,
            "size": size,
            "fps": fps,
        },
        sort_keys=True,
    ).encode()
    request_hash = hashlib.sha256(request_identity).hexdigest()[:10]
    run_id = uuid.uuid4().hex[:8]
    stem = f"dataset-preview_{_slug(spec)}_seed-{seed}_{request_hash}_{run_id}"
    video_path = save_video_mp4(video, destination / f"{stem}.mp4", fps=fps)
    metadata_path = destination / f"{stem}.json"
    _atomic_write_json(
        metadata_path,
        {
            "mode": "dataset renderer preview (not model output)",
            "prompt": prompt,
            "parsed_prompt": asdict(spec),
            "seed": seed,
            "frames": frames,
            "size": size,
            "fps": fps,
            "artifact_semantics": {
                "video_mp4": "temporal frames rendered by the synthetic curriculum",
                "denoising_gif": "not applicable: no diffusion sampling occurred",
            },
        },
    )
    return GenerationResult(
        video=video,
        video_path=video_path,
        denoising_path=None,
        metadata_path=metadata_path,
    )


class VideoDiTInference:
    """A loaded Video DiT, diffusion schedule, and their sampling contract."""

    def __init__(
        self,
        model: VideoDiT,
        schedule: DiffusionSchedule,
        *,
        device: torch.device,
        checkpoint_path: Path,
        weights_name: str,
        training_step: int | None,
        prediction_type: str = "epsilon",
    ) -> None:
        self.model = model
        self.schedule = schedule
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.weights_name = weights_name
        self.training_step = training_step
        self.prediction_type = prediction_type

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        use_ema: bool = True,
    ) -> "VideoDiTInference":
        """Load a training checkpoint, preferring EMA weights by default."""

        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Video DiT checkpoint not found: {path}. Train the model first "
                "or use the explicitly labelled dataset preview mode."
            )
        resolved_device = torch.device(device)
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise ValueError(
                "checkpoint is not a safe inference checkpoint; export tensor/state "
                "metadata only and never load untrusted pickle checkpoints"
            ) from error
        checkpoint = _require_checkpoint_mapping(checkpoint, "root")
        config_values = _require_checkpoint_mapping(
            checkpoint.get("config"), "config"
        )
        diffusion_values = _require_checkpoint_mapping(
            checkpoint.get("diffusion"), "diffusion"
        )
        config = VideoDiTConfig(**config_values)
        schedule = DiffusionSchedule(**diffusion_values).to(resolved_device)
        model = VideoDiT(config)

        if use_ema and "ema" in checkpoint:
            weights_name = "ema"
        elif "model" in checkpoint:
            weights_name = "model"
        else:
            raise ValueError("checkpoint must contain 'ema' or 'model' weights")
        state_dict = _require_checkpoint_mapping(
            checkpoint[weights_name], weights_name
        )
        model.load_state_dict(state_dict)
        model.to(resolved_device).eval().requires_grad_(False)
        step = checkpoint.get("step")
        return cls(
            model,
            schedule,
            device=resolved_device,
            checkpoint_path=path,
            weights_name=weights_name,
            training_step=int(step) if step is not None else None,
            prediction_type=str(checkpoint.get("prediction_type", "epsilon")),
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        output_dir: str | Path,
        *,
        seed: int = 0,
        sampling_steps: int = 50,
        cfg_scale: float = 2.5,
        trajectory_frames: int = 8,
        fps: float = 8,
    ) -> GenerationResult:
        """Generate one temporal clip plus a separate denoising diagnostic."""

        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if cfg_scale < 0:
            raise ValueError("cfg_scale must be non-negative")
        steps = min(sampling_steps, self.schedule.timesteps)
        if steps < 1:
            raise ValueError("sampling_steps must be positive")
        diagnostics = min(trajectory_frames, steps + 1)
        if diagnostics < 2:
            raise ValueError("trajectory_frames must be at least 2")

        spec = parse_prompt(prompt)
        started = time.perf_counter()
        cfg_was_trained = self.model.config.condition_dropout_prob > 0
        effective_cfg_scale = cfg_scale if cfg_was_trained else 1.0
        guided = effective_cfg_scale != 1.0
        batch = 2 if guided else 1
        condition_ids = spec.condition_ids().unsqueeze(0).repeat(batch, 1)
        condition_ids = condition_ids.to(self.device)

        # Create one CPU noise tensor for stable seed semantics on CPU/MPS/CUDA,
        # then duplicate it for the conditional/unconditional CFG pair.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        config = self.model.config
        single_noise = torch.randn(
            1,
            config.in_channels,
            config.frames,
            config.image_size,
            config.image_size,
            generator=generator,
        ).to(self.device)
        initial_noise = single_noise.repeat(batch, 1, 1, 1, 1)
        sampled, trajectory = ddim_sample_loop(
            self.model,
            self.schedule,
            shape=tuple(initial_noise.shape),
            condition_ids=condition_ids,
            device=self.device,
            cfg_scale=effective_cfg_scale,
            initial_noise=initial_noise,
            sampling_steps=steps,
            eta=0.0,
            trajectory_frames=diagnostics,
            prediction_type=self.prediction_type,
        )
        video = sampled[0].detach().cpu()

        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        request_identity = json.dumps(
            {
                "checkpoint": str(self.checkpoint_path),
                "step": self.training_step,
                "weights": self.weights_name,
                "prompt": asdict(spec),
                "seed": seed,
                "sampling_steps": steps,
                "cfg_scale": effective_cfg_scale,
                "prediction_type": self.prediction_type,
            },
            sort_keys=True,
        ).encode()
        request_hash = hashlib.sha256(request_identity).hexdigest()[:10]
        run_id = uuid.uuid4().hex[:8]
        stem = f"model_{_slug(spec)}_seed-{seed}_{request_hash}_{run_id}"
        video_path = save_video_mp4(video, destination / f"{stem}.mp4", fps=fps)
        denoising_path = save_denoising_gif(
            [state[0] for _, state in trajectory],
            destination / f"{stem}_denoising-diagnostic.gif",
        )
        metadata_path = destination / f"{stem}.json"
        metadata = {
            "mode": "trained Video DiT checkpoint",
            "prompt": prompt,
            "parsed_prompt": asdict(spec),
            "seed": seed,
            "sampling_steps": steps,
            "requested_cfg_scale": cfg_scale,
            "effective_cfg_scale": effective_cfg_scale,
            "cfg_note": (
                "enabled with a conditional/unconditional batch pair"
                if guided
                else (
                    "disabled: checkpoint was trained without condition dropout"
                    if not cfg_was_trained and cfg_scale != 1.0
                    else "disabled: scale is 1.0"
                )
            ),
            "checkpoint": str(self.checkpoint_path),
            "training_step": self.training_step,
            "weights": self.weights_name,
            "prediction_type": self.prediction_type,
            "elapsed_seconds": time.perf_counter() - started,
            "artifact_semantics": {
                "video_mp4": "temporal frames of the generated animation",
                "denoising_gif": "diagnostic diffusion steps, not animation time",
            },
        }
        _atomic_write_json(metadata_path, metadata)
        save_metrics(video, destination / f"{stem}_temporal-metrics.json")
        return GenerationResult(
            video=video,
            video_path=video_path,
            denoising_path=denoising_path,
            metadata_path=metadata_path,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/video_dit/inference"))
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--trajectory-frames", type=int, default=8)
    parser.add_argument("--fps", type=float, default=8)
    parser.add_argument("--no-ema", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> GenerationResult:
    args = parse_args(argv)
    engine = VideoDiTInference.from_checkpoint(
        args.checkpoint,
        device=args.device,
        use_ema=not args.no_ema,
    )
    result = engine.generate(
        args.prompt,
        args.output_dir,
        seed=args.seed,
        sampling_steps=args.sampling_steps,
        cfg_scale=args.cfg_scale,
        trajectory_frames=args.trajectory_frames,
        fps=args.fps,
    )
    print(f"Temporal animation: {result.video_path}")
    print(f"Denoising diagnostic (not animation time): {result.denoising_path}")
    print(f"Metadata: {result.metadata_path}")
    return result


if __name__ == "__main__":
    main()
