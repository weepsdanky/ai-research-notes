"""Checkpoint-driven DiT inference with deterministic trajectory visualization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from diffusion import DiffusionSchedule, ddim_sample_loop, sample_loop
from dit import DiT, DiTConfig
from train_cifar10_dit import choose_device


CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def tensor_to_pil(tensor: torch.Tensor, upscale: int = 8) -> Image.Image:
    image = tensor.detach().float().cpu().clamp(-1, 1)
    image = ((image + 1) * 127.5).round().to(torch.uint8)
    image = image.permute(1, 2, 0).numpy()
    if image.shape[-1] == 1:
        image = image[..., 0]
    result = Image.fromarray(image)
    if upscale > 1:
        result = result.resize(
            (result.width * upscale, result.height * upscale), Image.Resampling.NEAREST
        )
    return result


@dataclass
class GenerationResult:
    image: Image.Image
    trajectory: list[Image.Image]
    timesteps: list[int]


class DiTInference:
    def __init__(
        self,
        model: DiT,
        schedule: DiffusionSchedule,
        device: torch.device,
        metadata: dict[str, object] | None = None,
    ):
        self.model = model.eval()
        self.schedule = schedule
        self.device = device
        self.metadata = metadata or {}

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        use_ema: bool = True,
    ) -> "DiTInference":
        target = torch.device(device) if device is not None else choose_device()
        checkpoint = torch.load(checkpoint_path, map_location=target, weights_only=False)
        config = DiTConfig(**checkpoint["config"])
        model = DiT(config).to(target)
        state = checkpoint.get("ema") if use_ema else checkpoint.get("model")
        model.load_state_dict(state or checkpoint["model"])
        schedule = DiffusionSchedule(**checkpoint.get("diffusion", {})).to(target)
        metadata = {
            "step": int(checkpoint.get("step", 0)),
            "epoch": int(checkpoint.get("epoch", 0)),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "weights": "EMA" if use_ema and checkpoint.get("ema") is not None else "raw",
            "hidden_size": config.hidden_size,
            "depth": config.depth,
            "patch_size": config.patch_size,
            "diffusion_timesteps": schedule.timesteps,
        }
        return cls(model, schedule, target, metadata)

    @torch.inference_mode()
    def generate(
        self,
        class_id: int,
        seed: int = 42,
        cfg_scale: float = 2.0,
        trajectory_frames: int = 12,
        upscale: int = 1,
        sampler: str = "ddim",
        sampling_steps: int = 50,
    ) -> GenerationResult:
        if not 0 <= class_id < self.model.config.num_classes:
            raise ValueError(f"class_id must be in [0, {self.model.config.num_classes - 1}]")
        if trajectory_frames < 2:
            raise ValueError("trajectory_frames must be at least 2")
        if sampling_steps < 1:
            raise ValueError("sampling_steps must be positive")
        if upscale < 1:
            raise ValueError("upscale must be positive")
        # A CPU generator is portable across CPU, CUDA and MPS and keeps UI calls isolated.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(
            1,
            self.model.config.in_channels,
            self.model.config.image_size,
            self.model.config.image_size,
            generator=generator,
        ).to(self.device)
        if cfg_scale != 1.0:
            noise = noise.repeat(2, 1, 1, 1)
            labels = torch.tensor([class_id, class_id], device=self.device)
        else:
            labels = torch.tensor([class_id], device=self.device)
        common = dict(
            model=self.model, schedule=self.schedule, shape=tuple(noise.shape),
            labels=labels, device=self.device, cfg_scale=cfg_scale,
            initial_noise=noise, generator=generator,
        )
        if sampler == "ddim":
            steps = min(sampling_steps, self.schedule.timesteps)
            final, snapshots = ddim_sample_loop(
                **common, sampling_steps=steps,
                trajectory_frames=min(trajectory_frames, steps + 1),
            )
        elif sampler == "ddpm":
            final, snapshots = sample_loop(
                **common,
                trajectory_frames=min(
                    trajectory_frames, self.schedule.timesteps + 1
                ),
            )
        else:
            raise ValueError("sampler must be 'ddim' or 'ddpm'")
        return GenerationResult(
            image=tensor_to_pil(final[0], upscale),
            trajectory=[tensor_to_pil(value[0], upscale) for _, value in snapshots],
            timesteps=[step for step, _ in snapshots],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--class-id", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--trajectory-frames", type=int, default=12)
    parser.add_argument("--sampler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument(
        "--upscale", type=int, default=8,
        help="nearest-neighbor display scale; generation remains native 32x32",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("inference_output"))
    parser.add_argument("--device")
    parser.add_argument(
        "--use-ema", action=argparse.BooleanOptionalAction, default=True,
        help="load EMA weights (default); --no-use-ema loads raw training weights",
    )
    args = parser.parse_args()
    if args.trajectory_frames < 2:
        parser.error("--trajectory-frames must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = DiTInference.from_checkpoint(
        args.checkpoint, args.device, use_ema=args.use_ema
    )
    result = pipeline.generate(
        args.class_id, args.seed, args.cfg_scale, args.trajectory_frames,
        upscale=args.upscale, sampler=args.sampler, sampling_steps=args.sampling_steps,
    )
    result.image.save(args.output_dir / "sample.png")
    result.trajectory[0].save(
        args.output_dir / "denoising.gif",
        save_all=True,
        append_images=result.trajectory[1:],
        duration=350,
        loop=0,
    )
    print(f"saved {args.output_dir / 'sample.png'} and denoising.gif")


if __name__ == "__main__":
    main()
