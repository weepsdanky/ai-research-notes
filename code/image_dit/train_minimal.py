"""Train the smallest useful DiT: unconditional epsilon-prediction DDPM.

This entry point is intentionally separate from ``train_image_dit.py``.  It
keeps only the conceptual spine:

    image -> patches -> time-conditioned Transformer -> epsilon -> DDPM sample

Use the full trainer after this loop is understood; that is where class
conditioning, CFG, adaLN-Zero, learned variance/VB, EMA, and paper profiles live.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from diffusion import DiffusionSchedule
from minimal_dit import MinimalDiT, MinimalDiTConfig, minimal_diffusion_loss


OMITTED_TRICKS = [
    "class_conditioning_and_cfg",
    "adaln_zero",
    "learned_variance_and_vb",
    "ema",
    "special_initialization",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/minimal"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def choose_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loader(args: argparse.Namespace) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed)
    if args.dry_run:
        images = torch.rand(args.batch_size, 1, 28, 28, generator=generator) * 2 - 1
        return DataLoader(TensorDataset(images), batch_size=args.batch_size)

    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ),
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )


@torch.inference_mode()
def sample_ddpm(
    model: MinimalDiT,
    schedule: DiffusionSchedule,
    device: torch.device,
    *,
    seed: int,
    count: int = 16,
) -> torch.Tensor:
    """The ancestral reverse loop, kept explicit for line-by-line study."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    config = model.config
    x = torch.randn(
        count,
        config.in_channels,
        config.image_size,
        config.image_size,
        generator=generator,
    ).to(device)
    was_training = model.training
    model.eval()
    for step in reversed(range(schedule.timesteps)):
        t = torch.full((count,), step, dtype=torch.long, device=device)
        epsilon = model(x, t)
        x0 = schedule.predict_x0_from_eps(x, t, epsilon).clamp(-1, 1)
        mean, variance = schedule.q_posterior(x0, x, t)
        if step:
            noise = torch.randn(x.shape, generator=generator).to(device)
            x = mean + variance.clamp(min=0).sqrt() * noise
        else:
            x = mean
    model.train(was_training)
    return x


def train(args: argparse.Namespace) -> dict[str, object]:
    if args.epochs <= 0 or args.batch_size <= 0 or args.timesteps < 2:
        raise ValueError("epochs/batch-size must be positive and timesteps at least 2")
    if args.max_steps < 0:
        raise ValueError("max-steps must be non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    timesteps = 2 if args.dry_run else args.timesteps
    config = MinimalDiTConfig(
        image_size=28,
        patch_size=args.patch_size,
        in_channels=1,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
    )
    model = MinimalDiT(config).to(device)
    schedule = DiffusionSchedule(timesteps=timesteps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader = build_loader(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    final_loss: float | None = None
    stop = False
    for _ in range(1 if args.dry_run else args.epochs):
        for batch in loader:
            x0 = batch[0].to(device)
            t = torch.randint(0, timesteps, (x0.shape[0],), device=device)
            loss = minimal_diffusion_loss(model, x0, t, schedule.alpha_bar)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            final_loss = float(loss.detach())
            if args.dry_run or (args.max_steps and step >= args.max_steps):
                stop = True
                break
        if stop:
            break

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "diffusion": {
            "timesteps": schedule.timesteps,
            "beta_start": schedule.beta_start,
            "beta_end": schedule.beta_end,
        },
        "step": step,
        "recipe": "minimal_epsilon_ddpm",
        "omitted_tricks": OMITTED_TRICKS,
    }
    checkpoint_path = args.output_dir / "minimal_dit.pt"
    torch.save(checkpoint, checkpoint_path)
    samples = sample_ddpm(model, schedule, device, seed=args.seed)
    from torchvision.utils import save_image

    save_image(
        samples,
        args.output_dir / "sample.png",
        nrow=4,
        normalize=True,
        value_range=(-1, 1),
    )
    summary = {
        "device": str(device),
        "step": step,
        "final_loss": final_loss,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "dry_run": args.dry_run,
        "checkpoint": str(checkpoint_path),
        "sample": str(args.output_dir / "sample.png"),
        "recipe": checkpoint["recipe"],
        "omitted_tricks": OMITTED_TRICKS,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(train(parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
