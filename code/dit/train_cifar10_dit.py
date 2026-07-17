"""Train a small class-conditional DiT on CIFAR-10.

Fast verification without downloading data:
    python train_cifar10_dit.py --dry-run

Real training:
    python train_cifar10_dit.py --epochs 100 --batch-size 128
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from diffusion import DiffusionSchedule, diffusion_loss, sample_loop
from dit import DiT, DiTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("samples_dit"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="use synthetic images, two diffusion steps, and one optimizer update",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loader(args: argparse.Namespace) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed)
    if args.dry_run:
        images = torch.rand(args.batch_size, 3, 32, 32, generator=generator) * 2 - 1
        labels = torch.randint(0, 10, (args.batch_size,), generator=generator)
        return DataLoader(TensorDataset(images, labels), batch_size=args.batch_size)

    from torchvision import datasets, transforms

    dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def save_samples(
    model: DiT,
    schedule: DiffusionSchedule,
    output_path: Path,
    device: torch.device,
    cfg_scale: float,
) -> None:
    from torchvision.utils import save_image

    # CFG evaluates ten conditional samples followed by ten null-condition copies.
    labels = torch.arange(10, device=device).repeat(2)
    samples = sample_loop(
        model,
        schedule,
        shape=(20, 3, 32, 32),
        labels=labels,
        device=device,
        cfg_scale=cfg_scale,
    )
    save_image(samples[:10], output_path, nrow=10, normalize=True, value_range=(-1, 1))


def train(args: argparse.Namespace) -> dict[str, object]:
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    seed_everything(args.seed)
    device = choose_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = DiTConfig(
        patch_size=args.patch_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
    )
    model = DiT(config).to(device)
    schedule = DiffusionSchedule(timesteps=2 if args.dry_run else args.timesteps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    loader = build_loader(args)
    history: list[dict[str, float | int]] = []
    global_step = 0
    epochs = 1 if args.dry_run else args.epochs

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        model.train()
        for x0, labels in loader:
            x0, labels = x0.to(device), labels.to(device)
            t = torch.randint(0, schedule.timesteps, (x0.shape[0],), device=device)
            loss = diffusion_loss(model, schedule, x0, t, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            global_step += 1
            if args.dry_run:
                break
        row = {"epoch": epoch + 1, "step": global_step, "loss": total_loss / num_batches}
        history.append(row)
        print(f"epoch={row['epoch']} step={row['step']} loss={row['loss']:.6f}")

        if not args.dry_run and ((epoch + 1) % 5 == 0 or epoch + 1 == epochs):
            save_samples(
                model,
                schedule,
                args.output_dir / f"epoch_{epoch + 1:03d}.png",
                device,
                args.cfg_scale,
            )

    with (args.output_dir / "training_log.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "step", "loss"])
        writer.writeheader()
        writer.writerows(history)
    torch.save(
        {"model": model.state_dict(), "config": asdict(config)},
        args.output_dir / "dit_cifar10.pt",
    )
    summary = {
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "steps": global_step,
        "final_loss": history[-1]["loss"],
        "dry_run": args.dry_run,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
