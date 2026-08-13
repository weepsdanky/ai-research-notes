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
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from diffusion import DiffusionSchedule, ddim_sample_loop, diffusion_loss
from dit import DiT, DiTConfig


def arg(args: argparse.Namespace, name: str, default: object) -> object:
    """Read a CLI option while keeping programmatic callers backward compatible."""

    return getattr(args, name, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine"),
        default="constant",
        help="official DiT uses constant; cosine is provided as an ablation",
    )
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--learn-sigma", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-dropout-prob", type=float, default=0.1)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means no step limit")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
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

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    data_dir = Path(arg(args, "data_dir", Path("data")))
    image_folder = data_dir / "cifar10" / "train"
    if image_folder.is_dir():
        dataset = datasets.ImageFolder(image_folder, transform=transform)
        expected_classes = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ]
        if dataset.classes != expected_classes:
            raise ValueError("ImageFolder class order does not match CIFAR-10 labels")
    else:
        dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        drop_last=True,
    )


@torch.no_grad()
def update_ema(ema: DiT, model: DiT, decay: float) -> None:
    model_parameters = dict(model.named_parameters())
    for name, parameter in ema.named_parameters():
        parameter.mul_(decay).add_(model_parameters[name].detach(), alpha=1 - decay)


def learning_rate_at_step(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int,
    schedule: str = "constant",
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    if schedule == "constant":
        return base_lr
    if schedule != "cosine":
        raise ValueError("schedule must be 'constant' or 'cosine'")
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def save_checkpoint(
    path: Path,
    model: DiT,
    ema: DiT,
    optimizer: torch.optim.Optimizer,
    config: DiTConfig,
    schedule: DiffusionSchedule,
    epoch: int,
    step: int,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    rng_state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        rng_state["mps"] = torch.mps.get_rng_state()
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "diffusion": {
                "timesteps": schedule.timesteps,
                "beta_start": schedule.beta_start,
                "beta_end": schedule.beta_end,
            },
            "epoch": epoch,
            "step": step,
            "scaler": scaler.state_dict() if scaler is not None else {},
            "rng_state": rng_state,
        },
        path,
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
    generator = torch.Generator(device="cpu").manual_seed(0)
    initial_noise = torch.randn(10, 3, 32, 32, generator=generator).to(device)
    initial_noise = initial_noise.repeat(2, 1, 1, 1)
    samples = ddim_sample_loop(
        model,
        schedule,
        shape=(20, 3, 32, 32),
        labels=labels,
        device=device,
        cfg_scale=cfg_scale,
        initial_noise=initial_noise,
        sampling_steps=min(50, schedule.timesteps),
    )
    save_image(samples[:10], output_path, nrow=10, normalize=True, value_range=(-1, 1))


def train(args: argparse.Namespace) -> dict[str, object]:
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if int(arg(args, "gradient_accumulation", 1)) <= 0:
        raise ValueError("gradient_accumulation must be positive")
    if int(arg(args, "log_every", 50)) <= 0:
        raise ValueError("log_every must be positive")
    if int(arg(args, "warmup_steps", 0)) < 0:
        raise ValueError("warmup_steps must be non-negative")
    if int(arg(args, "sample_every", 1000)) < 0 or int(arg(args, "checkpoint_every", 1000)) < 0:
        raise ValueError("sample_every and checkpoint_every must be non-negative")
    if not 0 <= float(arg(args, "ema_decay", 0.9999)) < 1:
        raise ValueError("ema_decay must be in [0, 1)")
    if float(arg(args, "max_grad_norm", 1.0)) <= 0:
        raise ValueError("max_grad_norm must be positive")
    if int(arg(args, "max_steps", 0)) < 0:
        raise ValueError("max_steps must be non-negative")
    seed_everything(args.seed)
    device = choose_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = DiTConfig(
        patch_size=args.patch_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        class_dropout_prob=float(arg(args, "class_dropout_prob", 0.1)),
        learn_sigma=bool(arg(args, "learn_sigma", False)),
    )
    model = DiT(config).to(device)
    ema = DiT(config).to(device)
    ema.load_state_dict(model.state_dict())
    ema.eval().requires_grad_(False)
    schedule = DiffusionSchedule(timesteps=2 if args.dry_run else args.timesteps).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=float(arg(args, "weight_decay", 0.0)),
    )
    loader = build_loader(args)
    use_amp = bool(arg(args, "amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float | int]] = []
    global_step = 0
    start_epoch = 0
    resume = arg(args, "resume", None)
    resumed_from = None
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        if checkpoint["config"] != asdict(config):
            raise ValueError("resume checkpoint model config does not match CLI config")
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint.get("ema", checkpoint["model"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        rng_state = checkpoint.get("rng_state")
        if rng_state:
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch"].cpu())
            if torch.cuda.is_available() and "cuda" in rng_state:
                torch.cuda.set_rng_state_all(rng_state["cuda"])
            if torch.backends.mps.is_available() and "mps" in rng_state:
                torch.mps.set_rng_state(rng_state["mps"].cpu())
        global_step = int(checkpoint.get("step", 0))
        start_epoch = int(checkpoint.get("epoch", 0))
        resumed_from = str(resume)
    initial_global_step = global_step
    raw_model = model
    if bool(arg(args, "compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)
    epochs = 1 if args.dry_run else args.epochs
    accumulation = int(arg(args, "gradient_accumulation", 1))
    max_steps = int(arg(args, "max_steps", 0))
    requested_steps = epochs * max(1, math.ceil(len(loader) / accumulation))
    total_steps = max_steps or (global_step + requested_steps)
    optimizer.zero_grad(set_to_none=True)
    training_started = time.perf_counter()
    examples_seen = 0
    stop_training = False
    window_metrics = {"loss": 0.0, "mse": 0.0, "vb": 0.0}
    microbatches_in_window = 0
    step_log_path = args.output_dir / "step_log.csv"
    step_log_is_empty = not step_log_path.exists() or step_log_path.stat().st_size == 0
    step_log_file = step_log_path.open("a", newline="")
    step_writer = csv.DictWriter(
        step_log_file,
        fieldnames=["epoch", "step", "loss", "mse", "vb", "learning_rate", "samples_per_second"],
    )
    if step_log_is_empty:
        step_writer.writeheader()

    for local_epoch in range(epochs):
        if max_steps and global_step >= max_steps:
            break
        epoch = start_epoch + local_epoch
        total_loss = 0.0
        num_batches = 0
        model.train()
        for batch_index, (x0, labels) in enumerate(loader):
            if max_steps and global_step >= max_steps:
                stop_training = True
                break
            x0, labels = x0.to(device), labels.to(device)
            examples_seen += x0.shape[0]
            t = torch.randint(0, schedule.timesteps, (x0.shape[0],), device=device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss, metrics = diffusion_loss(
                    model, schedule, x0, t, labels, return_metrics=True
                )
                window_start = (batch_index // accumulation) * accumulation
                window_size = min(accumulation, len(loader) - window_start)
                scaled_loss = loss / window_size
            scaler.scale(scaled_loss).backward()
            for name in window_metrics:
                window_metrics[name] += metrics[name]
            microbatches_in_window += 1
            total_loss += loss.item()
            num_batches += 1
            if (batch_index + 1) % accumulation != 0 and batch_index + 1 != len(loader):
                continue
            next_step = global_step + 1
            lr = learning_rate_at_step(
                next_step,
                total_steps,
                args.learning_rate,
                int(arg(args, "warmup_steps", 0)),
                str(arg(args, "lr_schedule", "constant")),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(), float(arg(args, "max_grad_norm", 1.0))
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step = next_step
            update_ema(ema, raw_model, float(arg(args, "ema_decay", 0.9999)))
            averaged_metrics = {
                name: value / microbatches_in_window
                for name, value in window_metrics.items()
            }
            elapsed = time.perf_counter() - training_started
            step_writer.writerow(
                {
                    "epoch": epoch + 1,
                    "step": global_step,
                    **averaged_metrics,
                    "learning_rate": lr,
                    "samples_per_second": examples_seen / elapsed,
                }
            )
            step_log_file.flush()
            if global_step % int(arg(args, "log_every", 50)) == 0 or args.dry_run:
                print(
                    f"step={global_step} loss={averaged_metrics['loss']:.6f} "
                    f"mse={averaged_metrics['mse']:.6f} vb={averaged_metrics['vb']:.6f} "
                    f"lr={lr:.2e} samples/s={examples_seen / elapsed:.1f}"
                )
            window_metrics = {"loss": 0.0, "mse": 0.0, "vb": 0.0}
            microbatches_in_window = 0
            sample_every = int(arg(args, "sample_every", 1000))
            if not args.dry_run and sample_every and global_step % sample_every == 0:
                save_samples(
                    ema, schedule, args.output_dir / f"step_{global_step:07d}.png", device, args.cfg_scale
                )
            checkpoint_every = int(arg(args, "checkpoint_every", 1000))
            if checkpoint_every and global_step % checkpoint_every == 0:
                save_checkpoint(
                    args.output_dir / f"checkpoint_{global_step:07d}.pt",
                    raw_model, ema, optimizer, config, schedule, epoch, global_step,
                    scaler,
                )
            if args.dry_run:
                break
            if max_steps and global_step >= max_steps:
                stop_training = True
                break
        if num_batches == 0:
            break
        row = {"epoch": epoch + 1, "step": global_step, "loss": total_loss / num_batches}
        history.append(row)
        print(f"epoch={row['epoch']} step={row['step']} loss={row['loss']:.6f}")

        if not args.dry_run and not stop_training and ((local_epoch + 1) % 5 == 0 or local_epoch + 1 == epochs):
            save_samples(
                ema,
                schedule,
                args.output_dir / f"epoch_{epoch + 1:03d}.png",
                device,
                args.cfg_scale,
            )
        if stop_training:
            break

    step_log_file.close()
    epoch_log_path = args.output_dir / "training_log.csv"
    epoch_log_is_empty = not epoch_log_path.exists() or epoch_log_path.stat().st_size == 0
    with epoch_log_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "step", "loss"])
        if epoch_log_is_empty:
            writer.writeheader()
        writer.writerows(history)
    final_epoch = epoch if history else start_epoch
    save_checkpoint(
        args.output_dir / "dit_cifar10.pt",
        raw_model,
        ema,
        optimizer,
        config,
        schedule,
        final_epoch,
        global_step,
        scaler,
    )
    summary = {
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "steps": global_step,
        "updates_this_run": global_step - initial_global_step,
        "final_loss": history[-1]["loss"] if history else None,
        "dry_run": args.dry_run,
        "resumed_from": resumed_from,
        "checkpoint": str(args.output_dir / "dit_cifar10.pt"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
