"""Train a class-conditional image DiT on MNIST, FashionMNIST, or CIFAR-10.

Fast verification without downloading data:
    python train_image_dit.py --dry-run

Real training:
    python train_image_dit.py --dataset mnist --epochs 50 --batch-size 128
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
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, TensorDataset

from diffusion import DiffusionSchedule, ddim_sample_loop, diffusion_loss, sample_loop
from dit import DiT, DiTConfig
from profiles import DiffusionProfile, PROFILES, get_profile


class DatasetSpec(NamedTuple):
    image_size: int
    channels: int
    class_names: tuple[str, ...]
    horizontal_flip: bool


DATASET_SPECS = {
    "mnist": DatasetSpec(28, 1, tuple(str(i) for i in range(10)), False),
    "fashion_mnist": DatasetSpec(
        28,
        1,
        (
            "t-shirt", "trouser", "pullover", "dress", "coat",
            "sandal", "shirt", "sneaker", "bag", "ankle boot",
        ),
        False,
    ),
    "cifar10": DatasetSpec(
        32,
        3,
        (
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ),
        True,
    ),
}


class BalancedClassBatchSampler(BatchSampler):
    """Yield class-balanced batches, useful when class conditioning must be visible.

    DiT's paper recipe samples ImageNet normally because the dataset is already
    large and approximately balanced.  On a tiny teaching run, explicitly
    balancing each batch gives every class embedding equally frequent updates
    and makes CFG easier to verify without changing the diffusion objective.
    """

    def __init__(
        self,
        labels: torch.Tensor | list[int],
        batch_size: int,
        seed: int = 0,
    ) -> None:
        labels = torch.as_tensor(labels, dtype=torch.long)
        classes = torch.unique(labels, sorted=True)
        if classes.numel() < 2:
            raise ValueError("balanced sampling requires at least two classes")
        if batch_size < classes.numel() or batch_size % classes.numel():
            raise ValueError("batch_size must be a multiple of the class count")
        self.indices_by_class = [
            torch.where(labels == class_id)[0] for class_id in classes
        ]
        if any(indices.numel() == 0 for indices in self.indices_by_class):
            raise ValueError("every class must contain at least one sample")
        self.batch_size = batch_size
        self.per_class = batch_size // classes.numel()
        self.num_batches = labels.numel() // batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        pools = [
            indices[torch.randperm(indices.numel(), generator=generator)]
            for indices in self.indices_by_class
        ]
        offsets = [0] * len(pools)
        for _ in range(self.num_batches):
            batch: list[int] = []
            for class_index, pool in enumerate(pools):
                for _ in range(self.per_class):
                    offset = offsets[class_index]
                    if offset >= pool.numel():
                        pool = self.indices_by_class[class_index][
                            torch.randperm(
                                self.indices_by_class[class_index].numel(),
                                generator=generator,
                            )
                        ]
                        pools[class_index] = pool
                        offset = 0
                    batch.append(int(pool[offset]))
                    offsets[class_index] = offset + 1
            permutation = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[index] for index in permutation]


class CanonicalClassDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Repeat the first example of every class for an explicit overfit check."""

    def __init__(self, dataset: Dataset, num_classes: int, length: int = 60_000):
        if num_classes <= 1 or length < num_classes:
            raise ValueError("canonical dataset requires multiple classes and one cycle")
        prototypes: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for image, label in dataset:
            class_id = int(label)
            if 0 <= class_id < num_classes and class_id not in prototypes:
                prototypes[class_id] = (image, torch.tensor(class_id, dtype=torch.long))
            if len(prototypes) == num_classes:
                break
        if len(prototypes) != num_classes:
            raise ValueError("dataset does not contain every configured class")
        self.prototypes = tuple(prototypes[index] for index in range(num_classes))
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= index < self.length:
            raise IndexError(index)
        return self.prototypes[index % len(self.prototypes)]


def arg(args: argparse.Namespace, name: str, default: object) -> object:
    """Read a CLI option while keeping programmatic callers backward compatible."""

    return getattr(args, name, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="pixel_teaching")
    parser.add_argument("--dataset", choices=tuple(DATASET_SPECS), default="cifar10")
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
    parser.add_argument(
        "--max-grad-norm", type=float, default=None,
        help="override the profile; paper_compatible disables clipping",
    )
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument(
        "--prediction-type",
        choices=("epsilon", "velocity"),
        default="epsilon",
        help="denoising target ablation; the original DiT recipe uses epsilon",
    )
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--learn-sigma", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-dropout-prob", type=float, default=0.1)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--balanced-batches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="put an equal number of every class in each batch (teaching ablation)",
    )
    parser.add_argument(
        "--canonical-overfit",
        action="store_true",
        help="repeat one fixed training example per class to verify generation end to end",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="0 means no step limit")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=None,
        help="override the profile; paper_compatible disables AMP",
    )
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
    dataset_name = str(arg(args, "dataset", "cifar10"))
    spec = DATASET_SPECS[dataset_name]
    if args.dry_run:
        images = torch.rand(
            args.batch_size,
            spec.channels,
            spec.image_size,
            spec.image_size,
            generator=generator,
        ) * 2 - 1
        labels = torch.randint(0, 10, (args.batch_size,), generator=generator)
        return DataLoader(TensorDataset(images, labels), batch_size=args.batch_size)

    from torchvision import datasets, transforms

    operations: list[object] = []
    if spec.horizontal_flip:
        operations.append(transforms.RandomHorizontalFlip())
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * spec.channels, (0.5,) * spec.channels),
        ]
    )
    transform = transforms.Compose(operations)
    data_dir = Path(arg(args, "data_dir", Path("data")))
    image_folder = data_dir / "cifar10" / "train"
    if dataset_name == "cifar10" and image_folder.is_dir():
        dataset = datasets.ImageFolder(image_folder, transform=transform)
        if tuple(dataset.classes) != spec.class_names:
            raise ValueError("ImageFolder class order does not match CIFAR-10 labels")
    elif dataset_name == "mnist":
        dataset = datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
    elif dataset_name == "fashion_mnist":
        dataset = datasets.FashionMNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
    else:
        dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )
    if bool(arg(args, "canonical_overfit", False)):
        dataset = CanonicalClassDataset(
            dataset,
            num_classes=len(spec.class_names),
            length=max(60_000, args.batch_size),
        )
    loader_options = {
        "dataset": dataset,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "generator": generator,
    }
    if bool(arg(args, "balanced_batches", False)):
        labels = getattr(dataset, "targets", None)
        if labels is None:
            labels = [label for _, label in dataset]
        return DataLoader(
            **loader_options,
            batch_sampler=BalancedClassBatchSampler(
                labels, args.batch_size, args.seed
            ),
        )
    return DataLoader(
        **loader_options,
        batch_size=args.batch_size,
        shuffle=True,
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
    dataset: str = "cifar10",
    class_names: tuple[str, ...] = DATASET_SPECS["cifar10"].class_names,
    profile: DiffusionProfile = PROFILES["pixel_teaching"],
    prediction_type: str = "epsilon",
    data_policy: dict[str, object] | None = None,
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
            "dataset": dataset,
            "class_names": class_names,
            "profile": profile.name,
            "prediction_type": prediction_type,
            "data_policy": data_policy or {
                "curriculum": "full_dataset",
                "balanced_batches": False,
            },
            "recipe": profile.metadata(),
            "pipeline": {
                "space": "pixel",
                "vae_id": None,
                "vae_scaling": None,
            },
        },
        path,
    )


def save_inference_checkpoint(
    path: Path,
    model: DiT,
    ema: DiT,
    config: DiTConfig,
    schedule: DiffusionSchedule,
    epoch: int,
    step: int,
    dataset: str,
    class_names: tuple[str, ...],
    profile: DiffusionProfile,
    prediction_type: str,
    data_policy: dict[str, object],
) -> None:
    """Write a compact checkpoint accepted by safe weights-only inference."""

    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "config": asdict(config),
            "diffusion": {
                "timesteps": schedule.timesteps,
                "beta_start": schedule.beta_start,
                "beta_end": schedule.beta_end,
            },
            "epoch": epoch,
            "step": step,
            "dataset": dataset,
            "class_names": class_names,
            "profile": profile.name,
            "prediction_type": prediction_type,
            "data_policy": data_policy,
            "recipe": profile.metadata(),
            "pipeline": {
                "space": "pixel",
                "vae_id": None,
                "vae_scaling": None,
            },
        },
        path,
    )


def save_samples(
    model: DiT,
    schedule: DiffusionSchedule,
    output_path: Path,
    device: torch.device,
    cfg_scale: float,
    config: DiTConfig,
    profile: DiffusionProfile = PROFILES["pixel_teaching"],
    prediction_type: str = "epsilon",
) -> None:
    from torchvision.utils import save_image

    # CFG evaluates ten conditional samples followed by ten null-condition copies.
    class_count = config.num_classes
    labels = torch.arange(class_count, device=device).repeat(2)
    generator = torch.Generator(device="cpu").manual_seed(0)
    initial_noise = torch.randn(
        class_count,
        config.in_channels,
        config.image_size,
        config.image_size,
        generator=generator,
    ).to(device)
    initial_noise = initial_noise.repeat(2, 1, 1, 1)
    common = dict(
        model=model,
        schedule=schedule,
        shape=(2 * class_count, config.in_channels, config.image_size, config.image_size),
        labels=labels,
        device=device,
        cfg_scale=cfg_scale,
        initial_noise=initial_noise,
        clip_denoised=profile.sampling_clip_denoised,
        prediction_type=prediction_type,
    )
    steps = min(profile.sampling_steps, schedule.timesteps)
    if profile.sampler == "paper_ddpm":
        samples = sample_loop(**common, sampling_steps=steps)
    else:
        samples = ddim_sample_loop(**common, sampling_steps=steps)
    save_image(
        samples[:class_count], output_path, nrow=class_count,
        normalize=True, value_range=(-1, 1),
    )


def train(args: argparse.Namespace) -> dict[str, object]:
    profile = get_profile(str(arg(args, "profile", "pixel_teaching")))
    prediction_type = str(arg(args, "prediction_type", "epsilon"))
    if prediction_type not in {"epsilon", "velocity"}:
        raise ValueError("prediction_type must be 'epsilon' or 'velocity'")
    if profile.name == "paper_compatible" and prediction_type != "epsilon":
        raise ValueError("paper_compatible requires epsilon prediction")
    max_grad_norm_arg = arg(args, "max_grad_norm", None)
    max_grad_norm = (
        profile.max_grad_norm
        if max_grad_norm_arg is None
        else float(max_grad_norm_arg)
    )
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
    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive or omitted")
    if int(arg(args, "max_steps", 0)) < 0:
        raise ValueError("max_steps must be non-negative")
    seed_everything(args.seed)
    device = choose_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = str(arg(args, "dataset", "cifar10"))
    spec = DATASET_SPECS[dataset_name]
    data_policy: dict[str, object] = {
        "curriculum": (
            "canonical_overfit"
            if bool(arg(args, "canonical_overfit", False))
            else "full_dataset"
        ),
        "balanced_batches": bool(arg(args, "balanced_batches", False)),
    }
    config = DiTConfig(
        image_size=spec.image_size,
        patch_size=args.patch_size,
        in_channels=spec.channels,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        class_dropout_prob=float(arg(args, "class_dropout_prob", 0.1)),
        learn_sigma=bool(arg(args, "learn_sigma", True)),
        cfg_channels=profile.cfg_channels,
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
    amp_arg = arg(args, "amp", None)
    amp_enabled = profile.amp if amp_arg is None else bool(amp_arg)
    use_amp = amp_enabled and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float | int]] = []
    global_step = 0
    start_epoch = 0
    resume = arg(args, "resume", None)
    resumed_from = None
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        checkpoint_diffusion = checkpoint.get("diffusion", {})
        current_diffusion = {
            "timesteps": schedule.timesteps,
            "beta_start": schedule.beta_start,
            "beta_end": schedule.beta_end,
        }
        if checkpoint_diffusion != current_diffusion:
            raise ValueError("resume checkpoint diffusion schedule does not match CLI")
        checkpoint_config = dict(checkpoint["config"])
        checkpoint_config.setdefault("cfg_channels", "all")
        if checkpoint_config != asdict(config):
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
        if checkpoint.get("dataset", dataset_name) != dataset_name:
            raise ValueError("resume checkpoint dataset does not match --dataset")
        checkpoint_prediction_type = checkpoint.get("prediction_type", "epsilon")
        if checkpoint_prediction_type != prediction_type:
            raise ValueError("resume checkpoint prediction type does not match CLI")
        checkpoint_data_policy = checkpoint.get(
            "data_policy",
            {"curriculum": "full_dataset", "balanced_batches": False},
        )
        if checkpoint_data_policy != data_policy:
            raise ValueError("resume checkpoint data policy does not match CLI")
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
        if hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(epoch)
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
                    model,
                    schedule,
                    x0,
                    t,
                    labels,
                    vb_mode=profile.vb_mode,
                    clip_denoised=profile.loss_clip_denoised,
                    prediction_type=prediction_type,
                    return_metrics=True,
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
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_grad_norm)
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
                    ema, schedule, args.output_dir / f"step_{global_step:07d}.png",
                    device, args.cfg_scale, config, profile,
                    prediction_type,
                )
            checkpoint_every = int(arg(args, "checkpoint_every", 1000))
            if checkpoint_every and global_step % checkpoint_every == 0:
                save_checkpoint(
                    args.output_dir / f"checkpoint_{global_step:07d}.pt",
                    raw_model, ema, optimizer, config, schedule, epoch, global_step,
                    scaler, dataset_name, spec.class_names, profile,
                    prediction_type, data_policy,
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
                config,
                profile,
                prediction_type,
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
        args.output_dir / "dit_image.pt",
        raw_model,
        ema,
        optimizer,
        config,
        schedule,
        final_epoch,
        global_step,
        scaler,
        dataset_name,
        spec.class_names,
        profile,
        prediction_type,
        data_policy,
    )
    inference_checkpoint_path = args.output_dir / "dit_image_inference.pt"
    save_inference_checkpoint(
        inference_checkpoint_path,
        raw_model,
        ema,
        config,
        schedule,
        final_epoch,
        global_step,
        dataset_name,
        spec.class_names,
        profile,
        prediction_type,
        data_policy,
    )
    summary = {
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "steps": global_step,
        "updates_this_run": global_step - initial_global_step,
        "final_loss": history[-1]["loss"] if history else None,
        "dry_run": args.dry_run,
        "resumed_from": resumed_from,
        "dataset": dataset_name,
        "profile": profile.name,
        "prediction_type": prediction_type,
        "data_policy": data_policy,
        "checkpoint": str(args.output_dir / "dit_image.pt"),
        "inference_checkpoint": str(inference_checkpoint_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
