"""Train the educational Latte-style Video DiT on procedural motion clips.

Quick structural check (one optimizer update, no long training)::

    python train.py --dry-run --device cpu

Mac training can then be started explicitly with ``--device mps``.  The script
uses complete checkpoints and periodic MP4 validation samples so training can be
stopped and resumed without losing the optimizer or EMA state.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import (
    MOTIONS,
    CanonicalVideoDataset,
    SyntheticVideoDataset,
    parse_prompt,
)
from real_data import RealVideoDataset, load_manifest
from diffusion import DiffusionSchedule, ddim_sample_loop, diffusion_loss
from video_dit import VideoDiT, VideoDiTConfig
from video_io import save_denoising_gif, save_video_mp4


VALIDATION_PROMPTS = (
    "an orange cat walking right in the snow",
    "a brown dog walking left on grass",
    "a person in a red coat walking right in the snow",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("concept", "faithful"), default="faithful")
    parser.add_argument(
        "--curriculum",
        choices=("canonical", "factorial"),
        default="canonical",
        help=(
            "canonical repeats three fixed clips for an explicit overfit check; "
            "factorial samples the full compositional prompt space"
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=("synthetic", "real"),
        default="synthetic",
        help="real reads decoded clips and cached frozen text embeddings from --manifest",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--text-embedding-size", type=int)
    parser.add_argument(
        "--text-conditioning", choices=("pooled", "tokens"), default="pooled",
        help="pooled uses adaLN only; tokens adds per-block cross-attention",
    )
    parser.add_argument(
        "--space", choices=("pixel", "latent"), default="pixel",
        help="latent requires cached 4-channel VAE latents in every manifest record",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/video_dit"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--horizontal-flip-prob", type=float, default=0.5)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-pairs", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument(
        "--prediction-type",
        choices=("epsilon", "velocity"),
        default="epsilon",
        help="velocity is a teaching/quality ablation; Latte-faithful uses epsilon",
    )
    parser.add_argument(
        "--learn-sigma",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="defaults off in concept mode and on in faithful mode",
    )
    parser.add_argument(
        "--condition-dropout-prob",
        type=float,
        default=None,
        help="defaults to 0 in concept mode and 0.1 in faithful mode",
    )
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=1_000)
    parser.add_argument("--checkpoint-every", type=int, default=1_000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def horizontal_flip_batch(
    videos: torch.Tensor,
    conditions: torch.Tensor,
    probability: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flip videos while swapping left/right motion labels.

    Applying an image augmentation without adjusting the motion condition would
    teach the model a contradiction, so this transformation owns both tensors.
    """

    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    if videos.ndim != 5 or conditions.ndim != 2:
        raise ValueError("expected videos [B,C,T,H,W] and conditions [B,K]")
    if videos.shape[0] != conditions.shape[0]:
        raise ValueError("videos and conditions must have the same batch size")
    random_values = torch.rand(
        videos.shape[0], generator=generator, device="cpu"
    ).to(videos.device)
    mask = random_values < probability
    result_videos = videos.clone()
    result_videos[mask] = videos[mask].flip(-1)
    result_conditions = conditions.clone()
    right = MOTIONS.index("walk_right")
    left = MOTIONS.index("walk_left")
    motions = result_conditions[:, 2]
    swap_right = mask & (motions == right)
    swap_left = mask & (motions == left)
    motions[swap_right] = left
    motions[swap_left] = right
    return result_videos, result_conditions


@torch.no_grad()
def update_ema(ema: VideoDiT, model: VideoDiT, decay: float) -> None:
    parameters = dict(model.named_parameters())
    for name, ema_parameter in ema.named_parameters():
        ema_parameter.mul_(decay).add_(
            parameters[name].detach(), alpha=1 - decay
        )


def _rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        state["mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if (
        torch.backends.mps.is_available()
        and "mps" in state
        and hasattr(torch.mps, "set_rng_state")
    ):
        torch.mps.set_rng_state(state["mps"].cpu())


def save_checkpoint(
    path: Path,
    model: VideoDiT,
    ema: VideoDiT,
    optimizer: torch.optim.Optimizer,
    config: VideoDiTConfig,
    schedule: DiffusionSchedule,
    step: int,
    mode: str,
    curriculum: str,
    prediction_type: str,
    data_contract: dict[str, object],
    optimization_contract: dict[str, object],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            "step": step,
            "mode": mode,
            "curriculum": curriculum,
            "prediction_type": prediction_type,
            "data_contract": data_contract,
            "optimization_contract": optimization_contract,
            "rng_state": _rng_state(),
        },
        path,
    )
    return path


def save_inference_checkpoint(
    path: Path,
    model: VideoDiT,
    ema: VideoDiT,
    config: VideoDiTConfig,
    schedule: DiffusionSchedule,
    step: int,
    mode: str,
    curriculum: str,
    prediction_type: str,
    data_contract: dict[str, object] | None = None,
) -> Path:
    """Write a weights-only-safe checkpoint without optimizer or RNG objects."""

    path.parent.mkdir(parents=True, exist_ok=True)
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
            "step": step,
            "mode": mode,
            "curriculum": curriculum,
            "prediction_type": prediction_type,
            "data_contract": data_contract or {},
        },
        path,
    )
    return path


@torch.no_grad()
def save_validation_sample(
    model: VideoDiT,
    schedule: DiffusionSchedule,
    output_dir: Path,
    prompt: str,
    step: int,
    device: torch.device,
    cfg_scale: float,
    sampling_steps: int,
    seed: int,
    prediction_type: str = "epsilon",
) -> tuple[Path, Path]:
    """Save the final temporal animation and a separately labelled denoising GIF."""

    config = model.config
    spec = parse_prompt(prompt)
    guided = cfg_scale != 1.0 and config.condition_dropout_prob > 0
    batch = 2 if guided else 1
    condition = spec.condition_ids().unsqueeze(0).repeat(batch, 1).to(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    one_noise = torch.randn(
        1,
        config.in_channels,
        config.frames,
        config.image_size,
        config.image_size,
        generator=generator,
    ).to(device)
    initial_noise = one_noise.repeat(batch, 1, 1, 1, 1)
    sampled, trajectory = ddim_sample_loop(
        model,
        schedule,
        shape=tuple(initial_noise.shape),
        condition_ids=condition,
        device=device,
        cfg_scale=cfg_scale if guided else 1.0,
        initial_noise=initial_noise,
        sampling_steps=min(sampling_steps, schedule.timesteps),
        eta=0.0,
        trajectory_frames=min(8, sampling_steps + 1),
        prediction_type=prediction_type,
    )
    stem = f"step_{step:07d}_{spec.subject}_{spec.motion}_{spec.scene}"
    mp4_path = save_video_mp4(sampled[0], output_dir / f"{stem}.mp4")
    gif_path = save_denoising_gif(
        [state[0] for _, state in trajectory],
        output_dir / f"{stem}_denoising.gif",
    )
    return mp4_path, gif_path


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("max_steps", "batch_size", "log_every"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.sample_every < 0 or args.checkpoint_every < 0:
        raise ValueError("sample-every and checkpoint-every must be non-negative")
    if args.learning_rate <= 0 or args.max_grad_norm <= 0:
        raise ValueError("learning-rate and max-grad-norm must be positive")
    if not 0 <= args.ema_decay < 1:
        raise ValueError("ema-decay must be in [0, 1)")
    if not 0 <= args.horizontal_flip_prob <= 1:
        raise ValueError("horizontal-flip-prob must be in [0, 1]")
    if args.dataset == "real":
        if args.manifest is None or not args.manifest.is_file():
            raise ValueError("--dataset real requires an existing --manifest")
        if args.text_embedding_size is None or args.text_embedding_size <= 0:
            raise ValueError("--dataset real requires a positive --text-embedding-size")
        if args.horizontal_flip_prob != 0:
            raise ValueError(
                "real text training currently requires --horizontal-flip-prob 0"
            )
        if args.sample_every != 0:
            raise ValueError(
                "real text training currently requires --sample-every 0; "
                "use inference.py with the matching CLIP checkpoint"
            )
    elif args.space != "pixel":
        raise ValueError("latent space is only supported with --dataset real")
    effective_learn_sigma = (
        args.mode == "faithful" if args.learn_sigma is None else args.learn_sigma
    )
    if args.space == "latent" and effective_learn_sigma:
        raise ValueError(
            "latent training currently requires --no-learn-sigma because the "
            "implemented variational likelihood is pixel-specific"
        )


def _model_config(args: argparse.Namespace) -> VideoDiTConfig:
    faithful = args.mode == "faithful"
    learn_sigma = faithful if args.learn_sigma is None else args.learn_sigma
    dropout = (
        (0.1 if faithful else 0.0)
        if args.condition_dropout_prob is None
        else args.condition_dropout_prob
    )
    if args.dry_run:
        return VideoDiTConfig(
            frames=4,
            image_size=4 if args.space == "latent" else 32,
            patch_size=2 if args.space == "latent" else 8,
            in_channels=4 if args.space == "latent" else 3,
            hidden_size=32,
            num_heads=4,
            num_pairs=1,
            condition_mode=(
                "text_tokens" if args.dataset == "real" and args.text_conditioning == "tokens"
                else "text" if args.dataset == "real" else "structured"
            ),
            text_embedding_size=(
                args.text_embedding_size if args.dataset == "real" else None
            ),
            condition_dropout_prob=dropout,
            learn_sigma=learn_sigma,
        )
    return VideoDiTConfig(
        frames=args.frames,
        image_size=args.image_size,
        patch_size=args.patch_size,
        in_channels=4 if args.space == "latent" else 3,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_pairs=args.num_pairs,
        mlp_ratio=args.mlp_ratio,
        condition_mode=(
            "text_tokens" if args.dataset == "real" and args.text_conditioning == "tokens"
            else "text" if args.dataset == "real" else "structured"
        ),
        text_embedding_size=(args.text_embedding_size if args.dataset == "real" else None),
        condition_dropout_prob=dropout,
        learn_sigma=learn_sigma,
    )


def train(args: argparse.Namespace) -> dict[str, object]:
    _validate_args(args)
    seed_everything(args.seed)
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _model_config(args)
    curriculum = "real-manifest" if args.dataset == "real" else args.curriculum
    schedule = DiffusionSchedule(
        timesteps=8 if args.dry_run else args.timesteps
    ).to(device)
    model = VideoDiT(config).to(device)
    ema = VideoDiT(config).to(device)
    ema.load_state_dict(model.state_dict())
    ema.eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    global_step = 0
    resumed_from: str | None = None
    data_contract: dict[str, object] = {
        "device_type": device.type,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "horizontal_flip_prob": args.horizontal_flip_prob,
        "num_workers": args.num_workers,
    }
    if args.dataset == "real":
        manifest_records = load_manifest(args.manifest)
        encoder_ids = {
            record.text_encoder_id for record in manifest_records if record.split == "train"
        }
        if None in encoder_ids or len(encoder_ids) != 1:
            raise ValueError(
                "all train records must use one explicit text_encoder_id"
            )
        embedding_kinds = {
            record.text_embedding_kind or "pooled"
            for record in manifest_records if record.split == "train"
        }
        if embedding_kinds != {args.text_conditioning}:
            raise ValueError(
                "manifest text embedding kind does not match --text-conditioning"
            )
        if args.text_conditioning == "tokens" and not next(iter(encoder_ids)).endswith(
            "#tokens-v2"
        ):
            raise ValueError("token training requires the zero-padding #tokens-v2 contract")
        data_contract.update(
            {
                "dataset": "real",
                "space": args.space,
                "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                "text_encoder_id": next(iter(encoder_ids)),
                "text_embedding_kind": args.text_conditioning,
            }
        )
        if args.space == "latent":
            vae_contracts = {
                (record.vae_id, record.vae_scaling)
                for record in manifest_records if record.split == "train"
            }
            if any(None in contract for contract in vae_contracts) or len(vae_contracts) != 1:
                raise ValueError("all latent train records must use one VAE contract")
            vae_id, vae_scaling = next(iter(vae_contracts))
            data_contract.update({"vae_id": vae_id, "vae_scaling": vae_scaling})
    optimization_contract: dict[str, object] = {
        "effective_ema_decay": args.ema_decay if args.mode == "faithful" else 0.0,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
    }
    if args.resume is not None:
        checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        checkpoint_diffusion = checkpoint.get("diffusion", {})
        current_diffusion = {
            "timesteps": schedule.timesteps,
            "beta_start": schedule.beta_start,
            "beta_end": schedule.beta_end,
        }
        if checkpoint_diffusion != current_diffusion:
            raise ValueError("resume checkpoint diffusion schedule does not match CLI")
        if checkpoint["config"] != asdict(config):
            raise ValueError("resume checkpoint config does not match CLI config")
        if checkpoint.get("mode", "faithful") != args.mode:
            raise ValueError("resume checkpoint mode does not match --mode")
        if checkpoint.get("curriculum", "factorial") != curriculum:
            raise ValueError("resume checkpoint curriculum does not match --curriculum")
        if checkpoint.get("prediction_type", "epsilon") != args.prediction_type:
            raise ValueError("resume checkpoint prediction type does not match CLI")
        if checkpoint.get("data_contract") != data_contract:
            raise ValueError("resume checkpoint data contract does not match CLI")
        if checkpoint.get("optimization_contract") != optimization_contract:
            raise ValueError(
                "resume checkpoint optimization contract does not match CLI"
            )
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint.get("ema", checkpoint["model"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint["step"])
        if checkpoint.get("rng_state"):
            _restore_rng_state(checkpoint["rng_state"])
        resumed_from = str(args.resume)
    initial_step = global_step
    target_step = global_step + 1 if args.dry_run else args.max_steps
    # The sample at optimizer step k is always the same dataset index range.  This
    # makes a resumed run consume the exact next videos instead of reshuffling from
    # index zero and is what lets restored RNG state reproduce the next loss.
    if args.dataset == "real":
        dataset = RealVideoDataset(
            args.manifest,
            split="train",
            frames=config.frames,
            size=config.image_size,
            space=args.space,
        )
        remaining_indices = [
            index % len(dataset)
            for index in range(
                global_step * args.batch_size, target_step * args.batch_size
            )
        ]
    else:
        dataset_length = max(args.batch_size, target_step * args.batch_size)
        dataset_class = (
            CanonicalVideoDataset
            if args.curriculum == "canonical"
            else SyntheticVideoDataset
        )
        dataset = dataset_class(
            length=dataset_length,
            frames=config.frames,
            size=config.image_size,
            base_seed=args.seed,
        )
        remaining_indices = range(
            global_step * args.batch_size,
            target_step * args.batch_size,
        )
    # DataLoader creates a worker base seed even with num_workers=0.  Keeping
    # that bookkeeping on a private generator prevents iterator recreation on
    # resume from advancing the training RNG used by augmentation and timesteps.
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        Subset(dataset, remaining_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
        generator=loader_generator,
        persistent_workers=args.num_workers > 0,
    )
    log_path = output_dir / "step_log.csv"
    log_is_empty = not log_path.exists() or log_path.stat().st_size == 0
    log_file = log_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        log_file,
        fieldnames=["step", "loss", "mse", "vb", "samples_per_second"],
    )
    if log_is_empty:
        writer.writeheader()

    started = time.perf_counter()
    samples_seen = 0
    final_metrics = {"loss": math.nan, "mse": math.nan, "vb": math.nan}
    try:
        while global_step < target_step:
            for batch in loader:
                if global_step >= target_step:
                    break
                videos = batch["video"].to(device)
                conditions = batch["condition"].to(device)
                if args.dataset == "synthetic":
                    videos, conditions = horizontal_flip_batch(
                        videos, conditions, args.horizontal_flip_prob
                    )
                timesteps = torch.randint(
                    0,
                    schedule.timesteps,
                    (videos.shape[0],),
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                loss, final_metrics = diffusion_loss(
                    model,
                    schedule,
                    videos,
                    timesteps,
                    conditions,
                    prediction_type=args.prediction_type,
                    return_metrics=True,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
                optimizer.step()
                # Concept mode intentionally removes EMA as a trick while keeping
                # a common checkpoint interface: decay=0 copies raw weights.
                ema_decay = args.ema_decay if args.mode == "faithful" else 0.0
                update_ema(ema, model, ema_decay)
                global_step += 1
                samples_seen += videos.shape[0]
                elapsed = time.perf_counter() - started
                writer.writerow(
                    {
                        "step": global_step,
                        **final_metrics,
                        "samples_per_second": samples_seen / elapsed,
                    }
                )
                log_file.flush()
                if global_step % args.log_every == 0 or args.dry_run:
                    print(
                        f"step={global_step} loss={final_metrics['loss']:.6f} "
                        f"mse={final_metrics['mse']:.6f} "
                        f"vb={final_metrics['vb']:.6f} "
                        f"samples/s={samples_seen / elapsed:.1f} device={device}"
                    )
                if (
                    not args.dry_run
                    and args.sample_every
                    and global_step % args.sample_every == 0
                ):
                    prompt = VALIDATION_PROMPTS[
                        (global_step // args.sample_every - 1)
                        % len(VALIDATION_PROMPTS)
                    ]
                    save_validation_sample(
                        ema,
                        schedule,
                        output_dir,
                        prompt,
                        global_step,
                        device,
                        args.cfg_scale,
                        args.sampling_steps,
                        args.seed,
                        args.prediction_type,
                    )
                if (
                    args.checkpoint_every
                    and global_step % args.checkpoint_every == 0
                ):
                    save_checkpoint(
                        output_dir / f"checkpoint_{global_step:07d}.pt",
                        model,
                        ema,
                        optimizer,
                        config,
                        schedule,
                        global_step,
                        args.mode,
                        curriculum,
                        args.prediction_type,
                        data_contract,
                        optimization_contract,
                    )
    finally:
        log_file.close()

    checkpoint_path = save_checkpoint(
        output_dir / "video_dit.pt",
        model,
        ema,
        optimizer,
        config,
        schedule,
        global_step,
        args.mode,
        curriculum,
        args.prediction_type,
        data_contract,
        optimization_contract,
    )
    inference_checkpoint_path = save_inference_checkpoint(
        output_dir / "video_dit_inference.pt",
        model,
        ema,
        config,
        schedule,
        global_step,
        args.mode,
        curriculum,
        args.prediction_type,
        data_contract,
    )
    summary: dict[str, object] = {
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "steps": global_step,
        "updates_this_run": global_step - initial_step,
        "final_loss": final_metrics["loss"],
        "checkpoint": str(checkpoint_path),
        "inference_checkpoint": str(inference_checkpoint_path),
        "dry_run": args.dry_run,
        "mode": args.mode,
        "curriculum": curriculum,
        "prediction_type": args.prediction_type,
        "data_contract": data_contract,
        "optimization_contract": optimization_contract,
        "resumed_from": resumed_from,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
