"""Train the existing Video DiT with linear Conditional Flow Matching.

This is intentionally a parallel entry point to ``train.py``: the backbone,
synthetic/real datasets, cached text conditions, and cached VAE latents stay the
same, while the DDPM schedule/loss is replaced by a continuous flow path.

Quick structural check::

    uv run python train_flow.py --dry-run --device cpu
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from dataset import CanonicalVideoDataset, SyntheticVideoDataset
from flow_matching import (
    GENERATIVE_PROCESS,
    INFERENCE_TIME_GRID,
    PREDICTION_TYPE,
    TIME_CONVENTION,
    FlowMatchingSchedule,
    flow_matching_loss,
)
from real_data import RealVideoDataset, load_manifest
from train import (
    _restore_rng_state,
    _rng_state,
    choose_device,
    horizontal_flip_batch,
    seed_everything,
    update_ema,
)
from video_dit import VideoDiT, VideoDiTConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--curriculum", choices=("canonical", "factorial"), default="canonical")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--space", choices=("pixel", "latent"), default="pixel")
    parser.add_argument("--text-conditioning", choices=("pooled", "tokens"), default="pooled")
    parser.add_argument("--text-embedding-size", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/video_dit_flow"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--condition-dropout-prob", type=float, default=0.1)
    parser.add_argument("--horizontal-flip-prob", type=float, default=0.5)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-pairs", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--time-sampling", choices=("uniform", "logit_normal"), default="uniform")
    parser.add_argument("--time-shift", type=float, default=1.0)
    parser.add_argument("--logit-mean", type=float, default=0.0)
    parser.add_argument("--logit-std", type=float, default=1.0)
    parser.add_argument("--model-time-scale", type=float, default=1_000.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("max_steps", "batch_size", "log_every"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0 or args.max_grad_norm <= 0:
        raise ValueError("learning-rate and max-grad-norm must be positive")
    if not 0 <= args.ema_decay < 1:
        raise ValueError("ema-decay must be in [0, 1)")
    if not 0 <= args.condition_dropout_prob <= 1:
        raise ValueError("condition-dropout-prob must be in [0, 1]")
    if not 0 <= args.horizontal_flip_prob <= 1:
        raise ValueError("horizontal-flip-prob must be in [0, 1]")
    if args.dataset == "real":
        if args.manifest is None or not args.manifest.is_file():
            raise ValueError("--dataset real requires an existing --manifest")
        if args.text_embedding_size is None or args.text_embedding_size <= 0:
            raise ValueError("--dataset real requires --text-embedding-size")
        if args.horizontal_flip_prob != 0:
            raise ValueError("real text training requires --horizontal-flip-prob 0")
    elif args.space != "pixel":
        raise ValueError("latent space is only supported with --dataset real")


def _model_config(args: argparse.Namespace) -> VideoDiTConfig:
    condition_mode = (
        "text_tokens"
        if args.dataset == "real" and args.text_conditioning == "tokens"
        else "text" if args.dataset == "real" else "structured"
    )
    if args.dry_run:
        latent = args.space == "latent"
        return VideoDiTConfig(
            frames=4,
            image_size=4 if latent else 32,
            patch_size=2 if latent else 8,
            in_channels=4 if latent else 3,
            hidden_size=32,
            num_heads=4,
            num_pairs=1,
            condition_mode=condition_mode,
            text_embedding_size=args.text_embedding_size if args.dataset == "real" else None,
            condition_dropout_prob=args.condition_dropout_prob,
            learn_sigma=False,
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
        condition_mode=condition_mode,
        text_embedding_size=args.text_embedding_size if args.dataset == "real" else None,
        condition_dropout_prob=args.condition_dropout_prob,
        learn_sigma=False,
    )


def _data_contract(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    contract: dict[str, object] = {
        "device_type": device.type,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "horizontal_flip_prob": args.horizontal_flip_prob,
        "num_workers": args.num_workers,
    }
    if args.dataset != "real":
        contract.update({"dataset": "synthetic", "curriculum": args.curriculum})
        return contract
    records = [record for record in load_manifest(args.manifest) if record.split == "train"]
    encoder_contracts = {(r.text_encoder_id, r.text_embedding_kind or "pooled") for r in records}
    if len(encoder_contracts) != 1 or None in next(iter(encoder_contracts), (None, None)):
        raise ValueError("all train records must use one explicit text encoder contract")
    encoder_id, embedding_kind = next(iter(encoder_contracts))
    if embedding_kind != args.text_conditioning:
        raise ValueError("manifest text embedding kind does not match CLI")
    contract.update({
        "dataset": "real",
        "space": args.space,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "text_encoder_id": encoder_id,
        "text_embedding_kind": embedding_kind,
    })
    if args.space == "latent":
        vae_contracts = {(r.vae_id, r.vae_scaling) for r in records}
        if len(vae_contracts) != 1 or None in next(iter(vae_contracts), (None, None)):
            raise ValueError("all latent train records must use one VAE contract")
        contract.update(dict(zip(("vae_id", "vae_scaling"), next(iter(vae_contracts)))))
    return contract


def _checkpoint_payload(
    model: VideoDiT,
    ema: VideoDiT,
    config: VideoDiTConfig,
    schedule: FlowMatchingSchedule,
    step: int,
    data_contract: dict[str, object],
    optimization_contract: dict[str, object],
) -> dict[str, object]:
    return {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "config": asdict(config),
        "generative_process": GENERATIVE_PROCESS,
        "time_convention": TIME_CONVENTION,
        "prediction_type": PREDICTION_TYPE,
        "flow": asdict(schedule),
        "inference_defaults": {
            "time_grid": INFERENCE_TIME_GRID,
            "solver": "heun",
            "sampling_steps": 20,
            "sampling_time_shift": 1.0,
        },
        "step": step,
        "data_contract": data_contract,
        "optimization_contract": optimization_contract,
    }


def train(args: argparse.Namespace) -> dict[str, object]:
    _validate_args(args)
    seed_everything(args.seed)
    device = choose_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _model_config(args)
    schedule = FlowMatchingSchedule(
        model_time_scale=args.model_time_scale,
        time_sampling=args.time_sampling,
        time_shift=args.time_shift,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
    )
    data_contract = _data_contract(args, device)
    optimization_contract = {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "ema_decay": args.ema_decay,
    }
    model = VideoDiT(config).to(device)
    ema = VideoDiT(config).to(device)
    ema.load_state_dict(model.state_dict())
    ema.eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    global_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        expected = {
            "config": asdict(config),
            "generative_process": GENERATIVE_PROCESS,
            "time_convention": TIME_CONVENTION,
            "prediction_type": PREDICTION_TYPE,
            "flow": asdict(schedule),
            "data_contract": data_contract,
            "optimization_contract": optimization_contract,
        }
        if any(checkpoint.get(key) != value for key, value in expected.items()):
            raise ValueError("resume checkpoint contract does not match CLI")
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint["step"])
        if checkpoint.get("rng_state"):
            _restore_rng_state(checkpoint["rng_state"])
    initial_step = global_step
    target_step = global_step + 1 if args.dry_run else args.max_steps
    if args.dataset == "real":
        dataset = RealVideoDataset(
            args.manifest, split="train", frames=config.frames,
            size=config.image_size, space=args.space,
        )
        if len(dataset) == 0:
            raise ValueError("real manifest must contain at least one train sample")
        remaining_indices = [
            index % len(dataset)
            for index in range(
                global_step * args.batch_size, target_step * args.batch_size
            )
        ]
    else:
        dataset_type = CanonicalVideoDataset if args.curriculum == "canonical" else SyntheticVideoDataset
        dataset = dataset_type(
            length=max(args.batch_size, target_step * args.batch_size),
            frames=config.frames,
            size=config.image_size,
            base_seed=args.seed,
        )
        remaining_indices = range(
            global_step * args.batch_size, target_step * args.batch_size
        )
    loader = DataLoader(
        Subset(dataset, remaining_indices),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    started = time.perf_counter()
    final_metrics = {"loss": math.nan, "mse": math.nan, "time_mean": math.nan, "time_std": math.nan}
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
            times = schedule.sample_times(videos.shape[0], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss, final_metrics = flow_matching_loss(
                model, schedule, videos, times, conditions, return_metrics=True
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            update_ema(ema, model, args.ema_decay)
            global_step += 1
            if global_step % args.log_every == 0 or args.dry_run:
                print(
                    f"step={global_step} loss={final_metrics['loss']:.6f} "
                    f"t_mean={final_metrics['time_mean']:.3f} device={device}"
                )
        else:
            if global_step < target_step and len(loader) == 0:
                raise ValueError("dataset is smaller than one full batch")
            continue
    payload = _checkpoint_payload(
        model, ema, config, schedule, global_step,
        data_contract, optimization_contract,
    )
    inference_path = output_dir / "video_dit_flow_inference.pt"
    torch.save(payload, inference_path)
    payload.update({"optimizer": optimizer.state_dict(), "rng_state": _rng_state()})
    checkpoint_path = output_dir / "video_dit_flow.pt"
    torch.save(payload, checkpoint_path)
    summary = {
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "steps": global_step,
        "updates_this_run": global_step - initial_step,
        "final_loss": final_metrics["loss"],
        "time_mean": final_metrics["time_mean"],
        "generative_process": GENERATIVE_PROCESS,
        "time_convention": TIME_CONVENTION,
        "prediction_type": PREDICTION_TYPE,
        "flow": asdict(schedule),
        "checkpoint": str(checkpoint_path),
        "inference_checkpoint": str(inference_path),
        "data_contract": data_contract,
        "elapsed_seconds": time.perf_counter() - started,
        "dry_run": args.dry_run,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
