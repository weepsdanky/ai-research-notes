"""Check whether a checkpoint associates each training video with its prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from diffusion import DiffusionSchedule
from real_data import RealVideoDataset
from video_dit import VideoDiT, VideoDiTConfig


@torch.inference_mode()
def teacher_forced_retrieval(
    model: torch.nn.Module,
    clean: torch.Tensor,
    conditions: torch.Tensor,
    prompt_ids: list[str],
    *,
    sample_prompt_ids: list[str] | None = None,
    noisy_batches: list[torch.Tensor],
    timesteps: list[torch.Tensor],
    target_batches: list[torch.Tensor],
    channels: int,
    condition_batch_size: int = 1,
) -> dict[str, object]:
    """Rank every condition by denoising MSE on every ground-truth video."""

    samples = clean.shape[0]
    condition_count = conditions.shape[0]
    if samples < 2 or condition_count < 2:
        raise ValueError("clean videos and conditions need at least two entries")
    if len(prompt_ids) != condition_count or len(set(prompt_ids)) != condition_count:
        raise ValueError("prompt_ids must be unique and match condition count")
    sample_prompt_ids = prompt_ids if sample_prompt_ids is None else sample_prompt_ids
    if len(sample_prompt_ids) != samples or any(
        prompt not in prompt_ids for prompt in sample_prompt_ids
    ):
        raise ValueError("sample_prompt_ids must map every video to a known prompt")
    if not (len(noisy_batches) == len(timesteps) == len(target_batches)) or not noisy_batches:
        raise ValueError("noisy, timestep, and target batches must be non-empty and aligned")
    if condition_batch_size <= 0:
        raise ValueError("condition_batch_size must be positive")
    errors = torch.zeros(samples, condition_count, device=clean.device)
    for noisy, timestep, target in zip(
        noisy_batches, timesteps, target_batches, strict=True,
    ):
        if noisy.shape != clean.shape or target.shape != clean.shape or timestep.shape != (samples,):
            raise ValueError("teacher-forced batch shapes do not match clean videos")
        for start in range(0, condition_count, condition_batch_size):
            condition_block = conditions[start:start + condition_batch_size]
            block = condition_block.shape[0]
            repeated_noisy = noisy.repeat(block, *((1,) * (noisy.ndim - 1)))
            repeated_timesteps = timestep.repeat(block)
            repeated_conditions = condition_block.repeat_interleave(samples, dim=0)
            prediction = model(
                repeated_noisy, repeated_timesteps, repeated_conditions,
            )[:, :channels].reshape(block, samples, *clean.shape[1:])
            if not torch.isfinite(prediction).all():
                raise ValueError("model prediction must be finite")
            block_errors = (
                prediction - target.unsqueeze(0)
            ).square().flatten(2).mean(2).transpose(0, 1)
            errors[:, start:start + block] += block_errors
    errors /= len(noisy_batches)
    if not torch.isfinite(errors).all():
        raise ValueError("teacher-forced errors must be finite")
    prompt_index = {prompt: index for index, prompt in enumerate(prompt_ids)}
    expected = torch.tensor(
        [prompt_index[prompt] for prompt in sample_prompt_ids], device=errors.device,
    )
    predicted = errors.argmin(dim=1)
    minima = errors.min(dim=1, keepdim=True).values
    ambiguous = errors.eq(minima).sum(dim=1).gt(1)
    correct = errors.gather(1, expected[:, None]).squeeze(1)
    alternative = errors.masked_fill(
        torch.nn.functional.one_hot(expected, num_classes=condition_count).bool(),
        float("inf")
    ).min(dim=1).values
    return {
        "samples": samples,
        "conditions": condition_count,
        "evaluations_per_pair": len(noisy_batches),
        "prompt_ids": prompt_ids,
        "predicted_prompt_ids": [
            None if bool(tie) else prompt_ids[int(index)]
            for index, tie in zip(predicted.cpu(), ambiguous.cpu(), strict=True)
        ],
        "prompt_top1_accuracy": float(((predicted == expected) & ~ambiguous).float().mean()),
        "ambiguous_tie_rate": float(ambiguous.float().mean()),
        "error_margin_mean": float((alternative - correct).mean()),
        "error_margins": [float(value) for value in (alternative - correct).cpu()],
        "mse_matrix": errors.cpu().tolist(),
    }


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--weights", choices=("model", "ema"), default="model")
    parser.add_argument("--noise-seeds", default="11,29,47")
    parser.add_argument("--timesteps", default="100,300,500,700,900")
    parser.add_argument("--condition-batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = VideoDiTConfig(**checkpoint["config"])
    data_contract = checkpoint.get("data_contract", {})
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if data_contract.get("manifest_sha256") != manifest_hash:
        raise ValueError("manifest does not match checkpoint data contract")
    device = torch.device(args.device)
    model = VideoDiT(config)
    model.load_state_dict(checkpoint[args.weights])
    model.to(device).eval().requires_grad_(False)
    schedule = DiffusionSchedule(**checkpoint["diffusion"]).to(device)
    dataset = RealVideoDataset(
        args.manifest, split="train", frames=config.frames,
        size=config.image_size, space=str(data_contract.get("space", "pixel")),
    )
    items = [dataset[index] for index in range(len(dataset))]
    clean = torch.stack([item["video"] for item in items]).to(device)
    sample_prompts = [str(item["prompt"]) for item in items]
    prompts = list(dict.fromkeys(sample_prompts))
    condition_by_prompt: dict[str, torch.Tensor] = {}
    for item in items:
        prompt = str(item["prompt"])
        condition = item["condition"]
        if prompt in condition_by_prompt and not torch.equal(
            condition_by_prompt[prompt], condition
        ):
            raise ValueError("identical prompts must use identical cached conditions")
        condition_by_prompt[prompt] = condition
    conditions = torch.stack([condition_by_prompt[prompt] for prompt in prompts]).to(device)
    seed_values = [int(value) for value in args.noise_seeds.split(",")]
    timestep_values = [int(value) for value in args.timesteps.split(",")]
    if any(value < 0 or value >= schedule.timesteps for value in timestep_values):
        raise ValueError("timesteps must be inside the diffusion schedule")
    noisy_batches, timestep_batches, targets = [], [], []
    for seed in seed_values:
        generator = torch.Generator().manual_seed(seed)
        for timestep in timestep_values:
            noise = torch.randn(clean.shape, generator=generator).to(device)
            batch_timesteps = torch.full(
                (len(dataset),), timestep, dtype=torch.long, device=device,
            )
            noisy, _ = schedule.q_sample(clean, batch_timesteps, noise)
            target = (
                noise if checkpoint.get("prediction_type", "epsilon") == "epsilon"
                else schedule.velocity_target(clean, noise, batch_timesteps)
            )
            noisy_batches.append(noisy)
            timestep_batches.append(batch_timesteps)
            targets.append(target)
    report = teacher_forced_retrieval(
        model, clean, conditions, prompts, sample_prompt_ids=sample_prompts,
        noisy_batches=noisy_batches,
        timesteps=timestep_batches, target_batches=targets,
        channels=config.in_channels, condition_batch_size=args.condition_batch_size,
    )
    report.update({
        "metric": "teacher-forced denoising condition retrieval",
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": manifest_hash,
        "weights": args.weights,
        "noise_seeds": seed_values,
        "timesteps": timestep_values,
        "limitations": (
            "Training-video denoising retrieval diagnoses condition binding only; "
            "it does not measure free generation, held-out videos, or composition."
        ),
    })
    output = args.output or args.checkpoint.with_name(
        f"teacher-forced-{args.weights}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    print(main())
