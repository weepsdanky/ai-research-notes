"""Evaluate generated-video/prompt alignment with a pinned frozen CLIP model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from compositional_evaluation import prompt_retrieval_report
from real_data import decode_video_clip, load_manifest


@dataclass(frozen=True)
class Generation:
    prompt: str
    video_path: Path
    seed: int
    contract: tuple[object, ...]


def discover_generations(directory: Path) -> list[Generation]:
    """Find model-output metadata and its same-stem MP4, failing closed."""

    rows: list[Generation] = []
    expected_contract: tuple[object, ...] | None = None
    for metadata_path in sorted(directory.glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("mode") != "trained Video DiT checkpoint":
            continue
        video_path = metadata_path.with_suffix(".mp4")
        if not video_path.is_file():
            raise ValueError(f"generation metadata has no matching MP4: {metadata_path}")
        prompt = metadata.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"generation metadata has no prompt: {metadata_path}")
        checkpoint_value = metadata.get("checkpoint")
        checkpoint = Path(checkpoint_value) if isinstance(checkpoint_value, str) else None
        if checkpoint is None or not checkpoint.is_file():
            raise ValueError(f"generation checkpoint is unavailable: {metadata_path}")
        checkpoint_hash = metadata.get("checkpoint_sha256")
        actual_checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if (
            not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64
            or any(character not in "0123456789abcdef" for character in checkpoint_hash)
        ):
            raise ValueError(f"generation metadata lacks valid checkpoint SHA-256: {metadata_path}")
        if checkpoint_hash != actual_checkpoint_hash:
            raise ValueError(f"generation checkpoint hash mismatch: {metadata_path}")
        keys = (
            "training_step", "weights", "prediction_type", "sampling_steps",
            "start_timestep", "clip_denoised", "effective_cfg_scale",
            "condition_mode", "text_encoder_id",
        )
        if any(key not in metadata for key in keys):
            raise ValueError(f"generation metadata lacks evaluation contract: {metadata_path}")
        contract = (checkpoint_hash, *(metadata[key] for key in keys))
        if expected_contract is None:
            expected_contract = contract
        elif contract != expected_contract:
            raise ValueError("generation contract differs across evaluation directory")
        rows.append(Generation(prompt.strip(), video_path, int(metadata["seed"]), contract))
    return rows


def evaluate_alignment(
    generation_dir: Path,
    *,
    encoder_id: str,
    encode_videos: Callable[[list[Path]], torch.Tensor],
    encode_prompts: Callable[[list[str]], torch.Tensor],
    train_video_paths: list[Path] | None = None,
    train_ids: list[str] | None = None,
) -> dict[str, object]:
    rows = discover_generations(generation_dir)
    if len(rows) < 2:
        raise ValueError("alignment evaluation requires at least two generations")
    seed_rows: dict[int, list[Generation]] = {}
    for row in rows:
        seed_rows.setdefault(row.seed, []).append(row)
    prompt_set = {row.prompt for row in next(iter(seed_rows.values()))}
    if len(prompt_set) < 2 or any(
        len(group) != len(prompt_set) or {row.prompt for row in group} != prompt_set
        for group in seed_rows.values()
    ):
        raise ValueError("every seed must contain the same unique prompt set")
    prompt_order = sorted(prompt_set)
    ordered = [
        next(row for row in seed_rows[seed] if row.prompt == prompt)
        for seed in sorted(seed_rows) for prompt in prompt_order
    ]
    videos = encode_videos([row.video_path for row in ordered])
    prompts = encode_prompts(prompt_order)
    train = encode_videos(train_video_paths) if train_video_paths else None
    per_seed: dict[str, dict[str, object]] = {}
    block = len(prompt_order)
    for offset, seed in enumerate(sorted(seed_rows)):
        per_seed[str(seed)] = prompt_retrieval_report(
            videos[offset * block:(offset + 1) * block], prompts,
            prompt_ids=prompt_order, train_embeddings=train, train_ids=train_ids,
        )
    sample_count = block * len(per_seed)
    correct = sum(
        round(float(report["prompt_top1_accuracy"]) * block)
        for report in per_seed.values()
    )
    shuffled_correct = sum(
        round(float(report["shuffled_prompt_top1_accuracy"]) * block)
        for report in per_seed.values()
    )
    # Wilson interval is defined for the aggregate retrieval proportion and is
    # more honest than a normal interval for small diagnostic sample counts.
    z = 1.959963984540054
    proportion = correct / sample_count
    denominator = 1 + z * z / sample_count
    center = (proportion + z * z / (2 * sample_count)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / sample_count
        + z * z / (4 * sample_count * sample_count)
    ) / denominator
    margins = [
        float(value) for report in per_seed.values()
        for value in report["prompt_margins"]
    ]
    nearest = [
        float(value) for report in per_seed.values()
        for value in (report["nearest_train_similarities"] or [])
    ]
    return {
        "metric": "frozen CLIP mean-frame retrieval diagnostic",
        "encoder_id": encoder_id,
        "generation_directory": str(generation_dir.resolve()),
        "seeds": sorted(seed_rows),
        "limitations": (
            "CLIP frame-text similarity is a proxy: inspect videos for identity, "
            "action, temporal coherence, and training-set memorization."
        ),
        "samples": sample_count,
        "prompts_per_seed": block,
        "prompt_ids": prompt_order,
        "prompt_top1_accuracy": proportion,
        "prompt_top1_accuracy_wilson95": [max(0.0, center - radius), min(1.0, center + radius)],
        "shuffled_prompt_top1_accuracy": shuffled_correct / sample_count,
        "prompt_top1_gain_over_shuffled": (correct - shuffled_correct) / sample_count,
        "prompt_margin_mean": sum(margins) / len(margins),
        "nearest_train_similarity_mean": sum(nearest) / len(nearest) if nearest else None,
        "nearest_train_similarity_max": max(nearest) if nearest else None,
        "per_seed": per_seed,
    }


def openai_clip_features(
    checkpoint: Path, *, frames: int = 8, size: int = 96,
) -> tuple[str, Callable[[list[Path]], torch.Tensor], Callable[[list[str]], torch.Tensor]]:
    import clip

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    scripted = torch.jit.load(str(checkpoint), map_location="cpu")
    model = clip.model.build_model(scripted.state_dict()).float().eval()
    mean = torch.tensor((0.48145466, 0.4578275, 0.40821073))[None, :, None, None]
    std = torch.tensor((0.26862954, 0.26130258, 0.27577711))[None, :, None, None]

    def encode_videos(paths: list[Path]) -> torch.Tensor:
        features = []
        with torch.inference_mode():
            for path in paths:
                video = decode_video_clip(path, frames=frames, size=size)
                images = video.permute(1, 0, 2, 3).add(1).div(2)
                images = F.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False)
                frame_features = model.encode_image((images - mean) / std).float()
                frame_features = frame_features / frame_features.norm(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12)
                feature = frame_features.mean(dim=0)
                features.append(feature / feature.norm().clamp_min(1e-12))
        return torch.stack(features)

    def encode_prompts(prompts: list[str]) -> torch.Tensor:
        with torch.inference_mode():
            features = model.encode_text(clip.tokenize(prompts, truncate=True)).float()
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    return f"openai/CLIP-ViT-B-32@sha256:{digest}", encode_videos, encode_prompts


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generation_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.frames < 2 or args.size <= 0:
        raise ValueError("frames must be at least 2 and size must be positive")
    encoder_id, encode_videos, encode_prompts = openai_clip_features(
        args.checkpoint, frames=args.frames, size=args.size,
    )
    train_paths = None
    train_ids = None
    if args.train_manifest:
        records = load_manifest(args.train_manifest)
        train_paths = [
            (args.train_manifest.parent / record.clip_path).resolve()
            for record in records if record.split == "train"
        ]
        train_ids = [record.sample_id for record in records if record.split == "train"]
        if not train_paths:
            raise ValueError("training manifest contains no train records")
    report = evaluate_alignment(
        args.generation_dir, encoder_id=encoder_id,
        encode_videos=encode_videos, encode_prompts=encode_prompts,
        train_video_paths=train_paths, train_ids=train_ids,
    )
    output = args.output or args.generation_dir / "condition-alignment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    print(main())
