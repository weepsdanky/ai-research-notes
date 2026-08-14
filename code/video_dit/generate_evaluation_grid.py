"""Generate an exact preregistered prompt-by-seed Video DiT evaluation grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from cache_text_embeddings import openai_clip_encoder
from generation_blind_review import _evaluation_spec
from inference import VideoDiTInference
from vae_latents import DiffusersFrameVAE


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_engine(
    checkpoint: Path,
    *,
    device: str,
    text_checkpoint: Path,
    text_conditioning: str,
    vae_model: str,
    vae_revision: str,
    use_ema: bool,
) -> VideoDiTInference:
    """Load every frozen inference dependency once for the complete grid."""

    encoder_id, encode = openai_clip_encoder(
        text_checkpoint, embedding_kind=text_conditioning,
    )
    vae = DiffusersFrameVAE.from_pretrained(
        vae_model, revision=vae_revision, device=device,
    )
    return VideoDiTInference.from_checkpoint(
        checkpoint,
        device=device,
        use_ema=use_ema,
        text_encode=encode,
        text_encoder_id=encoder_id,
        vae_decoder=vae,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("evaluation_spec", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--text-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--text-conditioning", choices=("pooled", "tokens"), default="tokens",
    )
    parser.add_argument("--vae-model", required=True)
    parser.add_argument("--vae-revision", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--trajectory-frames", type=int, default=2)
    parser.add_argument("--fps", type=float, default=8.0)
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.expanduser().resolve()
    spec_path = args.evaluation_spec.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not checkpoint.is_file() or not spec_path.is_file():
        raise FileNotFoundError("checkpoint and evaluation spec must both exist")
    if output.exists():
        raise FileExistsError(f"evaluation grid output already exists: {output}")
    if args.trajectory_frames < 2 or args.fps <= 0:
        raise ValueError("trajectory_frames must be at least 2 and fps must be positive")

    spec_bytes = spec_path.read_bytes()
    prompt_records, spec = _evaluation_spec(spec_bytes)
    contract = spec["generation_contract"]
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != contract["checkpoint_sha256"]:
        raise ValueError("checkpoint bytes differ from the preregistered checkpoint")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        engine = load_engine(
            checkpoint,
            device=args.device,
            text_checkpoint=args.text_checkpoint.expanduser().resolve(),
            text_conditioning=args.text_conditioning,
            vae_model=args.vae_model,
            vae_revision=args.vae_revision,
            use_ema=contract["weights"] == "ema",
        )
        generated: list[dict[str, object]] = []
        for seed in spec["seeds"]:
            for prompt, record in prompt_records.items():
                result = engine.generate(
                    prompt,
                    staging,
                    seed=seed,
                    sampling_steps=contract["sampling_steps"],
                    cfg_scale=contract["effective_cfg_scale"],
                    trajectory_frames=args.trajectory_frames,
                    fps=args.fps,
                    start_timestep=contract["start_timestep"],
                    evaluation_spec=spec_bytes,
                )
                generated.append(
                    {
                        "prompt": prompt,
                        "seed": seed,
                        "split": record["split"],
                        "video_filename": result.video_path.name,
                        "video_sha256": _sha256(result.video_path),
                        "metadata_filename": result.metadata_path.name,
                        "metadata_sha256": _sha256(result.metadata_path),
                    }
                )

        expected = len(spec["seeds"]) * len(prompt_records)
        if len(generated) != expected:
            raise RuntimeError("generation grid is incomplete")
        inventory = {
            "protocol": "exact preregistered Video DiT evaluation grid v1",
            "evaluation_spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
            "checkpoint_sha256": checkpoint_sha256,
            "expected_generations": expected,
            "completed_generations": len(generated),
            "prompts": list(prompt_records),
            "seeds": spec["seeds"],
            "records": generated,
            "selection_policy": "complete prompt-by-seed grid; no post-hoc selection",
            "training_ready": False,
        }
        _write_json(staging / "grid-inventory.json", inventory)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(json.dumps({"output_dir": str(output), **inventory}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
