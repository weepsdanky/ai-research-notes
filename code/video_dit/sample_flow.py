"""Generate a closed-vocabulary video from a trained Flow Matching checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re

import torch

from dataset import parse_prompt
from cache_text_embeddings import openai_clip_encoder
from flow_matching import (
    GENERATIVE_PROCESS,
    PREDICTION_TYPE,
    TIME_CONVENTION,
    FlowMatchingSchedule,
    flow_sample_loop,
)
from train import choose_device
from vae_latents import DiffusersFrameVAE, decode_video_latents
from video_dit import VideoDiT, VideoDiTConfig
from video_io import save_denoising_gif, save_video_mp4


def sample(
    checkpoint_path: Path,
    prompt: str,
    output_dir: Path,
    *,
    device_name: str = "auto",
    seed: int = 42,
    steps: int = 20,
    solver: str = "heun",
    cfg_scale: float = 1.0,
    sampling_time_shift: float = 1.0,
    text_checkpoint: Path | None = None,
    text_conditioning: str = "pooled",
    vae_model: str | None = None,
    vae_revision: str | None = None,
) -> dict[str, object]:
    device = choose_device(device_name)
    checkpoint = torch.load(
        checkpoint_path.expanduser().resolve(), map_location="cpu", weights_only=True
    )
    if checkpoint.get("generative_process") != GENERATIVE_PROCESS:
        raise ValueError("checkpoint is not a linear Flow Matching checkpoint")
    if checkpoint.get("time_convention") != TIME_CONVENTION:
        raise ValueError("unsupported Flow Matching time convention")
    if checkpoint.get("prediction_type") != PREDICTION_TYPE:
        raise ValueError("unsupported Flow Matching prediction type")
    config = VideoDiTConfig(**checkpoint["config"])
    data_contract = checkpoint.get("data_contract", {})
    model = VideoDiT(config).to(device)
    model.load_state_dict(checkpoint.get("ema", checkpoint["model"]))
    model.eval().requires_grad_(False)
    schedule = FlowMatchingSchedule(**checkpoint["flow"])
    if config.condition_mode == "structured":
        spec = parse_prompt(prompt)
        condition = spec.condition_ids().unsqueeze(0)
        text_encoder_id = None
    else:
        spec = None
        if text_checkpoint is None:
            raise ValueError("text-conditioned checkpoint requires --text-checkpoint")
        expected_kind = "tokens" if config.condition_mode == "text_tokens" else "pooled"
        if text_conditioning != expected_kind:
            raise ValueError("--text-conditioning does not match checkpoint")
        text_encoder_id, encode = openai_clip_encoder(
            text_checkpoint, embedding_kind=text_conditioning
        )
        if text_encoder_id != data_contract.get("text_encoder_id"):
            raise ValueError("text encoder identity does not match checkpoint")
        condition = encode([prompt])
    space = data_contract.get("space", "pixel")
    vae_decoder = None
    if space == "latent":
        if not vae_model or not vae_revision:
            raise ValueError("latent checkpoint requires --vae-model and --vae-revision")
        vae_decoder = DiffusersFrameVAE.from_pretrained(
            vae_model, revision=vae_revision, device=device
        )
        if (
            vae_decoder.vae_id != data_contract.get("vae_id")
            or vae_decoder.scaling_factor != data_contract.get("vae_scaling")
        ):
            raise ValueError("VAE decoder identity or scaling does not match checkpoint")
    elif space != "pixel":
        raise ValueError(f"unsupported checkpoint space: {space}")
    guided = cfg_scale != 1.0 and config.condition_dropout_prob > 0
    batch = 2 if guided else 1
    conditions = condition.repeat(batch, *((1,) * (condition.ndim - 1))).to(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        1, config.in_channels, config.frames, config.image_size, config.image_size,
        generator=generator,
    ).repeat(batch, 1, 1, 1, 1).to(device)
    result, trajectory = flow_sample_loop(
        model,
        schedule,
        tuple(noise.shape),
        conditions,
        device,
        initial_noise=noise,
        sampling_steps=steps,
        solver=solver,
        cfg_scale=cfg_scale if guided else 1.0,
        trajectory_frames=6,
        sampling_time_shift=sampling_time_shift,
    )
    video = result[0].cpu()
    diagnostic_states = [state[0] for _, state in trajectory]
    if vae_decoder is not None:
        video = decode_video_latents(video, vae_decoder)
        diagnostic_states = [
            decode_video_latents(state, vae_decoder) for state in diagnostic_states
        ]
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_slug = (
        f"{spec.subject}_{spec.motion}"
        if spec is not None
        else re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")[:48]
    )
    stem = f"flow_{prompt_slug}_seed-{seed}_{solver}-{steps}"
    video_path = save_video_mp4(video, output_dir / f"{stem}.mp4", fps=8)
    diagnostic_path = save_denoising_gif(
        diagnostic_states,
        output_dir / f"{stem}_ode-diagnostic.gif",
    )
    metadata = {
        "prompt": prompt,
        "parsed_prompt": asdict(spec) if spec is not None else None,
        "condition_mode": config.condition_mode,
        "text_encoder_id": text_encoder_id,
        "space": space,
        "seed": seed,
        "solver": solver,
        "sampling_steps": steps,
        "nfe": steps if solver == "euler" else 2 * steps,
        "sampling_time_shift": sampling_time_shift,
        "cfg_scale": cfg_scale if guided else 1.0,
        "checkpoint_step": checkpoint.get("step"),
        "video": str(video_path),
        "ode_diagnostic": str(diagnostic_path),
    }
    metadata_path = output_dir / f"{stem}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {**metadata, "metadata": str(metadata_path)}


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/video_dit_flow/samples"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--solver", choices=("euler", "heun"), default="heun")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--sampling-time-shift", type=float, default=1.0)
    parser.add_argument("--text-checkpoint", type=Path)
    parser.add_argument("--text-conditioning", choices=("pooled", "tokens"), default="pooled")
    parser.add_argument("--vae-model")
    parser.add_argument("--vae-revision")
    args = parser.parse_args(argv)
    return sample(
        args.checkpoint, args.prompt, args.output_dir,
        device_name=args.device, seed=args.seed, steps=args.steps,
        solver=args.solver, cfg_scale=args.cfg_scale,
        sampling_time_shift=args.sampling_time_shift,
        text_checkpoint=args.text_checkpoint,
        text_conditioning=args.text_conditioning,
        vae_model=args.vae_model,
        vae_revision=args.vae_revision,
    )


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
