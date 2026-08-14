from dataclasses import asdict
import json

import torch

from diffusion import DiffusionSchedule
from inference import VideoDiTInference, parse_args, render_dataset_preview
from video_dit import VideoDiT, VideoDiTConfig


def _tiny_checkpoint(path, *, condition_dropout_prob=0.1):
    config = VideoDiTConfig(
        frames=4,
        image_size=32,
        patch_size=8,
        hidden_size=32,
        num_heads=4,
        num_pairs=1,
        condition_dropout_prob=condition_dropout_prob,
    )
    model = VideoDiT(config)
    schedule = DiffusionSchedule(timesteps=6)
    torch.save(
        {
            "config": asdict(config),
            "diffusion": {
                "timesteps": schedule.timesteps,
                "beta_start": schedule.beta_start,
                "beta_end": schedule.beta_end,
            },
            "model": model.state_dict(),
            "ema": model.state_dict(),
            "step": 7,
            "data_contract": {"space": "pixel", "manifest_sha256": "a" * 64},
        },
        path,
    )


def test_checkpoint_prompt_becomes_reproducible_temporal_video_and_separate_diagnostic(
    tmp_path,
):
    checkpoint = tmp_path / "tiny.pt"
    _tiny_checkpoint(checkpoint)
    engine = VideoDiTInference.from_checkpoint(checkpoint, device="cpu")

    first = engine.generate(
        "a person in a red coat walking in the snow to the right",
        tmp_path / "first",
        seed=123,
        sampling_steps=3,
        cfg_scale=2.5,
        trajectory_frames=3,
    )
    second = engine.generate(
        "a person in a red coat walking in the snow to the right",
        tmp_path / "second",
        seed=123,
        sampling_steps=3,
        cfg_scale=2.5,
        trajectory_frames=3,
    )

    assert torch.equal(first.video, second.video)
    assert first.video.shape == (3, 4, 32, 32)
    assert first.video_path.suffix == ".mp4" and first.video_path.is_file()
    assert first.denoising_path.suffix == ".gif" and first.denoising_path.is_file()
    metadata = json.loads(first.metadata_path.read_text())
    assert metadata["weights"] == "ema"
    assert metadata["parsed_prompt"] == {
        "subject": "person",
        "color": "red",
        "motion": "walk_right",
        "scene": "snow",
    }
    assert metadata["artifact_semantics"] == {
        "video_mp4": "temporal frames of the generated animation",
        "denoising_gif": "diagnostic diffusion steps, not animation time",
    }
    assert metadata["elapsed_seconds"] >= 0
    assert len(metadata["checkpoint_sha256"]) == 64
    assert metadata["video_filename"] == first.video_path.name
    assert metadata["video_sha256"] == __import__("hashlib").sha256(
        first.video_path.read_bytes()
    ).hexdigest()
    assert metadata["clip_denoised"] is True
    assert metadata["start_timestep"] == 5


def test_inference_can_export_each_denoising_state_as_a_labelled_video(tmp_path):
    checkpoint = tmp_path / "tiny.pt"
    _tiny_checkpoint(checkpoint)
    engine = VideoDiTInference.from_checkpoint(checkpoint, device="cpu")

    result = engine.generate(
        "an orange cat walking right in the snow", tmp_path / "trace",
        sampling_steps=2, trajectory_frames=3, save_trajectory_videos=True,
    )

    assert len(result.trajectory_video_paths) == 3
    for path in result.trajectory_video_paths:
        assert path.is_file()
        metadata = json.loads(path.with_suffix(".json").read_text())
        assert metadata["mode"] == "trained Video DiT trajectory state"
        assert isinstance(metadata["trajectory_timestep"], int)
        assert metadata["trajectory_phase"] in {"noisy", "final_clean"}


def test_inference_binds_and_enforces_preregistered_evaluation_spec(tmp_path):
    checkpoint = tmp_path / "tiny.pt"
    _tiny_checkpoint(checkpoint)
    engine = VideoDiTInference.from_checkpoint(checkpoint, device="cpu")
    prompt = "an orange cat walking right in the snow"
    contract = {
        "checkpoint_sha256": __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest(),
        "training_step": 7, "weights": "ema", "prediction_type": "epsilon",
        "sampling_steps": 3, "start_timestep": 5, "clip_denoised": True,
        "effective_cfg_scale": 2.5, "condition_mode": "structured",
        "text_encoder_id": None, "training_manifest_sha256": "a" * 64,
    }
    spec = json.dumps({
        "generation_contract": contract, "seeds": [17, 29],
        "records": [{"prompt": prompt}],
    }, sort_keys=True).encode()
    result = engine.generate(
        prompt, tmp_path / "bound", seed=17, sampling_steps=3,
        cfg_scale=2.5, evaluation_spec=spec,
    )
    metadata = json.loads(result.metadata_path.read_text())
    assert metadata["evaluation_spec_sha256"] == __import__("hashlib").sha256(spec).hexdigest()

    contract["sampling_steps"] = 4
    drifted = json.dumps({
        "generation_contract": contract, "seeds": [17, 29],
        "records": [{"prompt": prompt}],
    }).encode()
    with __import__("pytest").raises(ValueError, match="preregistered generation contract"):
        engine.generate(
            prompt, tmp_path / "rejected", seed=17, sampling_steps=3,
            cfg_scale=2.5, evaluation_spec=drifted,
        )


def test_dataset_preview_is_explicitly_labelled_as_renderer_not_model_output(tmp_path):
    result = render_dataset_preview(
        "an orange cat walking right on grass",
        tmp_path,
        seed=9,
        frames=4,
        size=32,
    )

    assert result.video.shape == (3, 4, 32, 32)
    assert result.video_path.is_file() and result.video_path.suffix == ".mp4"
    assert result.denoising_path is None
    metadata = json.loads(result.metadata_path.read_text())
    assert metadata["mode"] == "dataset renderer preview (not model output)"
    assert metadata["parsed_prompt"]["subject"] == "cat"


def test_identical_dataset_previews_do_not_overwrite_each_other(tmp_path):
    first = render_dataset_preview(
        "an orange cat walking right on grass", tmp_path, seed=9, frames=4, size=32
    )
    second = render_dataset_preview(
        "an orange cat walking right on grass", tmp_path, seed=9, frames=4, size=32
    )

    assert first.video_path != second.video_path
    assert first.metadata_path != second.metadata_path
    assert first.video_path.is_file() and second.video_path.is_file()
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]


def test_cli_accepts_readme_positional_checkpoint():
    args = parse_args(
        [
            "runs/faithful/video_dit.pt",
            "--prompt",
            "a red person walking in snow",
        ]
    )

    assert args.checkpoint.as_posix() == "runs/faithful/video_dit.pt"


def test_checkpoint_without_condition_dropout_disables_cfg(tmp_path):
    checkpoint = tmp_path / "concept.pt"
    _tiny_checkpoint(checkpoint, condition_dropout_prob=0.0)
    engine = VideoDiTInference.from_checkpoint(checkpoint, device="cpu")

    result = engine.generate(
        "an orange cat walking right in snow",
        tmp_path / "result",
        seed=4,
        sampling_steps=2,
        cfg_scale=5.0,
        trajectory_frames=2,
    )

    metadata = json.loads(result.metadata_path.read_text())
    assert metadata["requested_cfg_scale"] == 5.0
    assert metadata["effective_cfg_scale"] == 1.0
    assert metadata["cfg_note"] == "disabled: checkpoint was trained without condition dropout"


def test_velocity_checkpoint_selects_velocity_sampling_automatically(tmp_path):
    checkpoint = tmp_path / "velocity.pt"
    _tiny_checkpoint(checkpoint, condition_dropout_prob=0.0)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["prediction_type"] = "velocity"
    torch.save(payload, checkpoint)

    engine = VideoDiTInference.from_checkpoint(checkpoint, device="cpu")
    result = engine.generate(
        "a red person walking in snow",
        tmp_path / "velocity-result",
        sampling_steps=2,
        trajectory_frames=2,
    )

    metadata = json.loads(result.metadata_path.read_text())
    assert engine.prediction_type == "velocity"
    assert metadata["prediction_type"] == "velocity"


def test_different_sampling_requests_get_distinct_artifact_names(tmp_path):
    checkpoint = tmp_path / "tiny.pt"
    _tiny_checkpoint(checkpoint, condition_dropout_prob=0.0)
    engine = VideoDiTInference.from_checkpoint(checkpoint, device="cpu")

    first = engine.generate(
        "an orange cat walking right in snow",
        tmp_path / "result",
        seed=4,
        sampling_steps=2,
        cfg_scale=1,
        trajectory_frames=2,
    )
    second = engine.generate(
        "an orange cat walking right in snow",
        tmp_path / "result",
        seed=4,
        sampling_steps=3,
        cfg_scale=1,
        trajectory_frames=2,
    )

    assert first.video_path != second.video_path
    assert first.video_path.is_file() and second.video_path.is_file()
    assert not [
        path for path in (tmp_path / "result").iterdir()
        if path.name.startswith(".")
    ]


def test_identical_requests_do_not_overwrite_each_others_artifacts(tmp_path):
    checkpoint = tmp_path / "tiny.pt"
    _tiny_checkpoint(checkpoint, condition_dropout_prob=0.0)
    engine = VideoDiTInference.from_checkpoint(checkpoint, device="cpu")

    first = engine.generate(
        "an orange cat walking right in snow",
        tmp_path / "result",
        seed=4,
        sampling_steps=2,
        cfg_scale=1,
        trajectory_frames=2,
    )
    second = engine.generate(
        "an orange cat walking right in snow",
        tmp_path / "result",
        seed=4,
        sampling_steps=2,
        cfg_scale=1,
        trajectory_frames=2,
    )

    assert first.video_path != second.video_path
    assert first.video_path.is_file() and second.video_path.is_file()


def test_inference_rejects_checkpoint_that_is_not_safe_weights_only(tmp_path):
    checkpoint = tmp_path / "unsafe.pt"
    _tiny_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["unexpected_object"] = __import__("pathlib").Path("payload")
    torch.save(payload, checkpoint)

    with __import__("pytest").raises(ValueError, match="safe inference checkpoint"):
        VideoDiTInference.from_checkpoint(checkpoint, device="cpu")


def test_text_condition_checkpoint_generates_from_natural_prompt_embedding(tmp_path):
    config = VideoDiTConfig(
        frames=4, image_size=16, patch_size=4, hidden_size=32,
        num_heads=4, num_pairs=1, condition_mode="text",
        text_embedding_size=12, condition_dropout_prob=0.1,
    )
    model = VideoDiT(config)
    schedule = DiffusionSchedule(timesteps=4)
    checkpoint = tmp_path / "text.pt"
    torch.save(
        {
            "config": asdict(config),
            "diffusion": {
                "timesteps": 4, "beta_start": schedule.beta_start,
                "beta_end": schedule.beta_end,
            },
            "model": model.state_dict(), "ema": model.state_dict(),
            "step": 1,
            "data_contract": {
                "space": "pixel", "text_encoder_id": "test@sha256:abc"
            },
        },
        checkpoint,
    )
    seen = []

    def encode(prompts):
        seen.extend(prompts)
        return torch.arange(12, dtype=torch.float32).unsqueeze(0)

    engine = VideoDiTInference.from_checkpoint(
        checkpoint, device="cpu", text_encode=encode,
        text_encoder_id="test@sha256:abc",
    )
    result = engine.generate(
        "a cat jumping between concrete walls", tmp_path / "out",
        sampling_steps=2, trajectory_frames=2,
    )
    assert seen == ["a cat jumping between concrete walls"]
    assert result.video.shape == (3, 4, 16, 16)
    metadata = json.loads(result.metadata_path.read_text())
    assert metadata["condition_mode"] == "text"
    assert metadata["text_encoder_id"] == "test@sha256:abc"


def test_latent_checkpoint_requires_matching_decoder_and_decodes_sample(tmp_path):
    config = VideoDiTConfig(
        frames=4, image_size=2, patch_size=2, in_channels=4, hidden_size=32,
        num_heads=4, num_pairs=1, condition_mode="text", text_embedding_size=12,
    )
    model = VideoDiT(config)
    schedule = DiffusionSchedule(timesteps=4)
    checkpoint = tmp_path / "latent.pt"
    torch.save({
        "config": asdict(config),
        "diffusion": {"timesteps": 4, "beta_start": schedule.beta_start,
                      "beta_end": schedule.beta_end},
        "model": model.state_dict(), "ema": model.state_dict(), "step": 1,
        "data_contract": {"space": "latent", "text_encoder_id": "text@sha256:a",
                          "vae_id": "vae@sha256:b", "vae_scaling": 0.5},
    }, checkpoint)
    encode = lambda prompts: torch.ones(1, 12)

    import pytest
    with pytest.raises(ValueError, match="matching VAE decoder"):
        VideoDiTInference.from_checkpoint(
            checkpoint, text_encode=encode, text_encoder_id="text@sha256:a",
        )

    class Decoder:
        vae_id = "vae@sha256:b"
        scaling_factor = 0.5
        def decode(self, latents):
            return torch.zeros(latents.shape[0], 3, 16, 16)

    engine = VideoDiTInference.from_checkpoint(
        checkpoint, text_encode=encode, text_encoder_id="text@sha256:a",
        vae_decoder=Decoder(),
    )
    result = engine.generate(
        "a cat walking", tmp_path / "out", sampling_steps=2,
        trajectory_frames=2,
    )
    assert result.video.shape == (3, 4, 16, 16)
    metadata = json.loads(result.metadata_path.read_text())
    assert metadata["clip_denoised"] is False


def test_text_token_checkpoint_accepts_sequence_encoder(tmp_path):
    config = VideoDiTConfig(
        frames=4, image_size=16, patch_size=4, hidden_size=32,
        num_heads=4, num_pairs=1, condition_mode="text_tokens",
        text_embedding_size=12,
    )
    model = VideoDiT(config)
    checkpoint = tmp_path / "tokens.pt"
    torch.save({
        "config": asdict(config),
        "diffusion": {"timesteps": 4, "beta_start": 1e-4, "beta_end": 0.02},
        "model": model.state_dict(), "ema": model.state_dict(),
        "data_contract": {"space": "pixel", "text_encoder_id": "clip@rev#tokens-v2",
                          "text_embedding_kind": "tokens"},
    }, checkpoint)
    engine = VideoDiTInference.from_checkpoint(
        checkpoint, text_encode=lambda prompts: torch.ones(1, 7, 12),
        text_encoder_id="clip@rev#tokens-v2",
    )

    result = engine.generate(
        "a cat walking", tmp_path / "out", sampling_steps=2,
        trajectory_frames=2,
    )

    assert result.video.shape == (3, 4, 16, 16)

def test_text_checkpoint_rejects_wrong_encoder_identity(tmp_path):
    config = VideoDiTConfig(
        frames=4, image_size=16, patch_size=4, hidden_size=32,
        num_heads=4, num_pairs=1, condition_mode="text", text_embedding_size=12,
    )
    model = VideoDiT(config)
    checkpoint = tmp_path / "text.pt"
    torch.save(
        {
            "config": asdict(config),
            "diffusion": {"timesteps": 4, "beta_start": 1e-4, "beta_end": 0.02},
            "model": model.state_dict(),
            "data_contract": {"space": "pixel", "text_encoder_id": "right@sha256:1"},
        }, checkpoint,
    )

    with __import__("pytest").raises(ValueError, match="text encoder identity"):
        VideoDiTInference.from_checkpoint(
            checkpoint, text_encode=lambda prompts: torch.zeros(1, 12),
            text_encoder_id="wrong@sha256:2",
        )


def test_text_token_checkpoint_rejects_pooled_embedding_contract(tmp_path):
    config = VideoDiTConfig(
        frames=4, image_size=16, patch_size=4, hidden_size=32,
        num_heads=4, num_pairs=1, condition_mode="text_tokens",
        text_embedding_size=12,
    )
    model = VideoDiT(config)
    checkpoint = tmp_path / "tokens-with-pooled-contract.pt"
    torch.save({
        "config": asdict(config),
        "diffusion": {"timesteps": 4, "beta_start": 1e-4, "beta_end": 0.02},
        "model": model.state_dict(),
        "data_contract": {
            "space": "pixel", "text_encoder_id": "clip@rev",
            "text_embedding_kind": "pooled",
        },
    }, checkpoint)

    with __import__("pytest").raises(ValueError, match="embedding kind"):
        VideoDiTInference.from_checkpoint(
            checkpoint, text_encode=lambda prompts: torch.ones(1, 7, 12),
            text_encoder_id="clip@rev",
        )
