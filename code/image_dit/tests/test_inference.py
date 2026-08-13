from dataclasses import asdict

import torch

from dit import DiT, DiTConfig
import inference as inference_module
from inference import DiTInference


def test_inference_loads_checkpoint_and_replays_seed_with_trajectory(tmp_path):
    config = DiTConfig(
        image_size=8,
        patch_size=2,
        in_channels=3,
        hidden_size=16,
        depth=1,
        num_heads=4,
    )
    model = DiT(config)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "ema": model.state_dict(),
            "config": asdict(config),
            "diffusion": {"timesteps": 4, "beta_start": 1e-4, "beta_end": 0.02},
        },
        checkpoint,
    )

    pipeline = DiTInference.from_checkpoint(checkpoint, device="cpu")
    first = pipeline.generate(class_id=3, seed=7, cfg_scale=2.0, trajectory_frames=3)
    second = pipeline.generate(class_id=3, seed=7, cfg_scale=2.0, trajectory_frames=3)

    assert first.image.size == (8, 8)
    assert len(first.trajectory) == 3
    assert first.timesteps[-1] == 0
    assert first.image.tobytes() == second.image.tobytes()
    assert pipeline.metadata["prediction_type"] == "epsilon"


def test_inference_reads_velocity_parameterization_and_passes_it_to_sampler(
    tmp_path, monkeypatch
):
    config = DiTConfig(
        image_size=8, patch_size=2, in_channels=1,
        hidden_size=16, depth=1, num_heads=4,
    )
    model = DiT(config)
    checkpoint = tmp_path / "velocity.pt"
    torch.save(
        {
            "model": model.state_dict(), "config": asdict(config),
            "diffusion": {"timesteps": 4}, "prediction_type": "velocity",
        },
        checkpoint,
    )
    calls = {}

    def fake_ddim(**kwargs):
        calls.update(kwargs)
        value = torch.zeros(kwargs["shape"])
        return value, [(4, value), (0, value)]

    monkeypatch.setattr(inference_module, "ddim_sample_loop", fake_ddim)
    pipeline = DiTInference.from_checkpoint(checkpoint, "cpu")
    pipeline.generate(class_id=0, trajectory_frames=2)

    assert pipeline.metadata["prediction_type"] == "velocity"
    assert calls["prediction_type"] == "velocity"


def test_inference_loads_compact_ema_only_demo_checkpoint(tmp_path):
    config = DiTConfig(
        image_size=8, patch_size=2, in_channels=1,
        hidden_size=16, depth=1, num_heads=4,
    )
    model = DiT(config)
    checkpoint = tmp_path / "ema-only.pt"
    torch.save(
        {
            "ema": model.state_dict(),
            "config": asdict(config),
            "diffusion": {"timesteps": 4},
        },
        checkpoint,
    )

    pipeline = DiTInference.from_checkpoint(checkpoint, "cpu", use_ema=False)

    assert pipeline.metadata["weights"] == "EMA-only"


def test_inference_rejects_checkpoint_that_requires_unsafe_pickle(tmp_path):
    config = DiTConfig(
        image_size=8, patch_size=2, in_channels=1,
        hidden_size=16, depth=1, num_heads=4,
    )
    model = DiT(config)
    checkpoint = tmp_path / "unsafe.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "diffusion": {"timesteps": 4},
            "unexpected_object": __import__("pathlib").Path("payload"),
        },
        checkpoint,
    )

    with __import__("pytest").raises(ValueError, match="safe inference checkpoint"):
        DiTInference.from_checkpoint(checkpoint, "cpu")


def test_inference_supports_one_step_ddim_with_complete_two_frame_path(tmp_path):
    config = DiTConfig(
        image_size=8, patch_size=2, in_channels=3,
        hidden_size=16, depth=1, num_heads=4,
    )
    model = DiT(config)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "diffusion": {"timesteps": 4},
        },
        checkpoint,
    )
    result = DiTInference.from_checkpoint(checkpoint, "cpu").generate(
        class_id=0, sampling_steps=1, trajectory_frames=2
    )
    assert len(result.trajectory) == 2
    assert result.timesteps == [4, 3]


def test_inference_uses_dataset_class_names_from_checkpoint(tmp_path):
    config = DiTConfig(
        image_size=28, patch_size=4, in_channels=1,
        hidden_size=16, depth=1, num_heads=4,
    )
    model = DiT(config)
    checkpoint = tmp_path / "mnist.pt"
    names = tuple(str(index) for index in range(10))
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "diffusion": {"timesteps": 4},
            "dataset": "mnist",
            "class_names": names,
            "data_policy": {
                "curriculum": "canonical_overfit",
                "balanced_batches": False,
            },
        },
        checkpoint,
    )

    pipeline = DiTInference.from_checkpoint(checkpoint, "cpu")

    assert pipeline.class_names == names
    assert pipeline.metadata["dataset"] == "mnist"
    assert pipeline.metadata["data_policy"] == {
        "curriculum": "canonical_overfit",
        "balanced_batches": False,
    }


def test_inference_uses_checkpoint_paper_sampler_and_clip_policy(tmp_path, monkeypatch):
    config = DiTConfig(
        image_size=8, patch_size=2, in_channels=1, hidden_size=16,
        depth=1, num_heads=4, cfg_channels="first_3",
    )
    model = DiT(config)
    checkpoint = tmp_path / "paper.pt"
    recipe = {
        "name": "paper_compatible",
        "sampler": "paper_ddpm",
        "sampling_steps": 4,
        "sampling_clip_denoised": False,
    }
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "diffusion": {"timesteps": 10},
            "profile": "paper_compatible",
            "recipe": recipe,
            "pipeline": {"space": "pixel", "vae_id": None, "vae_scaling": None},
        },
        checkpoint,
    )
    calls = {}

    def fake_sample_loop(**kwargs):
        calls.update(kwargs)
        value = torch.zeros(kwargs["shape"])
        return value, [(10, value), (0, value)]

    def reject_ddim(**kwargs):
        raise AssertionError("paper profile must use ancestral DDPM")

    monkeypatch.setattr(inference_module, "sample_loop", fake_sample_loop)
    monkeypatch.setattr(inference_module, "ddim_sample_loop", reject_ddim)
    pipeline = DiTInference.from_checkpoint(checkpoint, "cpu")
    pipeline.generate(class_id=0, trajectory_frames=2)

    assert pipeline.metadata["profile"] == "paper_compatible"
    assert pipeline.metadata["pipeline"]["space"] == "pixel"
    assert calls["sampling_steps"] == 4
    assert calls["clip_denoised"] is False
