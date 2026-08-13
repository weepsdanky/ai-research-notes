from dataclasses import asdict

import torch

from dit import DiT, DiTConfig
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
