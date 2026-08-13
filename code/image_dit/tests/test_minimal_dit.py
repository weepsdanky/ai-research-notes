import torch
from pathlib import Path
import subprocess
import sys

from minimal_dit import MinimalDiT, MinimalDiTConfig, minimal_diffusion_loss


def tiny_model():
    return MinimalDiT(
        MinimalDiTConfig(
            image_size=8, patch_size=2, in_channels=1,
            hidden_size=16, depth=1, num_heads=4, mlp_ratio=2,
        )
    )


def test_minimal_dit_exposes_patch_transformer_image_pipeline():
    model = tiny_model()
    images = torch.randn(3, 1, 8, 8)
    assert model.patchify(images).shape == (3, 16, 16)
    assert model(images, torch.tensor([0, 1, 2])).shape == images.shape


def test_timestep_condition_changes_the_prediction():
    torch.manual_seed(0)
    model = tiny_model().eval()
    image = torch.randn(1, 1, 8, 8).repeat(2, 1, 1, 1)
    output = model(image, torch.tensor([0, 9]))
    assert not torch.allclose(output[0], output[1])


def test_minimal_ddpm_objective_is_finite_and_backpropagates():
    model = tiny_model()
    alpha_bar = torch.cumprod(1 - torch.linspace(1e-4, 0.02, 10), dim=0)
    images = torch.randn(2, 1, 8, 8)
    loss = minimal_diffusion_loss(
        model, images, torch.tensor([1, 8]), alpha_bar,
        noise=torch.ones_like(images),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.output.weight.grad is not None
    assert model.output.weight.grad.abs().sum() > 0


def test_minimal_cli_dry_run_writes_a_sample_and_self_describing_checkpoint(tmp_path):
    command = [
        sys.executable,
        "train_minimal.py",
        "--dry-run",
        "--device",
        "cpu",
        "--output-dir",
        str(tmp_path),
    ]

    completed = subprocess.run(
        command,
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "minimal_dit.pt").is_file()
    assert (tmp_path / "sample.png").is_file()
    checkpoint = torch.load(
        tmp_path / "minimal_dit.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["step"] == 1
    assert checkpoint["recipe"] == "minimal_epsilon_ddpm"
    assert checkpoint["omitted_tricks"] == [
        "class_conditioning_and_cfg",
        "adaln_zero",
        "learned_variance_and_vb",
        "ema",
        "special_initialization",
    ]
