from argparse import Namespace
import json

from train_cifar10_dit import train


def test_dry_run_writes_checkpoint_log_and_summary(tmp_path):
    args = Namespace(
        epochs=1,
        batch_size=2,
        learning_rate=1e-4,
        timesteps=10,
        hidden_size=16,
        depth=1,
        num_heads=4,
        patch_size=4,
        cfg_scale=2.0,
        seed=123,
        num_workers=0,
        output_dir=tmp_path,
        dry_run=True,
    )
    summary = train(args)
    persisted = json.loads((tmp_path / "summary.json").read_text())
    assert summary["steps"] == 1
    assert summary["dry_run"] is True
    assert persisted == summary
    assert (tmp_path / "training_log.csv").is_file()
    assert (tmp_path / "step_log.csv").is_file()
    assert (tmp_path / "dit_cifar10.pt").is_file()

    checkpoint = __import__("torch").load(
        tmp_path / "dit_cifar10.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["step"] == 1
    assert checkpoint["epoch"] == 0
    assert checkpoint["ema"]
    assert checkpoint["optimizer"]
    assert "scaler" in checkpoint
    assert set(checkpoint["rng_state"]) >= {"python", "numpy", "torch"}


def test_dry_run_can_resume_from_checkpoint(tmp_path):
    common = dict(
        epochs=1,
        batch_size=2,
        learning_rate=1e-4,
        timesteps=10,
        hidden_size=16,
        depth=1,
        num_heads=4,
        patch_size=4,
        cfg_scale=2.0,
        seed=123,
        num_workers=0,
        output_dir=tmp_path,
        dry_run=True,
    )
    train(Namespace(**common))
    resumed_dir = tmp_path / "resumed"
    summary = train(
        Namespace(
            **{**common, "output_dir": resumed_dir},
            resume=tmp_path / "dit_cifar10.pt",
        )
    )
    assert summary["steps"] == 2
    assert summary["resumed_from"].endswith("dit_cifar10.pt")


def test_resume_at_max_steps_does_not_update_model(tmp_path):
    common = dict(
        epochs=1, batch_size=2, learning_rate=1e-4, timesteps=10,
        hidden_size=16, depth=1, num_heads=4, patch_size=4,
        cfg_scale=2.0, seed=123, num_workers=0, output_dir=tmp_path,
        dry_run=True,
    )
    train(Namespace(**common))
    resumed_dir = tmp_path / "already-finished"
    summary = train(
        Namespace(
            **{**common, "output_dir": resumed_dir},
            resume=tmp_path / "dit_cifar10.pt",
            max_steps=1,
        )
    )
    assert summary["steps"] == 1
    assert summary["updates_this_run"] == 0
