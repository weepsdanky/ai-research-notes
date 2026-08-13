from argparse import Namespace
import json

import torch

from train_image_dit import (
    BalancedClassBatchSampler,
    CanonicalClassDataset,
    train,
)


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
    assert (tmp_path / "dit_image.pt").is_file()
    assert (tmp_path / "dit_image_inference.pt").is_file()

    checkpoint = __import__("torch").load(
        tmp_path / "dit_image.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["step"] == 1
    assert checkpoint["epoch"] == 0
    assert checkpoint["ema"]
    assert checkpoint["optimizer"]
    assert "scaler" in checkpoint
    assert set(checkpoint["rng_state"]) >= {"python", "numpy", "torch"}
    assert checkpoint["dataset"] == "cifar10"
    assert checkpoint["class_names"][3] == "cat"
    assert checkpoint["profile"] == "pixel_teaching"
    assert checkpoint["prediction_type"] == "epsilon"
    assert summary["prediction_type"] == "epsilon"
    assert checkpoint["data_policy"] == {
        "curriculum": "full_dataset",
        "balanced_batches": False,
    }
    assert summary["data_policy"] == checkpoint["data_policy"]
    assert checkpoint["recipe"]["sampling_clip_denoised"] is True
    assert checkpoint["pipeline"] == {
        "space": "pixel",
        "vae_id": None,
        "vae_scaling": None,
    }
    inference_checkpoint = torch.load(
        tmp_path / "dit_image_inference.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert "optimizer" not in inference_checkpoint
    assert "rng_state" not in inference_checkpoint
    assert {"model", "ema"} <= set(inference_checkpoint)
    assert inference_checkpoint["step"] == 1


def test_velocity_dry_run_persists_quality_ablation(tmp_path):
    args = Namespace(
        prediction_type="velocity", epochs=1, batch_size=2,
        learning_rate=1e-4, timesteps=10, hidden_size=16, depth=1,
        num_heads=4, patch_size=4, cfg_scale=2.0, seed=123,
        num_workers=0, output_dir=tmp_path, dry_run=True,
    )

    summary = train(args)
    checkpoint = torch.load(
        tmp_path / "dit_image.pt", map_location="cpu", weights_only=False
    )

    assert summary["prediction_type"] == "velocity"
    assert checkpoint["prediction_type"] == "velocity"


def test_canonical_curriculum_is_explicit_in_checkpoint_and_summary(tmp_path):
    args = Namespace(
        canonical_overfit=True, balanced_batches=False,
        dataset="mnist", epochs=1, batch_size=2, learning_rate=1e-4,
        timesteps=10, hidden_size=16, depth=1, num_heads=4, patch_size=4,
        cfg_scale=2.0, seed=123, num_workers=0, output_dir=tmp_path,
        dry_run=True,
    )

    summary = train(args)
    checkpoint = torch.load(
        tmp_path / "dit_image.pt", map_location="cpu", weights_only=False
    )

    expected = {"curriculum": "canonical_overfit", "balanced_batches": False}
    assert checkpoint["data_policy"] == expected
    assert summary["data_policy"] == expected


def test_resume_rejects_prediction_type_mismatch_and_old_checkpoint_defaults_epsilon(tmp_path):
    common = dict(
        epochs=1, batch_size=2, learning_rate=1e-4, timesteps=10,
        hidden_size=16, depth=1, num_heads=4, patch_size=4,
        cfg_scale=2.0, seed=123, num_workers=0, output_dir=tmp_path,
        dry_run=True,
    )
    train(Namespace(**common))
    checkpoint_path = tmp_path / "dit_image.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint.pop("prediction_type")
    old_checkpoint = tmp_path / "old.pt"
    torch.save(checkpoint, old_checkpoint)

    train(Namespace(**{**common, "output_dir": tmp_path / "epsilon"}, resume=old_checkpoint))
    with __import__("pytest").raises(ValueError, match="prediction type"):
        train(
            Namespace(
                **{**common, "output_dir": tmp_path / "velocity"},
                prediction_type="velocity", resume=old_checkpoint,
            )
        )


def test_legacy_checkpoint_without_data_policy_defaults_to_full_dataset(tmp_path):
    common = dict(
        dataset="mnist", epochs=1, batch_size=2, learning_rate=1e-4,
        timesteps=10, hidden_size=16, depth=1, num_heads=4, patch_size=4,
        cfg_scale=2.0, seed=123, num_workers=0, output_dir=tmp_path,
        dry_run=True,
    )
    train(Namespace(**common))
    checkpoint = torch.load(
        tmp_path / "dit_image.pt", map_location="cpu", weights_only=False
    )
    checkpoint.pop("data_policy")
    legacy = tmp_path / "legacy.pt"
    torch.save(checkpoint, legacy)

    summary = train(
        Namespace(
            **{**common, "output_dir": tmp_path / "full"},
            resume=legacy,
        )
    )
    assert summary["data_policy"]["curriculum"] == "full_dataset"

    with __import__("pytest").raises(ValueError, match="data policy"):
        train(
            Namespace(
                **{**common, "output_dir": tmp_path / "canonical"},
                canonical_overfit=True,
                resume=legacy,
            )
        )


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
            resume=tmp_path / "dit_image.pt",
        )
    )
    assert summary["steps"] == 2
    assert summary["resumed_from"].endswith("dit_image.pt")


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
            resume=tmp_path / "dit_image.pt",
            max_steps=1,
        )
    )
    assert summary["steps"] == 1
    assert summary["updates_this_run"] == 0


def test_resume_rejects_diffusion_schedule_mismatch(tmp_path):
    common = dict(
        epochs=1, batch_size=2, learning_rate=1e-4, timesteps=10,
        hidden_size=16, depth=1, num_heads=4, patch_size=4,
        cfg_scale=2.0, seed=123, num_workers=0, output_dir=tmp_path,
        dry_run=True,
    )
    train(Namespace(**common))
    checkpoint = torch.load(
        tmp_path / "dit_image.pt", map_location="cpu", weights_only=False
    )
    checkpoint["diffusion"]["beta_end"] = 0.03
    mismatch = tmp_path / "mismatch.pt"
    torch.save(checkpoint, mismatch)

    with __import__("pytest").raises(ValueError, match="diffusion schedule"):
        train(
            Namespace(
                **{**common, "output_dir": tmp_path / "resumed"},
                resume=mismatch,
            )
        )


def test_mnist_dry_run_persists_dataset_shape_and_digit_names(tmp_path):
    args = Namespace(
        dataset="mnist", epochs=1, batch_size=2, learning_rate=1e-4,
        timesteps=10, hidden_size=16, depth=1, num_heads=4, patch_size=4,
        cfg_scale=2.0, seed=123, num_workers=0, output_dir=tmp_path,
        dry_run=True,
    )
    summary = train(args)
    checkpoint = __import__("torch").load(
        tmp_path / "dit_image.pt", map_location="cpu", weights_only=False
    )

    assert summary["dataset"] == "mnist"
    assert checkpoint["config"]["image_size"] == 28
    assert checkpoint["config"]["in_channels"] == 1
    assert checkpoint["class_names"] == tuple(str(index) for index in range(10))


def test_paper_compatible_profile_persists_faithful_diffusion_policy(tmp_path):
    args = Namespace(
        profile="paper_compatible", dataset="mnist", epochs=1, batch_size=2,
        learning_rate=1e-4, timesteps=10, hidden_size=16, depth=1,
        num_heads=4, patch_size=4, cfg_scale=1.5, seed=123,
        num_workers=0, output_dir=tmp_path, dry_run=True,
    )
    train(args)
    checkpoint = __import__("torch").load(
        tmp_path / "dit_image.pt", map_location="cpu", weights_only=False
    )

    assert checkpoint["profile"] == "paper_compatible"
    assert checkpoint["config"]["cfg_channels"] == "first_3"
    assert checkpoint["recipe"] == {
        "name": "paper_compatible",
        "vb_mode": "raw",
        "loss_clip_denoised": False,
        "sampling_clip_denoised": False,
        "cfg_channels": "first_3",
        "sampler": "paper_ddpm",
        "sampling_steps": 250,
        "max_grad_norm": None,
        "amp": False,
    }


def test_paper_compatible_profile_rejects_velocity_ablation(tmp_path):
    args = Namespace(
        profile="paper_compatible", prediction_type="velocity",
        dataset="mnist", epochs=1, batch_size=2, learning_rate=1e-4,
        timesteps=10, hidden_size=16, depth=1, num_heads=4, patch_size=4,
        cfg_scale=1.5, seed=123, num_workers=0, output_dir=tmp_path,
        dry_run=True,
    )

    with __import__("pytest").raises(ValueError, match="requires epsilon"):
        train(args)


def test_balanced_batch_sampler_places_every_class_in_each_batch():
    labels = torch.arange(10).repeat_interleave(3)
    sampler = BalancedClassBatchSampler(labels, batch_size=20, seed=17)

    first_batch = next(iter(sampler))
    sampled_labels = labels[first_batch]

    assert len(first_batch) == 20
    assert torch.bincount(sampled_labels, minlength=10).tolist() == [2] * 10


def test_balanced_batch_sampler_changes_order_between_epochs_deterministically():
    labels = torch.arange(10).repeat_interleave(10)
    sampler = BalancedClassBatchSampler(labels, batch_size=20, seed=17)

    sampler.set_epoch(0)
    epoch_zero = list(iter(sampler))
    sampler.set_epoch(1)
    epoch_one = list(iter(sampler))
    sampler.set_epoch(0)

    assert epoch_zero != epoch_one
    assert list(iter(sampler)) == epoch_zero


def test_canonical_dataset_repeats_one_fixed_example_per_class():
    images = torch.arange(30, dtype=torch.float32).reshape(30, 1, 1, 1)
    labels = torch.arange(10).repeat(3)
    dataset = torch.utils.data.TensorDataset(images, labels)
    canonical = CanonicalClassDataset(dataset, num_classes=10)

    first_cycle = [canonical[index] for index in range(10)]
    second_cycle = [canonical[index + 10] for index in range(10)]

    assert [int(label) for _, label in first_cycle] == list(range(10))
    assert all(
        torch.equal(first_image, second_image)
        for (first_image, _), (second_image, _) in zip(
            first_cycle, second_cycle, strict=True
        )
    )
