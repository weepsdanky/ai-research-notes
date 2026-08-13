from pathlib import Path
import hashlib
import shutil

import torch

from dataset import MOTIONS
from train import horizontal_flip_batch, parse_args, train
from real_data import RealVideoRecord, write_manifest


def _small_training_args(output_dir, *, max_steps, resume=None):
    values = [
        "--mode",
        "concept",
        "--device",
        "cpu",
        "--max-steps",
        str(max_steps),
        "--batch-size",
        "2",
        "--frames",
        "4",
        "--patch-size",
        "8",
        "--hidden-size",
        "32",
        "--num-heads",
        "4",
        "--num-pairs",
        "1",
        "--sample-every",
        "0",
        "--checkpoint-every",
        "0",
        "--output-dir",
        str(output_dir),
    ]
    if resume is not None:
        values.extend(["--resume", str(resume)])
    return parse_args(values)


def test_horizontal_flip_keeps_motion_condition_semantically_correct():
    video = torch.arange(2 * 3 * 4 * 8 * 8).reshape(2, 3, 4, 8, 8)
    conditions = torch.tensor(
        [
            [0, 0, MOTIONS.index("walk_right"), 0],
            [1, 1, MOTIONS.index("bounce"), 1],
        ]
    )

    flipped, adjusted = horizontal_flip_batch(
        video, conditions, probability=1.0
    )

    assert torch.equal(flipped, video.flip(-1))
    assert adjusted[0, 2] == MOTIONS.index("walk_left")
    assert adjusted[1, 2] == MOTIONS.index("bounce")


def test_cpu_dry_run_performs_an_update_and_writes_a_complete_checkpoint(tmp_path):
    output_dir = tmp_path / "dry-run"
    args = parse_args(
        [
            "--dry-run",
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    )

    summary = train(args)

    checkpoint_path = Path(summary["checkpoint"])
    inference_checkpoint_path = Path(summary["inference_checkpoint"])
    assert summary["device"] == "cpu"
    assert summary["updates_this_run"] == 1
    assert checkpoint_path.is_file()
    assert inference_checkpoint_path.is_file()
    inference_checkpoint = torch.load(
        inference_checkpoint_path, map_location="cpu", weights_only=True
    )
    assert "optimizer" not in inference_checkpoint
    assert "rng_state" not in inference_checkpoint
    assert {"model", "ema"} <= set(inference_checkpoint)
    assert inference_checkpoint["step"] == 1
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert {"model", "ema", "optimizer", "config", "diffusion", "step"} <= set(
        checkpoint
    )
    assert checkpoint["step"] == 1
    assert checkpoint["config"]["frames"] == 4
    assert checkpoint["curriculum"] == "canonical"
    assert checkpoint["data_contract"] == {
        "device_type": "cpu",
        "seed": 42,
        "batch_size": 2,
        "horizontal_flip_prob": 0.5,
        "num_workers": 0,
    }
    assert checkpoint["optimization_contract"] == {
        "effective_ema_decay": 0.9999,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
    }
    assert summary["curriculum"] == "canonical"
    assert (output_dir / "step_log.csv").is_file()
    assert summary["elapsed_seconds"] >= 0


def test_default_training_profile_matches_the_documented_first_quality_run():
    args = parse_args([])

    assert args.mode == "faithful"
    assert args.curriculum == "canonical"
    assert args.max_steps == 3_000
    assert args.batch_size == 8
    assert args.sample_every == 1_000
    assert args.checkpoint_every == 1_000
    assert args.prediction_type == "epsilon"


def test_real_training_rejects_legacy_structured_periodic_sampler(tmp_path):
    args = parse_args([])
    args.dataset = "real"
    args.manifest = tmp_path / "manifest.jsonl"
    args.manifest.write_text("")
    args.text_embedding_size = 512
    args.horizontal_flip_prob = 0
    args.sample_every = 100

    with __import__("pytest").raises(ValueError, match="sample-every 0"):
        train(args)


def test_velocity_quality_ablation_is_explicit_and_checkpointed(tmp_path):
    args = _small_training_args(tmp_path / "velocity", max_steps=1)
    args.prediction_type = "velocity"

    summary = train(args)
    checkpoint = torch.load(
        summary["checkpoint"], map_location="cpu", weights_only=False
    )
    inference_checkpoint = torch.load(
        summary["inference_checkpoint"], map_location="cpu", weights_only=True
    )

    assert summary["prediction_type"] == "velocity"
    assert checkpoint["prediction_type"] == "velocity"


def test_resume_reproduces_the_second_loss_and_weights_of_an_uninterrupted_run(
    tmp_path,
):
    continuous = train(_small_training_args(tmp_path / "continuous", max_steps=2))
    first_leg = train(_small_training_args(tmp_path / "split", max_steps=1))
    resumed = train(
        _small_training_args(
            tmp_path / "resumed",
            max_steps=2,
            resume=first_leg["checkpoint"],
        )
    )

    assert resumed["final_loss"] == continuous["final_loss"]
    continuous_state = torch.load(
        continuous["checkpoint"], map_location="cpu", weights_only=False
    )["model"]
    resumed_state = torch.load(
        resumed["checkpoint"], map_location="cpu", weights_only=False
    )["model"]
    assert continuous_state.keys() == resumed_state.keys()
    for name in continuous_state:
        assert torch.equal(continuous_state[name], resumed_state[name]), name


def test_resume_rejects_mode_mismatch(tmp_path):
    first_leg = train(_small_training_args(tmp_path / "concept", max_steps=1))
    args = _small_training_args(
        tmp_path / "faithful", max_steps=2, resume=first_leg["checkpoint"]
    )
    args.mode = "faithful"
    args.learn_sigma = False
    args.condition_dropout_prob = 0.0

    with __import__("pytest").raises(ValueError, match="mode"):
        train(args)


def test_resume_rejects_diffusion_schedule_mismatch(tmp_path):
    first_leg = train(_small_training_args(tmp_path / "first", max_steps=1))
    args = _small_training_args(
        tmp_path / "mismatch", max_steps=2, resume=first_leg["checkpoint"]
    )
    args.timesteps = 9

    with __import__("pytest").raises(ValueError, match="diffusion schedule"):
        train(args)


def test_resume_rejects_data_contract_mismatch(tmp_path):
    first_leg = train(_small_training_args(tmp_path / "first", max_steps=1))
    for field, value in (
        ("seed", 99),
        ("batch_size", 4),
        ("horizontal_flip_prob", 0.25),
        ("num_workers", 1),
    ):
        args = _small_training_args(
            tmp_path / f"mismatch-{field}",
            max_steps=2,
            resume=first_leg["checkpoint"],
        )
        setattr(args, field, value)
        with __import__("pytest").raises(ValueError, match="data contract"):
            train(args)

    checkpoint = torch.load(
        first_leg["checkpoint"], map_location="cpu", weights_only=False
    )
    checkpoint["data_contract"]["device_type"] = "mps"
    wrong_device = tmp_path / "wrong-device.pt"
    torch.save(checkpoint, wrong_device)
    args = _small_training_args(
        tmp_path / "mismatch-device", max_steps=2, resume=wrong_device
    )
    with __import__("pytest").raises(ValueError, match="data contract"):
        train(args)


def test_resume_rejects_optimization_contract_mismatch(tmp_path):
    first_leg = train(_small_training_args(tmp_path / "first", max_steps=1))
    args = _small_training_args(
        tmp_path / "mismatch-grad-clip", max_steps=2,
        resume=first_leg["checkpoint"],
    )
    args.max_grad_norm = 0.5

    with __import__("pytest").raises(ValueError, match="optimization contract"):
        train(args)

    args = _small_training_args(
        tmp_path / "mismatch-learning-rate", max_steps=2,
        resume=first_leg["checkpoint"],
    )
    args.learning_rate = 2e-4
    with __import__("pytest").raises(ValueError, match="optimization contract"):
        train(args)


def test_real_manifest_text_embedding_performs_one_training_update(tmp_path):
    source = Path(__file__).parents[1] / "demo_artifacts" / "samples" / (
        "model_cat_orange_walk_right_snow_seed-42.mp4"
    )
    clip = tmp_path / "cat.mp4"
    shutil.copyfile(source, clip)
    embedding = tmp_path / "cat.text.pt"
    torch.save(torch.arange(12, dtype=torch.float32), embedding)
    manifest = write_manifest(
        tmp_path / "manifest.jsonl",
        [
            RealVideoRecord(
                sample_id="cat-1",
                source_id="demo-cat",
                clip_path=clip.name,
                prompt="an orange cat walking in snow",
                subject="cat",
                action="walking",
                scene="snow",
                source_url="repo://demo-cat",
                license="repository-demo",
                attribution="ai-research-notes",
                sha256=hashlib.sha256(clip.read_bytes()).hexdigest(),
                text_embedding_path=embedding.name,
                text_encoder_id="openai/clip-vit-base-patch32@main",
                split="train",
            )
        ],
    )
    args = parse_args(
        [
            "--dry-run", "--device", "cpu", "--dataset", "real",
            "--manifest", str(manifest), "--text-embedding-size", "12",
            "--batch-size", "1", "--horizontal-flip-prob", "0",
            "--sample-every", "0",
            "--output-dir", str(tmp_path / "run"),
        ]
    )

    summary = train(args)
    checkpoint = torch.load(
        summary["checkpoint"], map_location="cpu", weights_only=False
    )
    inference_checkpoint = torch.load(
        summary["inference_checkpoint"], map_location="cpu", weights_only=True
    )

    assert summary["updates_this_run"] == 1
    assert checkpoint["config"]["condition_mode"] == "text"
    assert checkpoint["curriculum"] == "real-manifest"
    assert checkpoint["data_contract"]["manifest_sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    assert checkpoint["data_contract"]["text_encoder_id"] == (
        "openai/clip-vit-base-patch32@main"
    )
    assert inference_checkpoint["data_contract"] == checkpoint["data_contract"]


def test_real_latent_manifest_records_vae_contract_in_checkpoint(tmp_path):
    text = tmp_path / "cat.text.pt"
    latent = tmp_path / "cat.latent.pt"
    torch.save(torch.ones(12), text)
    torch.save(torch.randn(4, 4, 4, 4), latent)
    manifest = write_manifest(
        tmp_path / "latent.jsonl",
        [
            RealVideoRecord(
                sample_id="cat-latent",
                source_id="source-cat",
                clip_path="unused.mp4",
                prompt="a cat running on grass",
                subject="cat", action="running", scene="grass",
                source_url="https://example.test/cat",
                license="CC0", attribution="Example", sha256="a" * 64,
                text_embedding_path=text.name,
                text_encoder_id="openai/clip@sha256:abc",
                latent_path=latent.name,
                vae_id="stabilityai/sd-vae-ft-mse@sha256:def",
                vae_scaling=0.18215,
                split="train",
            )
        ],
    )
    args = parse_args(
        [
            "--dry-run", "--device", "cpu", "--dataset", "real",
            "--space", "latent", "--no-learn-sigma", "--manifest", str(manifest),
            "--text-embedding-size", "12", "--batch-size", "1",
            "--horizontal-flip-prob", "0", "--sample-every", "0",
            "--output-dir", str(tmp_path / "run"),
        ]
    )

    summary = train(args)
    checkpoint = torch.load(summary["checkpoint"], weights_only=False)

    assert checkpoint["config"]["in_channels"] == 4
    assert checkpoint["config"]["image_size"] == 4
    assert checkpoint["data_contract"]["space"] == "latent"
    assert checkpoint["data_contract"]["vae_id"].endswith("sha256:def")
    assert checkpoint["data_contract"]["vae_scaling"] == 0.18215


def test_latent_training_rejects_pixel_specific_learned_variance(tmp_path):
    args = parse_args([])
    args.dataset = "real"
    args.space = "latent"
    args.manifest = tmp_path / "manifest.jsonl"
    args.manifest.write_text("")
    args.text_embedding_size = 512
    args.horizontal_flip_prob = 0
    args.sample_every = 0

    with __import__("pytest").raises(ValueError, match="pixel-specific"):
        train(args)
