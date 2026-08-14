import json

import torch

from condition_alignment import discover_generations, evaluate_alignment


def _metadata(path, prompt, *, mode="trained Video DiT checkpoint"):
    (path.parent / "model.pt").write_bytes(b"checkpoint")
    import hashlib
    path.with_suffix(".mp4").write_bytes(b"video")
    path.write_text(json.dumps({
        "mode": mode, "prompt": prompt, "seed": 42,
        "checkpoint": str(path.parent / "model.pt"), "training_step": 10,
        "weights": "ema", "prediction_type": "epsilon", "sampling_steps": 50,
        "start_timestep": 999,
        "clip_denoised": False,
        "effective_cfg_scale": 2.5, "condition_mode": "text_tokens",
        "text_encoder_id": "clip@sha256:test#tokens-v2",
        "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
        "video_filename": path.with_suffix(".mp4").name,
        "video_sha256": hashlib.sha256(b"video").hexdigest(),
    }), encoding="utf-8")


def test_discovery_ignores_metrics_and_requires_matching_video(tmp_path):
    _metadata(tmp_path / "cat.json", "a cat walking")
    (tmp_path / "cat_temporal-metrics.json").write_text("{}", encoding="utf-8")

    rows = discover_generations(tmp_path)

    assert [(row.prompt, row.video_path.name) for row in rows] == [
        ("a cat walking", "cat.mp4")
    ]


def test_discovery_can_lock_one_trajectory_timestep(tmp_path):
    _metadata(
        tmp_path / "cat.json", "a cat walking",
        mode="trained Video DiT trajectory state",
    )
    data = json.loads((tmp_path / "cat.json").read_text())
    data["trajectory_timestep"] = 500
    (tmp_path / "cat.json").write_text(json.dumps(data))

    rows = discover_generations(
        tmp_path, artifact_mode="trajectory", trajectory_timestep=500,
    )

    assert len(rows) == 1


def test_trajectory_report_records_its_selected_artifact_contract(tmp_path):
    for name, prompt in (("cat", "a cat walking"), ("dog", "a dog swimming")):
        _metadata(
            tmp_path / f"{name}.json", prompt,
            mode="trained Video DiT trajectory state",
        )
        data = json.loads((tmp_path / f"{name}.json").read_text())
        data["trajectory_timestep"] = 500
        (tmp_path / f"{name}.json").write_text(json.dumps(data))

    report = evaluate_alignment(
        tmp_path, encoder_id="clip@test", artifact_mode="trajectory",
        trajectory_timestep=500, encode_videos=lambda paths: torch.eye(2),
        encode_prompts=lambda prompts: torch.eye(2),
    )

    assert report["artifact_mode"] == "trajectory"
    assert report["trajectory_timestep"] == 500
    assert report["generation_contract"]


def test_alignment_uses_frozen_features_and_reports_encoder_identity(tmp_path):
    _metadata(tmp_path / "cat.json", "a cat walking")
    _metadata(tmp_path / "dog.json", "a dog swimming")
    features = {
        "cat.mp4": torch.tensor([1.0, 0.0]),
        "dog.mp4": torch.tensor([0.0, 1.0]),
    }

    report = evaluate_alignment(
        tmp_path,
        encoder_id="clip@sha256:test",
        encode_videos=lambda paths: torch.stack([features[path.name] for path in paths]),
        encode_prompts=lambda prompts: torch.eye(2),
    )

    assert report["encoder_id"] == "clip@sha256:test"
    assert report["prompt_top1_accuracy"] == 1.0
    assert report["prompt_margin_mean"] == 1.0


def test_discovery_fails_closed_on_missing_video(tmp_path):
    (tmp_path / "broken.json").write_text(json.dumps({
        "mode": "trained Video DiT checkpoint", "prompt": "a cat", "seed": 0,
    }), encoding="utf-8")

    with __import__("pytest").raises(ValueError, match="matching MP4"):
        discover_generations(tmp_path)


def test_alignment_aggregates_matched_prompt_blocks_across_seeds(tmp_path):
    _metadata(tmp_path / "cat.json", "a cat walking")
    _metadata(tmp_path / "dog.json", "a dog swimming")
    _metadata(tmp_path / "cat-7.json", "a cat walking")
    _metadata(tmp_path / "dog-7.json", "a dog swimming")
    for name in ("cat-7", "dog-7"):
        data = json.loads((tmp_path / f"{name}.json").read_text(encoding="utf-8"))
        data["seed"] = 7
        (tmp_path / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
    features = {
        "cat.mp4": torch.tensor([1.0, 0.0]), "dog.mp4": torch.tensor([0.0, 1.0]),
        "cat-7.mp4": torch.tensor([1.0, 0.0]), "dog-7.mp4": torch.tensor([0.0, 1.0]),
    }

    report = evaluate_alignment(
        tmp_path, encoder_id="clip@test",
        encode_videos=lambda paths: torch.stack([features[p.name] for p in paths]),
        encode_prompts=lambda prompts: torch.eye(2),
    )

    assert report["seeds"] == [7, 42]
    assert report["samples"] == 4
    assert report["prompt_top1_accuracy"] == 1.0
    assert report["shuffled_prompt_top1_accuracy"] == 0.0
    assert set(report["per_seed"]) == {"7", "42"}


def test_alignment_rejects_different_prompt_sets_across_seeds(tmp_path):
    _metadata(tmp_path / "cat.json", "a cat walking")
    _metadata(tmp_path / "dog.json", "a dog swimming")
    data = json.loads((tmp_path / "dog.json").read_text(encoding="utf-8"))
    data["seed"] = 7
    (tmp_path / "dog.json").write_text(json.dumps(data), encoding="utf-8")

    with __import__("pytest").raises(ValueError, match="same unique prompt set"):
        evaluate_alignment(
            tmp_path, encoder_id="clip@test",
            encode_videos=lambda paths: torch.eye(2),
            encode_prompts=lambda prompts: torch.eye(2),
        )


def test_discovery_rejects_mixed_generation_contracts(tmp_path):
    _metadata(tmp_path / "cat.json", "a cat walking")
    _metadata(tmp_path / "dog.json", "a dog swimming")
    data = json.loads((tmp_path / "dog.json").read_text(encoding="utf-8"))
    data["effective_cfg_scale"] = 7.0
    (tmp_path / "dog.json").write_text(json.dumps(data), encoding="utf-8")

    with __import__("pytest").raises(ValueError, match="generation contract"):
        discover_generations(tmp_path)


def test_discovery_rejects_unverifiable_legacy_metadata(tmp_path):
    _metadata(tmp_path / "cat.json", "a cat walking")
    data = json.loads((tmp_path / "cat.json").read_text(encoding="utf-8"))
    del data["checkpoint_sha256"]
    (tmp_path / "cat.json").write_text(json.dumps(data), encoding="utf-8")

    with __import__("pytest").raises(ValueError, match="valid checkpoint SHA-256"):
        discover_generations(tmp_path)


def test_discovery_rejects_generation_video_changed_after_metadata(tmp_path):
    _metadata(tmp_path / "cat.json", "a cat walking")
    (tmp_path / "cat.mp4").write_bytes(b"changed")

    with __import__("pytest").raises(ValueError, match="video SHA-256"):
        discover_generations(tmp_path)
