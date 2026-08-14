import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import generation_blind_review

from generation_blind_review import (
    _factor_derangement,
    build_generation_evaluation_spec,
    create_blind_generation_pool,
    reveal_blind_generation_results,
)
from real_data import CompositionalSplit, RealVideoRecord, validate_split


def _record(index, split, subject, action, scene, prompt):
    return RealVideoRecord(
        sample_id=f"sample-{index}", source_id=f"source-{index}",
        duplicate_group=f"group-{index}", clip_path=f"clips/{index}.mp4",
        prompt=prompt, subject=subject, action=action, scene=scene,
        source_url=f"https://example.test/{index}", license="CC BY 4.0",
        attribution="Tester", sha256=f"{index + 10:064x}",
        perceptual_hash=hashlib.sha256(f"phash-{index}".encode()).hexdigest()[:16],
        split=split,
    )


def _split_contract(checkpoint_sha="0" * 64, seeds=(42, 99)):
    records = []
    index = 1
    for subject, action, scene, prompt in (
        ("cat", "playing", "outdoors", "train cat playing outdoors"),
        ("dog", "playing", "indoors", "train dog playing indoors"),
        ("cat", "sitting", "indoors", "train cat sitting indoors"),
    ):
        for _ in range(10):
            records.append(_record(index, "train", subject, action, scene, prompt))
            index += 1
    records.append(_record(
        index, "train", "dog", "playing", "outdoors", "train dog playing outdoors",
    )); index += 1
    for _ in range(50):
        records.append(_record(
            index, "c1", "cat", "playing", "indoors", "a cat playing indoors",
        )); index += 1
    records.append(_record(
        index, "iid", "dog", "playing", "outdoors", "a dog playing outdoors",
    ))
    policy = CompositionalSplit(
        c1_triples={("cat", "playing", "indoors")}, c2_pairs=set(),
        iid_fraction=0, validation_fraction=0,
        min_c1_source_groups=50, min_train_pair_source_groups=10,
    )
    report = validate_split(records, policy)
    manifest_bytes = "".join(
        json.dumps(asdict(record), sort_keys=True) + "\n" for record in records
    ).encode()
    report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    generation_contract = {
        "checkpoint_sha256": checkpoint_sha,
        "training_step": 10000,
        "weights": "ema",
        "prediction_type": "velocity",
        "sampling_steps": 50,
        "start_timestep": 999,
        "clip_denoised": False,
        "effective_cfg_scale": 1.0,
        "condition_mode": "text",
        "text_encoder_id": "encoder#tokens-v2",
        "training_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    spec = build_generation_evaluation_spec(
        manifest_bytes, report_bytes,
        generation_contract=generation_contract, seeds=list(seeds),
    )
    return _canonical(spec), manifest_bytes, report_bytes


def _canonical(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def test_factor_derangement_finds_non_cyclic_perfect_mismatch():
    labels = ("a", "a", "b", "c", "b")
    rows = [
        {"prompt": f"p{index}", "expected_subject": label,
         "expected_action": "act", "expected_scene": "scene"}
        for index, label in enumerate(labels)
    ]
    replacements = _factor_derangement(rows)
    assert sorted(id(row) for row in replacements) == sorted(id(row) for row in rows)
    assert all(
        row["expected_subject"] != replacement["expected_subject"]
        for row, replacement in zip(rows, replacements, strict=True)
    )


def _write_generations(root: Path, *, omit_last: bool = False):
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "model.pt"
    checkpoint.write_bytes(b"locked model")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    prompts = (
        ("a cat playing indoors", "cat", "playing", "indoors", "c1"),
        ("a dog playing outdoors", "dog", "playing", "outdoors", "iid"),
    )
    spec_bytes, manifest_bytes, report_bytes = _split_contract(checkpoint_sha)
    generation_dir = root / "generations"
    generation_dir.mkdir()
    combinations = [(42, prompts[0]), (42, prompts[1]), (99, prompts[0]), (99, prompts[1])]
    if omit_last:
        combinations.pop()
    for index, (seed, (prompt, *_)) in enumerate(combinations):
        stem = f"generated-{index}"
        (generation_dir / f"{stem}.mp4").write_bytes(f"video-{index}".encode())
        video_bytes = f"video-{index}".encode()
        metadata = {
            "mode": "trained Video DiT checkpoint", "prompt": prompt,
            "parsed_prompt": None, "condition_mode": "text",
            "text_encoder_id": "encoder#tokens-v2", "seed": seed,
            "sampling_steps": 50, "start_timestep": 999,
            "effective_cfg_scale": 1.0, "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha, "training_step": 10000,
            "weights": "ema", "prediction_type": "velocity",
            "clip_denoised": False,
            "training_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "evaluation_spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
            "video_filename": f"{stem}.mp4",
            "video_sha256": hashlib.sha256(video_bytes).hexdigest(),
        }
        (generation_dir / f"{stem}.json").write_text(json.dumps(metadata))
    return spec_bytes, manifest_bytes, report_bytes, generation_dir


def _decision(queue: dict, reviewer: str, predictions: dict[str, tuple[str, str, str]]) -> bytes:
    value = {
        "protocol": "blinded generated-video factor review v1",
        "queue_sha256": hashlib.sha256(
            (json.dumps(queue, indent=2, sort_keys=True) + "\n").encode()
        ).hexdigest(),
        "reviewer_id": reviewer,
        "records": [{
            "blind_id": blind_id,
            "predicted_subject": factors[0],
            "predicted_action": factors[1],
            "predicted_scene": factors[2],
            "observed_video_sha256": next(
                row["video_sha256"] for row in queue["records"]
                if row["blind_id"] == blind_id
            ),
            "notes": "reviewed full temporal video",
        } for blind_id, factors in predictions.items()],
    }
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def test_create_pool_hides_prompt_split_seed_and_copies_locked_media(tmp_path):
    spec, manifest, report, generation_dir = _write_generations(tmp_path)
    media_dir = tmp_path / "blind-media"
    queue, key = create_blind_generation_pool(
        generation_dir, spec, manifest, report, media_dir, seed=7,
    )
    assert len(queue["records"]) == 4
    assert queue["seeds"] == [42, 99] and queue["prompts_per_seed"] == 2
    assert queue["factor_vocabulary"] == {
        "subjects": ["cat", "dog", "absent_or_unclear"],
        "actions": ["playing", "absent_or_unclear"],
        "scenes": ["indoors", "outdoors", "absent_or_unclear"],
    }
    assert all(
        not any(field in row for field in ("prompt", "split", "seed", "expected_subject"))
        for row in queue["records"]
    )
    assert {path.name for path in media_dir.glob("*.mp4")} == {
        f"{row['blind_id']}.mp4" for row in queue["records"]
    }
    assert key["queue_sha256"] == hashlib.sha256(
        (json.dumps(queue, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    assert all("prompt" in row and "split" in row and "seed" in row for row in key["records"])


def test_create_pool_rejects_incomplete_seed_blocks(tmp_path):
    spec, manifest, report, generation_dir = _write_generations(tmp_path, omit_last=True)
    with pytest.raises(ValueError, match="same prompt set"):
        create_blind_generation_pool(
            generation_dir, spec, manifest, report, tmp_path / "blind", seed=7,
        )
    assert not (tmp_path / "blind").exists()


def test_create_pool_rejects_seed_cherry_picking_and_training_manifest_drift(tmp_path):
    spec, manifest, report, generation_dir = _write_generations(tmp_path)
    for path in generation_dir.glob("generated-2.*"):
        path.unlink()
    for path in generation_dir.glob("generated-3.*"):
        path.unlink()
    with pytest.raises(ValueError, match="seed roster"):
        create_blind_generation_pool(
            generation_dir, spec, manifest, report, tmp_path / "blind", seed=7,
        )

    spec, manifest, report, generation_dir = _write_generations(tmp_path / "second")
    metadata_path = next(generation_dir.glob("*.json"))
    metadata = json.loads(metadata_path.read_text())
    metadata["training_manifest_sha256"] = "f" * 64
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="training manifest differs"):
        create_blind_generation_pool(
            generation_dir, spec, manifest, report, tmp_path / "blind-2", seed=7,
        )


def test_spec_requires_two_preregistered_seeds():
    _, manifest, report = _split_contract()
    contract = {
        "checkpoint_sha256": "0" * 64, "training_step": 1, "weights": "ema",
        "prediction_type": "velocity", "sampling_steps": 50,
        "start_timestep": 999, "clip_denoised": False,
        "effective_cfg_scale": 1.0, "condition_mode": "text",
        "text_encoder_id": "encoder#tokens-v2",
        "training_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }
    with pytest.raises(ValueError, match="at least two unique sorted seeds"):
        build_generation_evaluation_spec(
            manifest, report, generation_contract=contract, seeds=[42],
        )


def test_create_pool_rejects_a_fabricated_factor_spec(tmp_path):
    spec, manifest, report, generation_dir = _write_generations(tmp_path)
    value = json.loads(spec)
    value["records"][0]["scene"] = "snow"
    with pytest.raises(ValueError, match="validated split manifest"):
        create_blind_generation_pool(
            generation_dir, _canonical(value), manifest, report,
            tmp_path / "blind", seed=7,
        )
    assert not (tmp_path / "blind").exists()


def test_create_cli_cleans_media_when_queue_key_commit_fails(tmp_path, monkeypatch):
    spec, manifest, report, generation_dir = _write_generations(tmp_path)
    spec_path = tmp_path / "spec.json"; spec_path.write_bytes(spec)
    manifest_path = tmp_path / "split.jsonl"; manifest_path.write_bytes(manifest)
    report_path = tmp_path / "report.json"; report_path.write_bytes(report)
    media = tmp_path / "blind-media"
    queue = tmp_path / "queue.json"
    key = tmp_path / "key.json"

    def fail_commit(*args, **kwargs):
        raise OSError("simulated queue/key commit failure")

    monkeypatch.setattr(generation_blind_review, "_atomic_pair", fail_commit)
    with pytest.raises(OSError, match="simulated"):
        generation_blind_review.main([
            "create", str(generation_dir), str(spec_path), str(manifest_path),
            str(report_path), str(media), str(queue), str(key), "--seed", "7",
        ])
    assert not media.exists() and not queue.exists() and not key.exists()


def test_reveal_requires_two_raters_and_reports_matched_vs_shuffled(tmp_path):
    spec, manifest, split_report, generation_dir = _write_generations(tmp_path)
    media_dir = tmp_path / "blind-media"
    queue, key = create_blind_generation_pool(
        generation_dir, spec, manifest, split_report, media_dir, seed=7,
    )
    expected = {
        row["blind_id"]: (
            row["expected_subject"], row["expected_action"], row["expected_scene"],
        ) for row in key["records"]
    }
    decisions = [_decision(queue, "rater-a", expected), _decision(queue, "rater-b", expected)]
    queue_bytes = (json.dumps(queue, indent=2, sort_keys=True) + "\n").encode()
    key_bytes = (json.dumps(key, indent=2, sort_keys=True) + "\n").encode()
    report = reveal_blind_generation_results(
        queue_bytes, key_bytes, decisions, media_dir,
    )
    assert report["raters"] == 2
    assert report["overall"]["unanimous_triple_accuracy"] == 1.0
    assert report["overall"]["shuffled_triple_accuracy"] == 0.0
    assert report["overall"]["gain_over_shuffled"] == 1.0
    assert report["c1"]["samples"] == 2
    assert report["iid"]["samples"] == 2
    assert report["c2"]["status"] == "missing"

    with pytest.raises(ValueError, match="at least two"):
        reveal_blind_generation_results(queue_bytes, key_bytes, decisions[:1], media_dir)


def test_reveal_counts_disagreement_as_ambiguous_and_rejects_media_drift(tmp_path):
    spec, manifest, split_report, generation_dir = _write_generations(tmp_path)
    media_dir = tmp_path / "blind-media"
    queue, key = create_blind_generation_pool(
        generation_dir, spec, manifest, split_report, media_dir, seed=9,
    )
    expected = {
        row["blind_id"]: (
            row["expected_subject"], row["expected_action"], row["expected_scene"],
        ) for row in key["records"]
    }
    changed = dict(expected)
    first = next(iter(changed))
    changed[first] = ("dog" if changed[first][0] == "cat" else "cat", *changed[first][1:])
    queue_bytes = (json.dumps(queue, indent=2, sort_keys=True) + "\n").encode()
    key_bytes = (json.dumps(key, indent=2, sort_keys=True) + "\n").encode()
    report = reveal_blind_generation_results(
        queue_bytes, key_bytes,
        [_decision(queue, "a", expected), _decision(queue, "b", changed)], media_dir,
    )
    assert report["overall"]["unanimous_samples"] == 3
    assert report["overall"]["ambiguous_samples"] == 1
    assert report["overall"]["unanimous_triple_accuracy"] == 0.75

    (media_dir / f"{first}.mp4").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="media SHA-256"):
        reveal_blind_generation_results(
            queue_bytes, key_bytes,
            [_decision(queue, "a", expected), _decision(queue, "b", expected)], media_dir,
        )


def test_absent_or_unclear_is_available_and_never_counts_as_expected(tmp_path):
    spec, manifest, split_report, generation_dir = _write_generations(tmp_path)
    media_dir = tmp_path / "blind-media"
    queue, key = create_blind_generation_pool(
        generation_dir, spec, manifest, split_report, media_dir, seed=9,
    )
    unclear = {
        row["blind_id"]: ("absent_or_unclear",) * 3 for row in key["records"]
    }
    report = reveal_blind_generation_results(
        _canonical(queue), _canonical(key),
        [_decision(queue, "a", unclear), _decision(queue, "b", unclear)], media_dir,
    )
    assert report["overall"]["unanimous_triple_accuracy"] == 0.0


def test_reveal_rejects_shuffled_control_without_factor_derangement(tmp_path):
    spec, manifest, split_report, generation_dir = _write_generations(tmp_path)
    media_dir = tmp_path / "blind-media"
    queue, key = create_blind_generation_pool(
        generation_dir, spec, manifest, split_report, media_dir, seed=9,
    )
    for row in key["records"]:
        row["expected_subject"] = "cat"
        row["expected_action"] = "playing"
        row["expected_scene"] = "indoors"
    commitment = hashlib.sha256(
        (json.dumps(key["records"], separators=(",", ":"), sort_keys=True) + "\n").encode()
    ).hexdigest()
    queue["assignment_commitment_sha256"] = commitment
    key["assignment_commitment_sha256"] = commitment
    key["queue_sha256"] = hashlib.sha256(_canonical(queue)).hexdigest()
    expected = {
        row["blind_id"]: ("cat", "playing", "indoors") for row in key["records"]
    }
    with pytest.raises(ValueError, match="factor-triple derangement"):
        reveal_blind_generation_results(
            _canonical(queue), _canonical(key),
            [_decision(queue, "a", expected), _decision(queue, "b", expected)], media_dir,
        )
