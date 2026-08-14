import hashlib
import json
from dataclasses import replace
from pathlib import Path
import subprocess

from prepare_local_contribution import main
from real_data import RealVideoRecord, load_manifest, write_manifest


def _video(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
            "testsrc2=size=128x96:rate=8:duration=3", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )


def test_create_cli_builds_a_portable_rights_locked_contribution_bundle(tmp_path):
    media = tmp_path / "source.mp4"
    _video(media)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({
        "schema_version": 1,
        "contribution_id": "luna-play-001",
        "contributor_id": "household-017",
        "animal_id": "cat-luna",
        "capture_session_id": "2026-08-14-living-room-01",
        "rights_holder": "Example Contributor",
        "accepted_at": "2026-08-14T12:00:00Z",
        "recorded_at": "2026-08-14T11:30:00Z",
        "media_sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
        "license_id": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Example Contributor",
        "grants": {
            "model_training": True,
            "create_derivatives": True,
            "redistribute_normalized_clips": True,
        },
    }))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "schema_version": 1,
        "contribution_id": "luna-play-001",
        "media_path": "source.mp4",
        "subject": "cat",
        "action": "playing",
        "scene": "indoors",
        "prompt": "a cat playing indoors",
        "clip_start_seconds": 0.5,
    }))
    output = tmp_path / "bundle"

    assert main(["create", str(spec), str(receipt), str(output)]) == 0

    records = load_manifest(output / "source.jsonl")
    assert len(records) == 1
    record = records[0]
    assert record.sample_id == "local-luna-play-001"
    assert record.triple == ("cat", "playing", "indoors")
    assert record.source_id.startswith("local-contributor:")
    assert record.duplicate_group.startswith("local-session:")
    assert record.clip_path == "clips/local-luna-play-001.mp4"
    assert record.rights_receipt_path == "receipts/luna-play-001.json"
    assert record.rights_receipt_sha256 == hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert record.contribution_id == "luna-play-001"
    assert record.clip_start_seconds == 0.5
    assert record.clip_frames == 16
    assert record.clip_fps == 8
    assert record.clip_size == 96
    clip = output / record.clip_path
    assert hashlib.sha256(clip.read_bytes()).hexdigest() == record.sha256
    assert json.loads((output / record.rights_receipt_path).read_text())[
        "media_sha256"
    ] == hashlib.sha256(media.read_bytes()).hexdigest()
    evidence = json.loads((output / "evidence.json").read_text())
    assert evidence["training_ready"] is False
    assert evidence["training_blocker"] == "full-duration human factor review is required"
    source_media = output / evidence["source_media_path"]
    assert source_media.read_bytes() == media.read_bytes()
    assert evidence["source_media_sha256"] == hashlib.sha256(media.read_bytes()).hexdigest()
    assert (output / "review" / "full-contact.jpg").is_file()
    assert (output / "review" / "selected-window.jpg").is_file()
    assert evidence["source_manifest_sha256"] == hashlib.sha256(
        (output / "source.jsonl").read_bytes()
    ).hexdigest()


def test_create_cli_rejects_incomplete_rights_before_writing_outputs(tmp_path):
    media = tmp_path / "source.mp4"
    _video(media)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({
        "schema_version": 1,
        "contribution_id": "luna-play-001",
        "contributor_id": "household-017",
        "animal_id": "cat-luna",
        "capture_session_id": "2026-08-14-living-room-01",
        "rights_holder": "Example Contributor",
        "accepted_at": "2026-08-14T12:00:00Z",
        "recorded_at": "2026-08-14T11:30:00Z",
        "media_sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
        "license_id": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Example Contributor",
        "grants": {
            "model_training": True,
            "create_derivatives": True,
            "redistribute_normalized_clips": False,
        },
    }))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "schema_version": 1,
        "contribution_id": "luna-play-001",
        "media_path": "source.mp4",
        "subject": "cat",
        "action": "playing",
        "scene": "indoors",
        "prompt": "a cat playing indoors",
        "clip_start_seconds": 0.5,
    }))
    output = tmp_path / "bundle"

    try:
        main(["create", str(spec), str(receipt), str(output)])
    except ValueError as error:
        assert "explicitly grant" in str(error)
    else:
        raise AssertionError("an incomplete rights grant must be rejected")
    assert not output.exists()
    assert not output.with_name("bundle.tmp").exists()


def test_merge_cli_rebuilds_a_portable_dataset_without_mutating_the_base(tmp_path):
    base = tmp_path / "base"
    (base / "clips").mkdir(parents=True)
    base_clip = base / "clips" / "base.mp4"
    _video(base_clip)
    write_manifest(base / "source.jsonl", [RealVideoRecord(
        sample_id="base", source_id="source-base", clip_path="clips/base.mp4",
        prompt="a dog walking outdoors", subject="dog", action="walking",
        scene="outdoors", source_url="https://example.test/base", license="CC0",
        attribution="Base Author", sha256=hashlib.sha256(base_clip.read_bytes()).hexdigest(),
        duplicate_group="base-session", perceptual_hash="1234567890abcdef",
    )])
    contribution_input = tmp_path / "contribution-input"
    contribution_input.mkdir()
    media = contribution_input / "source.mp4"
    _video(media)
    receipt = contribution_input / "receipt.json"
    receipt.write_text(json.dumps({
        "schema_version": 1, "contribution_id": "luna-play-001",
        "contributor_id": "household-017", "animal_id": "cat-luna",
        "capture_session_id": "2026-08-14-living-room-01",
        "rights_holder": "Example Contributor",
        "accepted_at": "2026-08-14T12:00:00Z",
        "recorded_at": "2026-08-14T11:30:00Z",
        "media_sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
        "license_id": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Example Contributor",
        "grants": {"model_training": True, "create_derivatives": True,
                   "redistribute_normalized_clips": True},
    }))
    spec = contribution_input / "spec.json"
    spec.write_text(json.dumps({
        "schema_version": 1, "contribution_id": "luna-play-001",
        "media_path": "source.mp4", "subject": "cat", "action": "playing",
        "scene": "indoors", "prompt": "a cat playing indoors",
        "clip_start_seconds": 0.5,
    }))
    bundle = tmp_path / "bundle"
    assert main(["create", str(spec), str(receipt), str(bundle)]) == 0
    staged_evidence = json.loads((bundle / "evidence.json").read_text())
    staged_record = load_manifest(bundle / "source.jsonl")[0]
    review = tmp_path / "review.json"
    review.write_text(json.dumps({
        "schema_version": 1,
        "contribution_id": "luna-play-001",
        "reviewer_id": "reviewer-01",
        "reviewed_at": "2026-08-14T13:00:00Z",
        "source_manifest_sha256": staged_evidence["source_manifest_sha256"],
        "rights_receipt_sha256": staged_record.rights_receipt_sha256,
        "clip_sha256": staged_record.sha256,
        "full_contact_sha256": staged_evidence["review_artifacts"][
            "full_contact_sha256"
        ],
        "selected_window_sha256": staged_evidence["review_artifacts"][
            "selected_window_sha256"
        ],
        "subject": "cat", "action": "playing", "scene": "indoors",
        "prompt": "a cat playing indoors",
        "full_duration_reviewed": True,
        "single_subject_confirmed": True,
        "continuous_action_confirmed": True,
        "scene_confirmed": True,
        "no_edit_in_window": True,
        "review_notes": "One cat continuously paws a toy in the selected indoor window.",
    }))
    approved = tmp_path / "approved"
    assert main(["approve", str(bundle), str(review), str(approved)]) == 0
    assert json.loads((approved / "evidence.json").read_text())["training_ready"] is True
    merged = tmp_path / "merged"

    assert main([
        "merge", str(base / "source.jsonl"), str(approved), "--output", str(merged),
    ]) == 0

    rows = load_manifest(merged / "source.jsonl")
    assert [row.sample_id for row in rows] == ["base", "local-luna-play-001"]
    for row in rows:
        assert hashlib.sha256((merged / row.clip_path).read_bytes()).hexdigest() == row.sha256
    assert (merged / rows[1].rights_receipt_path).read_bytes() == receipt.read_bytes()
    assert (base / "source.jsonl").exists()
    assert json.loads((merged / "merge-evidence.json").read_text())[
        "contribution_bundles"
    ] == 1
    merge_evidence = json.loads((merged / "merge-evidence.json").read_text())
    audit = merged / merge_evidence["contribution_contracts"][0]["portable_audit_path"]
    assert (audit / "evidence.json").is_file()
    assert (audit / "reviews" / "luna-play-001.json").is_file()
    assert (audit / "review" / "full-contact.jpg").is_file()
    assert (audit / "source-media" / "luna-play-001.mp4").read_bytes() == media.read_bytes()
    try:
        main([
            "merge", str(merged / "source.jsonl"), str(approved),
            "--output", str(tmp_path / "incremental-merge"),
        ])
    except ValueError as error:
        assert "fixed pristine dataset" in str(error)
    else:
        raise AssertionError("incremental merge would drop prior portable audit evidence")

    # A coordinated rewrite of every mutable hash must not bypass receipt semantics.
    approved_receipt = approved / "receipts" / "luna-play-001.json"
    forged_receipt = json.loads(approved_receipt.read_text())
    forged_receipt["grants"]["model_training"] = False
    approved_receipt.write_text(json.dumps(forged_receipt))
    forged_receipt_sha = hashlib.sha256(approved_receipt.read_bytes()).hexdigest()
    forged_record = replace(
        load_manifest(approved / "source.jsonl")[0],
        rights_receipt_sha256=forged_receipt_sha,
    )
    write_manifest(approved / "source.jsonl", [forged_record])
    forged_manifest_sha = hashlib.sha256(
        (approved / "source.jsonl").read_bytes()
    ).hexdigest()
    approved_review = approved / "reviews" / "luna-play-001.json"
    forged_review = json.loads(approved_review.read_text())
    forged_review["rights_receipt_sha256"] = forged_receipt_sha
    forged_review["source_manifest_sha256"] = forged_manifest_sha
    approved_review.write_text(json.dumps(forged_review))
    forged_evidence_path = approved / "evidence.json"
    forged_evidence = json.loads(forged_evidence_path.read_text())
    forged_evidence["rights_receipt_sha256"] = forged_receipt_sha
    forged_evidence["source_manifest_sha256"] = forged_manifest_sha
    forged_evidence["human_review_sha256"] = hashlib.sha256(
        approved_review.read_bytes()
    ).hexdigest()
    forged_evidence_path.write_text(json.dumps(forged_evidence))
    try:
        main([
            "merge", str(base / "source.jsonl"), str(approved),
            "--output", str(tmp_path / "forged-merge"),
        ])
    except ValueError as error:
        assert "explicitly grant" in str(error)
    else:
        raise AssertionError("merge must revalidate receipt grants")


def test_create_cli_rejects_duplicate_json_keys_before_writing_outputs(tmp_path):
    media = tmp_path / "source.mp4"
    _video(media)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({
        "schema_version": 1, "contribution_id": "duplicate-id",
        "contributor_id": "household-017", "animal_id": "cat-luna",
        "capture_session_id": "session-1", "rights_holder": "Contributor",
        "accepted_at": "2026-08-14T12:00:00Z",
        "recorded_at": "2026-08-14T11:30:00Z",
        "media_sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
        "license_id": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "attribution": "Contributor",
        "grants": {"model_training": True, "create_derivatives": True,
                   "redistribute_normalized_clips": True},
    }))
    spec = tmp_path / "spec.json"
    spec.write_text(
        '{"schema_version":1,"contribution_id":"duplicate-id",'
        '"contribution_id":"duplicate-id","media_path":"source.mp4",'
        '"subject":"cat","action":"playing","scene":"indoors",'
        '"prompt":"a cat playing indoors","clip_start_seconds":0.5}'
    )
    output = tmp_path / "bundle"

    try:
        main(["create", str(spec), str(receipt), str(output)])
    except ValueError as error:
        assert "duplicate JSON key" in str(error)
    else:
        raise AssertionError("duplicate JSON keys must be rejected")
    assert not output.exists()


def test_approve_revalidates_the_receipt_instead_of_trusting_a_coordinated_rehash(tmp_path):
    media = tmp_path / "source.mp4"
    _video(media)
    receipt_path = tmp_path / "receipt.json"
    receipt = {
        "schema_version": 1, "contribution_id": "luna-play-001",
        "contributor_id": "household-017", "animal_id": "cat-luna",
        "capture_session_id": "session-1", "rights_holder": "Contributor",
        "accepted_at": "2026-08-14T12:00:00Z",
        "recorded_at": "2026-08-14T11:30:00Z",
        "media_sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
        "license_id": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "attribution": "Contributor",
        "grants": {"model_training": True, "create_derivatives": True,
                   "redistribute_normalized_clips": True},
    }
    receipt_path.write_text(json.dumps(receipt))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "schema_version": 1, "contribution_id": "luna-play-001",
        "media_path": "source.mp4", "subject": "cat", "action": "playing",
        "scene": "indoors", "prompt": "a cat playing indoors",
        "clip_start_seconds": 0.5,
    }))
    staged = tmp_path / "staged"
    assert main(["create", str(spec), str(receipt_path), str(staged)]) == 0

    staged_receipt = staged / "receipts" / "luna-play-001.json"
    receipt["grants"]["redistribute_normalized_clips"] = False
    staged_receipt.write_text(json.dumps(receipt))
    forged_receipt_sha = hashlib.sha256(staged_receipt.read_bytes()).hexdigest()
    record = load_manifest(staged / "source.jsonl")[0]
    record = replace(record, rights_receipt_sha256=forged_receipt_sha)
    write_manifest(staged / "source.jsonl", [record])
    evidence_path = staged / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["rights_receipt_sha256"] = forged_receipt_sha
    evidence["source_manifest_sha256"] = hashlib.sha256(
        (staged / "source.jsonl").read_bytes()
    ).hexdigest()
    evidence_path.write_text(json.dumps(evidence))
    review = tmp_path / "review.json"
    review.write_text(json.dumps({
        "schema_version": 1, "contribution_id": "luna-play-001",
        "reviewer_id": "reviewer-01", "reviewed_at": "2026-08-14T13:00:00Z",
        "source_manifest_sha256": evidence["source_manifest_sha256"],
        "rights_receipt_sha256": forged_receipt_sha,
        "clip_sha256": record.sha256,
        "full_contact_sha256": evidence["review_artifacts"]["full_contact_sha256"],
        "selected_window_sha256": evidence["review_artifacts"]["selected_window_sha256"],
        "subject": "cat", "action": "playing", "scene": "indoors",
        "prompt": "a cat playing indoors", "full_duration_reviewed": True,
        "single_subject_confirmed": True, "continuous_action_confirmed": True,
        "scene_confirmed": True, "no_edit_in_window": True,
        "review_notes": "One cat continuously paws a toy indoors.",
    }))

    try:
        main(["approve", str(staged), str(review), str(tmp_path / "approved")])
    except ValueError as error:
        assert "explicitly grant" in str(error)
    else:
        raise AssertionError("approval must revalidate the rights receipt contract")
