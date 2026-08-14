import hashlib
import json
from dataclasses import replace
from pathlib import Path
import subprocess

from prepare_peertube import main
from real_data import RealVideoRecord, decode_video_clip, load_manifest, perceptual_hash, write_manifest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _video(path: Path, duration: int = 3) -> None:
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
        f"testsrc2=size=128x96:rate=8:duration={duration}", "-an", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", str(path),
    ], check=True)


def _fixture(tmp_path: Path):
    uuid = "5163701d-4e33-4fe1-8f10-492dcdab3ee6"
    media = tmp_path / "source.mp4"
    _video(media)
    url = f"https://peer.example/download/web-videos/{uuid}-480.mp4"
    video = {
        "uuid": uuid, "name": "My cat plays", "description": "My cat Buster plays",
        "duration": 3, "publishedAt": "2026-08-14T12:00:00Z",
        "licence": {"id": 1, "label": "CC BY 4.0"},
        "account": {"name": "owner"},
        "files": [{"fileDownloadUrl": url, "size": media.stat().st_size,
                   "resolution": {"id": 480}}],
        "streamingPlaylists": [],
    }
    account = {
        "name": "owner", "displayName": "Cat Owner", "url": "https://peer.example/accounts/owner",
        "description": "Videos of my cat Buster", "createdAt": "2024-01-01T00:00:00Z",
    }
    video_path = tmp_path / "video.json"
    account_path = tmp_path / "account.json"
    video_path.write_text(json.dumps(video))
    account_path.write_text(json.dumps(account))
    preview = tmp_path / "preview.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-ss", "0.5", "-i", str(media), "-t", "2",
        "-an", "-vf", "fps=8,scale=96:96:force_original_aspect_ratio=increase,crop=96:96",
        "-frames:v", "16", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags",
        "+faststart", str(preview),
    ], check=True)
    full = tmp_path / "full.jpg"
    selected = tmp_path / "selected.jpg"
    full.write_bytes(b"full review")
    selected.write_bytes(b"selected review")
    evidence = {
        "schema_version": 1,
        "video": {
            "uuid": uuid, "origin": "https://peer.example",
            "watch_url": f"https://peer.example/videos/watch/{uuid}",
            "api_snapshot_sha256": _sha(video_path), "name": "My cat plays",
            "description": "My cat Buster plays", "duration_seconds": 3,
            "published_at": "2026-08-14T12:00:00Z", "licence_id": 1,
            "licence_label": "CC BY 4.0",
        },
        "account": {
            "name": "owner", "display_name": "Cat Owner",
            "url": "https://peer.example/accounts/owner", "description": "Videos of my cat Buster",
            "created_at": "2024-01-01T00:00:00Z", "api_snapshot_sha256": _sha(account_path),
            "originality_basis": "Account says the uploader owns and films Buster.",
        },
        "rendition": {"url": url, "size_bytes": media.stat().st_size,
                      "sha256": _sha(media), "resolution_id": 480},
        "decision": {
            "status": "review-qualified-pending-import", "subject": "cat", "action": "playing",
            "scene": "indoors", "clip_start_seconds": 0.5, "clip_stop_seconds": 2.5,
            "reviewer_id": "reviewer", "reviewed_at": "2026-08-14T12:30:00Z",
            "full_duration_reviewed": True, "single_animal_confirmed": True,
            "continuous_action_confirmed": True, "scene_confirmed": True,
            "no_edit_in_window": True, "notes": "One cat continuously plays indoors.",
        },
        "grouping": {"conservative_group": "peertube:peer.example:owner-buster",
                     "reason": "All videos of Buster share one animal group."},
        "review_artifacts": {
            "full_contact": {"path": "full.jpg", "sha256": _sha(full)},
            "selected_window": {"path": "selected.jpg", "sha256": _sha(selected)},
            "normalized_preview": {"path": "preview.mp4", "sha256": _sha(preview),
                "perceptual_hash": perceptual_hash(decode_video_clip(preview, frames=16, size=96)),
                "frames": 16, "fps": 8, "size": 96},
        },
        "validation": {"nearest_existing_phash_distance": 20,
                       "remaining_blocker": "exact portable bundle"},
        "training_ready": False,
    }
    evidence_path = tmp_path / "review.json"
    evidence_path.write_text(json.dumps(evidence))
    return video_path, account_path, media, evidence_path


def test_create_and_merge_peer_tube_bundle(tmp_path):
    video, account, media, review = _fixture(tmp_path)
    bundle = tmp_path / "bundle"
    assert main(["create", str(video), str(account), str(media), str(review), str(bundle)]) == 0
    candidate = load_manifest(bundle / "source.jsonl")[0]
    assert candidate.triple == ("cat", "playing", "indoors")
    assert candidate.license == "CC BY 4.0"

    base = tmp_path / "base"
    (base / "clips").mkdir(parents=True)
    base_clip = base / "clips/base.mp4"
    _video(base_clip)
    base_hash = f"{int(candidate.perceptual_hash, 16) ^ ((1 << 20) - 1):016x}"
    evidence = json.loads(review.read_text())
    evidence["validation"]["nearest_existing_phash_distance"] = 20
    review.write_text(json.dumps(evidence))
    # Bundle is immutable; the base is constructed to match its locked distance.
    write_manifest(base / "source.jsonl", [RealVideoRecord(
        sample_id="base", source_id="base", clip_path="clips/base.mp4",
        prompt="a dog walking outdoors", subject="dog", action="walking", scene="outdoors",
        source_url="https://example.test/base", license="CC0", attribution="Base",
        sha256=_sha(base_clip), duplicate_group="base", perceptual_hash=base_hash,
    )])
    output = tmp_path / "merged"
    assert main(["merge", str(base / "source.jsonl"), str(bundle), "--output", str(output)]) == 0
    assert len(load_manifest(output / "source.jsonl")) == 2
    assert (output / "peertube-audit").is_dir()


def test_create_rejects_noncanonical_download_origin(tmp_path):
    video, account, media, review = _fixture(tmp_path)
    payload = json.loads(review.read_text())
    payload["rendition"]["url"] = "https://evil.example/download/source.mp4"
    review.write_text(json.dumps(payload))
    try:
        main(["create", str(video), str(account), str(media), str(review), str(tmp_path / "out")])
    except ValueError as error:
        assert "canonical download" in str(error)
    else:
        raise AssertionError("cross-origin media must be rejected")


def test_create_rejects_a_coordinated_noncommercial_license_rewrite(tmp_path):
    video, account, media, review = _fixture(tmp_path)
    video_payload = json.loads(video.read_text())
    video_payload["licence"] = {"id": 4, "label": "Attribution - Non Commercial"}
    video.write_text(json.dumps(video_payload))
    review_payload = json.loads(review.read_text())
    review_payload["video"]["licence_id"] = 4
    review_payload["video"]["licence_label"] = "Attribution - Non Commercial"
    review_payload["video"]["api_snapshot_sha256"] = _sha(video)
    review.write_text(json.dumps(review_payload))
    try:
        main(["create", str(video), str(account), str(media), str(review), str(tmp_path / "out")])
    except ValueError as error:
        assert "allowed set" in str(error)
    else:
        raise AssertionError("noncommercial license must remain excluded after coordinated rehash")


def test_merge_rejects_a_coordinated_manifest_rewrite(tmp_path):
    video, account, media, review = _fixture(tmp_path)
    bundle = tmp_path / "bundle"
    main(["create", str(video), str(account), str(media), str(review), str(bundle)])
    rows = load_manifest(bundle / "source.jsonl")
    rows[0] = replace(rows[0], prompt="a cat sleeping indoors")
    write_manifest(bundle / "source.jsonl", rows)
    report_path = bundle / "evidence.json"
    report = json.loads(report_path.read_text())
    report["source_manifest_sha256"] = _sha(bundle / "source.jsonl")
    report_path.write_text(json.dumps(report))
    try:
        from prepare_peertube import _verify_bundle
        _verify_bundle(bundle)
    except ValueError as error:
        assert "prompt" in str(error)
    else:
        raise AssertionError("coordinated manifest rewrite must be rejected")


def test_bundle_verification_rejects_untracked_extra_files(tmp_path):
    video, account, media, review = _fixture(tmp_path)
    bundle = tmp_path / "bundle"
    main(["create", str(video), str(account), str(media), str(review), str(bundle)])
    (bundle / "untracked.txt").write_text("not part of the portable audit")
    try:
        from prepare_peertube import _verify_bundle
        _verify_bundle(bundle)
    except ValueError as error:
        assert "unexpected files" in str(error)
    else:
        raise AssertionError("bundle verification must reject ambiguous extra files")
