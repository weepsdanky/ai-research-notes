import csv
import hashlib
import io
import json
from io import BytesIO

import pytest

from ego4d_media import (
    build_download_plan,
    download_exact_media,
    verify_downloaded_media,
)
from prepare_ego4d import main


def _frozen_scan():
    candidates = [
        {"video_uid": "uid-a", "annotation_uid": "ann-a"},
        {"video_uid": "uid-b", "annotation_uid": "ann-b"},
    ]
    candidate_bytes = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in candidates
    ).encode()
    uid_bytes = b"uid-a\nuid-b\n"
    metadata_bytes = json.dumps({
        "version": "v2.1", "date": "2026-01-01",
        "videos": [
            {"video_uid": "uid-a", "duration_sec": 12.5},
            {"video_uid": "uid-b", "duration_sec": 12.5},
        ],
    }, sort_keys=True).encode()
    report = {
        "dataset": "ego4d",
        "dataset_release": "v2_1",
        "dataset_version": "v2.1",
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "narrations_sha256": "2" * 64,
        "license_sha256": "3" * 64,
        "license_receipt_sha256": "4" * 64,
        "license_accepted_at": "2026-08-14T00:00:00Z",
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "uid_file_sha256": hashlib.sha256(uid_bytes).hexdigest(),
        "selected_video_uids": ["uid-a", "uid-b"],
        "download_contract": {
            "dataset": "video_540ss",
            "max_bytes": 12_000_000_000,
            "on_budget_exceeded": "stop_without_fallback",
        },
        "training_ready": False,
    }
    return (
        candidate_bytes, uid_bytes,
        (json.dumps(report, sort_keys=True) + "\n").encode(), metadata_bytes,
    )


def _manifest_bytes():
    output = io.StringIO()
    writer = csv.DictWriter(
        output, fieldnames=("video_uid", "canonical_s3_location", "type"),
    )
    writer.writeheader()
    writer.writerows([
        {"video_uid": "uid-a", "canonical_s3_location": "s3://ego4d-fair/a.mp4", "type": "video"},
        {"video_uid": "uid-b", "canonical_s3_location": "s3://ego4d-cmu/b.mp4", "type": "video"},
    ])
    return output.getvalue().encode()


def _manifest_identity(data):
    return {
        "s3_path": "s3://ego4d-consortium-sharing/public/v2_1/video_540ss/manifest.csv",
        "version_id": "manifest-version",
        "content_length": len(data),
        "etag": "manifest-etag",
    }


def test_preflight_binds_remote_versions_and_enforces_budget():
    candidates, uids, scan_report, _ = _frozen_scan()
    manifest = _manifest_bytes()
    heads = {
        "s3://ego4d-fair/a.mp4": {
            "version_id": "version-a", "content_length": 40, "etag": "etag-a",
        },
        "s3://ego4d-cmu/b.mp4": {
            "version_id": "version-b", "content_length": 60, "etag": "etag-b",
        },
    }
    plan = build_download_plan(
        candidates, uids, scan_report, manifest, _manifest_identity(manifest),
        heads.__getitem__,
    )
    assert plan["selected_video_uids"] == ["uid-a", "uid-b"]
    assert plan["total_content_length"] == 100
    assert plan["remote_objects"][0] == {
        "video_uid": "uid-a", "filename": "uid-a.mp4",
        "s3_path": "s3://ego4d-fair/a.mp4", "version_id": "version-a",
        "content_length": 40, "etag": "etag-a",
    }
    assert plan["training_ready"] is False

    wrong_identity = {**_manifest_identity(manifest), "s3_path": "s3://other/manifest.csv"}
    with pytest.raises(ValueError, match="fixed official path"):
        build_download_plan(
            candidates, uids, scan_report, manifest, wrong_identity, heads.__getitem__,
        )
    unsafe_manifest = manifest.replace(b"s3://ego4d-fair/", b"s3://private-bucket/")
    unsafe_identity = _manifest_identity(unsafe_manifest)
    with pytest.raises(ValueError, match="non-official S3 path"):
        build_download_plan(
            candidates, uids, scan_report, unsafe_manifest, unsafe_identity,
            heads.__getitem__,
        )

    unlicensed = json.loads(scan_report); unlicensed.pop("license_receipt_sha256")
    with pytest.raises(ValueError, match="license_receipt_sha256"):
        build_download_plan(
            candidates, uids, json.dumps(unlicensed).encode(), manifest,
            _manifest_identity(manifest), heads.__getitem__,
        )

    wrong_release = json.loads(scan_report); wrong_release["dataset_version"] = "v1"
    with pytest.raises(ValueError, match="v2.1"):
        build_download_plan(
            candidates, uids, json.dumps(wrong_release).encode(), manifest,
            _manifest_identity(manifest), heads.__getitem__,
        )

    heads["s3://ego4d-cmu/b.mp4"]["content_length"] = 12_000_000_000
    with pytest.raises(ValueError, match="12,000,000,000"):
        build_download_plan(
            candidates, uids, scan_report, manifest, _manifest_identity(manifest),
            heads.__getitem__,
        )


def test_verify_downloads_binds_version_size_hash_and_probe(tmp_path):
    candidates, uids, scan_report, metadata = _frozen_scan()
    manifest = _manifest_bytes()
    heads = {
        "s3://ego4d-fair/a.mp4": {
            "version_id": "version-a", "content_length": 4, "etag": "etag-a",
        },
        "s3://ego4d-cmu/b.mp4": {
            "version_id": "version-b", "content_length": 6, "etag": "etag-b",
        },
    }
    plan = build_download_plan(
        candidates, uids, scan_report, manifest, _manifest_identity(manifest),
        heads.__getitem__,
    )
    media = tmp_path / "v2" / "video_540ss"
    bodies = {
        ("s3://ego4d-fair/a.mp4", "version-a"): b"aaaa",
        ("s3://ego4d-cmu/b.mp4", "version-b"): b"bbbbbb",
    }
    def get_object(path, version):
        body = bodies[(path, version)]
        head = heads[path]
        return {**head, "body": BytesIO(body)}

    plan_bytes = (json.dumps(plan, sort_keys=True) + "\n").encode()
    receipt = download_exact_media(plan_bytes, media, get_object, sdk_version="test-sdk")
    receipt_bytes = (json.dumps(receipt, sort_keys=True) + "\n").encode()
    versions = (media / "manifest.ver").read_bytes()

    report = verify_downloaded_media(
        candidates, uids, scan_report, metadata, plan_bytes, receipt_bytes,
        versions, media,
        probe=lambda path: {
            "container": "mov,mp4", "video_codec": "h264",
            "audio_codecs": ["aac"], "frame_count": 375,
            "width": 960, "height": 540, "duration_seconds": 12.5, "fps": 30.0,
        },
    )
    assert report["total_local_bytes"] == 10
    assert report["media"][0]["sha256"] == hashlib.sha256(b"aaaa").hexdigest()
    assert report["media"][0]["s3_version_id"] == "version-a"
    assert report["media"][0]["duration_seconds"] == 12.5
    assert report["media"][0]["frame_count"] == 375
    assert report["media"][0]["audio_codecs"] == ["aac"]
    assert report["media"][0]["metadata_duration_seconds"] == 12.5
    assert report["training_ready"] is False

    (media / "uid-a.mp4").write_bytes(b"zzzz")
    with pytest.raises(ValueError, match="download receipt SHA"):
        verify_downloaded_media(
            candidates, uids, scan_report, metadata, plan_bytes, receipt_bytes,
            versions, media,
            probe=lambda _: {
                "container": "mov,mp4", "video_codec": "h264",
                "audio_codecs": [], "frame_count": 375,
                "width": 960, "height": 540, "duration_seconds": 12.5, "fps": 30.0,
            },
        )
    (media / "uid-a.mp4").write_bytes(b"aaaa")

    with pytest.raises(ValueError, match="metadata duration"):
        verify_downloaded_media(
            candidates, uids, scan_report, metadata, plan_bytes, receipt_bytes,
            versions, media,
            probe=lambda _: {
                "container": "mov,mp4", "video_codec": "h264",
                "audio_codecs": [], "frame_count": 600,
                "width": 960, "height": 540, "duration_seconds": 20.0, "fps": 30.0,
            },
        )

    (media / "uid-b.mp4").write_bytes(b"short")
    with pytest.raises(ValueError, match="size"):
        verify_downloaded_media(
            candidates, uids, scan_report, metadata, plan_bytes, receipt_bytes,
            versions, media,
            probe=lambda _: {
                "container": "mov,mp4", "video_codec": "h264",
                "audio_codecs": ["aac"], "frame_count": 375,
                "width": 960, "height": 540, "duration_seconds": 12.5, "fps": 30.0,
            },
        )

    (media / "uid-b.mp4").write_bytes(b"bbbbbb")
    (media / "extra.mp4").write_bytes(b"extra")
    with pytest.raises(ValueError, match="unexpected MP4"):
        verify_downloaded_media(
            candidates, uids, scan_report, metadata, plan_bytes, receipt_bytes,
            versions, media,
            probe=lambda _: {
                "container": "mov,mp4", "video_codec": "h264",
                "audio_codecs": [], "frame_count": 375,
                "width": 960, "height": 540, "duration_seconds": 12.5, "fps": 30.0,
            },
        )


def test_verifier_rejects_wrong_scan_or_s3_version_before_media_is_trusted(tmp_path):
    candidates, uids, scan_report, metadata = _frozen_scan()
    manifest = _manifest_bytes()
    heads = {
        "s3://ego4d-fair/a.mp4": {"version_id": "a", "content_length": 1, "etag": "a"},
        "s3://ego4d-cmu/b.mp4": {"version_id": "b", "content_length": 1, "etag": "b"},
    }
    plan = build_download_plan(
        candidates, uids, scan_report, manifest, _manifest_identity(manifest),
        heads.__getitem__,
    )
    plan_bytes = (json.dumps(plan, sort_keys=True) + "\n").encode()
    media = tmp_path / "video_540ss"
    bodies = {
        ("s3://ego4d-fair/a.mp4", "a"): b"a",
        ("s3://ego4d-cmu/b.mp4", "b"): b"b",
    }
    receipt = download_exact_media(
        plan_bytes, media,
        lambda path, version: {**heads[path], "body": BytesIO(bodies[(path, version)])},
        sdk_version="test-sdk",
    )
    receipt_bytes = (json.dumps(receipt, sort_keys=True) + "\n").encode()
    versions = b"uid,version,filename\nuid-a,a,uid-a.mp4\nuid-b,wrong,uid-b.mp4\n"
    with pytest.raises(ValueError, match="manifest.ver"):
        verify_downloaded_media(
            candidates, uids, scan_report, metadata, plan_bytes, receipt_bytes,
            versions, media,
            probe=lambda _: {"container": "mp4", "video_codec": "h264",
                             "audio_codecs": [], "frame_count": 1,
                             "width": 1, "height": 1, "duration_seconds": 1.0,
                             "fps": 1.0},
        )

    corrupted = json.loads(scan_report); corrupted["selected_video_uids"] = ["uid-a"]
    with pytest.raises(ValueError, match="UID"):
        build_download_plan(
            candidates, uids, json.dumps(corrupted).encode(), manifest,
            _manifest_identity(manifest), heads.__getitem__,
        )


def test_exact_downloader_rejects_a_changed_s3_version_without_fallback(tmp_path):
    candidates, uids, scan_report, _ = _frozen_scan()
    manifest = _manifest_bytes()
    heads = {
        "s3://ego4d-fair/a.mp4": {"version_id": "a", "content_length": 1, "etag": "a"},
        "s3://ego4d-cmu/b.mp4": {"version_id": "b", "content_length": 1, "etag": "b"},
    }
    plan = build_download_plan(
        candidates, uids, scan_report, manifest, _manifest_identity(manifest),
        heads.__getitem__,
    )
    with pytest.raises(ValueError, match="version"):
        download_exact_media(
            (json.dumps(plan, sort_keys=True) + "\n").encode(), tmp_path / "media",
            lambda path, version: {
                **heads[path], "version_id": "changed", "body": BytesIO(b"x"),
            },
            sdk_version="test-sdk",
        )
    assert not list((tmp_path / "media").glob("*.mp4"))


def test_exact_downloader_rejects_unsafe_uid_before_writing_outside_media_dir(tmp_path):
    candidates, uids, scan_report, _ = _frozen_scan()
    manifest = _manifest_bytes()
    heads = {
        "s3://ego4d-fair/a.mp4": {"version_id": "a", "content_length": 1, "etag": "a"},
        "s3://ego4d-cmu/b.mp4": {"version_id": "b", "content_length": 1, "etag": "b"},
    }
    plan = build_download_plan(
        candidates, uids, scan_report, manifest, _manifest_identity(manifest),
        heads.__getitem__,
    )
    plan["selected_video_uids"] = ["../escape"]
    plan["remote_objects"] = [{
        "video_uid": "../escape", "filename": "../escape.mp4",
        "s3_path": "s3://ego4d-fair/a.mp4", "version_id": "a",
        "content_length": 1, "etag": "a",
    }]
    plan["total_content_length"] = 1
    outside = tmp_path / "escape.mp4"
    outside.write_bytes(b"keep")

    with pytest.raises(ValueError, match="unsafe video_uid"):
        download_exact_media(
            (json.dumps(plan, sort_keys=True) + "\n").encode(), tmp_path / "media",
            lambda path, version: {**heads[path], "body": BytesIO(b"x")},
            sdk_version="test-sdk",
        )
    assert outside.read_bytes() == b"keep"
    assert not (tmp_path / "media").exists()

    plan["selected_video_uids"] = ["uid-a"]
    plan["remote_objects"] = [{
        "video_uid": "uid-a", "filename": "uid-a.mp4",
        "s3_path": "s3://ego4d-fair/a.mp4", "version_id": "a",
        "content_length": 1, "etag": "a",
    }]
    plan["dataset_version"] = "v1"
    with pytest.raises(ValueError, match="fixed contract"):
        download_exact_media(
            (json.dumps(plan, sort_keys=True) + "\n").encode(), tmp_path / "media",
            lambda path, version: {**heads[path], "body": BytesIO(b"x")},
            sdk_version="test-sdk",
        )


def test_cli_writes_bound_preflight_and_verification_reports(tmp_path):
    candidates, uids, scan_report, metadata = _frozen_scan()
    manifest = _manifest_bytes()
    inputs = {
        "candidates.jsonl": candidates,
        "uids.txt": uids,
        "scan.json": scan_report,
        "ego4d.json": metadata,
    }
    for name, data in inputs.items():
        (tmp_path / name).write_bytes(data)
    heads = {
        "s3://ego4d-fair/a.mp4": {"version_id": "a", "content_length": 1, "etag": "a"},
        "s3://ego4d-cmu/b.mp4": {"version_id": "b", "content_length": 1, "etag": "b"},
    }
    plan_path = tmp_path / "download-plan.json"
    assert main([
        "preflight", str(tmp_path / "candidates.jsonl"), str(tmp_path / "uids.txt"),
        str(tmp_path / "scan.json"), str(plan_path),
    ], head_object=heads.__getitem__,
       snapshot_object=lambda _: (manifest, _manifest_identity(manifest))) == 0
    assert json.loads(plan_path.read_text())["total_content_length"] == 2

    media = tmp_path / "raw" / "v2" / "video_540ss"
    receipt_path = tmp_path / "download-receipt.json"
    bodies = {
        ("s3://ego4d-fair/a.mp4", "a"): b"a",
        ("s3://ego4d-cmu/b.mp4", "b"): b"b",
    }
    assert main([
        "download", str(plan_path), str(media), str(receipt_path),
    ], get_object=lambda path, version: {
        **heads[path], "body": BytesIO(bodies[(path, version)]),
    }, sdk_version="test-sdk") == 0
    versions = media / "manifest.ver"
    verified_path = tmp_path / "verified-media.json"
    assert main([
        "verify", str(tmp_path / "candidates.jsonl"), str(tmp_path / "uids.txt"),
        str(tmp_path / "scan.json"), str(tmp_path / "ego4d.json"), str(plan_path),
        str(receipt_path), str(versions), str(media), str(verified_path),
    ], probe=lambda _: {"container": "mp4", "video_codec": "h264",
                        "audio_codecs": [], "frame_count": 1,
                        "width": 1, "height": 1, "duration_seconds": 12.5,
                        "fps": 1.0}) == 0
    assert json.loads(verified_path.read_text())["total_local_bytes"] == 2

    with pytest.raises(ValueError, match="output"):
        main([
            "verify", str(tmp_path / "candidates.jsonl"), str(tmp_path / "uids.txt"),
            str(tmp_path / "scan.json"), str(tmp_path / "ego4d.json"), str(plan_path),
            str(receipt_path), str(versions), str(media), str(media / "uid-a.mp4"),
        ], probe=lambda _: {})
