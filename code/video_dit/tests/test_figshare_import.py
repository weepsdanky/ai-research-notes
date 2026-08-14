import hashlib
import json
from dataclasses import replace
from pathlib import Path
import subprocess

from prepare_figshare import main
from real_data import RealVideoRecord, load_manifest, write_manifest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _video(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
            "testsrc2=size=128x96:rate=8:duration=3", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    media = tmp_path / "example.mp4"
    _video(media)
    preview = tmp_path / "preview.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-ss", "0.5", "-i", str(media),
            "-t", "2", "-an", "-vf",
            "fps=8,scale=96:96:force_original_aspect_ratio=increase,crop=96:96",
            "-frames:v", "16", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(preview),
        ],
        check=True,
    )
    full = tmp_path / "full.jpg"
    selected = tmp_path / "selected.jpg"
    full.write_bytes(b"locked full-duration navigation sheet")
    selected.write_bytes(b"locked selected-window sheet")
    media_md5 = hashlib.md5(media.read_bytes(), usedforsecurity=False).hexdigest()
    article = {
        "id": 28287896,
        "doi": "10.25377/sussex.28287896.v2",
        "version": 2,
        "published_date": "2025-09-11T08:53:10Z",
        "modified_date": "2025-09-11T08:53:10Z",
        "title": "Cat object permanence",
        "license": {
            "name": "CC BY 4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/",
        },
        "authors": [{"full_name": "Example Researcher"}],
        "files": [{
            "id": 57852742,
            "name": "example.mp4",
            "size": media.stat().st_size,
            "computed_md5": media_md5,
            "download_url": "https://ndownloader.figshare.com/files/57852742",
        }],
    }
    article_path = tmp_path / "article.json"
    article_path.write_text(json.dumps(article))
    evidence = {
        "schema_version": 1,
        "article": {
            "id": 28287896,
            "version": 2,
            "doi": "10.25377/sussex.28287896.v2",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "published_at": "2025-09-11T08:53:10Z",
            "modified_at": "2025-09-11T08:53:10Z",
            "url": "https://api.figshare.com/v2/articles/28287896",
            "api_snapshot_sha256": _sha(article_path),
        },
        "file": {
            "figshare_file_id": 57852742,
            "name": "example.mp4",
            "size_bytes": media.stat().st_size,
            "md5": media_md5,
            "sha256": _sha(media),
            "download_url": "https://ndownloader.figshare.com/files/57852742",
        },
        "decision": {
            "status": "review-qualified-pending-import",
            "subject": "cat", "action": "playing", "scene": "indoors",
            "clip_start_seconds": 0.5, "clip_stop_seconds": 2.5,
            "reviewer_id": "reviewer-01",
            "reviewed_at": "2026-08-14T18:30:00Z",
            "full_duration_reviewed": True,
            "single_animal_confirmed": True,
            "continuous_action_confirmed": True,
            "scene_confirmed": True,
            "no_edit_in_window": True,
            "notes": "One cat continuously bats a toy indoors.",
        },
        "grouping": {
            "conservative_group": "figshare-article:28287896",
            "group_upper_bound": 1,
            "reason": "No public stable cat/session mapping.",
        },
        "review_artifacts": {
            "full_contact": {"path": "full.jpg", "sha256": _sha(full)},
            "selected_window": {"path": "selected.jpg", "sha256": _sha(selected)},
            "normalized_preview": {
                "path": "preview.mp4", "sha256": _sha(preview),
                "perceptual_hash": "00f0f0f0f0f0f000", "frames": 16,
                "fps": 8, "size": 96,
            },
        },
        "training_ready": False,
        "validation": {
            "nearest_existing_phash_distance": 20,
            "remaining_blocker": "exact importer",
        },
    }
    # The importer recomputes pHash; use its public media result as the fixture truth.
    from real_data import decode_video_clip, perceptual_hash
    evidence["review_artifacts"]["normalized_preview"]["perceptual_hash"] = (
        perceptual_hash(decode_video_clip(preview, frames=16, size=96))
    )
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    return article_path, media, evidence_path


def test_create_cli_builds_a_rights_and_review_locked_figshare_bundle(tmp_path):
    article, media, evidence = _fixture(tmp_path)
    output = tmp_path / "bundle"

    assert main(["create", str(article), str(media), str(evidence), str(output)]) == 0

    rows = load_manifest(output / "source.jsonl")
    assert len(rows) == 1
    record = rows[0]
    assert record.sample_id == "figshare-28287896-file-57852742"
    assert record.triple == ("cat", "playing", "indoors")
    assert record.source_id == "figshare-article:28287896"
    assert record.duplicate_group == "figshare-article:28287896"
    assert record.license == "CC BY 4.0"
    assert _sha(output / record.clip_path) == record.sha256
    bundled = json.loads((output / "evidence.json").read_text())
    assert bundled["training_ready"] is True
    assert _sha(output / bundled["article_path"]) == bundled["article_sha256"]
    assert _sha(output / bundled["source_media_path"]) == bundled["source_media_sha256"]
    assert (output / "review" / "full-contact.jpg").is_file()


def test_create_cli_rejects_an_article_license_rewrite_before_writing(tmp_path):
    article, media, evidence = _fixture(tmp_path)
    payload = json.loads(article.read_text())
    payload["license"]["name"] = "CC BY-NC 4.0"
    article.write_text(json.dumps(payload))
    review = json.loads(evidence.read_text())
    review["article"]["api_snapshot_sha256"] = _sha(article)
    review["article"]["license"] = "CC BY-NC 4.0"
    evidence.write_text(json.dumps(review))
    output = tmp_path / "bundle"

    try:
        main(["create", str(article), str(media), str(evidence), str(output)])
    except ValueError as error:
        assert "CC BY 4.0" in str(error)
    else:
        raise AssertionError("non-redistributable license rewrite must be rejected")
    assert not output.exists()


def test_merge_cli_adds_the_bundle_without_dropping_portable_audit(tmp_path):
    article, media, evidence = _fixture(tmp_path)
    bundle = tmp_path / "bundle"
    assert main(["create", str(article), str(media), str(evidence), str(bundle)]) == 0
    base = tmp_path / "base"
    (base / "clips").mkdir(parents=True)
    base_clip = base / "clips/base.mp4"
    _video(base_clip)
    candidate = load_manifest(bundle / "source.jsonl")[0]
    base_phash = f"{int(candidate.perceptual_hash, 16) ^ ((1 << 20) - 1):016x}"
    write_manifest(base / "source.jsonl", [RealVideoRecord(
        sample_id="base", source_id="source-base", clip_path="clips/base.mp4",
        prompt="a dog walking outdoors", subject="dog", action="walking",
        scene="outdoors", source_url="https://example.test/base", license="CC0",
        attribution="Base Author", sha256=_sha(base_clip),
        duplicate_group="base-session", perceptual_hash=base_phash,
    )])
    output = tmp_path / "merged"

    assert main([
        "merge", str(base / "source.jsonl"), str(bundle), "--output", str(output),
    ]) == 0

    rows = load_manifest(output / "source.jsonl")
    assert [row.sample_id for row in rows] == ["base", "figshare-28287896-file-57852742"]
    for row in rows:
        assert _sha(output / row.clip_path) == row.sha256
    report = json.loads((output / "merge-evidence.json").read_text())
    audit = output / report["figshare_contracts"][0]["portable_audit_path"]
    assert (audit / "source-media/figshare-57852742.mp4").read_bytes() == media.read_bytes()
    assert (audit / "provenance/review-evidence.json").is_file()
    try:
        main([
            "merge", str(output / "source.jsonl"), str(bundle),
            "--output", str(tmp_path / "incremental"),
        ])
    except ValueError as error:
        assert "fixed pristine dataset" in str(error)
    else:
        raise AssertionError("incremental merge would drop prior audit evidence")

    bundled_article = bundle / "provenance/article.json"
    forged_article = json.loads(bundled_article.read_text())
    forged_article["license"]["name"] = "CC BY-NC 4.0"
    bundled_article.write_text(json.dumps(forged_article))
    bundled_review = bundle / "provenance/review-evidence.json"
    forged_review = json.loads(bundled_review.read_text())
    forged_review["article"]["license"] = "CC BY-NC 4.0"
    forged_review["article"]["api_snapshot_sha256"] = _sha(bundled_article)
    bundled_review.write_text(json.dumps(forged_review))
    bundle_evidence_path = bundle / "evidence.json"
    forged_bundle_evidence = json.loads(bundle_evidence_path.read_text())
    forged_bundle_evidence["article_sha256"] = _sha(bundled_article)
    forged_bundle_evidence["review_evidence_sha256"] = _sha(bundled_review)
    bundle_evidence_path.write_text(json.dumps(forged_bundle_evidence))
    try:
        main([
            "merge", str(base / "source.jsonl"), str(bundle),
            "--output", str(tmp_path / "forged"),
        ])
    except ValueError as error:
        assert "CC BY 4.0" in str(error)
    else:
        raise AssertionError("merge must revalidate license semantics after coordinated rehash")


def test_merge_rejects_a_coordinated_noncanonical_download_url_rehash(tmp_path):
    article, media, evidence = _fixture(tmp_path)
    bundle = tmp_path / "bundle"
    assert main(["create", str(article), str(media), str(evidence), str(bundle)]) == 0
    record = load_manifest(bundle / "source.jsonl")[0]
    base = tmp_path / "base"
    (base / "clips").mkdir(parents=True)
    base_clip = base / "clips/base.mp4"
    _video(base_clip)
    base_phash = f"{int(record.perceptual_hash, 16) ^ ((1 << 20) - 1):016x}"
    write_manifest(base / "source.jsonl", [RealVideoRecord(
        sample_id="base", source_id="source-base", clip_path="clips/base.mp4",
        prompt="a dog walking outdoors", subject="dog", action="walking",
        scene="outdoors", source_url="https://example.test/base", license="CC0",
        attribution="Base Author", sha256=_sha(base_clip),
        duplicate_group="base-session", perceptual_hash=base_phash,
    )])

    evil_url = "https://evil.test/files/57852742"
    article_path = bundle / "provenance/article.json"
    article_payload = json.loads(article_path.read_text())
    article_payload["files"][0]["download_url"] = evil_url
    article_path.write_text(json.dumps(article_payload))
    review_path = bundle / "provenance/review-evidence.json"
    review_payload = json.loads(review_path.read_text())
    review_payload["article"]["api_snapshot_sha256"] = _sha(article_path)
    review_payload["file"]["download_url"] = evil_url
    review_path.write_text(json.dumps(review_payload))
    record = replace(
        record, original_media_url=evil_url, downloaded_media_url=evil_url,
    )
    write_manifest(bundle / "source.jsonl", [record])
    bundle_evidence_path = bundle / "evidence.json"
    bundle_evidence = json.loads(bundle_evidence_path.read_text())
    bundle_evidence["article_sha256"] = _sha(article_path)
    bundle_evidence["review_evidence_sha256"] = _sha(review_path)
    bundle_evidence["source_manifest_sha256"] = _sha(bundle / "source.jsonl")
    bundle_evidence_path.write_text(json.dumps(bundle_evidence))

    try:
        main([
            "merge", str(base / "source.jsonl"), str(bundle),
            "--output", str(tmp_path / "forged-url"),
        ])
    except ValueError as error:
        assert "canonical download URL" in str(error)
    else:
        raise AssertionError("merge must reject a coordinated noncanonical URL rewrite")
