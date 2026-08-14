"""Build a portable, review-locked bundle from an exact Figshare video file."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlparse

from real_data import (
    RealVideoRecord,
    decode_video_clip,
    load_manifest,
    perceptual_hash,
    write_manifest,
)


_HEX32 = re.compile(r"[0-9a-f]{32}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_TOP_FIELDS = {
    "schema_version", "article", "file", "decision", "grouping",
    "review_artifacts", "training_ready", "validation",
}
_ARTICLE_FIELDS = {
    "id", "version", "doi", "license", "license_url", "published_at",
    "modified_at", "url", "api_snapshot_sha256",
}
_FILE_FIELDS = {
    "figshare_file_id", "name", "size_bytes", "md5", "sha256", "download_url",
}
_DECISION_FIELDS = {
    "status", "subject", "action", "scene", "clip_start_seconds",
    "clip_stop_seconds", "reviewer_id", "reviewed_at", "full_duration_reviewed",
    "single_animal_confirmed", "continuous_action_confirmed", "scene_confirmed",
    "no_edit_in_window", "notes",
}
_GROUP_FIELDS = {"conservative_group", "group_upper_bound", "reason"}
_ARTIFACT_FIELDS = {"full_contact", "selected_window", "normalized_preview"}
_VALIDATION_FIELDS = {"nearest_existing_phash_distance", "remaining_blocker"}


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path, name: str) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=_object_without_duplicate_keys)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {name} JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value, raw


def _exact(value: object, fields: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} must contain exactly {sorted(fields)}")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _utc(value: object, name: str) -> str:
    text = _text(value, name)
    if not text.endswith("Z") or "T" not in text:
        raise ValueError(f"{name} must be a UTC RFC3339 timestamp")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError:
        raise ValueError(f"{name} must be a UTC RFC3339 timestamp") from None
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{name} must be a UTC RFC3339 timestamp")
    return text


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _locked_relative(root: Path, relative: object, expected: object, name: str) -> Path:
    relative_path = Path(_text(relative, f"{name} path"))
    digest = _text(expected, f"{name} sha256")
    if relative_path.is_absolute() or ".." in relative_path.parts or not _HEX64.fullmatch(digest):
        raise ValueError(f"{name} path or SHA-256 is invalid")
    path = root / relative_path
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"{name} must be a regular file inside the review evidence directory")
    if _sha256(path) != digest:
        raise ValueError(f"{name} SHA-256 does not match review evidence")
    return path


def _normalize(source: Path, destination: Path, start: float) -> None:
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", str(source),
            "-t", "2", "-an", "-vf",
            "fps=8,scale=96:96:force_original_aspect_ratio=increase,crop=96:96",
            "-frames:v", "16", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(destination),
        ],
        check=True,
        timeout=120,
    )


def _probe_duration(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=True, capture_output=True, text=True, timeout=120,
    )
    try:
        duration = float(completed.stdout.strip())
    except ValueError:
        raise ValueError("ffprobe returned an invalid source duration") from None
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("source duration must be positive and finite")
    return duration


def _validate_normalized_clip(path: Path) -> None:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate",
            "-show_entries", "format=duration", "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True, timeout=120,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise ValueError("normalized clip must contain exactly one video stream")
    stream = streams[0]
    try:
        duration = float(payload["format"]["duration"])
        frames = int(stream["nb_read_frames"])
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("normalized clip has an invalid media contract") from None
    if frames != 16 or width != 96 or height != 96 \
            or stream.get("r_frame_rate") != "8/1" or abs(duration - 2) > 1e-3:
        raise ValueError("normalized clip must be exactly 16 frames, 8 fps, 96x96 and 2 seconds")


def create_bundle(
    article_path: Path, media_path: Path, review_evidence_path: Path, output_dir: Path,
) -> dict[str, object]:
    article, article_bytes = _load_json(article_path, "Figshare article snapshot")
    evidence, evidence_bytes = _load_json(review_evidence_path, "Figshare review evidence")
    _exact(evidence, _TOP_FIELDS, "review evidence")
    if evidence["schema_version"] != 1 or evidence["training_ready"] is not False:
        raise ValueError("review evidence must be schema 1 and not training-ready")
    expected_article = _exact(evidence["article"], _ARTICLE_FIELDS, "article contract")
    expected_file = _exact(evidence["file"], _FILE_FIELDS, "file contract")
    decision = _exact(evidence["decision"], _DECISION_FIELDS, "review decision")
    grouping = _exact(evidence["grouping"], _GROUP_FIELDS, "grouping contract")
    artifacts = _exact(evidence["review_artifacts"], _ARTIFACT_FIELDS, "review artifacts")
    validation = _exact(evidence["validation"], _VALIDATION_FIELDS, "validation contract")

    if _sha256(article_path) != expected_article["api_snapshot_sha256"]:
        raise ValueError("Figshare article snapshot SHA-256 does not match review evidence")
    article_id = _positive_int(article.get("id"), "article id")
    version = _positive_int(article.get("version"), "article version")
    if article_id != expected_article["id"] or version != expected_article["version"]:
        raise ValueError("Figshare article identity does not match review evidence")
    doi = _text(article.get("doi"), "article DOI")
    if doi != expected_article["doi"]:
        raise ValueError("Figshare DOI does not match review evidence")
    license_data = article.get("license")
    if not isinstance(license_data, dict) or license_data.get("name") != "CC BY 4.0" \
            or license_data.get("url") != "https://creativecommons.org/licenses/by/4.0/" \
            or expected_article["license"] != "CC BY 4.0" \
            or expected_article["license_url"] != license_data.get("url"):
        raise ValueError("Figshare source must use exact CC BY 4.0")
    published = _utc(article.get("published_date"), "article published_date")
    modified = _utc(article.get("modified_date"), "article modified_date")
    if published != expected_article["published_at"] or modified != expected_article["modified_at"]:
        raise ValueError("Figshare article timestamps do not match review evidence")
    if expected_article["url"] != f"https://api.figshare.com/v2/articles/{article_id}":
        raise ValueError("Figshare article URL is not canonical")
    authors = article.get("authors")
    if not isinstance(authors, list) or not authors:
        raise ValueError("Figshare article must have at least one attributed author")
    author_names = [_text(author.get("full_name") if isinstance(author, dict) else None,
                          "author full_name") for author in authors]

    file_id = _positive_int(expected_file["figshare_file_id"], "Figshare file id")
    files = article.get("files")
    if not isinstance(files, list):
        raise ValueError("Figshare article files must be a list")
    matches = [item for item in files if isinstance(item, dict) and item.get("id") == file_id]
    if len(matches) != 1:
        raise ValueError("Figshare article must contain exactly one requested file id")
    source_file = matches[0]
    for article_key, evidence_key in (
        ("name", "name"), ("size", "size_bytes"), ("computed_md5", "md5"),
        ("download_url", "download_url"),
    ):
        if source_file.get(article_key) != expected_file[evidence_key]:
            raise ValueError(f"Figshare file {article_key} does not match review evidence")
    md5 = _text(expected_file["md5"], "file md5")
    media_sha256 = _text(expected_file["sha256"], "file sha256")
    if not _HEX32.fullmatch(md5) or not _HEX64.fullmatch(media_sha256):
        raise ValueError("Figshare file digests must be lowercase hexadecimal")
    download_url = _text(expected_file["download_url"], "download URL")
    parsed = urlparse(download_url)
    if parsed.scheme != "https" or parsed.hostname != "ndownloader.figshare.com" \
            or parsed.path != f"/files/{file_id}":
        raise ValueError("Figshare download URL is not the canonical file endpoint")
    if media_path.is_symlink() or not media_path.is_file():
        raise ValueError("Figshare media must be a regular local file")
    media_bytes = media_path.read_bytes()
    if len(media_bytes) != expected_file["size_bytes"] \
            or hashlib.md5(media_bytes, usedforsecurity=False).hexdigest() != md5 \
            or hashlib.sha256(media_bytes).hexdigest() != media_sha256:
        raise ValueError("Figshare media bytes do not match the locked file contract")

    if decision["status"] != "review-qualified-pending-import":
        raise ValueError("Figshare decision must be review-qualified-pending-import")
    for field in ("subject", "action", "scene", "notes", "reviewer_id"):
        _text(decision[field], field)
    _utc(decision["reviewed_at"], "reviewed_at")
    for field in (
        "full_duration_reviewed", "single_animal_confirmed",
        "continuous_action_confirmed", "scene_confirmed", "no_edit_in_window",
    ):
        if decision[field] is not True:
            raise ValueError(f"review must explicitly confirm {field}")
    try:
        start = float(decision["clip_start_seconds"])
        stop = float(decision["clip_stop_seconds"])
    except (TypeError, ValueError):
        raise ValueError("review window must contain finite numeric seconds") from None
    if not math.isfinite(start) or not math.isfinite(stop) or start < 0 \
            or abs(stop - start - 2) > 1e-6:
        raise ValueError("review window must be exactly two finite non-negative seconds")
    if stop > _probe_duration(media_path) + 1e-6:
        raise ValueError("review window exceeds the locked Figshare source duration")
    group = _text(grouping["conservative_group"], "conservative_group")
    if group != f"figshare-article:{article_id}" or grouping["group_upper_bound"] != 1:
        raise ValueError("Figshare files without stable cat/session IDs must share one article group")
    _text(grouping["reason"], "group reason")
    nearest_distance = _positive_int(
        validation["nearest_existing_phash_distance"],
        "nearest_existing_phash_distance",
    )
    if nearest_distance <= 4:
        raise ValueError("review evidence identifies a near-duplicate at Hamming distance <= 4")
    _text(validation["remaining_blocker"], "remaining blocker")

    full_contract = _exact(artifacts["full_contact"], {"path", "sha256"}, "full contact")
    selected_contract = _exact(
        artifacts["selected_window"], {"path", "sha256"}, "selected window",
    )
    preview_contract = _exact(
        artifacts["normalized_preview"],
        {"path", "sha256", "perceptual_hash", "frames", "fps", "size"},
        "normalized preview",
    )
    evidence_root = review_evidence_path.parent
    full_contact = _locked_relative(
        evidence_root, full_contract["path"], full_contract["sha256"], "full contact",
    )
    selected_window = _locked_relative(
        evidence_root, selected_contract["path"], selected_contract["sha256"],
        "selected window",
    )
    preview = _locked_relative(
        evidence_root, preview_contract["path"], preview_contract["sha256"],
        "normalized preview",
    )
    if preview_contract["frames"] != 16 or preview_contract["fps"] != 8 \
            or preview_contract["size"] != 96:
        raise ValueError("normalized preview contract must be 16 frames, 8 fps and 96x96")
    expected_phash = _text(preview_contract["perceptual_hash"], "perceptual hash")
    if perceptual_hash(decode_video_clip(preview, frames=16, size=96)) != expected_phash:
        raise ValueError("normalized preview pHash does not match review evidence")

    if output_dir.exists():
        raise ValueError("Figshare bundle output must not already exist")
    staging = output_dir.with_name(output_dir.name + ".tmp")
    if staging.exists():
        raise ValueError("Figshare bundle staging directory already exists")
    try:
        (staging / "clips").mkdir(parents=True)
        (staging / "source-media").mkdir()
        (staging / "review").mkdir()
        (staging / "provenance").mkdir()
        sample_id = f"figshare-{article_id}-file-{file_id}"
        clip_relative = Path("clips") / f"{sample_id}.mp4"
        _normalize(media_path, staging / clip_relative, start)
        _validate_normalized_clip(staging / clip_relative)
        clip_sha256 = _sha256(staging / clip_relative)
        if clip_sha256 != preview_contract["sha256"]:
            raise ValueError("normalized clip does not reproduce the reviewed preview")
        source_relative = Path("source-media") / f"figshare-{file_id}.mp4"
        shutil.copy2(media_path, staging / source_relative)
        shutil.copy2(full_contact, staging / "review/full-contact.jpg")
        shutil.copy2(selected_window, staging / "review/selected-window.jpg")
        article_relative = Path("provenance/article.json")
        review_relative = Path("provenance/review-evidence.json")
        (staging / article_relative).write_bytes(article_bytes)
        (staging / review_relative).write_bytes(evidence_bytes)
        attribution = ", ".join(author_names) + f"; {doi}; CC BY 4.0"
        record = RealVideoRecord(
            sample_id=sample_id,
            source_id=group,
            clip_path=clip_relative.as_posix(),
            prompt=f"a {decision['subject']} {decision['action']} {decision['scene']}",
            subject=str(decision["subject"]), action=str(decision["action"]),
            scene=str(decision["scene"]), source_url=f"https://doi.org/{doi}",
            license="CC BY 4.0", attribution=attribution, sha256=clip_sha256,
            license_url="https://creativecommons.org/licenses/by/4.0/",
            original_media_url=download_url, page_revision=f"figshare-v{version}:{modified}",
            original_media_timestamp=modified, downloaded_media_url=download_url,
            downloaded_media_kind="figshare_original", downloaded_media_sha256=media_sha256,
            downloaded_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            clip_start_seconds=start, clip_frames=16, clip_fps=8, clip_size=96,
            duplicate_group=group, perceptual_hash=expected_phash,
        )
        manifest = staging / "source.jsonl"
        write_manifest(manifest, [record])
        output_evidence = {
            "article_id": article_id,
            "article_path": article_relative.as_posix(),
            "article_sha256": hashlib.sha256(article_bytes).hexdigest(),
            "clip_sha256": clip_sha256,
            "conservative_group": group,
            "figshare_file_id": file_id,
            "full_contact_sha256": str(full_contract["sha256"]),
            "review_evidence_path": review_relative.as_posix(),
            "review_evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
            "selected_window_sha256": str(selected_contract["sha256"]),
            "source_manifest_sha256": _sha256(manifest),
            "source_media_path": source_relative.as_posix(),
            "source_media_sha256": media_sha256,
            "training_ready": True,
        }
        (staging / "evidence.json").write_text(
            json.dumps(output_evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output_dir)
        return output_evidence
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _bundle_contract(
    bundle: Path,
) -> tuple[RealVideoRecord, dict[str, object], bytes, int]:
    manifest_path = bundle / "source.jsonl"
    output_evidence_path = bundle / "evidence.json"
    if not manifest_path.is_file() or not output_evidence_path.is_file():
        raise ValueError("Figshare bundle lacks source.jsonl or evidence.json")
    output_evidence, output_evidence_bytes = _load_json(
        output_evidence_path, "Figshare bundle evidence",
    )
    expected_output_fields = {
        "article_id", "article_path", "article_sha256", "clip_sha256",
        "conservative_group", "figshare_file_id", "full_contact_sha256",
        "review_evidence_path", "review_evidence_sha256", "selected_window_sha256",
        "source_manifest_sha256", "source_media_path", "source_media_sha256",
        "training_ready",
    }
    _exact(output_evidence, expected_output_fields, "Figshare bundle evidence")
    if output_evidence["training_ready"] is not True \
            or output_evidence["source_manifest_sha256"] != _sha256(manifest_path):
        raise ValueError("Figshare bundle evidence does not bind a training-ready manifest")
    records = load_manifest(manifest_path)
    if len(records) != 1:
        raise ValueError("Figshare bundle must contain exactly one record")
    record = records[0]
    article_path = _locked_relative(
        bundle, output_evidence["article_path"], output_evidence["article_sha256"],
        "Figshare article",
    )
    review_path = _locked_relative(
        bundle, output_evidence["review_evidence_path"],
        output_evidence["review_evidence_sha256"], "Figshare review evidence",
    )
    source_media = _locked_relative(
        bundle, output_evidence["source_media_path"],
        output_evidence["source_media_sha256"], "Figshare source media",
    )
    _locked_relative(bundle, record.clip_path, record.sha256, "Figshare clip")
    _locked_relative(
        bundle, "review/full-contact.jpg", output_evidence["full_contact_sha256"],
        "full contact",
    )
    _locked_relative(
        bundle, "review/selected-window.jpg",
        output_evidence["selected_window_sha256"], "selected window",
    )
    article, _ = _load_json(article_path, "bundled Figshare article")
    review, _ = _load_json(review_path, "bundled Figshare review evidence")
    _exact(review, _TOP_FIELDS, "bundled Figshare review evidence")
    if review["schema_version"] != 1 or review["training_ready"] is not False:
        raise ValueError("bundled Figshare review must remain schema 1 and pre-import")
    article_contract = _exact(review.get("article"), _ARTICLE_FIELDS, "article contract")
    file_contract = _exact(review.get("file"), _FILE_FIELDS, "file contract")
    decision = _exact(review.get("decision"), _DECISION_FIELDS, "review decision")
    grouping = _exact(review.get("grouping"), _GROUP_FIELDS, "grouping contract")
    artifacts = _exact(review.get("review_artifacts"), _ARTIFACT_FIELDS, "review artifacts")
    validation = _exact(review.get("validation"), _VALIDATION_FIELDS, "validation contract")
    article_id = _positive_int(output_evidence["article_id"], "article id")
    file_id = _positive_int(output_evidence["figshare_file_id"], "Figshare file id")
    if article.get("id") != article_id or article_contract["id"] != article_id \
            or file_contract["figshare_file_id"] != file_id:
        raise ValueError("bundled Figshare article/file identity is inconsistent")
    version = _positive_int(article.get("version"), "bundled article version")
    expected_version = _positive_int(
        article_contract["version"], "bundled review article version",
    )
    doi = _text(article.get("doi"), "bundled article DOI")
    expected_doi = _text(article_contract["doi"], "bundled review DOI")
    if version != expected_version or doi != expected_doi:
        raise ValueError("bundled Figshare release identity is inconsistent")
    published = _utc(article.get("published_date"), "bundled article published_date")
    modified = _utc(article.get("modified_date"), "bundled article modified_date")
    if article_contract["url"] != f"https://api.figshare.com/v2/articles/{article_id}" \
            or published != _utc(
                article_contract["published_at"], "bundled review published_at",
            ) \
            or modified != _utc(
                article_contract["modified_at"], "bundled review modified_at",
            ):
        raise ValueError("bundled Figshare article URL or timestamps are inconsistent")
    license_data = article.get("license")
    if not isinstance(license_data, dict) or license_data.get("name") != "CC BY 4.0" \
            or license_data.get("url") != "https://creativecommons.org/licenses/by/4.0/" \
            or article_contract["license"] != "CC BY 4.0" \
            or article_contract["license_url"] != license_data.get("url"):
        raise ValueError("bundled Figshare source must retain exact CC BY 4.0")
    authors = article.get("authors")
    if not isinstance(authors, list) or not authors:
        raise ValueError("bundled Figshare article must retain its authors")
    author_names = [
        _text(author.get("full_name") if isinstance(author, dict) else None,
              "author full_name")
        for author in authors
    ]
    if _sha256(article_path) != article_contract["api_snapshot_sha256"]:
        raise ValueError("bundled Figshare review does not bind its article snapshot")
    files = article.get("files")
    matches = [
        item for item in files if isinstance(item, dict) and item.get("id") == file_id
    ] if isinstance(files, list) else []
    if len(matches) != 1:
        raise ValueError("bundled Figshare article lacks its exact file id")
    article_file = matches[0]
    file_name = _text(file_contract["name"], "bundled Figshare file name")
    file_size = _positive_int(file_contract["size_bytes"], "bundled Figshare file size")
    file_md5 = _text(file_contract["md5"], "bundled Figshare file MD5")
    file_sha256 = _text(file_contract["sha256"], "bundled Figshare file SHA-256")
    download_url = _text(file_contract["download_url"], "bundled Figshare download URL")
    parsed_download = urlparse(download_url)
    if not _HEX32.fullmatch(file_md5) or not _HEX64.fullmatch(file_sha256) \
            or parsed_download.scheme != "https" \
            or parsed_download.hostname != "ndownloader.figshare.com" \
            or parsed_download.path != f"/files/{file_id}":
        raise ValueError("bundled Figshare file digests or canonical download URL are invalid")
    for article_key, review_key in (
        ("name", "name"), ("size", "size_bytes"), ("computed_md5", "md5"),
        ("download_url", "download_url"),
    ):
        if article_file.get(article_key) != file_contract[review_key]:
            raise ValueError(f"bundled Figshare file {article_key} contract is inconsistent")
    source_bytes = source_media.read_bytes()
    if article_file.get("name") != file_name or len(source_bytes) != file_size \
            or hashlib.md5(source_bytes, usedforsecurity=False).hexdigest() != file_md5 \
            or file_sha256 != hashlib.sha256(source_bytes).hexdigest() \
            or file_sha256 != output_evidence["source_media_sha256"]:
        raise ValueError("bundled Figshare review does not bind its source media")
    if grouping["conservative_group"] != f"figshare-article:{article_id}" \
            or grouping["group_upper_bound"] != 1 \
            or output_evidence["conservative_group"] != grouping["conservative_group"]:
        raise ValueError("bundled Figshare grouping contract is invalid")
    for field in (
        "full_duration_reviewed", "single_animal_confirmed",
        "continuous_action_confirmed", "scene_confirmed", "no_edit_in_window",
    ):
        if decision[field] is not True:
            raise ValueError(f"bundled review must explicitly confirm {field}")
    if decision["status"] != "review-qualified-pending-import":
        raise ValueError("bundled review status is not import-qualified")
    for field in ("subject", "action", "scene", "notes", "reviewer_id"):
        _text(decision[field], field)
    _utc(decision["reviewed_at"], "reviewed_at")
    try:
        start = float(decision["clip_start_seconds"])
        stop = float(decision["clip_stop_seconds"])
    except (TypeError, ValueError):
        raise ValueError("bundled review window is invalid") from None
    if not math.isfinite(start) or not math.isfinite(stop) or start < 0 \
            or abs(stop - start - 2) > 1e-6 or stop > _probe_duration(source_media) + 1e-6:
        raise ValueError("bundled review window must be an in-bounds two-second interval")
    full_contract = _exact(artifacts["full_contact"], {"path", "sha256"}, "full contact")
    selected_contract = _exact(
        artifacts["selected_window"], {"path", "sha256"}, "selected window",
    )
    preview_contract = _exact(
        artifacts["normalized_preview"],
        {"path", "sha256", "perceptual_hash", "frames", "fps", "size"},
        "normalized preview",
    )
    if preview_contract["frames"] != 16 or preview_contract["fps"] != 8 \
            or preview_contract["size"] != 96:
        raise ValueError("bundled normalized preview contract is invalid")
    if output_evidence["full_contact_sha256"] != full_contract["sha256"] \
            or output_evidence["selected_window_sha256"] != selected_contract["sha256"] \
            or output_evidence["clip_sha256"] != preview_contract["sha256"] \
            or record.sha256 != preview_contract["sha256"]:
        raise ValueError("bundled Figshare review artifacts are inconsistent")
    if perceptual_hash(decode_video_clip(bundle / record.clip_path, frames=16, size=96)) \
            != preview_contract["perceptual_hash"]:
        raise ValueError("bundled Figshare clip pHash does not match review evidence")
    _validate_normalized_clip(bundle / record.clip_path)
    with tempfile.TemporaryDirectory(prefix="figshare-verify-") as temporary:
        reproduced = Path(temporary) / "clip.mp4"
        _normalize(source_media, reproduced, start)
        _validate_normalized_clip(reproduced)
        if _sha256(reproduced) != record.sha256:
            raise ValueError("bundled Figshare clip cannot be reproduced from source and review")
    expected_record = {
        "sample_id": f"figshare-{article_id}-file-{file_id}",
        "source_id": grouping["conservative_group"],
        "duplicate_group": grouping["conservative_group"],
        "subject": decision["subject"], "action": decision["action"],
        "scene": decision["scene"], "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "prompt": f"a {decision['subject']} {decision['action']} {decision['scene']}",
        "source_url": f"https://doi.org/{article_contract['doi']}",
        "attribution": (
            ", ".join(author_names)
            + f"; {article_contract['doi']}; CC BY 4.0"
        ),
        "original_media_url": file_contract["download_url"],
        "page_revision": (
            f"figshare-v{article_contract['version']}:{article_contract['modified_at']}"
        ),
        "original_media_timestamp": article_contract["modified_at"],
        "downloaded_media_url": file_contract["download_url"],
        "downloaded_media_kind": "figshare_original",
        "downloaded_media_sha256": file_contract["sha256"],
        "perceptual_hash": preview_contract["perceptual_hash"],
        "clip_start_seconds": decision["clip_start_seconds"],
        "clip_frames": 16, "clip_fps": 8, "clip_size": 96,
    }
    for field, value in expected_record.items():
        if getattr(record, field) != value:
            raise ValueError(f"Figshare manifest {field} does not match its audit contract")
    _utc(record.downloaded_at, "manifest downloaded_at")
    nearest_distance = _positive_int(
        validation["nearest_existing_phash_distance"],
        "nearest_existing_phash_distance",
    )
    if nearest_distance <= 4:
        raise ValueError("bundled review identifies a near-duplicate at distance <= 4")
    _text(validation["remaining_blocker"], "remaining blocker")
    expected_files = {
        "source.jsonl", "evidence.json", record.clip_path,
        str(output_evidence["source_media_path"]), str(output_evidence["article_path"]),
        str(output_evidence["review_evidence_path"]), "review/full-contact.jpg",
        "review/selected-window.jpg",
    }
    actual_files = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*") if path.is_file() or path.is_symlink()
    }
    if actual_files != expected_files or any(path.is_symlink() for path in bundle.rglob("*")):
        raise ValueError("Figshare bundle contains unexpected files or symbolic links")
    return record, output_evidence, output_evidence_bytes, nearest_distance


def merge_bundle(
    base_manifest: Path, bundle: Path, output_dir: Path,
) -> dict[str, object]:
    if (base_manifest.parent / "merge-evidence.json").exists() \
            or (base_manifest.parent / "figshare-audit").exists() \
            or (base_manifest.parent / "contribution-audit").exists():
        raise ValueError(
            "merge base must be the fixed pristine dataset; rebuild from that base and "
            "all approved source bundles"
        )
    if output_dir.exists():
        raise ValueError("merged output directory must not already exist")
    input_roots = (base_manifest.parent.resolve(), bundle.resolve())
    output_resolved = output_dir.resolve()
    if any(output_resolved == root or output_resolved.is_relative_to(root) for root in input_roots):
        raise ValueError("merged output must be outside every input dataset tree")
    staging = output_dir.with_name(output_dir.name + ".tmp")
    if staging.exists():
        raise ValueError("merged output staging directory already exists")
    base_bytes = base_manifest.read_bytes()
    base_records = load_manifest(base_manifest)
    contribution, bundle_evidence, bundle_evidence_bytes, expected_nearest = (
        _bundle_contract(bundle)
    )
    all_records = [*base_records, contribution]
    sample_ids = [record.sample_id for record in all_records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("base and Figshare bundle contain duplicate sample_id values")
    base_phashes = [record.perceptual_hash for record in base_records if record.perceptual_hash]
    if not base_phashes:
        raise ValueError("base manifest must contain perceptual hashes for near-duplicate audit")
    actual_nearest = min(
        (int(contribution.perceptual_hash, 16) ^ int(base_hash, 16)).bit_count()
        for base_hash in base_phashes
    )
    if actual_nearest != expected_nearest or actual_nearest <= 4:
        raise ValueError("Figshare candidate-to-base pHash distance does not match review evidence")
    try:
        (staging / "clips").mkdir(parents=True)
        (staging / "figshare-audit").mkdir()
        merged_records: list[RealVideoRecord] = []
        for record, root in [
            *((record, base_manifest.parent) for record in base_records),
            (contribution, bundle),
        ]:
            source = _locked_relative(root, record.clip_path, record.sha256, "clip")
            destination = Path("clips") / (
                hashlib.sha256(record.sample_id.encode()).hexdigest()[:24]
                + source.suffix.lower()
            )
            if (staging / destination).exists():
                raise ValueError("merged clip destination collision")
            shutil.copy2(source, staging / destination)
            merged_records.append(replace(record, clip_path=destination.as_posix()))
        audit_name = hashlib.sha256(
            str(bundle_evidence["source_manifest_sha256"]).encode()
        ).hexdigest()[:24]
        audit_destination = staging / "figshare-audit" / audit_name
        shutil.copytree(bundle, audit_destination)
        manifest = staging / "source.jsonl"
        write_manifest(manifest, merged_records)
        contract = {
            "bundle_evidence_sha256": hashlib.sha256(bundle_evidence_bytes).hexdigest(),
            "bundle_manifest_sha256": bundle_evidence["source_manifest_sha256"],
            "portable_audit_path": audit_destination.relative_to(staging).as_posix(),
        }
        report = {
            "base_manifest_sha256": hashlib.sha256(base_bytes).hexdigest(),
            "figshare_contracts": [contract],
            "records": len(merged_records),
            "source_manifest_sha256": _sha256(manifest),
            "training_ready": True,
        }
        (staging / "merge-evidence.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        staging.replace(output_dir)
        return report
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("article", type=Path)
    create.add_argument("media", type=Path)
    create.add_argument("review_evidence", type=Path)
    create.add_argument("output", type=Path)
    merge = subparsers.add_parser("merge")
    merge.add_argument("base_manifest", type=Path)
    merge.add_argument("bundle", type=Path)
    merge.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "create":
        create_bundle(args.article, args.media, args.review_evidence, args.output)
        return 0
    if args.command == "merge":
        merge_bundle(args.base_manifest, args.bundle, args.output)
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    sys.exit(main())
