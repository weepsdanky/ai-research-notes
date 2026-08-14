"""Create and merge review-locked PeerTube video bundles."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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

from real_data import RealVideoRecord, decode_video_clip, load_manifest, perceptual_hash, write_manifest


HEX64 = re.compile(r"[0-9a-f]{64}")
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
LICENSES = {
    1: ("CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/"),
    2: ("CC BY-SA 4.0", "https://creativecommons.org/licenses/by-sa/4.0/"),
    7: ("CC0 1.0", "https://creativecommons.org/publicdomain/zero/1.0/"),
}
DECISION_FIELDS = {
    "status", "subject", "action", "scene", "clip_start_seconds", "clip_stop_seconds",
    "reviewer_id", "reviewed_at", "full_duration_reviewed", "single_animal_confirmed",
    "continuous_action_confirmed", "scene_confirmed", "no_edit_in_window", "notes",
}


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path: Path, name: str) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=_object)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {name}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value, raw


def _exact(value: object, fields: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} must contain exactly {sorted(fields)}")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _utc(value: object, name: str) -> str:
    text = _text(value, name)
    if not text.endswith("Z") or "T" not in text:
        raise ValueError(f"{name} must be UTC RFC3339")
    try:
        datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError:
        raise ValueError(f"{name} must be UTC RFC3339") from None
    return text


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _locked(root: Path, relative: object, digest: object, name: str) -> Path:
    rel = Path(_text(relative, f"{name} path"))
    expected = _text(digest, f"{name} sha256")
    if rel.is_absolute() or ".." in rel.parts or not HEX64.fullmatch(expected):
        raise ValueError(f"invalid {name} path or SHA-256")
    path = root / rel
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"{name} must be a regular file inside its bundle")
    if _sha(path) != expected:
        raise ValueError(f"{name} SHA-256 mismatch")
    return path


def _duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
         "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    duration = float(result.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("media duration must be positive and finite")
    return duration


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _normalize(source: Path, destination: Path, start: float) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", str(source),
         "-t", "2", "-an", "-vf",
         "fps=8,scale=96:96:force_original_aspect_ratio=increase,crop=96:96",
         "-frames:v", "16", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(destination)],
        check=True, timeout=120,
    )


def _renditions(video: dict[str, object]) -> list[dict[str, object]]:
    result = [item for item in video.get("files", []) if isinstance(item, dict)] \
        if isinstance(video.get("files"), list) else []
    playlists = video.get("streamingPlaylists")
    if isinstance(playlists, list):
        for playlist in playlists:
            if isinstance(playlist, dict) and isinstance(playlist.get("files"), list):
                result.extend(item for item in playlist["files"] if isinstance(item, dict))
    return result


def _contract(
    video_path: Path, account_path: Path, media_path: Path, evidence_path: Path,
) -> tuple[RealVideoRecord, dict[str, object], dict[str, bytes | Path]]:
    video, video_bytes = _load(video_path, "PeerTube video snapshot")
    account, account_bytes = _load(account_path, "PeerTube account snapshot")
    evidence, evidence_bytes = _load(evidence_path, "PeerTube review evidence")
    top = {
        "schema_version", "video", "account", "rendition", "decision", "grouping",
        "review_artifacts", "validation", "training_ready",
    }
    _exact(evidence, top, "review evidence")
    if evidence["schema_version"] != 1 or evidence["training_ready"] is not False:
        raise ValueError("review evidence must be schema 1 and pre-import")
    expected_video = _exact(evidence["video"], {
        "uuid", "origin", "watch_url", "api_snapshot_sha256", "name", "description",
        "duration_seconds", "published_at", "licence_id", "licence_label",
    }, "video contract")
    expected_account = _exact(evidence["account"], {
        "name", "display_name", "url", "description", "created_at", "api_snapshot_sha256",
        "originality_basis",
    }, "account contract")
    rendition = _exact(evidence["rendition"], {
        "url", "size_bytes", "sha256", "resolution_id",
    }, "rendition contract")
    decision = _exact(evidence["decision"], DECISION_FIELDS, "decision")
    grouping = _exact(evidence["grouping"], {
        "conservative_group", "reason",
    }, "grouping")
    artifacts = _exact(evidence["review_artifacts"], {
        "full_contact", "selected_window", "normalized_preview",
    }, "review artifacts")
    validation = _exact(evidence["validation"], {
        "nearest_existing_phash_distance", "remaining_blocker",
    }, "validation")

    if hashlib.sha256(video_bytes).hexdigest() != expected_video["api_snapshot_sha256"] \
            or hashlib.sha256(account_bytes).hexdigest() != expected_account["api_snapshot_sha256"]:
        raise ValueError("PeerTube snapshots do not match review evidence")
    video_uuid = _text(video.get("uuid"), "video UUID")
    if not UUID.fullmatch(video_uuid) or video_uuid != expected_video["uuid"]:
        raise ValueError("invalid or inconsistent PeerTube UUID")
    origin = _text(expected_video["origin"], "origin")
    parsed_origin = urlparse(origin)
    if parsed_origin.scheme != "https" or parsed_origin.path not in ("", "/") \
            or parsed_origin.username or parsed_origin.password or parsed_origin.query \
            or parsed_origin.fragment:
        raise ValueError("PeerTube origin must be an HTTPS origin")
    watch_url = f"{origin.rstrip('/')}/videos/watch/{video_uuid}"
    if expected_video["watch_url"] != watch_url:
        raise ValueError("PeerTube watch URL is not canonical")
    scalar_fields = {
        "name": "name", "description": "description", "duration": "duration_seconds",
        "publishedAt": "published_at",
    }
    for source, expected in scalar_fields.items():
        if video.get(source) != expected_video[expected]:
            raise ValueError(f"PeerTube video {source} mismatch")
    _utc(expected_video["published_at"], "published_at")
    licence = video.get("licence")
    licence_id = expected_video["licence_id"]
    if not isinstance(licence_id, int) or isinstance(licence_id, bool) or licence_id not in LICENSES \
            or not isinstance(licence, dict) or licence.get("id") != licence_id \
            or licence.get("label") != expected_video["licence_label"]:
        raise ValueError("PeerTube licence is absent or not in the allowed set")

    for source, expected in (
        ("name", "name"), ("displayName", "display_name"), ("url", "url"),
        ("description", "description"), ("createdAt", "created_at"),
    ):
        if account.get(source) != expected_account[expected]:
            raise ValueError(f"PeerTube account {source} mismatch")
    account_name = _text(expected_account["name"], "account name")
    if expected_account["url"] != f"{origin.rstrip('/')}/accounts/{account_name}":
        raise ValueError("PeerTube account URL is not canonical")
    _text(expected_account["originality_basis"], "originality basis")
    _utc(expected_account["created_at"], "account created_at")
    embedded_account = video.get("account")
    if not isinstance(embedded_account, dict) or embedded_account.get("name") != account_name:
        raise ValueError("video and account snapshots disagree")

    media_url = _text(rendition["url"], "rendition URL")
    parsed_media = urlparse(media_url)
    size_bytes = _positive_int(rendition["size_bytes"], "rendition size")
    resolution_id = _positive_int(rendition["resolution_id"], "rendition resolution")
    if parsed_media.scheme != "https" or parsed_media.netloc != parsed_origin.netloc \
            or not parsed_media.path.startswith("/download/"):
        raise ValueError("rendition URL must be a canonical download on the PeerTube origin")
    matches = [item for item in _renditions(video) if item.get("fileDownloadUrl") == media_url]
    if len(matches) != 1 or matches[0].get("size") != size_bytes \
            or not isinstance(matches[0].get("resolution"), dict) \
            or matches[0]["resolution"].get("id") != resolution_id:
        raise ValueError("rendition does not match the API snapshot")
    if media_path.is_symlink() or not media_path.is_file():
        raise ValueError("PeerTube rendition must be a regular local file")
    media_bytes = media_path.read_bytes()
    media_sha = _text(rendition["sha256"], "rendition SHA-256")
    if not HEX64.fullmatch(media_sha) or len(media_bytes) != size_bytes \
            or hashlib.sha256(media_bytes).hexdigest() != media_sha:
        raise ValueError("media bytes do not match the rendition contract")
    source_duration = _duration(media_path)
    expected_duration = _positive_int(expected_video["duration_seconds"], "video duration")
    if abs(source_duration - expected_duration) > 1:
        raise ValueError("rendition duration does not match the PeerTube video contract")

    if decision["status"] != "review-qualified-pending-import":
        raise ValueError("decision is not import-qualified")
    for field in ("full_duration_reviewed", "single_animal_confirmed",
                  "continuous_action_confirmed", "scene_confirmed", "no_edit_in_window"):
        if decision[field] is not True:
            raise ValueError(f"review must confirm {field}")
    for field in ("subject", "action", "scene", "reviewer_id", "notes"):
        _text(decision[field], field)
    _utc(decision["reviewed_at"], "reviewed_at")
    start = _finite_number(decision["clip_start_seconds"], "clip_start_seconds")
    stop = _finite_number(decision["clip_stop_seconds"], "clip_stop_seconds")
    if not math.isfinite(start) or not math.isfinite(stop) or start < 0 \
            or abs(stop - start - 2) > 1e-6 or stop > source_duration + 1e-6:
        raise ValueError("review window must be an in-bounds two-second interval")
    group = _text(grouping["conservative_group"], "conservative group")
    if not group.startswith(f"peertube:{parsed_origin.hostname}:"):
        raise ValueError("conservative group must be bound to the PeerTube origin")
    _text(grouping["reason"], "group reason")
    nearest = validation["nearest_existing_phash_distance"]
    if not isinstance(nearest, int) or isinstance(nearest, bool) or nearest <= 4:
        raise ValueError("candidate must not be a pHash near-duplicate of the base")
    _text(validation["remaining_blocker"], "remaining blocker")

    root = evidence_path.parent
    full = _exact(artifacts["full_contact"], {"path", "sha256"}, "full contact")
    selected = _exact(artifacts["selected_window"], {"path", "sha256"}, "selected window")
    preview = _exact(artifacts["normalized_preview"], {
        "path", "sha256", "perceptual_hash", "frames", "fps", "size",
    }, "normalized preview")
    full_path = _locked(root, full["path"], full["sha256"], "full contact")
    selected_path = _locked(root, selected["path"], selected["sha256"], "selected window")
    preview_path = _locked(root, preview["path"], preview["sha256"], "normalized preview")
    if len({full_path.resolve(), selected_path.resolve(), preview_path.resolve()}) != 3:
        raise ValueError("review artifacts must be three distinct files")
    if (preview["frames"], preview["fps"], preview["size"]) != (16, 8, 96):
        raise ValueError("normalized preview must be 16 frames, 8 fps and 96x96")
    phash = _text(preview["perceptual_hash"], "perceptual hash")
    if perceptual_hash(decode_video_clip(preview_path, frames=16, size=96)) != phash:
        raise ValueError("normalized preview pHash mismatch")

    license_name, license_url = LICENSES[licence_id]
    sample_id = f"peertube-{video_uuid}"
    record = RealVideoRecord(
        sample_id=sample_id, source_id=group, clip_path=f"clips/{sample_id}.mp4",
        prompt=f"a {decision['subject']} {decision['action']} {decision['scene']}",
        subject=str(decision["subject"]), action=str(decision["action"]),
        scene=str(decision["scene"]), source_url=watch_url, license=license_name,
        attribution=f"{expected_account['display_name']} ({account_name}); {watch_url}; {license_name}",
        sha256=str(preview["sha256"]), license_url=license_url,
        original_media_url=media_url,
        page_revision=f"peertube:{video_uuid}:{expected_video['published_at']}",
        original_media_timestamp=str(expected_video["published_at"]),
        downloaded_media_url=media_url, downloaded_media_kind="peertube_rendition",
        downloaded_media_sha256=media_sha,
        downloaded_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        clip_start_seconds=start, clip_frames=16, clip_fps=8, clip_size=96,
        duplicate_group=group, perceptual_hash=phash,
    )
    locked = {
        "video": video_path, "video_bytes": video_bytes, "account": account_path,
        "account_bytes": account_bytes, "media": media_path, "evidence": evidence_path,
        "evidence_bytes": evidence_bytes, "full": full_path, "selected": selected_path,
        "preview": preview_path,
    }
    return record, evidence, locked


def create_bundle(video: Path, account: Path, media: Path, review: Path, output: Path) -> None:
    record, _, locked = _contract(video, account, media, review)
    if output.exists():
        raise ValueError("PeerTube bundle output already exists")
    staging = output.with_name(output.name + ".tmp")
    if staging.exists():
        raise ValueError("PeerTube staging directory already exists")
    try:
        for directory in ("clips", "source-media", "review", "provenance"):
            (staging / directory).mkdir(parents=True, exist_ok=True)
        _normalize(Path(locked["media"]), staging / record.clip_path, record.clip_start_seconds)
        if _sha(staging / record.clip_path) != record.sha256:
            raise ValueError("reviewed preview is not reproducible from the source rendition")
        shutil.copy2(Path(locked["media"]), staging / "source-media/source.mp4")
        shutil.copy2(Path(locked["full"]), staging / "review/full-contact.jpg")
        shutil.copy2(Path(locked["selected"]), staging / "review/selected-window.jpg")
        (staging / "provenance/video.json").write_bytes(locked["video_bytes"])
        (staging / "provenance/account.json").write_bytes(locked["account_bytes"])
        (staging / "provenance/review.json").write_bytes(locked["evidence_bytes"])
        # Keep the original relative artifact paths valid so merge-time semantic
        # validation replays the exact same review contract.
        review_payload = json.loads(locked["evidence_bytes"])
        for key, source_key in (
            ("full_contact", "full"), ("selected_window", "selected"),
            ("normalized_preview", "preview"),
        ):
            relative = Path(review_payload["review_artifacts"][key]["path"])
            destination = staging / "provenance" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(locked[source_key]), destination)
        write_manifest(staging / "source.jsonl", [record])
        report = {
            "account_sha256": hashlib.sha256(locked["account_bytes"]).hexdigest(),
            "clip_sha256": record.sha256,
            "full_contact_sha256": _sha(Path(locked["full"])),
            "review_sha256": hashlib.sha256(locked["evidence_bytes"]).hexdigest(),
            "selected_window_sha256": _sha(Path(locked["selected"])),
            "source_manifest_sha256": _sha(staging / "source.jsonl"),
            "source_media_sha256": _sha(Path(locked["media"])),
            "training_ready": True,
            "video_sha256": hashlib.sha256(locked["video_bytes"]).hexdigest(),
        }
        (staging / "evidence.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        staging.replace(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _verify_bundle(bundle: Path) -> tuple[RealVideoRecord, int]:
    report, _ = _load(bundle / "evidence.json", "bundle evidence")
    _exact(report, {
        "account_sha256", "clip_sha256", "full_contact_sha256", "review_sha256",
        "selected_window_sha256", "source_manifest_sha256", "source_media_sha256",
        "training_ready", "video_sha256",
    }, "bundle evidence")
    if report["training_ready"] is not True or _sha(bundle / "source.jsonl") != report["source_manifest_sha256"]:
        raise ValueError("bundle is not training-ready or its manifest changed")
    records = load_manifest(bundle / "source.jsonl")
    if len(records) != 1:
        raise ValueError("PeerTube bundle must contain exactly one record")
    record, review, _ = _contract(
        _locked(bundle, "provenance/video.json", report["video_sha256"], "video snapshot"),
        _locked(bundle, "provenance/account.json", report["account_sha256"], "account snapshot"),
        _locked(bundle, "source-media/source.mp4", report["source_media_sha256"], "source media"),
        _locked(bundle, "provenance/review.json", report["review_sha256"], "review evidence"),
    )
    stored = records[0]
    _utc(stored.downloaded_at, "manifest downloaded_at")
    ignored = {"clip_path", "downloaded_at"}
    for field, value in asdict(record).items():
        if field not in ignored and getattr(stored, field) != value:
            raise ValueError(f"PeerTube manifest {field} diverges from its contract")
    _locked(bundle, stored.clip_path, stored.sha256, "normalized clip")
    _locked(bundle, "review/full-contact.jpg", report["full_contact_sha256"], "full contact")
    _locked(bundle, "review/selected-window.jpg", report["selected_window_sha256"], "selected window")
    if any(path.is_symlink() for path in bundle.rglob("*")):
        raise ValueError("PeerTube bundle must not contain symbolic links")
    review_payload, _ = _load(bundle / "provenance/review.json", "bundled review evidence")
    artifact_paths = {
        f"provenance/{review_payload['review_artifacts'][key]['path']}"
        for key in ("full_contact", "selected_window", "normalized_preview")
    }
    expected_files = {
        "evidence.json", "source.jsonl", stored.clip_path, "source-media/source.mp4",
        "review/full-contact.jpg", "review/selected-window.jpg", "provenance/video.json",
        "provenance/account.json", "provenance/review.json", *artifact_paths,
    }
    actual_files = {
        path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("PeerTube bundle contains missing or unexpected files")
    with tempfile.TemporaryDirectory(prefix="peertube-verify-") as temporary:
        reproduced = Path(temporary) / "clip.mp4"
        _normalize(bundle / "source-media/source.mp4", reproduced, stored.clip_start_seconds)
        if _sha(reproduced) != stored.sha256:
            raise ValueError("PeerTube clip cannot be reproduced")
    return stored, int(review["validation"]["nearest_existing_phash_distance"])


def merge_bundles(base_manifest: Path, bundles: list[Path], output: Path) -> None:
    if not bundles:
        raise ValueError("at least one PeerTube bundle is required")
    if output.exists():
        raise ValueError("merged output already exists")
    roots = [base_manifest.parent.resolve(), *(bundle.resolve() for bundle in bundles)]
    resolved_output = output.resolve()
    if any(resolved_output == root or resolved_output.is_relative_to(root) for root in roots):
        raise ValueError("merged output must be outside every input tree")
    base_records = load_manifest(base_manifest)
    additions = [_verify_bundle(bundle) for bundle in bundles]
    ids = [record.sample_id for record in base_records] + [item[0].sample_id for item in additions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample IDs in merge inputs")
    base_hashes = [record.perceptual_hash for record in base_records]
    if any(value is None for value in base_hashes):
        raise ValueError("base manifest lacks perceptual hashes")
    for record, expected in additions:
        actual = min((int(record.perceptual_hash, 16) ^ int(value, 16)).bit_count() for value in base_hashes)
        if actual != expected or actual <= 4:
            raise ValueError("candidate-to-base pHash distance changed")
    for index, (left, _) in enumerate(additions):
        for right, _ in additions[index + 1:]:
            if (int(left.perceptual_hash, 16) ^ int(right.perceptual_hash, 16)).bit_count() <= 4:
                raise ValueError("PeerTube bundles contain a cross-bundle pHash near-duplicate")
    staging = output.with_name(output.name + ".tmp")
    if staging.exists():
        raise ValueError("merge staging directory exists")
    try:
        (staging / "clips").mkdir(parents=True)
        (staging / "peertube-audit").mkdir()
        prior_evidence = base_manifest.parent / "merge-evidence.json"
        prior_evidence_sha256 = None
        if prior_evidence.is_file():
            if prior_evidence.is_symlink():
                raise ValueError("base merge evidence must not be a symbolic link")
            prior_report, _ = _load(prior_evidence, "base merge evidence")
            if prior_report.get("training_ready") is not True \
                    or prior_report.get("source_manifest_sha256") != _sha(base_manifest):
                raise ValueError("base merge evidence does not bind the base manifest")
            (staging / "base-audit").mkdir()
            shutil.copy2(prior_evidence, staging / "base-audit/merge-evidence.json")
            prior_evidence_sha256 = _sha(prior_evidence)
            for audit_name in ("figshare-audit", "contribution-audit"):
                source_audit = base_manifest.parent / audit_name
                if source_audit.exists():
                    if source_audit.is_symlink() or any(path.is_symlink() for path in source_audit.rglob("*")):
                        raise ValueError("base portable audit must not contain symbolic links")
                    shutil.copytree(source_audit, staging / "base-audit" / audit_name)
        records: list[RealVideoRecord] = []
        inputs = [(record, base_manifest.parent) for record in base_records]
        inputs.extend((item[0], bundle) for item, bundle in zip(additions, bundles))
        for record, root in inputs:
            source = _locked(root, record.clip_path, record.sha256, "clip")
            destination = Path("clips") / (hashlib.sha256(record.sample_id.encode()).hexdigest()[:24] + ".mp4")
            if (staging / destination).exists():
                raise ValueError("merged clip destination collision")
            shutil.copy2(source, staging / destination)
            records.append(replace(record, clip_path=destination.as_posix()))
        for bundle in bundles:
            name = hashlib.sha256((bundle / "source.jsonl").read_bytes()).hexdigest()[:24]
            shutil.copytree(bundle, staging / "peertube-audit" / name)
        write_manifest(staging / "source.jsonl", records)
        report = {
            "base_manifest_sha256": _sha(base_manifest), "bundles": len(bundles),
            "base_merge_evidence_sha256": prior_evidence_sha256,
            "records": len(records), "source_manifest_sha256": _sha(staging / "source.jsonl"),
            "training_ready": True,
        }
        (staging / "merge-evidence.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        staging.replace(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    for name in ("video", "account", "media", "review", "output"):
        create.add_argument(name, type=Path)
    merge = commands.add_parser("merge")
    merge.add_argument("base_manifest", type=Path)
    merge.add_argument("bundles", nargs="+", type=Path)
    merge.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "create":
        create_bundle(args.video, args.account, args.media, args.review, args.output)
    else:
        merge_bundles(args.base_manifest, args.bundles, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
