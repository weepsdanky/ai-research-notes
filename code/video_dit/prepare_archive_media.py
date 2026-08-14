"""Download and verify review-only Internet Archive media candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request

from archive_import import license_allowed


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX32 = re.compile(r"[0-9a-f]{32}")
_VIDEO_SUFFIXES = {".mp4", ".m4v", ".mov", ".webm", ".ogv"}


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_candidates(path: Path) -> tuple[list[dict[str, object]], str]:
    raw = path.read_bytes()
    rows: list[dict[str, object]] = []
    identifiers: set[str] = set()
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, object_pairs_hook=_object_without_duplicate_keys)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid candidate JSONL line {line_number}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"candidate line {line_number} must be an object")
        identifier = value.get("identifier")
        original_file = value.get("original_file")
        if not isinstance(identifier, str) or not _IDENTIFIER.fullmatch(identifier):
            raise ValueError(f"candidate line {line_number} has an invalid identifier")
        if identifier in identifiers:
            raise ValueError(f"duplicate Internet Archive identifier: {identifier}")
        if not isinstance(original_file, str) or Path(original_file).name != original_file:
            raise ValueError(f"candidate {identifier} has an unsafe original filename")
        suffix = Path(original_file).suffix.lower()
        if suffix not in _VIDEO_SUFFIXES:
            raise ValueError(f"candidate {identifier} does not name a supported video")
        expected_source = f"https://archive.org/details/{urllib.parse.quote(identifier, safe='')}"
        expected_media = (
            f"https://archive.org/download/{urllib.parse.quote(identifier, safe='')}/"
            f"{urllib.parse.quote(original_file, safe='')}"
        )
        if value.get("source_url") != expected_source or value.get("media_url") != expected_media:
            raise ValueError(f"candidate {identifier} has an unbound Archive URL")
        if value.get("review_status") != "pending" or value.get("action") != "review_required" \
                or value.get("scene") != "review_required":
            raise ValueError(f"candidate {identifier} must remain review-only pending")
        if value.get("subject") not in {"cat", "dog"}:
            raise ValueError(f"candidate {identifier} has an unsupported subject")
        if not _HEX40.fullmatch(str(value.get("original_sha1", ""))) \
                or not _HEX32.fullmatch(str(value.get("original_md5", ""))):
            raise ValueError(f"candidate {identifier} has invalid original digests")
        size = value.get("original_size")
        mtime = value.get("original_mtime")
        if not isinstance(size, int) or isinstance(size, bool) or size < 1 \
                or not isinstance(mtime, int) or isinstance(mtime, bool) or mtime < 1:
            raise ValueError(f"candidate {identifier} has invalid original size/mtime")
        for field in ("title", "creator", "license_url", "source_id", "duplicate_group", "query"):
            if not isinstance(value.get(field), str) or not str(value[field]).strip():
                raise ValueError(f"candidate {identifier} lacks {field}")
        creator = str(value["creator"])
        if "@" in creator or not license_allowed(str(value["license_url"])):
            raise ValueError(f"candidate {identifier} has an incompatible license or creator")
        creator_group = "internet-archive-creator:" + hashlib.sha256(
            creator.casefold().encode()
        ).hexdigest()
        if value["source_id"] != creator_group or value["duplicate_group"] != creator_group:
            raise ValueError(f"candidate {identifier} has an invalid creator group")
        identifiers.add(identifier)
        rows.append(value)
    if not rows:
        raise ValueError("candidate JSONL must contain at least one record")
    return rows, hashlib.sha256(raw).hexdigest()


def _download_verified(row: dict[str, object], destination: Path, byte_cap: int) -> str:
    request = urllib.request.Request(
        str(row["media_url"]),
        headers={"User-Agent": "ai-research-notes-video-dit/1.0"},
    )
    sha256 = hashlib.sha256()
    sha1 = hashlib.sha1()
    md5 = hashlib.md5()
    written = 0
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("xb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > int(row["original_size"]) or written > byte_cap:
                raise ValueError(f"Archive media exceeded its byte contract: {row['identifier']}")
            output.write(chunk)
            sha256.update(chunk)
            sha1.update(chunk)
            md5.update(chunk)
    if written != int(row["original_size"]):
        raise ValueError(f"Archive media size mismatch: {row['identifier']}")
    if sha1.hexdigest() != row["original_sha1"] or md5.hexdigest() != row["original_md5"]:
        raise ValueError(f"Archive media digest mismatch: {row['identifier']}")
    return sha256.hexdigest()


def _positive_int(value: object, name: str) -> int:
    try:
        result = int(str(value))
    except (TypeError, ValueError):
        raise ValueError(f"ffprobe lacks a positive {name}") from None
    if result < 1:
        raise ValueError(f"ffprobe lacks a positive {name}")
    return result


def _probe_media(path: Path) -> dict[str, object]:
    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries",
                "stream=codec_name,width,height,avg_frame_rate,nb_read_frames,nb_frames:format=duration",
                "-of", "json", str(path),
            ],
            check=True, capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise ValueError("ffprobe timed out while validating Archive media") from None
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
        raise ValueError("ffprobe must return exactly one selected video stream")
    stream = streams[0]
    codec = stream.get("codec_name")
    if not isinstance(codec, str) or not codec:
        raise ValueError("ffprobe lacks a video codec")
    try:
        duration = float(payload["format"]["duration"])
        numerator, denominator = str(stream["avg_frame_rate"]).split("/", 1)
        fps = float(numerator) / float(denominator)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        raise ValueError("ffprobe returned an invalid duration/frame rate") from None
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("ffprobe returned a non-positive duration/frame rate")
    frames_value = stream.get("nb_read_frames", stream.get("nb_frames"))
    return {
        "codec": codec,
        "width": _positive_int(stream.get("width"), "width"),
        "height": _positive_int(stream.get("height"), "height"),
        "fps": fps,
        "frames": _positive_int(frames_value, "frame count"),
        "duration_seconds": duration,
    }


def download_archive_media(
    candidates_path: Path, media_dir: Path, manifest_path: Path, *, max_total_bytes: int,
) -> dict[str, object]:
    if max_total_bytes < 1:
        raise ValueError("max_total_bytes must be positive")
    candidates_resolved = candidates_path.resolve()
    media_resolved = media_dir.resolve()
    manifest_resolved = manifest_path.resolve()
    if len({candidates_resolved, media_resolved, manifest_resolved}) != 3:
        raise ValueError("candidate, media, and manifest paths must differ")
    if media_resolved.parent != manifest_resolved.parent:
        raise ValueError("media directory must be a direct child of the manifest directory")
    if media_dir.exists() or manifest_path.exists():
        raise ValueError("media directory and manifest output must not already exist")

    rows, candidate_sha256 = _load_candidates(candidates_path)
    expected_total = sum(int(row["original_size"]) for row in rows)
    if expected_total > max_total_bytes:
        raise ValueError("Archive candidate bytes exceed the configured hard cap")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = media_dir.with_name(media_dir.name + ".tmp")
    staging_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    if staging_dir.exists() or staging_manifest.exists():
        raise ValueError("staging outputs already exist")
    media_rows: list[dict[str, object]] = []
    committed_media = False
    try:
        staging_dir.mkdir()
        for row in sorted(rows, key=lambda item: str(item["identifier"])):
            suffix = Path(str(row["original_file"])).suffix.lower()
            filename = hashlib.sha256(str(row["identifier"]).encode()).hexdigest()[:24] + suffix
            staged_path = staging_dir / filename
            sha256 = _download_verified(row, staged_path, max_total_bytes)
            probe = _probe_media(staged_path)
            media_rows.append({
                "identifier": row["identifier"], "title": row["title"],
                "subject": row["subject"], "query": row["query"],
                "source_id": row["source_id"], "duplicate_group": row["duplicate_group"],
                "source_url": row["source_url"], "media_url": row["media_url"],
                "license_url": row["license_url"], "creator": row["creator"],
                "original_file": row["original_file"], "original_sha1": row["original_sha1"],
                "original_md5": row["original_md5"], "original_mtime": row["original_mtime"],
                "bytes": row["original_size"], "sha256": sha256,
                "local_path": (Path(media_dir.name) / filename).as_posix(),
                **probe,
            })
        report = {
            "candidate_sha256": candidate_sha256,
            "records": len(media_rows),
            "total_bytes": expected_total,
            "max_total_bytes": max_total_bytes,
            "media": media_rows,
            "training_ready": False,
            "training_blocker": "downloaded media still requires full-duration human factor review",
        }
        staging_manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        staging_dir.replace(media_dir)
        committed_media = True
        staging_manifest.replace(manifest_path)
        return report
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        if staging_manifest.exists():
            staging_manifest.unlink()
        if committed_media and media_dir.exists() and not manifest_path.exists():
            shutil.rmtree(media_dir)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("media_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--max-total-bytes", type=int, required=True)
    args = parser.parse_args(argv)
    download_archive_media(
        args.candidates, args.media_dir, args.manifest,
        max_total_bytes=args.max_total_bytes,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
