"""Preflight and verify license-gated Ego4D ``video_540ss`` downloads."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Callable, Mapping
from urllib.parse import urlparse


HeadObject = Callable[[str], Mapping[str, object]]
GetObject = Callable[[str, str], Mapping[str, object]]
ProbeVideo = Callable[[Path], Mapping[str, object]]
OFFICIAL_MANIFEST_PATH = (
    "s3://ego4d-consortium-sharing/public/v2_1/video_540ss/manifest.csv"
)
OFFICIAL_MEDIA_BUCKETS = {
    "ego4d-bristol", "ego4d-cmu", "ego4d-consortium-sharing",
    "ego4d-georgiatech", "ego4d-iiith", "ego4d-indiana", "ego4d-kaust",
    "ego4d-minnesota",
    "ego4d-speac", "ego4d-sfu", "ego4d-unc", "ego4d-unict-milan",
    "ego4d-utokyo", "ego4d-university-sa", "ego4d-universityaf",
    "ego4d-penn", "ego4d-fair",
}
VIDEO_UID_PATTERN = re.compile(r"\A[a-z0-9][a-z0-9_-]{0,127}\Z")
EXPECTED_METADATA_VERSION = "v2.1"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_object(data: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _json_lines(data: bytes, label: str) -> list[dict[str, object]]:
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8 JSONL") from error
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid {label} line {line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {line_number} must be an object")
        rows.append(row)
    return rows


def _csv_rows(data: bytes, label: str) -> list[dict[str, str]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8 CSV") from error
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError(f"{label} requires a header")
    return [dict(row) for row in reader]


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _positive_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _lower_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _utc_timestamp(value: object, label: str) -> datetime:
    text = _nonempty_string(value, label)
    if not text.endswith("Z") or "T" not in text:
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(text.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp") from error
    return parsed


def _selected_scan_contract(
    candidate_bytes: bytes, uid_file_bytes: bytes, scan_report_bytes: bytes,
) -> tuple[list[str], dict[str, object]]:
    report = _json_object(scan_report_bytes, "Ego4D scan report")
    if (
        report.get("dataset") != "ego4d"
        or report.get("dataset_release") != "v2_1"
        or report.get("dataset_version") != EXPECTED_METADATA_VERSION
        or report.get("training_ready") is not False
    ):
        raise ValueError(
            "Ego4D scan report must bind v2_1 metadata version v2.1 and remain training_ready=false"
        )
    if report.get("candidate_sha256") != _sha256(candidate_bytes):
        raise ValueError("Ego4D candidate SHA-256 does not match the scan report")
    if report.get("uid_file_sha256") != _sha256(uid_file_bytes):
        raise ValueError("Ego4D UID file SHA-256 does not match the scan report")
    for field in (
        "metadata_sha256", "narrations_sha256", "license_sha256",
        "license_receipt_sha256", "candidate_sha256", "uid_file_sha256",
    ):
        _lower_sha256(report.get(field), f"scan report {field}")
    _utc_timestamp(
        report.get("license_accepted_at"), "scan report license_accepted_at",
    )
    try:
        uid_lines = uid_file_bytes.decode("utf-8").split()
    except UnicodeDecodeError as error:
        raise ValueError("Ego4D UID file must be UTF-8") from error
    if not uid_lines or len(uid_lines) != len(set(uid_lines)):
        raise ValueError("Ego4D UID file must contain unique non-empty UIDs")
    if any(VIDEO_UID_PATTERN.fullmatch(uid) is None for uid in uid_lines):
        raise ValueError("Ego4D UID file contains an unsafe video_uid")
    selected = report.get("selected_video_uids")
    if not isinstance(selected, list) or selected != uid_lines:
        raise ValueError("Ego4D selected UID list does not match the UID file")
    if len(selected) > 10:
        raise ValueError("Ego4D download plan is limited to 10 UIDs")
    candidate_uids = {
        _nonempty_string(row.get("video_uid"), "candidate video_uid")
        for row in _json_lines(candidate_bytes, "Ego4D candidates")
    }
    if not set(uid_lines).issubset(candidate_uids):
        raise ValueError("Ego4D UID file contains a UID absent from the candidates")
    contract = report.get("download_contract")
    if not isinstance(contract, dict) or (
        contract.get("dataset") != "video_540ss"
        or contract.get("on_budget_exceeded") != "stop_without_fallback"
    ):
        raise ValueError("Ego4D scan report has an invalid download contract")
    maximum = _positive_integer(contract.get("max_bytes"), "download max_bytes")
    if maximum > 12_000_000_000:
        raise ValueError("Ego4D download max_bytes may not exceed 12,000,000,000")
    _nonempty_string(report.get("dataset_version"), "Ego4D dataset_version")
    return uid_lines, report


def build_download_plan(
    candidate_bytes: bytes,
    uid_file_bytes: bytes,
    scan_report_bytes: bytes,
    manifest_bytes: bytes,
    manifest_identity: Mapping[str, object],
    head_object: HeadObject,
) -> dict[str, object]:
    """HEAD selected S3 objects and freeze a hard-budget download plan."""

    selected, scan = _selected_scan_contract(
        candidate_bytes, uid_file_bytes, scan_report_bytes,
    )
    if manifest_identity.get("s3_path") != OFFICIAL_MANIFEST_PATH:
        raise ValueError("Ego4D dataset manifest must come from the fixed official path")
    manifest_version = _nonempty_string(
        manifest_identity.get("version_id"), "Ego4D manifest S3 version",
    )
    manifest_size = _positive_integer(
        manifest_identity.get("content_length"), "Ego4D manifest content length",
    )
    if manifest_size != len(manifest_bytes):
        raise ValueError("Ego4D manifest byte size does not match its S3 snapshot")
    manifest_etag = _nonempty_string(
        manifest_identity.get("etag"), "Ego4D manifest S3 ETag",
    )
    rows = _csv_rows(manifest_bytes, "Ego4D dataset manifest")
    by_uid: dict[str, dict[str, str]] = {}
    selected_set = set(selected)
    for row in rows:
        uid = row.get("video_uid") or row.get("file_uid") or ""
        if uid not in selected_set:
            continue
        if uid in by_uid:
            raise ValueError(f"duplicate selected UID in Ego4D dataset manifest: {uid}")
        by_uid[uid] = row
    missing = selected_set - set(by_uid)
    if missing:
        raise ValueError(f"selected UIDs absent from Ego4D dataset manifest: {sorted(missing)}")

    objects: list[dict[str, object]] = []
    for uid in selected:
        row = by_uid[uid]
        media_type = row.get("type", "")
        if media_type and media_type not in {"mp4", "video"}:
            raise ValueError(f"selected Ego4D UID is not a video: {uid}")
        s3_path = row.get("canonical_s3_location") or row.get("s3_path")
        if not isinstance(s3_path, str):
            raise ValueError(f"selected Ego4D UID lacks an S3 media path: {uid}")
        parsed = urlparse(s3_path)
        if (
            parsed.scheme != "s3" or parsed.netloc not in OFFICIAL_MEDIA_BUCKETS
            or not parsed.path.lstrip("/")
        ):
            raise ValueError(f"selected Ego4D UID uses a non-official S3 path: {uid}")
        head = head_object(s3_path)
        version = _nonempty_string(head.get("version_id"), f"S3 version for {uid}")
        length = _positive_integer(head.get("content_length"), f"S3 size for {uid}")
        etag = _nonempty_string(head.get("etag"), f"S3 ETag for {uid}")
        objects.append({
            "video_uid": uid,
            "filename": f"{uid}.mp4",
            "s3_path": s3_path,
            "version_id": version,
            "content_length": length,
            "etag": etag,
        })
    total = sum(int(row["content_length"]) for row in objects)
    maximum = int(scan["download_contract"]["max_bytes"])  # type: ignore[index]
    if total > maximum:
        raise ValueError(
            f"Ego4D selection exceeds the {maximum:,}-byte download budget"
        )
    return {
        "dataset": "ego4d",
        "dataset_version": scan["dataset_version"],
        "dataset_name": "video_540ss",
        "reference_ego4d_cli_revision": "4bd10ed40b4f8d8ad26344afc2c8526f7d1dedeb",
        "candidate_sha256": _sha256(candidate_bytes),
        "uid_file_sha256": _sha256(uid_file_bytes),
        "scan_report_sha256": _sha256(scan_report_bytes),
        "dataset_manifest_sha256": _sha256(manifest_bytes),
        "dataset_manifest_s3_path": OFFICIAL_MANIFEST_PATH,
        "dataset_manifest_s3_version_id": manifest_version,
        "dataset_manifest_content_length": manifest_size,
        "dataset_manifest_s3_etag": manifest_etag,
        "selected_video_uids": selected,
        "remote_objects": objects,
        "total_content_length": total,
        "max_bytes": maximum,
        "on_budget_exceeded": "stop_without_fallback",
        "training_ready": False,
    }


def _plan_objects(
    plan: dict[str, object],
) -> tuple[list[str], dict[str, dict[str, object]], int]:
    selected = plan.get("selected_video_uids")
    if (
        plan.get("dataset") != "ego4d"
        or plan.get("dataset_version") != EXPECTED_METADATA_VERSION
        or plan.get("dataset_name") != "video_540ss"
        or plan.get("reference_ego4d_cli_revision")
        != "4bd10ed40b4f8d8ad26344afc2c8526f7d1dedeb"
        or plan.get("dataset_manifest_s3_path") != OFFICIAL_MANIFEST_PATH
        or plan.get("on_budget_exceeded") != "stop_without_fallback"
        or plan.get("training_ready") is not False
        or not isinstance(selected, list)
        or not selected
        or len(selected) > 10
        or len(selected) != len(set(selected))
        or any(not isinstance(uid, str) or not uid for uid in selected)
    ):
        raise ValueError("Ego4D download plan has an invalid fixed contract")
    if any(VIDEO_UID_PATTERN.fullmatch(uid) is None for uid in selected):
        raise ValueError("Ego4D download plan contains an unsafe video_uid")
    _lower_sha256(plan.get("dataset_manifest_sha256"), "dataset manifest SHA-256")
    _nonempty_string(
        plan.get("dataset_manifest_s3_version_id"), "dataset manifest S3 version",
    )
    _nonempty_string(plan.get("dataset_manifest_s3_etag"), "dataset manifest S3 ETag")
    _positive_integer(
        plan.get("dataset_manifest_content_length"), "dataset manifest content length",
    )
    maximum = _positive_integer(plan.get("max_bytes"), "download plan max_bytes")
    if maximum > 12_000_000_000:
        raise ValueError("Ego4D download plan exceeds the hard 12 GB budget")
    objects_value = plan.get("remote_objects")
    if not isinstance(objects_value, list) or len(objects_value) != len(selected):
        raise ValueError("Ego4D download plan remote objects do not match selected UIDs")
    objects: dict[str, dict[str, object]] = {}
    for value in objects_value:
        if not isinstance(value, dict):
            raise ValueError("Ego4D remote object entries must be objects")
        uid = _nonempty_string(value.get("video_uid"), "remote object video_uid")
        if uid in objects or uid not in selected:
            raise ValueError(f"invalid duplicate/unselected Ego4D remote object: {uid}")
        if value.get("filename") != f"{uid}.mp4":
            raise ValueError(f"unexpected planned filename for Ego4D UID: {uid}")
        s3_path = _nonempty_string(value.get("s3_path"), f"S3 path for {uid}")
        parsed = urlparse(s3_path)
        if (
            parsed.scheme != "s3" or parsed.netloc not in OFFICIAL_MEDIA_BUCKETS
            or not parsed.path.lstrip("/")
        ):
            raise ValueError(f"invalid planned S3 path for Ego4D UID: {uid}")
        _nonempty_string(value.get("version_id"), f"S3 version for {uid}")
        _nonempty_string(value.get("etag"), f"S3 ETag for {uid}")
        _positive_integer(value.get("content_length"), f"planned content length for {uid}")
        objects[uid] = value
    if set(objects) != set(selected):
        raise ValueError("Ego4D remote object UID set does not match the frozen selection")
    total = sum(int(objects[uid]["content_length"]) for uid in selected)
    if plan.get("total_content_length") != total or total > maximum:
        raise ValueError("Ego4D download plan total violates the frozen byte budget")
    return selected, objects, maximum


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False,
    ) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def download_exact_media(
    plan_bytes: bytes,
    media_dir: Path,
    get_object: GetObject,
    *,
    sdk_version: str,
) -> dict[str, object]:
    """GET the exact preflighted S3 versions without an unversioned fallback."""

    plan = _json_object(plan_bytes, "Ego4D download plan")
    selected, objects, maximum = _plan_objects(plan)
    sdk_version = _nonempty_string(sdk_version, "S3 SDK version")
    if media_dir.is_symlink():
        raise ValueError("Ego4D media directory may not be a symlink")
    media_dir.mkdir(parents=True, exist_ok=True)
    resolved_media_dir = media_dir.resolve()
    expected_names = {f"{uid}.mp4" for uid in selected}
    extras = {
        path.name for path in media_dir.iterdir()
        if path.name.lower().endswith(".mp4") and path.name not in expected_names
    }
    if extras:
        raise ValueError(f"unexpected MP4 files in Ego4D media directory: {sorted(extras)}")
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    downloaded: list[dict[str, object]] = []
    cumulative = 0
    for uid in selected:
        expected = objects[uid]
        response = get_object(str(expected["s3_path"]), str(expected["version_id"]))
        if response.get("version_id") != expected["version_id"]:
            raise ValueError(f"version-specific S3 GET returned a different version: {uid}")
        if response.get("etag") != expected["etag"]:
            raise ValueError(f"version-specific S3 GET returned a different ETag: {uid}")
        if response.get("content_length") != expected["content_length"]:
            raise ValueError(f"version-specific S3 GET returned a different size: {uid}")
        body = response.get("body")
        if not hasattr(body, "read"):
            raise ValueError(f"version-specific S3 GET lacks a streaming body: {uid}")
        destination = media_dir / str(expected["filename"])
        if destination.parent.resolve() != resolved_media_dir:
            raise ValueError(f"Ego4D destination must be directly inside media_dir: {uid}")
        try:
            with tempfile.NamedTemporaryFile(
                dir=media_dir, prefix=f".{destination.name}.", suffix=".partial",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                written = 0
                digest = hashlib.sha256()
                try:
                    while True:
                        chunk = body.read(1024 * 1024)  # type: ignore[union-attr]
                        if not chunk:
                            break
                        if not isinstance(chunk, bytes):
                            raise ValueError(f"S3 body returned non-byte data: {uid}")
                        if (
                            written + len(chunk) > int(expected["content_length"])
                            or cumulative + written + len(chunk) > maximum
                        ):
                            raise ValueError(f"Ego4D S3 stream exceeds the frozen byte budget: {uid}")
                        handle.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                    handle.flush()
                    if written != expected["content_length"]:
                        raise ValueError(f"Ego4D S3 stream ended at the wrong size: {uid}")
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
        cumulative += written
        downloaded.append({
            "video_uid": uid,
            "filename": destination.name,
            "version_id": expected["version_id"],
            "content_length": written,
            "sha256": digest.hexdigest(),
        })
    if cumulative != plan.get("total_content_length"):
        raise ValueError("Ego4D downloaded bytes do not match the frozen plan")
    version_buffer = io.StringIO()
    writer = csv.DictWriter(version_buffer, fieldnames=("uid", "version", "filename"))
    writer.writeheader()
    writer.writerows(
        {"uid": row["video_uid"], "version": row["version_id"], "filename": row["filename"]}
        for row in downloaded
    )
    version_bytes = version_buffer.getvalue().encode()
    _atomic_bytes(media_dir / "manifest.ver", version_bytes)
    finished_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "dataset": "ego4d",
        "dataset_name": "video_540ss",
        "downloader": "version-specific-s3-get-v1",
        "sdk": "boto3",
        "sdk_version": sdk_version,
        "download_plan_sha256": _sha256(plan_bytes),
        "version_manifest_sha256": _sha256(version_bytes),
        "selected_video_uids": selected,
        "downloaded": downloaded,
        "total_downloaded_bytes": cumulative,
        "started_at": started_at,
        "finished_at": finished_at,
        "training_ready": False,
    }


def _default_probe(path: Path) -> Mapping[str, object]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames",
            "-show_entries",
            (
                "stream=codec_type,codec_name,width,height,avg_frame_rate,nb_read_frames:"
                "format=format_name,duration"
            ),
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    value = json.loads(result.stdout)
    streams = value.get("streams", [])
    if not isinstance(streams, list):
        raise ValueError(f"ffprobe returned invalid streams for: {path.name}")
    video_streams = [row for row in streams if row.get("codec_type") == "video"]
    if len(video_streams) != 1:
        raise ValueError(f"Ego4D media must contain exactly one video stream: {path.name}")
    stream = video_streams[0]
    try:
        numerator, denominator = str(stream.get("avg_frame_rate", "0/1")).split("/", 1)
        fps = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"ffprobe returned invalid frame rate for: {path.name}") from error
    return {
        "container": value.get("format", {}).get("format_name"),
        "video_codec": stream.get("codec_name"),
        "audio_codecs": [
            row.get("codec_name") for row in streams if row.get("codec_type") == "audio"
        ],
        "frame_count": int(stream.get("nb_read_frames", 0)),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "fps": fps,
        "duration_seconds": float(value.get("format", {}).get("duration", 0)),
    }


def _validated_probe(value: Mapping[str, object], uid: str) -> dict[str, object]:
    container = _nonempty_string(value.get("container"), f"container for {uid}")
    codec = _nonempty_string(value.get("video_codec"), f"video codec for {uid}")
    audio_value = value.get("audio_codecs")
    if not isinstance(audio_value, list) or any(
        not isinstance(item, str) or not item.strip() for item in audio_value
    ):
        raise ValueError(f"audio_codecs for {uid} must be a list of non-empty strings")
    frame_count = _positive_integer(value.get("frame_count"), f"frame count for {uid}")
    width = _positive_integer(value.get("width"), f"video width for {uid}")
    height = _positive_integer(value.get("height"), f"video height for {uid}")
    output: dict[str, object] = {
        "container": container,
        "video_codec": codec,
        "audio_codecs": [str(item).strip() for item in audio_value],
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }
    for key in ("duration_seconds", "fps"):
        number = value.get(key)
        if (
            not isinstance(number, (int, float)) or isinstance(number, bool)
            or not math.isfinite(float(number)) or float(number) <= 0
        ):
            raise ValueError(f"{key} for {uid} must be positive and finite")
        output[key] = float(number)
    return output


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_downloaded_media(
    candidate_bytes: bytes,
    uid_file_bytes: bytes,
    scan_report_bytes: bytes,
    metadata_bytes: bytes,
    plan_bytes: bytes,
    download_receipt_bytes: bytes,
    version_manifest_bytes: bytes,
    media_dir: Path,
    *,
    probe: ProbeVideo = _default_probe,
) -> dict[str, object]:
    """Verify exact-version downloader output against every frozen contract."""

    selected, scan = _selected_scan_contract(
        candidate_bytes, uid_file_bytes, scan_report_bytes,
    )
    if _sha256(metadata_bytes) != scan.get("metadata_sha256"):
        raise ValueError("Ego4D metadata SHA-256 does not match the scan report")
    metadata = _json_object(metadata_bytes, "Ego4D metadata")
    videos_value = metadata.get("videos")
    if not isinstance(videos_value, list):
        raise ValueError("Ego4D metadata requires a videos array")
    durations: dict[str, float] = {}
    for row in videos_value:
        if not isinstance(row, dict):
            raise ValueError("Ego4D metadata video rows must be objects")
        uid = _nonempty_string(row.get("video_uid"), "metadata video_uid")
        duration = row.get("duration_sec")
        if (
            not isinstance(duration, (int, float)) or isinstance(duration, bool)
            or not math.isfinite(float(duration)) or float(duration) <= 0
        ):
            raise ValueError(f"invalid Ego4D metadata duration: {uid}")
        if uid in durations:
            raise ValueError(f"duplicate Ego4D metadata video_uid: {uid}")
        durations[uid] = float(duration)
    if not set(selected).issubset(durations):
        raise ValueError("Ego4D metadata is missing selected video UIDs")
    plan = _json_object(plan_bytes, "Ego4D download plan")
    expected_bindings = {
        "candidate_sha256": _sha256(candidate_bytes),
        "uid_file_sha256": _sha256(uid_file_bytes),
        "scan_report_sha256": _sha256(scan_report_bytes),
    }
    if any(plan.get(field) != digest for field, digest in expected_bindings.items()):
        raise ValueError("Ego4D download plan does not bind the frozen scan inputs")
    plan_selected, objects, maximum = _plan_objects(plan)
    if (
        plan.get("dataset_version") != scan.get("dataset_version")
        or plan_selected != selected
    ):
        raise ValueError("Ego4D download plan has an invalid dataset/UID contract")
    if maximum != scan["download_contract"]["max_bytes"]:  # type: ignore[index]
        raise ValueError("Ego4D download plan does not match the scan byte budget")
    manifest_digest = plan.get("dataset_manifest_sha256")
    _lower_sha256(manifest_digest, "download plan dataset_manifest_sha256")

    receipt = _json_object(download_receipt_bytes, "Ego4D download receipt")
    if (
        receipt.get("dataset") != "ego4d"
        or receipt.get("dataset_name") != "video_540ss"
        or receipt.get("downloader") != "version-specific-s3-get-v1"
        or receipt.get("sdk") != "boto3"
        or receipt.get("download_plan_sha256") != _sha256(plan_bytes)
        or receipt.get("selected_video_uids") != selected
        or receipt.get("total_downloaded_bytes") != plan.get("total_content_length")
        or receipt.get("training_ready") is not False
    ):
        raise ValueError("Ego4D download receipt does not bind the exact-version download")
    _nonempty_string(receipt.get("sdk_version"), "download receipt SDK version")
    started_at = _utc_timestamp(receipt.get("started_at"), "download receipt started_at")
    finished_at = _utc_timestamp(receipt.get("finished_at"), "download receipt finished_at")
    if finished_at < started_at:
        raise ValueError("Ego4D download receipt finishes before it starts")

    if receipt.get("version_manifest_sha256") != _sha256(version_manifest_bytes):
        raise ValueError("Ego4D manifest.ver does not match the download receipt")
    downloaded_value = receipt.get("downloaded")
    if not isinstance(downloaded_value, list) or len(downloaded_value) != len(selected):
        raise ValueError("Ego4D download receipt has an invalid downloaded UID set")
    receipt_downloads: dict[str, dict[str, object]] = {}
    for row in downloaded_value:
        if not isinstance(row, dict):
            raise ValueError("Ego4D downloaded receipt entries must be objects")
        uid = _nonempty_string(row.get("video_uid"), "download receipt video_uid")
        if uid in receipt_downloads or uid not in selected:
            raise ValueError("Ego4D download receipt has an invalid downloaded UID set")
        receipt_downloads[uid] = row
    if set(receipt_downloads) != set(selected):
        raise ValueError("Ego4D download receipt has an invalid downloaded UID set")
    version_rows = _csv_rows(version_manifest_bytes, "Ego4D manifest.ver")
    versions: dict[str, dict[str, str]] = {}
    for row in version_rows:
        uid = _nonempty_string(row.get("uid"), "manifest.ver UID")
        if uid not in objects:
            raise ValueError(f"unexpected UID in Ego4D manifest.ver: {uid}")
        if uid in versions:
            raise ValueError(f"duplicate selected UID in Ego4D manifest.ver: {uid}")
        versions[uid] = row
    if set(versions) != set(selected):
        raise ValueError("Ego4D manifest.ver is missing selected UIDs")
    if not media_dir.is_dir():
        raise ValueError("Ego4D media directory does not exist")
    expected_filenames = {f"{uid}.mp4" for uid in selected}
    actual_filenames = {
        path.name for path in media_dir.iterdir() if path.name.lower().endswith(".mp4")
    }
    unexpected = actual_filenames - expected_filenames
    if unexpected:
        raise ValueError(f"unexpected MP4 files in Ego4D media directory: {sorted(unexpected)}")
    missing_files = expected_filenames - actual_filenames
    if missing_files:
        raise ValueError(f"missing MP4 files in Ego4D media directory: {sorted(missing_files)}")

    media: list[dict[str, object]] = []
    total = 0
    for uid in selected:
        remote = objects[uid]
        expected_filename = f"{uid}.mp4"
        version = versions[uid]
        if version.get("filename") != expected_filename:
            raise ValueError(f"unexpected manifest.ver filename for Ego4D UID: {uid}")
        if version.get("version") != remote.get("version_id"):
            raise ValueError(f"S3 version mismatch for Ego4D UID: {uid}")
        receipt_row = receipt_downloads[uid]
        if (
            receipt_row.get("filename") != expected_filename
            or receipt_row.get("version_id") != remote.get("version_id")
            or receipt_row.get("content_length") != remote.get("content_length")
        ):
            raise ValueError(f"download receipt object mismatch for Ego4D UID: {uid}")
        receipt_digest = _lower_sha256(
            receipt_row.get("sha256"), f"download receipt SHA-256 for {uid}",
        )
        path = media_dir / expected_filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Ego4D media must be a regular non-symlink file: {uid}")
        before = path.stat()
        expected_size = _positive_integer(
            remote.get("content_length"), f"planned content length for {uid}",
        )
        if before.st_size != expected_size:
            raise ValueError(f"downloaded Ego4D media size mismatch for UID: {uid}")
        digest = _file_sha256(path)
        if digest != receipt_digest:
            raise ValueError(f"downloaded media does not match download receipt SHA: {uid}")
        video = _validated_probe(probe(path), uid)
        metadata_duration = durations[uid]
        duration_delta = abs(float(video["duration_seconds"]) - metadata_duration)
        duration_tolerance = max(1.0, metadata_duration * 0.01)
        if duration_delta > duration_tolerance:
            raise ValueError(f"downloaded Ego4D media disagrees with metadata duration: {uid}")
        after = path.stat()
        if (
            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
        ) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise ValueError(f"Ego4D media changed during verification: {uid}")
        total += before.st_size
        media.append({
            "video_uid": uid,
            "filename": expected_filename,
            "sha256": digest,
            "content_length": before.st_size,
            "s3_version_id": remote["version_id"],
            "s3_etag": remote["etag"],
            "metadata_duration_seconds": metadata_duration,
            "duration_delta_seconds": duration_delta,
            "duration_tolerance_seconds": duration_tolerance,
            **video,
        })
    if total > maximum or total != plan.get("total_content_length"):
        raise ValueError("downloaded Ego4D media violates the frozen byte budget")
    return {
        "dataset": "ego4d",
        "dataset_version": plan.get("dataset_version"),
        "dataset_name": "video_540ss",
        **expected_bindings,
        "metadata_sha256": _sha256(metadata_bytes),
        "download_plan_sha256": _sha256(plan_bytes),
        "download_receipt_sha256": _sha256(download_receipt_bytes),
        "dataset_manifest_sha256": manifest_digest,
        "version_manifest_sha256": _sha256(version_manifest_bytes),
        "selected_video_uids": selected,
        "total_local_bytes": total,
        "media": media,
        "training_ready": False,
        "training_blocker": "human clip review and normalized manifest import are required",
    }
