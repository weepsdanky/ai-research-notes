"""Freeze license-bound Ego4D narration hits before any restricted video download."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile


SUBJECT_PATTERN = re.compile(r"\b(cat|kitten|dog|puppy)\b", re.IGNORECASE)
ACTION_PATTERN = re.compile(
    r"\b(play(?:s|ed|ing)?|chas(?:e|es|ed|ing)|toy|fetch(?:es|ed|ing)?|tug(?:s|ged|ging)?)\b",
    re.IGNORECASE,
)
INDOOR_PATTERN = re.compile(
    r"\b(inside|indoor|indoors|room|home|house|apartment|kitchen|living room)\b",
    re.IGNORECASE,
)
OUTDOOR_PATTERN = re.compile(
    r"\b(outside|outdoor|outdoors|garden|park|street|yard|beach|forest|field)\b",
    re.IGNORECASE,
)
UTC_TIMESTAMP_PATTERN = re.compile(
    r"\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\Z"
)


def _json_object(data: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _required_string(row: dict[str, object], field: str, label: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} requires non-empty {field}")
    return value.strip()


def _license_contract(
    license_bytes: bytes, receipt_bytes: bytes,
) -> tuple[str, str, str]:
    receipt = _json_object(receipt_bytes, "license receipt")
    license_sha = hashlib.sha256(license_bytes).hexdigest()
    if receipt.get("dataset") != "ego4d" or receipt.get("license_sha256") != license_sha:
        raise ValueError("license receipt does not bind the supplied Ego4D license")
    if receipt.get("permits_local_noncommercial_training") is not True:
        raise ValueError("license receipt must explicitly permit local noncommercial training")
    accepted_at = _required_string(receipt, "accepted_at", "license receipt")
    if UTC_TIMESTAMP_PATTERN.fullmatch(accepted_at) is None:
        raise ValueError("license receipt accepted_at must be an ISO-8601 UTC timestamp")
    try:
        datetime.fromisoformat(accepted_at.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise ValueError(
            "license receipt accepted_at must be an ISO-8601 UTC timestamp"
        ) from error
    _required_string(receipt, "accepted_by", "license receipt")
    return license_sha, hashlib.sha256(receipt_bytes).hexdigest(), accepted_at


def scan_ego4d_annotations(
    metadata_bytes: bytes,
    narration_bytes: bytes,
    license_bytes: bytes,
    receipt_bytes: bytes,
    *,
    max_video_uids: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Return deterministic, review-required narration hits from frozen snapshots."""

    if not isinstance(max_video_uids, int) or isinstance(max_video_uids, bool) or not 1 <= max_video_uids <= 10:
        raise ValueError("max_video_uids must be an integer from 1 to 10")
    license_sha, receipt_sha, accepted_at = _license_contract(license_bytes, receipt_bytes)
    metadata = _json_object(metadata_bytes, "Ego4D metadata")
    narrations = _json_object(narration_bytes, "Ego4D narrations")
    dataset_version = _required_string(metadata, "version", "Ego4D metadata")
    metadata_date = _required_string(metadata, "date", "Ego4D metadata")
    videos_value = metadata.get("videos")
    if not isinstance(videos_value, list):
        raise ValueError("Ego4D metadata requires a videos array")
    videos: dict[str, dict[str, object]] = {}
    for row in videos_value:
        if not isinstance(row, dict):
            raise ValueError("Ego4D video metadata rows must be objects")
        video_uid = _required_string(row, "video_uid", "Ego4D video")
        if video_uid in videos:
            raise ValueError(f"duplicate Ego4D video_uid: {video_uid}")
        duration = row.get("duration_sec")
        if (
            not isinstance(duration, (int, float)) or isinstance(duration, bool)
            or not math.isfinite(float(duration)) or float(duration) <= 0
        ):
            raise ValueError(f"Ego4D video duration must be positive and finite: {video_uid}")
        _required_string(row, "video_source", "Ego4D video")
        _required_string(row, "origin_video_id", "Ego4D video")
        videos[video_uid] = row

    matches: list[dict[str, object]] = []
    annotation_uids: set[str] = set()
    for video_uid, value in narrations.items():
        if video_uid not in videos:
            raise ValueError(f"Ego4D narration video_uid is absent from metadata: {video_uid}")
        if not isinstance(value, dict):
            raise ValueError(f"Ego4D narration entry must be an object: {video_uid}")
        for pass_name in ("narration_pass_1", "narration_pass_2"):
            pass_value = value.get(pass_name)
            if pass_value is None:
                continue
            if not isinstance(pass_value, dict) or not isinstance(pass_value.get("narrations"), list):
                raise ValueError(f"Ego4D {pass_name} requires a narrations array")
            normalized_narrations: list[tuple[str, float, str]] = []
            for narration in pass_value["narrations"]:
                if not isinstance(narration, dict):
                    raise ValueError("Ego4D narration rows must be objects")
                annotation_uid = _required_string(narration, "annotation_uid", "Ego4D narration")
                if annotation_uid in annotation_uids:
                    raise ValueError(f"duplicate Ego4D annotation_uid: {annotation_uid}")
                annotation_uids.add(annotation_uid)
                text = _required_string(narration, "narration_text", "Ego4D narration")
                timestamp = narration.get("timestamp_sec")
                duration = float(videos[video_uid]["duration_sec"])
                if (
                    not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool)
                    or not math.isfinite(float(timestamp)) or not 0 <= float(timestamp) <= duration
                ):
                    raise ValueError(f"Ego4D narration timestamp is outside its video: {annotation_uid}")
                normalized_narrations.append((annotation_uid, float(timestamp), text))
            normalized_narrations.sort(key=lambda row: (row[1], row[0]))
            for index, (annotation_uid, timestamp, text) in enumerate(normalized_narrations):
                subject_match = SUBJECT_PATTERN.search(text)
                action_match = ACTION_PATTERN.search(text)
                if subject_match is None or action_match is None:
                    continue
                video = videos[video_uid]
                video_source = str(video["video_source"])
                participant = video.get("fb_participant_id")
                if isinstance(participant, int) and not isinstance(participant, bool):
                    duplicate_group = f"ego4d-participant:{video_source}:{participant}"
                else:
                    duplicate_group = (
                        f"ego4d-origin:{video_source}:{video['origin_video_id']}"
                    )
                matches.append({
                    "video_uid": video_uid,
                    "annotation_uid": annotation_uid,
                    "narration_pass": pass_name,
                    "timestamp_sec": timestamp,
                    "narration_text": text,
                    "context_before": (
                        {
                            "annotation_uid": normalized_narrations[index - 1][0],
                            "timestamp_sec": normalized_narrations[index - 1][1],
                            "narration_text": normalized_narrations[index - 1][2],
                        }
                        if index > 0 else None
                    ),
                    "context_after": (
                        {
                            "annotation_uid": normalized_narrations[index + 1][0],
                            "timestamp_sec": normalized_narrations[index + 1][1],
                            "narration_text": normalized_narrations[index + 1][2],
                        }
                        if index + 1 < len(normalized_narrations) else None
                    ),
                    "subject_hint": subject_match.group(1).casefold(),
                    "action_hint": action_match.group(1).casefold(),
                    "indoor_hint": bool(INDOOR_PATTERN.search(text)),
                    "outdoor_hint": bool(OUTDOOR_PATTERN.search(text)),
                    "video_source": video_source,
                    "origin_video_id": str(video["origin_video_id"]),
                    "physical_setting_name": video.get("physical_setting_name"),
                    "scenarios": video.get("scenarios", []),
                    "source_id": f"ego4d:{video_uid}",
                    "duplicate_group": duplicate_group,
                    "review_status": "pending",
                    "subject": "review_required",
                    "action": "review_required",
                    "scene": "review_required",
                    "prompt": "review the animal subject, action, and scene",
                    "license_sha256": license_sha,
                    "license_receipt_sha256": receipt_sha,
                })

    uid_priority: dict[str, tuple[int, int, str]] = {}
    for row in matches:
        video_uid = str(row["video_uid"])
        priority = (
            -int(bool(row["indoor_hint"])), int(bool(row["outdoor_hint"])), video_uid,
        )
        uid_priority[video_uid] = min(priority, uid_priority.get(video_uid, priority))
    ranked_uids = [
        video_uid for video_uid, _ in sorted(uid_priority.items(), key=lambda item: item[1])
    ]
    uid_groups: dict[str, str] = {}
    for row in matches:
        video_uid = str(row["video_uid"])
        duplicate_group = str(row["duplicate_group"])
        previous = uid_groups.setdefault(video_uid, duplicate_group)
        if previous != duplicate_group:
            raise ValueError(f"Ego4D video has inconsistent duplicate groups: {video_uid}")
    selected_uids: list[str] = []
    selected_groups: set[str] = set()
    for video_uid in ranked_uids:
        duplicate_group = uid_groups[video_uid]
        if duplicate_group not in selected_groups:
            selected_uids.append(video_uid)
            selected_groups.add(duplicate_group)
        if len(selected_uids) == max_video_uids:
            break
    if len(selected_uids) < max_video_uids:
        for video_uid in ranked_uids:
            if video_uid not in selected_uids:
                selected_uids.append(video_uid)
            if len(selected_uids) == max_video_uids:
                break
    selected = sorted(
        (row for row in matches if row["video_uid"] in set(selected_uids)),
        key=lambda row: (selected_uids.index(str(row["video_uid"])), float(row["timestamp_sec"]), str(row["annotation_uid"])),
    )
    candidate_bytes = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in selected
    ).encode()
    uid_file_bytes = "".join(f"{video_uid}\n" for video_uid in selected_uids).encode()
    report = {
        "dataset": "ego4d",
        "dataset_version": dataset_version,
        "metadata_date": metadata_date,
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "narrations_sha256": hashlib.sha256(narration_bytes).hexdigest(),
        "license_sha256": license_sha,
        "license_receipt_sha256": receipt_sha,
        "license_accepted_at": accepted_at,
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "matching_narrations": len(matches),
        "matching_video_uids": len(uid_priority),
        "selected_video_uids": selected_uids,
        "uid_file_sha256": hashlib.sha256(uid_file_bytes).hexdigest(),
        "max_video_uids": max_video_uids,
        "query_contract": {
            "subject_terms": ["cat", "kitten", "dog", "puppy"],
            "action_terms": ["play", "chase", "toy", "fetch", "tug"],
            "cooccurrence": "same narration",
            "selection": (
                "indoor hint first, outdoor hint last, distinct duplicate groups first, "
                "then remaining video_uid"
            ),
        },
        "download_contract": {
            "dataset": "video_540ss",
            "max_bytes": 12_000_000_000,
            "on_budget_exceeded": "stop_without_fallback",
        },
        "training_ready": False,
        "training_blocker": "restricted videos require UID download, media hash verification, and human review",
    }
    return selected, report


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("narrations", type=Path)
    parser.add_argument("license", type=Path)
    parser.add_argument("license_receipt", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("uid_file", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--max-video-uids", type=int, default=10)
    args = parser.parse_args(argv)
    inputs = [args.metadata, args.narrations, args.license, args.license_receipt]
    outputs = [args.candidates, args.uid_file, args.report]
    resolved_inputs = {path.resolve() for path in inputs}
    resolved_outputs = [path.resolve() for path in outputs]
    if len(set(resolved_outputs)) != len(outputs) or any(
        path in resolved_inputs for path in resolved_outputs
    ):
        raise ValueError("Ego4D outputs must differ from every input and each other")
    candidates, report = scan_ego4d_annotations(
        args.metadata.read_bytes(), args.narrations.read_bytes(), args.license.read_bytes(),
        args.license_receipt.read_bytes(), max_video_uids=args.max_video_uids,
    )
    candidate_bytes = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in candidates
    ).encode()
    _atomic_write(args.candidates, candidate_bytes)
    _atomic_write(
        args.uid_file,
        "".join(f"{video_uid}\n" for video_uid in report["selected_video_uids"]).encode(),
    )
    _atomic_write(
        args.report, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
