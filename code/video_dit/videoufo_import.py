"""Metadata-first, provenance-locked selection from the VideoUFO dataset."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import shutil
import tarfile
import uuid
from collections import Counter


VIDEOUFO_REPO = "WenhaoWang/VideoUFO"
VIDEOUFO_REVISION = "683425832ad5b35d57d213963ecec18d5b689652"
VIDEOUFO_LICENSE = "CC BY 4.0"
VIDEOUFO_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"

_FACTORS = {
    "subject": {
        "cat": ("cat", "cats", "kitten", "kittens", "feline"),
        "dog": ("dog", "dogs", "puppy", "puppies", "canine"),
        "person": (
            "person", "people", "man", "men", "woman", "women", "boy",
            "boys", "girl", "girls", "child", "children", "human",
        ),
    },
    "action": {
        "running": ("run", "runs", "running", "jog", "jogs", "jogging"),
        "walking": ("walk", "walks", "walking", "stroll", "strolls", "strolling"),
        "jumping": ("jump", "jumps", "jumping", "leap", "leaps", "leaping"),
    },
    "scene": {
        "snow": ("snow", "snowy", "snowfield", "winter landscape"),
        "beach": ("beach", "beaches", "seashore", "shoreline", "sandy shore"),
        "grass": ("grass", "grassy", "lawn", "meadow"),
        "forest": ("forest", "forests", "woodland", "woods"),
    },
}

_QUALITY_FIELDS = (
    "Aesthetic Quality", "Background Consistency", "Dynamic Degree",
    "Imaging Quality", "Motion Smoothness", "Subject Consistency",
)
_REQUIRED_FIELDS = {
    "ID", "Topic", "Brief Caption", "Detailed Caption", "Start Time", "End Time",
    *_QUALITY_FIELDS,
}

_PATTERNS = {
    kind: {
        label: re.compile(
            r"(?<![a-z])(?:" + "|".join(re.escape(phrase) for phrase in phrases) + r")(?![a-z])"
        )
        for label, phrases in labels.items()
    }
    for kind, labels in _FACTORS.items()
}


@dataclass(frozen=True)
class VideoUFOCandidate:
    video_id: str
    source_id: str
    topic: str
    brief_caption: str
    detailed_caption: str
    start_seconds: float
    end_seconds: float
    subject: str
    action: str
    scene: str
    prompt: str
    quality: dict[str, float]
    quality_score: float
    repository: str = VIDEOUFO_REPO
    revision: str = VIDEOUFO_REVISION
    license: str = VIDEOUFO_LICENSE
    license_url: str = VIDEOUFO_LICENSE_URL
    archive_shard: str | None = None
    archive_member: str | None = None
    review_status: str = "pending"
    review_notes: str = ""

    def as_json(self) -> dict[str, object]:
        return asdict(self)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _seconds(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) != 3:
        raise ValueError("timestamp must use HH:MM:SS")
    hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
    if hours < 0 or minutes not in range(60) or not 0 <= seconds < 60:
        raise ValueError("timestamp is out of range")
    return hours * 3600 + minutes * 60 + seconds


def _factor(text: str, kind: str) -> str | None:
    matches = {
        label for label, pattern in _PATTERNS[kind].items() if pattern.search(text)
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _parse_candidate(row: dict[str, str]) -> tuple[VideoUFOCandidate | None, str | None]:
    try:
        start, end = _seconds(row["Start Time"]), _seconds(row["End Time"])
    except (KeyError, TypeError, ValueError):
        return None, "invalid_time"
    if end - start < 2:
        return None, "invalid_time"
    text = f"{row.get('Brief Caption', '')} {row.get('Detailed Caption', '')}".lower()
    factors = tuple(_factor(text, kind) for kind in ("subject", "action", "scene"))
    if any(value is None for value in factors):
        return None, "ambiguous_factors"
    try:
        quality = {name: float(row[name]) for name in _QUALITY_FIELDS}
    except (KeyError, TypeError, ValueError):
        return None, "quality"
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in quality.values()):
        return None, "quality"
    if (
        quality["Dynamic Degree"] < 0.15
        or quality["Motion Smoothness"] < 0.5
        or quality["Subject Consistency"] < 0.5
        or quality["Imaging Quality"] < 0.3
        or quality["Aesthetic Quality"] < 0.3
    ):
        return None, "quality"
    subject, action, scene = factors
    identifier = row.get("ID", "").strip()
    if not identifier:
        return None, "missing_id"
    caption = row.get("Brief Caption", "").strip()
    score = sum(quality.values()) / len(quality)
    return VideoUFOCandidate(
        video_id=identifier,
        source_id=f"videoufo:{re.sub(r'\.\d+$', '', identifier)}",
        topic=row.get("Topic", "").strip(), brief_caption=caption,
        detailed_caption=row.get("Detailed Caption", "").strip(),
        start_seconds=start, end_seconds=end, subject=subject, action=action,
        scene=scene, prompt=caption, quality=quality, quality_score=score,
    ), None


def scan_metadata(
    path: Path,
    *,
    expected_sha256: str | None,
    max_per_triple: int = 500,
) -> tuple[list[VideoUFOCandidate], dict[str, object]]:
    """Scan the CSV once and retain the best bounded set for each exact triple."""

    if max_per_triple < 1:
        raise ValueError("max_per_triple must be positive")
    source_sha256 = sha256_file(path)
    if expected_sha256 is not None and source_sha256 != expected_sha256.lower():
        raise ValueError("VideoUFO metadata SHA-256 does not match the locked source")
    heaps: dict[str, list[tuple[float, str, VideoUFOCandidate]]] = {}
    rejected: Counter[str] = Counter()
    rows_scanned = 0
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if not _REQUIRED_FIELDS <= fields:
            missing = sorted(_REQUIRED_FIELDS - fields)
            raise ValueError(f"VideoUFO CSV schema is missing columns: {missing}")
        for row in reader:
            rows_scanned += 1
            candidate, reason = _parse_candidate(row)
            if candidate is None:
                rejected[reason or "invalid"] += 1
                continue
            triple = "|".join((candidate.subject, candidate.action, candidate.scene))
            heap = heaps.setdefault(triple, [])
            item = (candidate.quality_score, candidate.video_id, candidate)
            if len(heap) < max_per_triple:
                heapq.heappush(heap, item)
            elif item[:2] > heap[0][:2]:
                heapq.heapreplace(heap, item)
    candidates = sorted(
        (item[2] for heap in heaps.values() for item in heap),
        key=lambda item: (item.subject, item.action, item.scene, -item.quality_score, item.video_id),
    )
    triple_counts = Counter(
        "|".join((item.subject, item.action, item.scene)) for item in candidates
    )
    report: dict[str, object] = {
        "dataset": VIDEOUFO_REPO,
        "revision": VIDEOUFO_REVISION,
        "source_path": str(path),
        "source_sha256": source_sha256,
        "rows_scanned": rows_scanned,
        "candidates_retained": len(candidates),
        "max_per_triple": max_per_triple,
        "triple_counts": dict(sorted(triple_counts.items())),
        "rejections": dict(sorted(rejected.items())),
        "archive_mapping_status": "unresolved",
        "training_ready": False,
        "training_blocker": "candidate IDs have not yet been mapped to verified tar shards and human-reviewed",
    }
    return candidates, report


def write_scan(
    candidates_path: Path,
    report_path: Path,
    candidates: list[VideoUFOCandidate],
    report: dict[str, object],
) -> None:
    candidates_payload = "".join(
        json.dumps(row.as_json(), sort_keys=True) + "\n" for row in candidates
    )
    report = dict(report)
    report.update({
        "candidates_count": len(candidates),
        "candidates_sha256": hashlib.sha256(candidates_payload.encode()).hexdigest(),
    })
    for path, payload in ((candidates_path, candidates_payload), (
        report_path, json.dumps(report, indent=2, sort_keys=True) + "\n",
    )):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)


def index_tar_shard(path: Path) -> dict[str, str]:
    """Return VideoUFO ID to safe MP4 member names without extracting data."""

    index: dict[str, str] = {}
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            if not member.isfile() or Path(member.name).suffix.lower() != ".mp4":
                continue
            video_id = Path(member.name).stem
            if not video_id or video_id in index:
                raise ValueError(f"duplicate video ID in archive: {video_id!r}")
            index[video_id] = member.name
    return index


def resolve_candidates(
    candidates: list[VideoUFOCandidate],
    report: dict[str, object],
    archives: list[Path],
    *,
    expected_sha256: dict[str, str],
) -> tuple[list[VideoUFOCandidate], dict[str, object]]:
    """Bind candidate IDs to immutable local tar shards; never extract members."""

    from dataclasses import replace

    locations: dict[str, tuple[str, str]] = {}
    archive_records: list[dict[str, object]] = []
    for archive in archives:
        digest = sha256_file(archive)
        expected = expected_sha256.get(archive.name)
        if expected is None or digest != expected.lower():
            raise ValueError(f"VideoUFO archive SHA-256 mismatch: {archive.name}")
        shard_index = index_tar_shard(archive)
        archive_records.append({
            "path": str(archive), "name": archive.name, "sha256": digest,
            "mp4_members": len(shard_index),
        })
        for video_id, member in shard_index.items():
            if video_id in locations:
                raise ValueError(f"video ID occurs in multiple archives: {video_id}")
            locations[video_id] = (archive.name, member)
    resolved = [
        replace(item, archive_shard=locations[item.video_id][0],
                archive_member=locations[item.video_id][1])
        for item in candidates if item.video_id in locations
    ]
    updated = dict(report)
    updated.update({
        "archive_shards": archive_records,
        "archive_mapping_status": "complete" if len(resolved) == len(candidates) else "partial",
        "candidates_mapped": len(resolved),
        "candidates_unmapped": len(candidates) - len(resolved),
    })
    return resolved, updated


def extract_resolved_clip(
    candidate: VideoUFOCandidate,
    archive_path: Path,
    output_dir: Path,
    *,
    expected_archive_sha256: str,
) -> Path:
    """Extract exactly one locked regular MP4 member to an ID-derived filename."""

    if sha256_file(archive_path) != expected_archive_sha256.lower():
        raise ValueError("VideoUFO archive SHA-256 does not match the locked shard")
    if candidate.archive_shard != archive_path.name or not candidate.archive_member:
        raise ValueError("candidate does not belong to the supplied archive")
    expected_member = candidate.archive_member
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = hashlib.sha256(candidate.video_id.encode()).hexdigest()[:24] + ".mp4"
    destination = output_dir / safe_name
    temporary = output_dir / f".{safe_name}.{uuid.uuid4().hex}.tmp"
    try:
        with tarfile.open(archive_path, "r:") as archive:
            member = archive.getmember(expected_member)
            if not member.isfile() or Path(member.name).suffix.lower() != ".mp4":
                raise ValueError("resolved archive member is not a regular MP4")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("resolved archive member has no data")
            with source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
