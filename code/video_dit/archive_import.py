"""Fail-closed, metadata-only candidates from Internet Archive items."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import urllib.parse


_VIDEO_SUFFIXES = {".mp4", ".m4v", ".mov", ".webm", ".ogv"}
_ALLOWED_LICENSE_PATHS = (
    re.compile(r"^/licenses/by/(?:1\.0|2\.0|2\.5|3\.0|4\.0)/?$"),
    re.compile(r"^/licenses/by-sa/(?:1\.0|2\.0|2\.5|3\.0|4\.0)/?$"),
    re.compile(r"^/publicdomain/zero/(?:1\.0)/?$"),
    re.compile(r"^/publicdomain/mark/(?:1\.0)/?$"),
    re.compile(r"^/licenses/publicdomain/?$"),
)


@dataclass(frozen=True)
class ArchiveItemCandidate:
    identifier: str
    title: str
    creator: str
    license_url: str
    original_file: str
    original_sha1: str
    original_md5: str
    original_mtime: int
    original_size: int
    source_url: str
    media_url: str
    subject: str
    query: str
    review_status: str = "pending"
    review_notes: str = ""

    def as_json(self) -> dict[str, object]:
        creator_sha256 = hashlib.sha256(self.creator.casefold().encode()).hexdigest()
        return {
            **asdict(self),
            "source_id": f"internet-archive-creator:{creator_sha256}",
            "duplicate_group": f"internet-archive-creator:{creator_sha256}",
            "action": "review_required", "scene": "review_required",
            "prompt": f"review the {self.subject} action and scene",
        }


def _license_allowed(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname != "creativecommons.org":
        return False
    return any(pattern.fullmatch(parsed.path) for pattern in _ALLOWED_LICENSE_PATHS)


def parse_archive_item(
    payload: dict[str, object], *, subject: str, query: str,
) -> ArchiveItemCandidate:
    """Validate one official /metadata response without downloading media."""

    metadata = payload.get("metadata")
    files = payload.get("files")
    if not isinstance(metadata, dict) or not isinstance(files, list):
        raise ValueError("Internet Archive response lacks metadata/files")
    if metadata.get("mediatype") != "movies":
        raise ValueError("Internet Archive item must use mediatype movies")
    identifier_value, title_value = metadata.get("identifier"), metadata.get("title")
    if not isinstance(identifier_value, str) or not isinstance(title_value, str):
        raise ValueError("Internet Archive identifier and title must be strings")
    identifier, title = identifier_value.strip(), title_value.strip()
    creator_value = metadata.get("creator")
    if not isinstance(creator_value, str):
        raise ValueError("Internet Archive creator must be a string")
    creator = creator_value.strip()
    uploader_value = metadata.get("uploader")
    if not isinstance(uploader_value, str):
        raise ValueError("Internet Archive uploader must be a string")
    uploader = uploader_value.strip().casefold()
    if not identifier or not title:
        raise ValueError("Internet Archive item requires identifier and title")
    if not creator:
        raise ValueError("Internet Archive item requires a creator for attribution")
    if "@" in creator:
        raise ValueError("Internet Archive creator must not expose an email address")
    if not uploader:
        raise ValueError("Internet Archive item requires an uploader for source grouping")
    license_value = metadata.get("licenseurl")
    if not isinstance(license_value, str):
        raise ValueError("Internet Archive license must be a string")
    license_url = license_value.strip()
    if not _license_allowed(license_url):
        raise ValueError("Internet Archive item license is absent or incompatible")
    originals = [
        row for row in files
        if isinstance(row, dict) and row.get("source") == "original"
        and Path(str(row.get("name", ""))).suffix.lower() in _VIDEO_SUFFIXES
    ]
    if len(originals) != 1:
        raise ValueError("Internet Archive item requires exactly one original video")
    original = originals[0]
    name = str(original.get("name", ""))
    sha1, md5 = str(original.get("sha1", "")), str(original.get("md5", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", sha1) or not re.fullmatch(r"[0-9a-f]{32}", md5):
        raise ValueError("Internet Archive original requires lowercase SHA-1 and MD5 digest")
    try:
        mtime, size = int(str(original["mtime"])), int(str(original["size"]))
    except (KeyError, TypeError, ValueError):
        raise ValueError("Internet Archive original requires integer mtime and size") from None
    if mtime <= 0 or size <= 0:
        raise ValueError("Internet Archive original mtime and size must be positive")
    quoted_id = urllib.parse.quote(identifier, safe="")
    quoted_name = urllib.parse.quote(name, safe="")
    return ArchiveItemCandidate(
        identifier=identifier, title=title, creator=creator, license_url=license_url,
        original_file=name, original_sha1=sha1, original_md5=md5,
        original_mtime=mtime, original_size=size,
        source_url=f"https://archive.org/details/{quoted_id}",
        media_url=f"https://archive.org/download/{quoted_id}/{quoted_name}",
        subject=subject, query=query,
    )


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def write_archive_candidates(
    candidates: list[ArchiveItemCandidate], output: Path, report: Path,
    *, report_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    """Write pending candidates and a content-bound report."""

    if output.resolve() == report.resolve():
        raise ValueError("candidate and report outputs must differ")
    unique: dict[str, ArchiveItemCandidate] = {}
    for candidate in candidates:
        previous = unique.get(candidate.identifier)
        if previous is not None and previous != candidate:
            raise ValueError(f"conflicting Internet Archive item: {candidate.identifier}")
        unique[candidate.identifier] = candidate
    rows = [unique[key].as_json() for key in sorted(unique)]
    candidate_bytes = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows
    ).encode()
    summary = {
        "records": len(rows),
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "training_ready": False,
        "training_blocker": "metadata candidates require media download verification and human review",
    }
    if report_fields:
        overlap = summary.keys() & report_fields.keys()
        if overlap:
            raise ValueError(f"report fields cannot replace locked fields: {sorted(overlap)}")
        summary.update(report_fields)
    _atomic_write(output, candidate_bytes)
    _atomic_write(report, (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode())
    return summary
