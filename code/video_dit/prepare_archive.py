"""Create a review-only Internet Archive candidate set from exact item IDs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.parse
import urllib.request

from archive_import import parse_archive_item, write_archive_candidates


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def fetch_archive_metadata(identifier: str) -> dict[str, object]:
    quoted = urllib.parse.quote(identifier, safe="")
    request = urllib.request.Request(
        f"https://archive.org/metadata/{quoted}",
        headers={"User-Agent": "ai-research-notes-video-dit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError("Internet Archive metadata response must be an object")
    return payload


def _load_specs(path: Path) -> tuple[list[dict[str, str]], str]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid Archive spec JSON: {error}") from error
    if not isinstance(value, list) or not value:
        raise ValueError("Archive specs must be a non-empty JSON list")
    specs: list[dict[str, str]] = []
    identifiers: set[str] = set()
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError(f"Archive spec {index} must be an object")
        if set(row) != {"identifier", "subject", "query"}:
            raise ValueError(f"Archive spec {index} requires only identifier, subject, query")
        identifier = str(row["identifier"]).strip()
        subject = str(row["subject"]).strip()
        query = str(row["query"]).strip()
        if not _IDENTIFIER.fullmatch(identifier):
            raise ValueError(f"invalid Internet Archive identifier: {identifier!r}")
        if identifier in identifiers:
            raise ValueError("Internet Archive identifiers must be unique")
        if subject not in {"cat", "dog"}:
            raise ValueError("Archive subject must be cat or dog")
        if not query:
            raise ValueError("Archive query must be non-empty")
        identifiers.add(identifier)
        specs.append({"identifier": identifier, "subject": subject, "query": query})
    return specs, hashlib.sha256(raw).hexdigest()


def _rejection_category(error: ValueError) -> str:
    message = str(error).lower()
    if "license" in message:
        return "license"
    if "creator" in message or "email" in message:
        return "creator"
    if "mediatype" in message or "movies" in message:
        return "mediatype"
    if "original" in message or "digest" in message or "mtime" in message or "size" in message:
        return "original_file"
    return "invalid_metadata"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specs", type=Path, help="Exact item IDs and intended query labels")
    parser.add_argument("output", type=Path, help="Pending candidate JSONL")
    parser.add_argument("report", type=Path, help="Content-bound scan report JSON")
    args = parser.parse_args(argv)
    resolved = [args.specs.resolve(), args.output.resolve(), args.report.resolve()]
    if len(set(resolved)) != len(resolved):
        raise ValueError("spec, candidate, and report paths must differ")

    specs, spec_sha256 = _load_specs(args.specs)
    candidates = []
    rejections: Counter[str] = Counter()
    rejected_items: list[dict[str, str]] = []
    for spec in specs:
        payload = fetch_archive_metadata(spec["identifier"])
        try:
            candidate = parse_archive_item(
                payload, subject=spec["subject"], query=spec["query"],
            )
        except ValueError as error:
            category = _rejection_category(error)
            rejections[category] += 1
            rejected_items.append({"identifier": spec["identifier"], "reason": category})
        else:
            if candidate.identifier != spec["identifier"]:
                raise ValueError(
                    f"metadata identifier mismatch: requested {spec['identifier']!r}, "
                    f"received {candidate.identifier!r}"
                )
            candidates.append(candidate)

    write_archive_candidates(
        candidates, args.output, args.report,
        report_fields={
            "specs": len(specs),
            "spec_sha256": spec_sha256,
            "rejections": dict(sorted(rejections.items())),
            "rejected_items": sorted(rejected_items, key=lambda row: row["identifier"]),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
