"""Prepare a bounded, review-only VideoUFO metadata candidate set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from videoufo_import import (
    VideoUFOCandidate, extract_review_set, resolve_candidates, scan_metadata,
    write_scan,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan")
    scan.add_argument("metadata", type=Path)
    scan.add_argument("candidates", type=Path)
    scan.add_argument("report", type=Path)
    scan.add_argument("--expected-sha256", required=True)
    scan.add_argument("--max-per-triple", type=int, default=500)
    resolve = commands.add_parser("resolve")
    resolve.add_argument("candidates", type=Path)
    resolve.add_argument("report", type=Path)
    resolve.add_argument("resolved_candidates", type=Path)
    resolve.add_argument("resolved_report", type=Path)
    resolve.add_argument("--shard-checksums", type=Path, required=True)
    resolve.add_argument("archives", type=Path, nargs="+")
    extract = commands.add_parser("extract")
    extract.add_argument("candidates", type=Path)
    extract.add_argument("output", type=Path)
    extract.add_argument("media_dir", type=Path)
    extract.add_argument("--shard-checksums", type=Path, required=True)
    extract.add_argument("archives", type=Path, nargs="+")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "extract":
        candidates = [
            VideoUFOCandidate(**json.loads(line))
            for line in args.candidates.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        checksums = json.loads(args.shard_checksums.read_text(encoding="utf-8"))
        if not isinstance(checksums, dict):
            raise ValueError("shard checksum manifest must be a JSON object")
        rows = extract_review_set(
            candidates, args.archives, args.media_dir,
            expected_sha256={str(key): str(value) for key, value in checksums.items()},
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for row in rows:
            row["review_media_path"] = os.path.relpath(
                Path(str(row["review_media_path"])).resolve(),
                args.output.parent.resolve(),
            )
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        temporary.replace(args.output)
        return 0
    if args.command == "resolve":
        candidates = [
            VideoUFOCandidate(**json.loads(line))
            for line in args.candidates.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        report = json.loads(args.report.read_text(encoding="utf-8"))
        candidates_digest = hashlib.sha256(args.candidates.read_bytes()).hexdigest()
        contract_mismatch = (
            report.get("candidates_sha256") != candidates_digest
            or report.get("candidates_count") != len(candidates)
            or report.get("dataset") != "WenhaoWang/VideoUFO"
            or bool(candidates and report.get("revision") != candidates[0].revision)
        )
        if contract_mismatch:
            raise ValueError("candidate file does not match its locked scan report")
        checksums = json.loads(args.shard_checksums.read_text(encoding="utf-8"))
        if not isinstance(checksums, dict):
            raise ValueError("shard checksum manifest must be a JSON object")
        resolved, updated = resolve_candidates(
            candidates, report, args.archives,
            expected_sha256={str(key): str(value) for key, value in checksums.items()},
        )
        write_scan(args.resolved_candidates, args.resolved_report, resolved, updated)
        return 0
    candidates, report = scan_metadata(
        args.metadata,
        expected_sha256=args.expected_sha256,
        max_per_triple=args.max_per_triple,
    )
    write_scan(args.candidates, args.report, candidates, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
