"""Search, review, and normalize license-auditable Wikimedia Commons videos."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import urllib.parse

from commons_import import (
    CommonsCandidate, candidate_sample_id, normalize_clip, search_candidates,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search")
    search.add_argument("specs", type=Path)
    search.add_argument("output", type=Path)
    search.add_argument("--state", type=Path)
    download = commands.add_parser("download")
    download.add_argument("candidates", type=Path)
    download.add_argument("clips", type=Path)
    download.add_argument("manifest", type=Path)
    download.add_argument("--frames", type=int, default=16)
    download.add_argument("--fps", type=int, default=8)
    download.add_argument("--size", type=int, default=96)
    return parser.parse_args(argv)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "search":
        specs = json.loads(args.specs.read_text(encoding="utf-8"))
        if not isinstance(specs, list):
            raise ValueError("search specs must be a JSON list")
        rows = _read_jsonl(args.output) if args.output.is_file() else []
        seen = {str(row["title"]) for row in rows}
        state_path = args.state or args.output.with_suffix(args.output.suffix + ".state.json")
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            completed_queries = set(state.get("completed_queries", []))
        else:
            completed_queries = {str(row["query"]) for row in rows if row.get("query")}
        for spec in specs:
            required = ("query", "subject", "action", "scene", "prompt")
            if not isinstance(spec, dict) or not all(spec.get(key) for key in required):
                raise ValueError(f"each search spec requires {required}")
            if str(spec["query"]) in completed_queries:
                continue
            candidates = search_candidates(
                str(spec["query"]),
                subject=str(spec["subject"]), action=str(spec["action"]),
                scene=str(spec["scene"]), prompt=str(spec["prompt"]),
                limit=int(spec.get("limit", 100)),
            )
            for candidate in candidates:
                if candidate.title in seen:
                    continue
                seen.add(candidate.title)
                rows.append({
                    **asdict(candidate), "query": spec["query"],
                    "review_status": "pending", "review_notes": "",
                })
            _write_jsonl(args.output, rows)
            completed_queries.add(str(spec["query"]))
            temporary = state_path.with_suffix(state_path.suffix + ".tmp")
            temporary.write_text(json.dumps({
                "completed_queries": sorted(completed_queries),
            }, indent=2) + "\n", encoding="utf-8")
            temporary.replace(state_path)
        return 0

    approved = _read_jsonl(args.manifest) if args.manifest.is_file() else []
    completed_ids = {str(record["sample_id"]) for record in approved}
    for row in _read_jsonl(args.candidates):
        if row.pop("review_status", None) != "approved":
            continue
        row.pop("review_notes", None)
        row.pop("query", None)
        if "review_required" in (row.get("action"), row.get("scene")):
            raise ValueError("approved candidate still contains placeholder labels")
        media = urllib.parse.urlparse(str(row.get("media_url", "")))
        if media.scheme != "https" or media.hostname != "upload.wikimedia.org":
            raise ValueError("approved candidate requires an HTTPS Wikimedia upload URL")
        description = urllib.parse.urlparse(str(row.get("description_url", "")))
        if description.scheme != "https" or description.hostname != "commons.wikimedia.org":
            raise ValueError("approved candidate requires a Commons description URL")
        candidate = CommonsCandidate(**row)
        expected_sample_id = candidate_sample_id(candidate.title)
        if expected_sample_id in completed_ids:
            continue
        record = normalize_clip(
            candidate, args.clips, frames=args.frames, fps=args.fps, size=args.size,
        )
        approved.append(asdict(record))
        completed_ids.add(expected_sample_id)
        _write_jsonl(args.manifest, approved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
