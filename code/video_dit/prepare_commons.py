"""Search, review, and normalize license-auditable Wikimedia Commons videos."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import urllib.parse

from commons_import import (
    CommonsCandidate, candidate_sample_id, category_candidates, normalize_clip,
    search_candidates,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search")
    search.add_argument("specs", type=Path)
    search.add_argument("output", type=Path)
    search.add_argument("--state", type=Path)
    category = commands.add_parser("category")
    category.add_argument("category")
    category.add_argument("subject")
    category.add_argument("output", type=Path)
    category.add_argument("--max-depth", type=int, default=1)
    category.add_argument("--limit", type=int, default=500)
    category.add_argument("--state", type=Path)
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


def _write_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "category":
        rows = _read_jsonl(args.output) if args.output.is_file() else []
        seen = {str(row["title"]) for row in rows}
        state_key = hashlib.sha256(json.dumps({
            "category": args.category, "subject": args.subject,
            "max_depth": args.max_depth, "limit": args.limit,
        }, sort_keys=True).encode()).hexdigest()[:12]
        state_path = args.state or args.output.with_suffix(
            args.output.suffix + f".category-{state_key}.state.json"
        )
        saved = json.loads(state_path.read_text()) if state_path.is_file() else {}
        contract = {
            "category": args.category, "subject": args.subject,
            "max_depth": args.max_depth, "limit": args.limit,
        }
        if saved and saved.get("contract") != contract:
            raise ValueError("category state contract does not match this traversal")
        if saved.get("completed"):
            return 0
        traversal = saved.get("traversal")

        def persist_batch(
            candidates: list[CommonsCandidate], next_state: dict[str, object],
        ) -> None:
            for candidate in candidates:
                if candidate.title in seen:
                    continue
                seen.add(candidate.title)
                rows.append({
                    **asdict(candidate), "query": f"category:{args.category}",
                    "category_traversal_id": state_key,
                    "review_status": "pending", "review_notes": "",
                })
            _write_jsonl(args.output, rows)
            _write_state(state_path, {
                "completed": False, "contract": contract, "traversal": next_state,
            })

        durable_count = sum(
            1 for row in rows if row.get("category_traversal_id") == state_key
        )
        while durable_count < args.limit:
            returned = category_candidates(
                args.category, subject=args.subject, max_depth=args.max_depth,
                limit=args.limit - durable_count, state=traversal,
                on_batch=persist_batch,
            )
            if returned and any(candidate.title not in seen for candidate in returned):
                persist_batch(returned, {})
            durable_count = sum(
                1 for row in rows if row.get("category_traversal_id") == state_key
            )
            progress = json.loads(state_path.read_text())["traversal"]
            traversal = progress
            if not any((progress.get("pending_titles"), progress.get("current"),
                        progress.get("queue"))):
                break
        _write_state(state_path, {
            "completed": True, "contract": contract,
            "traversal": traversal or {},
        })
        return 0
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
            active_queries = dict(state.get("active_queries", {}))
        else:
            completed_queries = {str(row["query"]) for row in rows if row.get("query")}
            active_queries: dict[str, object] = {}
        for spec in specs:
            required = ("query", "subject", "action", "scene", "prompt")
            if not isinstance(spec, dict) or not all(spec.get(key) for key in required):
                raise ValueError(f"each search spec requires {required}")
            if str(spec["query"]) in completed_queries:
                continue
            query = str(spec["query"])
            active = active_queries.get(query, {})
            if not isinstance(active, dict):
                raise ValueError("search state active query must be a mapping")
            # The candidates file is replaced before the state file. If the
            # process dies between those replacements, the durable rows are the
            # authoritative lower bound for progress.
            durable_count = sum(1 for row in rows if str(row.get("query", "")) == query)
            accepted = max(int(active.get("accepted", 0)), durable_count)
            requested = int(spec.get("limit", 100))
            if accepted >= requested:
                completed_queries.add(query)
                active_queries.pop(query, None)
                _write_state(state_path, {
                    "completed_queries": sorted(completed_queries),
                    "active_queries": active_queries,
                })
                continue

            # Write-ahead marker: a first-page crash must not make rows already
            # written to the candidate file look like a completed query.
            initial_continuation = dict(active.get("continuation", {}))
            active_queries[query] = {
                "accepted": accepted, "continuation": initial_continuation,
            }
            _write_state(state_path, {
                "completed_queries": sorted(completed_queries),
                "active_queries": active_queries,
            })

            def persist_page(
                page_candidates: list[CommonsCandidate],
                continuation: dict[str, object] | None,
            ) -> None:
                nonlocal accepted
                for candidate in page_candidates:
                    if candidate.title in seen:
                        continue
                    seen.add(candidate.title)
                    rows.append({
                        **asdict(candidate), "query": query,
                        "review_status": "pending", "review_notes": "",
                    })
                    accepted += 1
                _write_jsonl(args.output, rows)
                active_queries[query] = {
                    "accepted": accepted, "continuation": continuation or {},
                }
                _write_state(state_path, {
                    "completed_queries": sorted(completed_queries),
                    "active_queries": active_queries,
                })

            continuation = initial_continuation
            while accepted < requested:
                returned = search_candidates(
                    query,
                    subject=str(spec["subject"]), action=str(spec["action"]),
                    scene=str(spec["scene"]), prompt=str(spec["prompt"]),
                    limit=requested - accepted,
                    continuation=continuation,
                    on_page=persist_page,
                )
                # Keep compatibility with simple/custom search providers that return
                # one materialized page instead of invoking the streaming callback.
                if returned and any(candidate.title not in seen for candidate in returned):
                    persist_page(returned, None)
                progress = active_queries.get(query, {})
                if not isinstance(progress, dict):
                    raise ValueError("search state active query must be a mapping")
                continuation = dict(progress.get("continuation", {}))
                if not continuation:
                    break
            completed_queries.add(query)
            active_queries.pop(query, None)
            _write_state(state_path, {
                "completed_queries": sorted(completed_queries),
                "active_queries": active_queries,
            })
        return 0

    approved = _read_jsonl(args.manifest) if args.manifest.is_file() else []
    completed_ids = {str(record["sample_id"]) for record in approved}
    for row in _read_jsonl(args.candidates):
        if row.pop("review_status", None) != "approved":
            continue
        row.pop("review_notes", None)
        row.pop("query", None)
        preview_url = row.pop("review_preview_url", None)
        row.pop("duration_seconds", None)
        row.pop("review_media_error", None)
        if preview_url:
            row["media_url"] = preview_url
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
        if hasattr(record, "clip_path"):
            clip_path = Path(record.clip_path)
            try:
                relative_clip = clip_path.resolve().relative_to(
                    args.manifest.parent.resolve()
                )
            except ValueError:
                raise ValueError(
                    "normalized clips must be inside the manifest directory"
                )
            else:
                record = replace(record, clip_path=str(relative_clip))
        approved.append(asdict(record))
        completed_ids.add(expected_sample_id)
        _write_jsonl(args.manifest, approved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
