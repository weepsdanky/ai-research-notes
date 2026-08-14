"""Freeze a blinded high-score versus random-control video review pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile


VISUAL_METRIC = "OpenAI CLIP ViT-B/32 single-poster retrieval margin"
OFFICIAL_VIT_B32_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"


def _snapshot(path: Path) -> tuple[bytes, str]:
    data = path.read_bytes()
    return data, hashlib.sha256(data).hexdigest()


def _jsonl(data: bytes) -> list[dict[str, object]]:
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def _unique(rows: list[dict[str, object]], label: str) -> dict[str, dict[str, object]]:
    result = {}
    for row in rows:
        title = row.get("title")
        if not isinstance(title, str) or not title or title in result:
            raise ValueError(f"{label} titles must be non-empty and unique")
        result[title] = row
    return result


def _assignment_commitment(rows: list[dict[str, object]]) -> str:
    canonical = (json.dumps(rows, separators=(",", ":"), sort_keys=True) + "\n").encode()
    return hashlib.sha256(canonical).hexdigest()


def create_blind_pool(
    candidate_bytes: bytes, provenance_bytes: bytes, report_bytes: bytes, *,
    subject: str, high_count: int, control_count: int, seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    if high_count < 1 or control_count < 1 or not subject:
        raise ValueError("subject and positive arm sizes are required")
    candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
    provenance_sha = hashlib.sha256(provenance_bytes).hexdigest()
    report_sha = hashlib.sha256(report_bytes).hexdigest()
    candidates = _unique(_jsonl(candidate_bytes), "candidate")
    provenance = _unique(_jsonl(provenance_bytes), "provenance")
    report = json.loads(report_bytes)
    if (
        not isinstance(report, dict)
        or report.get("metric") != VISUAL_METRIC
        or report.get("candidate_sha256") != candidate_sha
        or report.get("provenance_sha256") != provenance_sha
        or report.get("training_ready") is not False
    ):
        raise ValueError("visual report does not bind the input snapshots")
    checkpoint_sha = report.get("checkpoint_sha256")
    if checkpoint_sha != OFFICIAL_VIT_B32_SHA256:
        raise ValueError("visual report requires the official locked ViT-B/32 checkpoint")
    for field in ("positive_prompts", "negative_prompts"):
        prompts = report.get(field)
        if not isinstance(prompts, list) or not prompts or any(
            not isinstance(prompt, str) or not prompt.strip() for prompt in prompts
        ):
            raise ValueError("visual report requires frozen non-empty prompts")
    records = report.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("visual report records are required")
    expected_titles = {
        title for title, candidate in candidates.items()
        if candidate.get("review_status") == "pending"
        and candidate.get("subject") == subject and title in provenance
        and isinstance(provenance[title].get("review_preview_url"), str)
        and isinstance(provenance[title].get("duration_seconds"), (int, float))
        and not isinstance(provenance[title].get("duration_seconds"), bool)
        and math.isfinite(float(provenance[title]["duration_seconds"]))
        and float(provenance[title]["duration_seconds"]) > 0
    }
    eligible = []
    seen_titles: set[str] = set()
    for rank, record in enumerate(records, 1):
        if not isinstance(record, dict) or record.get("rank") != rank:
            raise ValueError("visual report ranks must be contiguous")
        title = record.get("title")
        margin = record.get("margin")
        if (
            not isinstance(title, str) or title not in candidates or title not in provenance
            or title in seen_titles
            or not isinstance(margin, (int, float)) or isinstance(margin, bool)
            or not math.isfinite(float(margin))
        ):
            raise ValueError("visual report records must be finite and input-bound")
        seen_titles.add(title)
        candidate, source = candidates[title], provenance[title]
        if candidate.get("review_status") != "pending" or candidate.get("subject") != subject:
            raise ValueError("visual report must contain only the requested pending subject")
        for field in ("original_media_sha1", "original_media_timestamp", "poster_sha256"):
            expected = source.get(field) if field != "poster_sha256" else record.get(field)
            if record.get(field) != expected or not isinstance(expected, str) or not expected:
                raise ValueError(f"visual report lacks locked {field}")
        poster_sha = record["poster_sha256"]
        if len(poster_sha) != 64 or any(char not in "0123456789abcdef" for char in poster_sha):
            raise ValueError("visual report poster SHA-256 must be lowercase hexadecimal")
        eligible.append(record)
    if seen_titles != expected_titles:
        raise ValueError("visual report must cover the complete playable pending subject pool")
    if high_count + control_count > len(eligible):
        raise ValueError("blind pool arms exceed the eligible report")
    high = eligible[:high_count]
    remainder = eligible[high_count:]
    rng = random.Random(seed)
    control = rng.sample(remainder, control_count)
    selected = [("high", row) for row in high] + [("random_control", row) for row in control]
    rng.shuffle(selected)
    key_rows = []
    queue_rows = []
    for index, (arm, row) in enumerate(selected, 1):
        title = str(row["title"])
        blind_id = f"blind-{index:03d}"
        source = provenance[title]
        queue_rows.append({
            "blind_id": blind_id, "title": title,
            "description_url": source.get("description_url"),
            "duration_seconds": source.get("duration_seconds"),
            "review_preview_url": source.get("review_preview_url"),
            "original_media_sha1": source.get("original_media_sha1"),
            "original_media_timestamp": source.get("original_media_timestamp"),
        })
        key_rows.append({
            "blind_id": blind_id, "title": title, "arm": arm,
            "source_rank": row["rank"], "source_margin": row["margin"],
        })
    assignment_commitment = _assignment_commitment(key_rows)
    queue = {
        "protocol": "blinded full-video review queue",
        "candidate_sha256": candidate_sha, "provenance_sha256": provenance_sha,
        "visual_report_sha256": report_sha, "subject": subject,
        "assignment_commitment_sha256": assignment_commitment,
        "records": queue_rows, "training_ready": False,
    }
    queue_bytes = (json.dumps(queue, indent=2, sort_keys=True) + "\n").encode()
    key = {
        "protocol": "sealed high-score versus random-control assignment",
        "queue_sha256": hashlib.sha256(queue_bytes).hexdigest(),
        "candidate_sha256": candidate_sha, "provenance_sha256": provenance_sha,
        "visual_report_sha256": report_sha, "subject": subject,
        "assignment_commitment_sha256": assignment_commitment,
        "seed": seed, "high_count": high_count, "control_count": control_count,
        "records": key_rows, "training_ready": False,
    }
    return queue, key


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total < 1 or successes < 0 or successes > total:
        raise ValueError("Wilson interval counts are invalid")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def reveal_results(
    queue_bytes: bytes, key_bytes: bytes, candidate_bytes: bytes,
    audit_bytes: bytes, media_manifest_bytes: bytes, *,
    target_triple: tuple[str, str, str],
) -> dict[str, object]:
    queue, key = json.loads(queue_bytes), json.loads(key_bytes)
    audit, media_manifest = json.loads(audit_bytes), json.loads(media_manifest_bytes)
    candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
    queue_sha = hashlib.sha256(queue_bytes).hexdigest()
    if (
        key.get("queue_sha256") != queue_sha
        or queue.get("candidate_sha256") != key.get("candidate_sha256")
        or queue.get("visual_report_sha256") != key.get("visual_report_sha256")
        or queue.get("subject") != key.get("subject")
        or target_triple[0] != queue.get("subject")
        or queue.get("assignment_commitment_sha256")
        != _assignment_commitment(key.get("records", []))
        or key.get("assignment_commitment_sha256")
        != queue.get("assignment_commitment_sha256")
    ):
        raise ValueError("blind key does not bind the review queue")
    decisions = _unique(_jsonl(candidate_bytes), "decision candidate")
    queue_records, key_records = queue.get("records"), key.get("records")
    if not isinstance(queue_records, list) or not isinstance(key_records, list):
        raise ValueError("blind pool records are required")
    queue_ids = [row.get("blind_id") for row in queue_records if isinstance(row, dict)]
    assignments = {row.get("blind_id"): row for row in key_records if isinstance(row, dict)}
    if len(assignments) != len(key_records) or len(queue_records) != len(key_records):
        raise ValueError("blind assignments must be complete and unique")
    if len(queue_ids) != len(queue_records) or len(set(queue_ids)) != len(queue_ids):
        raise ValueError("review queue blind IDs must be complete and unique")
    if set(queue_ids) != set(assignments):
        raise ValueError("review queue and sealed key IDs differ")
    if key.get("high_count") != sum(row.get("arm") == "high" for row in key_records):
        raise ValueError("sealed high arm count differs from its contract")
    if key.get("control_count") != sum(
        row.get("arm") == "random_control" for row in key_records
    ):
        raise ValueError("sealed control arm count differs from its contract")
    audit_records = audit.get("records")
    media_records = media_manifest.get("records")
    if (
        audit.get("candidate_sha256") != candidate_sha
        or not isinstance(audit_records, list)
        or media_manifest.get("queue_sha256") != queue_sha
        or media_manifest.get("training_ready") is not False
        or not isinstance(media_records, list)
    ):
        raise ValueError("review audit and media manifest must bind the frozen pool")
    audit_by_title = _unique(audit_records, "review audit")
    media_by_id = {
        row.get("blind_id"): row for row in media_records if isinstance(row, dict)
    }
    if (
        len(media_by_id) != len(media_records)
        or set(audit_by_title) != {str(row.get("title")) for row in queue_records}
        or set(media_by_id) != set(queue_ids)
    ):
        raise ValueError("review audit and media manifest must cover the complete pool")
    revealed = []
    for row in queue_records:
        if not isinstance(row, dict) or row.get("blind_id") not in assignments:
            raise ValueError("queue blind ID is absent from the sealed key")
        assignment = assignments[row["blind_id"]]
        if assignment.get("title") != row.get("title"):
            raise ValueError("sealed title differs from review queue")
        decision = decisions.get(str(row["title"]))
        if decision is None or decision.get("review_status") not in {"approved", "rejected"}:
            raise ValueError(f"blind review is incomplete: {row.get('blind_id')}")
        audited = audit_by_title[str(row["title"])]
        media = media_by_id[row["blind_id"]]
        review_fields = (
            "review_status", "subject", "action", "scene", "prompt", "start_seconds",
            "duplicate_group", "review_notes",
        )
        if (
            any(audited.get(field) != decision.get(field) for field in review_fields)
            or audited.get("original_media_sha1") != row.get("original_media_sha1")
            or audited.get("original_media_timestamp") != row.get("original_media_timestamp")
            or media.get("title") != row.get("title")
            or not isinstance(media.get("derivative_path"), str)
            or not str(media["derivative_path"]).strip()
            or not isinstance(media.get("bytes"), int)
            or isinstance(media.get("bytes"), bool)
            or int(media["bytes"]) < 1
            or not isinstance(media.get("derivative_sha256"), str)
            or len(str(media["derivative_sha256"])) != 64
            or any(char not in "0123456789abcdef" for char in str(media["derivative_sha256"]))
        ):
            raise ValueError("review decision does not bind the locked queue media")
        is_target = (
            decision.get("review_status") == "approved"
            and tuple(decision.get(field) for field in ("subject", "action", "scene"))
            == target_triple
        )
        revealed.append({
            **assignment, "review_status": decision["review_status"],
            "is_target_triple": is_target,
        })
    arms = {}
    for arm in ("high", "random_control"):
        rows = [row for row in revealed if row["arm"] == arm]
        successes = sum(bool(row["is_target_triple"]) for row in rows)
        arms[arm] = {
            "records": len(rows), "target_hits": successes,
            "yield": successes / len(rows),
            "wilson_95": wilson_interval(successes, len(rows)),
        }
    return {
        "protocol": "revealed blinded candidate-ranking evaluation",
        "queue_sha256": queue_sha,
        "key_sha256": hashlib.sha256(key_bytes).hexdigest(),
        "decision_candidate_sha256": candidate_sha,
        "review_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "media_manifest_sha256": hashlib.sha256(media_manifest_bytes).hexdigest(),
        "target_triple": "|".join(target_triple), "arms": arms,
        "yield_difference": arms["high"]["yield"] - arms["random_control"]["yield"],
        "records": revealed, "training_ready": False,
        "limitation": "partial-pool yield/precision estimate; not full-pool recall",
    }


def _staged_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp",
        dir=path.parent, delete=False,
    ) as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        return Path(handle.name)


def _write_json(path: Path, value: object) -> None:
    temporary = _staged_json(path, value)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_pair(first_path: Path, first: object, second_path: Path, second: object) -> None:
    staged_first = _staged_json(first_path, first)
    try:
        staged_second = _staged_json(second_path, second)
    except Exception:
        staged_first.unlink(missing_ok=True)
        raise
    old_first = first_path.read_bytes() if first_path.exists() else None
    old_second = second_path.read_bytes() if second_path.exists() else None
    try:
        staged_first.replace(first_path)
        staged_second.replace(second_path)
    except Exception:
        if old_first is None:
            first_path.unlink(missing_ok=True)
        else:
            first_path.write_bytes(old_first)
        if old_second is None:
            second_path.unlink(missing_ok=True)
        else:
            second_path.write_bytes(old_second)
        raise
    finally:
        staged_first.unlink(missing_ok=True)
        staged_second.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("candidates", type=Path); create.add_argument("provenance", type=Path)
    create.add_argument("visual_report", type=Path); create.add_argument("queue", type=Path)
    create.add_argument("key", type=Path); create.add_argument("--subject", required=True)
    create.add_argument("--high-count", type=int, required=True)
    create.add_argument("--control-count", type=int, required=True)
    create.add_argument("--seed", type=int, required=True)
    reveal = sub.add_parser("reveal")
    reveal.add_argument("queue", type=Path); reveal.add_argument("key", type=Path)
    reveal.add_argument("candidates", type=Path); reveal.add_argument("audit", type=Path)
    reveal.add_argument("media_manifest", type=Path); reveal.add_argument("output", type=Path)
    reveal.add_argument("--target-triple", required=True)
    args = parser.parse_args(argv)
    inputs = (
        (args.candidates, args.provenance, args.visual_report)
        if args.command == "create" else (
            args.queue, args.key, args.candidates, args.audit, args.media_manifest,
        )
    )
    outputs = (args.queue, args.key) if args.command == "create" else (args.output,)
    resolved_inputs = [path.resolve() for path in inputs]
    resolved_outputs = [path.resolve() for path in outputs]
    if len(set(resolved_outputs)) != len(resolved_outputs) or any(
        output in resolved_inputs for output in resolved_outputs
    ):
        raise ValueError("blind pool outputs must differ from all other paths")
    if args.command == "create":
        queue, key = create_blind_pool(
            args.candidates.read_bytes(), args.provenance.read_bytes(),
            args.visual_report.read_bytes(), subject=args.subject,
            high_count=args.high_count, control_count=args.control_count, seed=args.seed,
        )
        _write_json_pair(args.queue, queue, args.key, key)
        return 0
    triple = tuple(args.target_triple.split("|"))
    if len(triple) != 3:
        raise ValueError("target triple must be subject|action|scene")
    report = reveal_results(
        args.queue.read_bytes(), args.key.read_bytes(), args.candidates.read_bytes(),
        args.audit.read_bytes(), args.media_manifest.read_bytes(),
        target_triple=triple,
    )
    _write_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
