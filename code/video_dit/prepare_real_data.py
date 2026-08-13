"""Prepare and validate leakage-safe real-video experiment manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from real_data import (
    CompositionalSplit,
    load_manifest,
    split_manifest,
    validate_split,
    write_manifest,
)


def _triple(value: str) -> tuple[str, str, str]:
    parts = tuple(part.strip() for part in value.split(","))
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("C1 must be subject,action,scene")
    return parts


def _pair(value: str) -> tuple[str, str]:
    parts = tuple(part.strip() for part in value.split(","))
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError("C2 must contain two comma-separated factors")
    return parts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    split = subparsers.add_parser("split")
    split.add_argument("source", type=Path)
    split.add_argument("output", type=Path)
    split.add_argument("--c1", type=_triple, action="append", default=[])
    split.add_argument("--c2", type=_pair, action="append", default=[])
    split.add_argument("--iid-fraction", type=float, default=0.15)
    split.add_argument("--validation-fraction", type=float, default=0.10)
    split.add_argument("--seed", type=int, default=0)
    split.add_argument("--min-c1-source-groups", type=int, default=50)
    split.add_argument("--min-c2-source-groups", type=int, default=1)
    split.add_argument("--min-iid-source-groups", type=int, default=1)
    split.add_argument("--min-train-pair-source-groups", type=int, default=1)
    split.add_argument("--report", type=Path)
    validate = subparsers.add_parser("validate")
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--c1", type=_triple, action="append", default=[])
    validate.add_argument("--c2", type=_pair, action="append", default=[])
    validate.add_argument("--min-c1-source-groups", type=int, default=50)
    validate.add_argument("--min-c2-source-groups", type=int, default=1)
    validate.add_argument("--min-iid-source-groups", type=int, default=1)
    validate.add_argument("--min-train-pair-source-groups", type=int, default=1)
    validate.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def _policy(args: argparse.Namespace) -> CompositionalSplit:
    return CompositionalSplit(
        c1_triples=set(args.c1),
        c2_pairs=set(args.c2),
        iid_fraction=getattr(args, "iid_fraction", 0),
        validation_fraction=getattr(args, "validation_fraction", 0),
        seed=getattr(args, "seed", 0),
        min_c1_source_groups=args.min_c1_source_groups,
        min_c2_source_groups=args.min_c2_source_groups,
        min_iid_source_groups=args.min_iid_source_groups,
        min_train_pair_source_groups=args.min_train_pair_source_groups,
    )


def _write_report(path: Path | None, report: dict[str, object]) -> None:
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(text, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    policy = _policy(args)
    if args.command == "split":
        records = split_manifest(load_manifest(args.source), policy)
        report = validate_split(records, policy)
        write_manifest(args.output, records)
    else:
        report = validate_split(load_manifest(args.manifest), policy)
    _write_report(args.report, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
