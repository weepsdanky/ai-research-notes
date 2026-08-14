"""Auditable manifests and leakage-safe compositional video splits."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import math
import shutil
import subprocess
import tempfile
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


TRIPLE = tuple[str, str, str]
PAIR = tuple[str, str]
SPLITS = {"train", "validation", "iid", "c1", "c2"}


@dataclass(frozen=True)
class RealVideoRecord:
    """One normalized clip plus the provenance needed to audit its use."""

    sample_id: str
    source_id: str
    clip_path: str
    prompt: str
    subject: str
    action: str
    scene: str
    source_url: str
    license: str
    attribution: str
    sha256: str
    license_url: str | None = None
    original_media_url: str | None = None
    page_revision: str | None = None
    original_media_sha1: str | None = None
    original_media_timestamp: str | None = None
    downloaded_media_url: str | None = None
    downloaded_media_kind: str | None = None
    downloaded_media_sha256: str | None = None
    downloaded_at: str | None = None
    clip_start_seconds: float | None = None
    clip_frames: int | None = None
    clip_fps: int | None = None
    clip_size: int | None = None
    duplicate_group: str | None = None
    perceptual_hash: str | None = None
    text_embedding_path: str | None = None
    text_encoder_id: str | None = None
    text_embedding_kind: str | None = None
    latent_path: str | None = None
    vae_id: str | None = None
    vae_scaling: float | None = None
    split: str | None = None

    def __post_init__(self) -> None:
        required = (
            "sample_id", "source_id", "clip_path", "prompt", "subject",
            "action", "scene", "source_url", "license", "attribution", "sha256",
        )
        for name in required:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in self.sha256
        ):
            raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
        if self.downloaded_media_sha256 is not None and (
            len(self.downloaded_media_sha256) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in self.downloaded_media_sha256)
        ):
            raise ValueError("downloaded_media_sha256 must contain exactly 64 hexadecimal characters")
        if self.clip_start_seconds is not None and (
            not math.isfinite(self.clip_start_seconds) or self.clip_start_seconds < 0
        ):
            raise ValueError("clip_start_seconds must be finite and non-negative")
        for name in ("clip_frames", "clip_fps", "clip_size"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        clip_contract = (
            self.clip_start_seconds, self.clip_frames, self.clip_fps, self.clip_size,
        )
        if any(value is not None for value in clip_contract) and not all(
            value is not None for value in clip_contract
        ):
            raise ValueError("clip contract must be all present or all absent")
        if self.split is not None and self.split not in SPLITS:
            raise ValueError(f"split must be one of {sorted(SPLITS)}")
        if self.perceptual_hash is not None and (
            len(self.perceptual_hash) != 16
            or any(character not in "0123456789abcdefABCDEF" for character in self.perceptual_hash)
        ):
            raise ValueError("perceptual_hash must be 16 hexadecimal characters")
        if bool(self.text_embedding_path) != bool(self.text_encoder_id):
            raise ValueError(
                "text_embedding_path and text_encoder_id must be provided together"
            )
        if self.text_embedding_path is not None:
            kind = self.text_embedding_kind or "pooled"
            if kind not in ("pooled", "tokens"):
                raise ValueError("text_embedding_kind must be pooled or tokens")
        latent_fields = (self.latent_path, self.vae_id, self.vae_scaling)
        if any(value is not None for value in latent_fields) and not all(
            value is not None for value in latent_fields
        ):
            raise ValueError("latent_path, vae_id, and vae_scaling must be provided together")
        if self.vae_scaling is not None and (
            not math.isfinite(self.vae_scaling) or self.vae_scaling <= 0
        ):
            raise ValueError("vae_scaling must be positive and finite")

    @property
    def triple(self) -> TRIPLE:
        return self.subject, self.action, self.scene


@dataclass(frozen=True)
class CompositionalSplit:
    """Policy for ordinary, held-triple (C1), and held-pair (C2) tests."""

    c1_triples: set[TRIPLE]
    c2_pairs: set[PAIR]
    iid_fraction: float = 0.15
    validation_fraction: float = 0.10
    seed: int = 0
    near_duplicate_hamming: int = 4
    min_c1_source_groups: int = 1
    min_c2_source_groups: int = 1
    min_iid_source_groups: int = 1
    min_train_pair_source_groups: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.iid_fraction < 1:
            raise ValueError("iid_fraction must be in [0, 1)")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.iid_fraction + self.validation_fraction >= 1:
            raise ValueError("iid_fraction + validation_fraction must be below 1")
        if self.near_duplicate_hamming < 0:
            raise ValueError("near_duplicate_hamming must be non-negative")
        for name in (
            "min_c1_source_groups", "min_c2_source_groups",
            "min_iid_source_groups", "min_train_pair_source_groups",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")


def write_manifest(path: Path, records: Iterable[RealVideoRecord]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(asdict(record), sort_keys=True) for record in records]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    temporary.replace(path)
    return path


def load_manifest(path: Path) -> list[RealVideoRecord]:
    records: list[RealVideoRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(RealVideoRecord(**json.loads(line)))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid manifest line {line_number}: {error}") from error
    sample_ids = [record.sample_id for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id values must be unique")
    return records


class _Groups:
    def __init__(
        self, records: list[RealVideoRecord], near_duplicate_hamming: int | None = None,
    ) -> None:
        self.parent = list(range(len(records)))
        source_owner: dict[str, int] = {}
        duplicate_owner: dict[str, int] = {}
        for index, record in enumerate(records):
            self._join_owner(source_owner, record.source_id, index)
            if record.duplicate_group:
                self._join_owner(duplicate_owner, record.duplicate_group, index)
        if near_duplicate_hamming is not None:
            tree: tuple[int, list[int], dict[int, object]] | None = None

            def neighbors(
                node: tuple[int, list[int], dict[int, object]], value: int,
            ) -> list[int]:
                found: list[int] = []
                stack = [node]
                while stack:
                    node_value, node_indices, children = stack.pop()
                    distance = (node_value ^ value).bit_count()
                    if distance <= near_duplicate_hamming:
                        found.extend(node_indices)
                    lower = distance - near_duplicate_hamming
                    upper = distance + near_duplicate_hamming
                    stack.extend(
                        child  # type: ignore[misc]
                        for edge, child in children.items()
                        if lower <= edge <= upper
                    )
                return found

            def insert(
                node: tuple[int, list[int], dict[int, object]], value: int, index: int,
            ) -> None:
                current = node
                while True:
                    distance = (current[0] ^ value).bit_count()
                    if distance == 0:
                        current[1].append(index)
                        return
                    child = current[2].get(distance)
                    if child is None:
                        current[2][distance] = (value, [index], {})
                        return
                    current = child  # type: ignore[assignment]

            for index, record in enumerate(records):
                if record.perceptual_hash is None:
                    continue
                value = int(record.perceptual_hash, 16)
                if tree is None:
                    tree = (value, [index], {})
                    continue
                for neighbor in neighbors(tree, value):
                    self._union(neighbor, index)
                insert(tree, value, index)

    def _root(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def _union(self, left: int, right: int) -> None:
        left_root, right_root = self._root(left), self._root(right)
        if left_root != right_root:
            self.parent[right_root] = left_root

    def _join_owner(self, owners: dict[str, int], key: str, index: int) -> None:
        if key in owners:
            self._union(owners[key], index)
        else:
            owners[key] = index

    def members(self) -> list[list[int]]:
        groups: dict[int, list[int]] = {}
        for index in range(len(self.parent)):
            groups.setdefault(self._root(index), []).append(index)
        return list(groups.values())


def _matches_c2(record: RealVideoRecord, pairs: set[PAIR]) -> bool:
    projections = {
        (record.subject, record.action),
        (record.subject, record.scene),
        (record.action, record.scene),
    }
    return bool(projections & pairs)


def _unit_interval(seed: int, group_key: str) -> float:
    digest = hashlib.sha256(f"{seed}:{group_key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_manifest(
    records: Iterable[RealVideoRecord], policy: CompositionalSplit
) -> list[RealVideoRecord]:
    """Assign whole source/duplicate groups using a deterministic policy."""

    materialized = list(records)
    assigned: list[RealVideoRecord | None] = [None] * len(materialized)
    for members in _Groups(materialized, policy.near_duplicate_hamming).members():
        group = [materialized[index] for index in members]
        c1_matches = [record.triple in policy.c1_triples for record in group]
        if any(c1_matches) and not all(c1_matches):
            raise ValueError(
                "one source/duplicate group mixes held-out and ordinary triples"
            )
        has_c1 = any(c1_matches)
        c2_matches = [_matches_c2(record, policy.c2_pairs) for record in group]
        if any(c2_matches) and not all(c2_matches):
            raise ValueError(
                "one source/duplicate group mixes held-out C2 and ordinary records"
            )
        has_c2 = any(c2_matches)
        if has_c1 and has_c2:
            raise ValueError("one source/duplicate group matches both C1 and C2")
        if has_c1:
            split = "c1"
        elif has_c2:
            split = "c2"
        else:
            group_key = min(record.sample_id for record in group)
            draw = _unit_interval(policy.seed, group_key)
            if draw < policy.validation_fraction:
                split = "validation"
            elif draw < policy.validation_fraction + policy.iid_fraction:
                split = "iid"
            else:
                split = "train"
        for index in members:
            assigned[index] = replace(materialized[index], split=split)
    return [record for record in assigned if record is not None]


def _pair_projections(triple: TRIPLE) -> set[PAIR]:
    subject, action, scene = triple
    return {(subject, action), (subject, scene), (action, scene)}


def _independent_group_ids(
    records: list[RealVideoRecord], near_duplicate_hamming: int,
) -> dict[int, int]:
    """Map records to transitive source/duplicate groups."""

    result: dict[int, int] = {}
    for group_id, members in enumerate(
        _Groups(records, near_duplicate_hamming).members()
    ):
        for index in members:
            result[index] = group_id
    return result


def _coverage_by_triple(
    records: list[RealVideoRecord], group_ids: dict[int, int], split: str,
) -> dict[TRIPLE, set[int]]:
    coverage: dict[TRIPLE, set[int]] = {}
    for index, record in enumerate(records):
        if record.split == split:
            coverage.setdefault(record.triple, set()).add(group_ids[index])
    return coverage


def independent_coverage(
    records: Iterable[RealVideoRecord], *, target_per_triple: int = 50,
    near_duplicate_hamming: int = 4, min_train_pair_source_groups: int = 1,
) -> dict[str, object]:
    """Report C1 readiness using the same transitive groups as split validation."""

    if target_per_triple < 1:
        raise ValueError("target_per_triple must be at least 1")
    if min_train_pair_source_groups < 1:
        raise ValueError("min_train_pair_source_groups must be at least 1")
    materialized = list(records)
    missing = [record.sample_id for record in materialized if record.perceptual_hash is None]
    if missing:
        raise ValueError(f"coverage requires perceptual_hash: {missing[:3]}")
    group_ids = _independent_group_ids(materialized, near_duplicate_hamming)
    coverage: dict[TRIPLE, set[int]] = {}
    for index, record in enumerate(materialized):
        coverage.setdefault(record.triple, set()).add(group_ids[index])
    counts = _json_coverage(coverage)
    all_triples = set(coverage)
    atoms_by_group: dict[str, set[int]] = {}
    pairs_by_group: dict[PAIR, set[int]] = {}
    blockers: dict[str, dict[str, list[str]]] = {}
    train_support: dict[str, dict[str, object]] = {}
    eligible: list[str] = []
    meets_target = sorted(
        triple for triple, groups in coverage.items() if len(groups) >= target_per_triple
    )
    meets_target_set = set(meets_target)
    for held in sorted(all_triples):
        held_group_ids = coverage[held]
        train_triples = all_triples - {held}
        for triple in train_triples:
            groups = coverage[triple] - held_group_ids
            for atom in triple:
                atoms_by_group.setdefault(atom, set()).update(groups)
            for pair in _pair_projections(triple):
                pairs_by_group.setdefault(pair, set()).update(groups)
        missing_atoms = sorted(atom for atom in held if not atoms_by_group.get(atom))
        missing_pairs = sorted(
            "|".join(pair) for pair in _pair_projections(held)
            if len(pairs_by_group.get(pair, set())) < min_train_pair_source_groups
        )
        key = "|".join(held)
        mixed_triples = sorted(
            "|".join(triple) for triple in train_triples
            if coverage[triple] & held_group_ids
        )
        atom_counts = {
            atom: len(atoms_by_group.get(atom, set())) for atom in held
        }
        pair_counts = {
            "|".join(pair): len(pairs_by_group.get(pair, set()))
            for pair in _pair_projections(held)
        }
        train_support[key] = {
            "train_atom_source_groups": dict(sorted(atom_counts.items())),
            "train_pair_source_groups": dict(sorted(pair_counts.items())),
            "train_pair_deficits": {
                pair: max(0, min_train_pair_source_groups - count)
                for pair, count in sorted(pair_counts.items())
            },
        }
        if mixed_triples:
            train_support[key]["mixed_held_group_triples"] = mixed_triples
        if held not in meets_target_set:
            pass
        elif mixed_triples or missing_atoms or missing_pairs:
            blockers[key] = {
                "missing_train_atoms": missing_atoms,
                "missing_train_pairs": missing_pairs,
            }
            if mixed_triples:
                blockers[key]["mixed_held_group_triples"] = mixed_triples
        else:
            eligible.append(key)
        atoms_by_group.clear()
        pairs_by_group.clear()
    return {
        "records": len(materialized),
        "independent_source_groups": len(set(group_ids.values())),
        "near_duplicate_hamming": near_duplicate_hamming,
        "target_per_triple": target_per_triple,
        "min_train_pair_source_groups": min_train_pair_source_groups,
        "triple_source_groups": counts,
        "triple_deficits": {
            triple: max(0, target_per_triple - count)
            for triple, count in counts.items()
        },
        "meets_target_triples": ["|".join(triple) for triple in meets_target],
        "eligible_c1_triples": eligible,
        "c1_blockers": blockers,
        "c1_train_support": train_support,
    }


def _coverage_by_pair(
    records: list[RealVideoRecord], group_ids: dict[int, int], split: str,
) -> dict[PAIR, set[int]]:
    coverage: dict[PAIR, set[int]] = {}
    for index, record in enumerate(records):
        if record.split == split:
            for pair in _pair_projections(record.triple):
                coverage.setdefault(pair, set()).add(group_ids[index])
    return coverage


def _json_coverage(values: dict[tuple[str, ...], set[int]]) -> dict[str, int]:
    return {
        "|".join(key): len(group_ids)
        for key, group_ids in sorted(values.items())
    }


def validate_split(
    records: Iterable[RealVideoRecord], policy: CompositionalSplit
) -> dict[str, object]:
    """Fail closed unless the manifest proves the stated C1/C2 experiment."""

    materialized = list(records)
    if any(record.split is None for record in materialized):
        raise ValueError("every record must have a split")
    missing_fingerprints = [
        record.sample_id for record in materialized if record.perceptual_hash is None
    ]
    if missing_fingerprints:
        raise ValueError(
            "every record requires perceptual_hash before validation: "
            f"{missing_fingerprints[:3]}"
        )
    for field in ("source_id", "duplicate_group"):
        memberships: dict[str, set[str]] = {}
        for record in materialized:
            key = getattr(record, field)
            if key:
                memberships.setdefault(key, set()).add(record.split or "")
        leaked = sorted(key for key, splits in memberships.items() if len(splits) > 1)
        if leaked:
            raise ValueError(f"{field} crosses splits: {leaked[:3]}")
    fingerprinted = [record for record in materialized if record.perceptual_hash]
    for index, left in enumerate(fingerprinted):
        for right in fingerprinted[index + 1:]:
            if left.split == right.split:
                continue
            distance = (
                int(left.perceptual_hash or "0", 16)
                ^ int(right.perceptual_hash or "0", 16)
            ).bit_count()
            if distance <= policy.near_duplicate_hamming:
                raise ValueError(
                    "near-duplicate perceptual hashes cross splits: "
                    f"{left.sample_id}, {right.sample_id}, hamming={distance}"
                )

    train = [record for record in materialized if record.split == "train"]
    group_ids = _independent_group_ids(materialized, policy.near_duplicate_hamming)
    triple_coverage = {
        split: _coverage_by_triple(materialized, group_ids, split)
        for split in ("train", "validation", "iid", "c1", "c2")
    }
    train_pair_coverage = _coverage_by_pair(materialized, group_ids, "train")
    train_triples = {record.triple for record in train}
    train_atoms = {
        value for record in train for value in record.triple
    }
    train_pairs = {pair for record in train for pair in _pair_projections(record.triple)}
    c1_triples = {
        record.triple for record in materialized if record.split == "c1"
    }
    unexpected_c1 = c1_triples - policy.c1_triples
    if unexpected_c1:
        raise ValueError(f"C1 records contain undeclared triples: {sorted(unexpected_c1)}")
    for triple in policy.c1_triples:
        if triple not in c1_triples:
            raise ValueError(f"C1 triple has no evaluation records: {triple}")
        if triple in train_triples:
            raise ValueError(f"C1 triple leaked into training: {triple}")
        missing_atoms = set(triple) - train_atoms
        if missing_atoms:
            raise ValueError(f"C1 atom absent from training: {sorted(missing_atoms)}")
        missing_pairs = _pair_projections(triple) - train_pairs
        if missing_pairs:
            raise ValueError(f"C1 pair projection absent from training: {sorted(missing_pairs)}")
        c1_sources = len(triple_coverage["c1"].get(triple, set()))
        if c1_sources < policy.min_c1_source_groups:
            raise ValueError(
                f"C1 triple {triple} requires {policy.min_c1_source_groups} "
                f"independent source groups; found {c1_sources}"
            )
        for pair in _pair_projections(triple):
            source_count = len(train_pair_coverage.get(pair, set()))
            if source_count < policy.min_train_pair_source_groups:
                raise ValueError(
                    f"C1 pair projection {pair} requires "
                    f"{policy.min_train_pair_source_groups} independent training "
                    f"source groups; found {source_count}"
                )
    c2_records = [record for record in materialized if record.split == "c2"]
    unexpected_c2 = [
        record.sample_id
        for record in c2_records
        if not (_pair_projections(record.triple) & policy.c2_pairs)
    ]
    if unexpected_c2:
        raise ValueError(f"C2 records match no declared pair: {unexpected_c2[:3]}")
    for pair in policy.c2_pairs:
        if not any(pair in _pair_projections(record.triple) for record in c2_records):
            raise ValueError(f"C2 pair has no evaluation records: {pair}")
        if pair in train_pairs:
            raise ValueError(f"C2 pair leaked into training: {pair}")
        if not set(pair) <= train_atoms:
            raise ValueError(f"C2 atoms must each occur in training: {pair}")
        c2_sources = {
            group_ids[index]
            for index, record in enumerate(materialized)
            if record.split == "c2" and pair in _pair_projections(record.triple)
        }
        if len(c2_sources) < policy.min_c2_source_groups:
            raise ValueError(
                f"C2 pair {pair} requires {policy.min_c2_source_groups} "
                f"independent source groups; found {len(c2_sources)}"
            )

    iid_triples = set(triple_coverage["iid"])
    unseen_iid = iid_triples - train_triples
    if unseen_iid:
        raise ValueError(f"IID triple absent from training: {sorted(unseen_iid)}")
    for triple, sources in triple_coverage["iid"].items():
        if len(sources) < policy.min_iid_source_groups:
            raise ValueError(
                f"IID triple {triple} requires {policy.min_iid_source_groups} "
                f"independent source groups; found {len(sources)}"
            )

    counts = {split: 0 for split in sorted(SPLITS)}
    for record in materialized:
        counts[record.split or "train"] += 1
    return {
        "valid": True,
        "records": len(materialized),
        "counts": counts,
        "c1_triples": sorted(policy.c1_triples),
        "c2_pairs": sorted(policy.c2_pairs),
        "coverage": {
            **{
                f"{split}_triples": _json_coverage(coverage)
                for split, coverage in triple_coverage.items()
            },
            "train_pairs": _json_coverage(train_pair_coverage),
        },
        "minimums": {
            "c1_source_groups": policy.min_c1_source_groups,
            "c2_source_groups": policy.min_c2_source_groups,
            "iid_source_groups": policy.min_iid_source_groups,
            "train_pair_source_groups": policy.min_train_pair_source_groups,
            "near_duplicate_hamming": policy.near_duplicate_hamming,
        },
    }


def decode_video_clip(path: Path, *, frames: int, size: int) -> torch.Tensor:
    """Decode evenly sampled RGB frames through ffmpeg into ``[-1, 1]``."""

    if not path.is_file():
        raise FileNotFoundError(path)
    if frames < 2 or size <= 0:
        raise ValueError("frames must be at least 2 and size must be positive")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to decode real videos")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to inspect real videos")
    probe = subprocess.run(
        [
            ffprobe, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = float(probe.stdout.strip())
    if duration <= 0:
        raise ValueError(f"clip {path} has no positive duration")
    with tempfile.TemporaryDirectory(prefix="video-dit-decode-") as directory:
        frame_pattern = Path(directory) / "%06d.png"
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
            "-vf",
            (
                f"fps={frames / duration},scale={size}:{size}:force_original_aspect_ratio=increase,"
                f"crop={size}:{size}"
            ),
            "-frames:v", str(frames), str(frame_pattern),
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"ffmpeg could not decode {path}") from error
        images = sorted(Path(directory).glob("*.png"))
        if len(images) != frames:
            raise ValueError(
                f"clip {path} yielded {len(images)} frames; expected {frames}"
            )
        arrays = [
            np.asarray(Image.open(image).convert("RGB"), dtype=np.uint8).copy()
            for image in images
        ]
    return (
        torch.from_numpy(np.stack(arrays))
        .permute(3, 0, 1, 2)
        .float()
        .div(127.5)
        .sub(1.0)
    )


class RealVideoDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Read one split of an auditable real-video JSONL manifest."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        split: str,
        frames: int,
        size: int,
        space: str = "pixel",
        verify_sha256: bool = True,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {sorted(SPLITS)}")
        self.root = manifest_path.parent
        self.records = [
            record for record in load_manifest(manifest_path) if record.split == split
        ]
        if not self.records:
            raise ValueError(f"manifest contains no {split!r} records")
        self.frames = frames
        self.size = size
        if space not in ("pixel", "latent"):
            raise ValueError("space must be pixel or latent")
        self.space = space
        self.verify_sha256 = verify_sha256
        self.subjects = sorted({record.subject for record in self.records})
        self.actions = sorted({record.action for record in self.records})
        self.scenes = sorted({record.scene for record in self.records})

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        path = Path(record.clip_path)
        if not path.is_absolute():
            path = self.root / path
        if self.space == "pixel" and self.verify_sha256:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual.lower() != record.sha256.lower():
                raise ValueError(f"sha256 mismatch for {record.sample_id}")
        factors = torch.tensor(
            [
                self.subjects.index(record.subject),
                self.actions.index(record.action),
                self.scenes.index(record.scene),
            ],
            dtype=torch.long,
        )
        return {
            "video": self._load_video(record, path),
            "prompt": record.prompt,
            "factors": factors,
            "condition": self._load_text_embedding(record),
            "text_encoder_id": record.text_encoder_id or "",
            "vae_id": record.vae_id or "",
            "sample_id": record.sample_id,
        }

    def _load_video(self, record: RealVideoRecord, clip_path: Path) -> torch.Tensor:
        if self.space == "pixel":
            return decode_video_clip(clip_path, frames=self.frames, size=self.size)
        if record.latent_path is None:
            raise ValueError(f"latent record {record.sample_id} lacks latent_path")
        path = Path(record.latent_path)
        if not path.is_absolute():
            path = self.root / path
        latent = torch.load(path, map_location="cpu", weights_only=True)
        expected = (4, self.frames, self.size, self.size)
        if not isinstance(latent, torch.Tensor) or tuple(latent.shape) != expected:
            raise ValueError(f"latent for {record.sample_id} must have shape {expected}")
        if not torch.isfinite(latent).all():
            raise ValueError(f"latent for {record.sample_id} must be finite")
        return latent.float()

    def _load_text_embedding(self, record: RealVideoRecord) -> torch.Tensor:
        if record.text_embedding_path is None:
            return torch.empty(0, dtype=torch.float32)
        path = Path(record.text_embedding_path)
        if not path.is_absolute():
            path = self.root / path
        embedding = torch.load(path, map_location="cpu", weights_only=True)
        kind = record.text_embedding_kind or "pooled"
        if kind == "tokens" and not (record.text_encoder_id or "").endswith("#tokens-v2"):
            raise ValueError(f"token embedding for {record.sample_id} requires #tokens-v2")
        expected_ndim = 1 if kind == "pooled" else 2
        if not isinstance(embedding, torch.Tensor) or embedding.ndim != expected_ndim:
            raise ValueError(
                f"{kind} text embedding for {record.sample_id} must be "
                f"a {expected_ndim}-D tensor"
            )
        if not torch.isfinite(embedding).all():
            raise ValueError(f"text embedding for {record.sample_id} must be finite")
        if kind == "tokens" and not torch.any(embedding.abs().sum(dim=-1).ne(0)):
            raise ValueError(f"token embedding for {record.sample_id} has no valid token")
        return embedding.float()


def perceptual_hash(video: torch.Tensor) -> str:
    """Compute a compact temporal-average difference hash for leakage screening."""

    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError("video must have shape [3, T, H, W]")
    gray = (
        0.299 * video[0].float()
        + 0.587 * video[1].float()
        + 0.114 * video[2].float()
    ).mean(dim=0)
    image = Image.fromarray(
        ((gray.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy(), mode="L"
    ).resize((9, 8), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.int16)
    bits = (array[:, 1:] > array[:, :-1]).reshape(-1)
    value = sum(int(bit) << index for index, bit in enumerate(bits))
    return f"{value:016x}"
