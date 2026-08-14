"""Create and reveal a prompt-hidden human review of generated videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile

import torch

from blind_review_pool import wilson_interval
from condition_alignment import discover_generations
from real_data import CompositionalSplit, RealVideoRecord, validate_split


PROTOCOL = "frozen compositional generation evaluation v1"
QUEUE_PROTOCOL = "prompt-hidden generated-video factor review queue v1"
DECISION_PROTOCOL = "blinded generated-video factor review v1"
SPLITS = ("iid", "c1", "c2")
UNCLEAR_FACTOR = "absent_or_unclear"
GENERATION_CONTRACT_FIELDS = (
    "checkpoint_sha256", "training_step", "weights", "prediction_type",
    "sampling_steps", "start_timestep", "clip_denoised", "effective_cfg_scale",
    "condition_mode", "text_encoder_id", "training_manifest_sha256",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _lower_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _json_object(data: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _within(path: Path, directory: Path) -> bool:
    resolved_path = path.resolve()
    resolved_directory = directory.resolve()
    return resolved_path == resolved_directory or resolved_directory in resolved_path.parents


def _assignment_commitment(rows: list[dict[str, object]]) -> str:
    return _sha256(
        (json.dumps(rows, separators=(",", ":"), sort_keys=True) + "\n").encode()
    )


def _factor_derangement(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return a deterministic perfect mismatch, or reject when none exists."""

    factors = ("subject", "action", "scene")
    triples = [
        tuple(row.get(f"expected_{factor}") for factor in factors) for row in rows
    ]
    candidates = [
        [replacement for replacement in range(len(rows)) if triples[index] != triples[replacement]]
        for index in range(len(rows))
    ]
    matched_original: dict[int, int] = {}

    def assign(original: int, seen: set[int]) -> bool:
        for replacement in candidates[original]:
            if replacement in seen:
                continue
            seen.add(replacement)
            previous = matched_original.get(replacement)
            if previous is None or assign(previous, seen):
                matched_original[replacement] = original
                return True
        return False

    for original in sorted(range(len(rows)), key=lambda index: (len(candidates[index]), index)):
        if not assign(original, set()):
            raise ValueError("cannot construct a factor-triple derangement for shuffled control")
    replacement_for_original = {
        original: replacement for replacement, original in matched_original.items()
    }
    return [rows[replacement_for_original[index]] for index in range(len(rows))]


def _evaluation_spec(data: bytes) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    spec = _json_object(data, "generation evaluation spec")
    if spec.get("protocol") != PROTOCOL or spec.get("training_ready") is not False:
        raise ValueError("generation evaluation spec has an invalid frozen protocol")
    for field in ("training_manifest_sha256", "split_report_sha256"):
        _lower_sha(spec.get(field), f"generation evaluation spec {field}")
    seeds = spec.get("seeds")
    if (
        not isinstance(seeds, list) or len(seeds) < 2
        or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds) or seeds != sorted(seeds)
    ):
        raise ValueError("generation evaluation spec requires at least two unique sorted seeds")
    contract = spec.get("generation_contract")
    if not isinstance(contract, dict) or set(contract) != set(GENERATION_CONTRACT_FIELDS):
        raise ValueError("generation evaluation spec has an invalid generation contract")
    _lower_sha(contract.get("checkpoint_sha256"), "generation checkpoint SHA-256")
    _lower_sha(contract.get("training_manifest_sha256"), "checkpoint training manifest SHA-256")
    if contract["training_manifest_sha256"] != spec["training_manifest_sha256"]:
        raise ValueError("checkpoint training manifest differs from the evaluation manifest")
    value = spec.get("records")
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError("generation evaluation spec requires at least two prompts")
    records: dict[str, dict[str, str]] = {}
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("generation evaluation records must be objects")
        row = {
            field: _nonempty(item.get(field), f"generation evaluation {field}")
            for field in ("prompt", "split", "subject", "action", "scene")
        }
        if row["split"] not in SPLITS:
            raise ValueError(f"generation evaluation split must be one of {SPLITS}")
        if row["prompt"] in records:
            raise ValueError("generation evaluation prompts must be unique")
        if UNCLEAR_FACTOR in (row["subject"], row["action"], row["scene"]):
            raise ValueError("absent_or_unclear is reserved for blinded human judgments")
        records[row["prompt"]] = row
    if not any(row["split"] == "c1" for row in records.values()):
        raise ValueError("generation evaluation spec must include a C1 prompt")
    return records, spec


def _manifest_records(data: bytes) -> list[RealVideoRecord]:
    records: list[RealVideoRecord] = []
    sample_ids: set[str] = set()
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("split manifest must be UTF-8 JSONL") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            record = RealVideoRecord(**value)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid split manifest line {line_number}: {error}") from error
        if record.sample_id in sample_ids:
            raise ValueError("split manifest sample_id values must be unique")
        sample_ids.add(record.sample_id)
        records.append(record)
    if not records:
        raise ValueError("split manifest must not be empty")
    return records


def build_generation_evaluation_spec(
    manifest_bytes: bytes, split_report_bytes: bytes, *,
    generation_contract: dict[str, object], seeds: list[int],
) -> dict[str, object]:
    """Derive prompt/factor expectations from a fully revalidated split manifest."""

    records = _manifest_records(manifest_bytes)
    report = _json_object(split_report_bytes, "split validation report")
    minimums = report.get("minimums")
    if report.get("valid") is not True or not isinstance(minimums, dict):
        raise ValueError("split validation report must be valid and include minimums")
    try:
        policy = CompositionalSplit(
            c1_triples={tuple(value) for value in report["c1_triples"]},
            c2_pairs={tuple(value) for value in report["c2_pairs"]},
            iid_fraction=0, validation_fraction=0,
            near_duplicate_hamming=int(minimums["near_duplicate_hamming"]),
            min_c1_source_groups=int(minimums["c1_source_groups"]),
            min_c2_source_groups=int(minimums["c2_source_groups"]),
            min_iid_source_groups=int(minimums["iid_source_groups"]),
            min_train_pair_source_groups=int(minimums["train_pair_source_groups"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("split validation report has an invalid policy contract") from error
    if policy.min_c1_source_groups < 50 or policy.min_train_pair_source_groups < 10:
        raise ValueError(
            "formal generation evaluation requires at least 50 C1 source groups "
            "and 10 training source groups per held pair"
        )
    recomputed = validate_split(records, policy)
    if json.loads(json.dumps(recomputed, sort_keys=True)) != report:
        raise ValueError("split validation report does not match the supplied manifest")
    prompts: dict[str, dict[str, str]] = {}
    for record in records:
        if record.split not in SPLITS:
            continue
        value = {
            "prompt": record.prompt, "split": str(record.split),
            "subject": record.subject, "action": record.action, "scene": record.scene,
        }
        previous = prompts.setdefault(record.prompt, value)
        if previous != value:
            raise ValueError("one evaluation prompt maps to multiple factor/split contracts")
    spec = {
        "protocol": PROTOCOL,
        "training_manifest_sha256": _sha256(manifest_bytes),
        "split_report_sha256": _sha256(split_report_bytes),
        "generation_contract": generation_contract,
        "seeds": sorted(seeds),
        "records": sorted(prompts.values(), key=lambda row: (row["split"], row["prompt"])),
        "training_ready": False,
    }
    _evaluation_spec(_canonical(spec))
    return spec


def checkpoint_generation_contract(
    checkpoint_path: Path, *, weights: str, sampling_steps: int,
    start_timestep: int | None, cfg_scale: float,
) -> dict[str, object]:
    """Freeze the exact inference contract before any evaluation video is sampled."""

    checkpoint_bytes = checkpoint_path.read_bytes()
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("generation checkpoint must be a safe state-only checkpoint") from error
    if not isinstance(checkpoint, dict):
        raise ValueError("generation checkpoint root must be a mapping")
    config = checkpoint.get("config")
    diffusion = checkpoint.get("diffusion")
    data = checkpoint.get("data_contract")
    if not all(isinstance(value, dict) for value in (config, diffusion, data)):
        raise ValueError("generation checkpoint lacks config/diffusion/data contracts")
    if weights not in checkpoint or not isinstance(checkpoint[weights], dict):
        raise ValueError(f"generation checkpoint lacks {weights!r} weights")
    try:
        timesteps = int(diffusion["timesteps"])
        start = timesteps - 1 if start_timestep is None else start_timestep
        dropout = float(config["condition_dropout_prob"])
        space = str(data["space"])
        manifest_sha = _lower_sha(data["manifest_sha256"], "checkpoint manifest SHA-256")
        training_step = int(checkpoint["step"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("generation checkpoint has an invalid inference contract") from error
    if not 0 <= start < timesteps or sampling_steps < 1 or cfg_scale < 0:
        raise ValueError("invalid preregistered sampling parameters")
    return {
        "checkpoint_sha256": _sha256(checkpoint_bytes),
        "training_step": training_step, "weights": weights,
        "prediction_type": str(checkpoint.get("prediction_type", "epsilon")),
        "sampling_steps": min(sampling_steps, start + 1), "start_timestep": start,
        "clip_denoised": space == "pixel",
        "effective_cfg_scale": cfg_scale if dropout > 0 else 1.0,
        "condition_mode": str(config["condition_mode"]),
        "text_encoder_id": data.get("text_encoder_id"),
        "training_manifest_sha256": manifest_sha,
    }


def create_blind_generation_pool(
    generation_dir: Path,
    spec_bytes: bytes,
    manifest_bytes: bytes,
    split_report_bytes: bytes,
    media_dir: Path,
    *,
    seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Freeze a complete multi-seed generation grid into prompt-hidden MP4 names."""

    specs, spec = _evaluation_spec(spec_bytes)
    derived_spec = build_generation_evaluation_spec(
        manifest_bytes, split_report_bytes,
        generation_contract=dict(spec["generation_contract"]),
        seeds=list(spec["seeds"]),
    )
    if spec != derived_spec:
        raise ValueError("generation evaluation spec does not match the validated split manifest")
    rows = discover_generations(generation_dir)
    if len(rows) < 2:
        raise ValueError("blind generation review requires at least two generations")
    expected_spec_sha = _sha256(spec_bytes)
    if {row.evaluation_spec_sha256 for row in rows} != {expected_spec_sha}:
        raise ValueError("every generation must bind the preregistered evaluation spec")
    by_seed: dict[int, dict[str, object]] = {}
    for row in rows:
        prompt_rows = by_seed.setdefault(row.seed, {})
        if row.prompt in prompt_rows:
            raise ValueError("each seed may contain each prompt only once")
        prompt_rows[row.prompt] = row
    expected_prompts = set(specs)
    if set(by_seed) != set(spec["seeds"]):
        raise ValueError("generation directory does not exactly match the preregistered seed roster")
    if any(set(prompt_rows) != expected_prompts for prompt_rows in by_seed.values()):
        raise ValueError("every seed must contain the same prompt set from the evaluation spec")
    contracts = {row.contract for row in rows}
    if len(contracts) != 1:
        raise ValueError("generation contract differs across the blind review pool")
    training_manifests = {row.training_manifest_sha256 for row in rows}
    if len(training_manifests) != 1:
        raise ValueError("generation training manifest differs across the blind review pool")
    actual_contract_tuple = next(iter(contracts))
    actual_contract = dict(zip(GENERATION_CONTRACT_FIELDS[:-1], actual_contract_tuple, strict=True))
    actual_contract["training_manifest_sha256"] = training_manifests.pop()
    if actual_contract != spec["generation_contract"]:
        raise ValueError("generation contract differs from the preregistered evaluation contract")
    if media_dir.exists():
        raise ValueError("blind media directory must not already exist")
    resolved_generation = generation_dir.resolve()
    resolved_parent = media_dir.parent.resolve()
    if media_dir.resolve() == resolved_generation or resolved_generation in media_dir.resolve().parents:
        raise ValueError("blind media directory must be outside the generation directory")
    media_dir.parent.mkdir(parents=True, exist_ok=True)

    ordered = [
        by_seed[seed_value][prompt]
        for seed_value in sorted(by_seed) for prompt in sorted(expected_prompts)
    ]
    rng = random.Random(seed)
    rng.shuffle(ordered)
    key_rows: list[dict[str, object]] = []
    queue_rows: list[dict[str, object]] = []
    stage = Path(tempfile.mkdtemp(prefix=f".{media_dir.name}.", dir=resolved_parent))
    try:
        for index, generation in enumerate(ordered, 1):
            blind_id = f"generation-{index:04d}"
            source = generation.video_path
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"generated video must be a regular non-symlink file: {source}")
            video_bytes = source.read_bytes()
            if not video_bytes:
                raise ValueError(f"generated video is empty: {source}")
            digest = _sha256(video_bytes)
            if digest != generation.video_sha256:
                raise ValueError(f"generated video changed after discovery: {source}")
            filename = f"{blind_id}.mp4"
            (stage / filename).write_bytes(video_bytes)
            expected = specs[generation.prompt]
            queue_rows.append({
                "blind_id": blind_id, "filename": filename,
                "bytes": len(video_bytes), "video_sha256": digest,
            })
            key_rows.append({
                "blind_id": blind_id, "source_video": str(source.resolve()),
                "video_sha256": digest, "prompt": generation.prompt,
                "split": expected["split"], "seed": generation.seed,
                "expected_subject": expected["subject"],
                "expected_action": expected["action"],
                "expected_scene": expected["scene"],
            })
        os.replace(stage, media_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    vocabulary = {
        "subjects": sorted({row["subject"] for row in specs.values()}) + [UNCLEAR_FACTOR],
        "actions": sorted({row["action"] for row in specs.values()}) + [UNCLEAR_FACTOR],
        "scenes": sorted({row["scene"] for row in specs.values()}) + [UNCLEAR_FACTOR],
    }
    commitment = _assignment_commitment(key_rows)
    queue = {
        "protocol": QUEUE_PROTOCOL,
        "evaluation_spec_sha256": _sha256(spec_bytes),
        "training_manifest_sha256": spec["training_manifest_sha256"],
        "split_report_sha256": spec["split_report_sha256"],
        "generation_contract": spec["generation_contract"],
        "seeds": list(spec["seeds"]), "prompts_per_seed": len(specs),
        "factor_vocabulary": vocabulary,
        "assignment_commitment_sha256": commitment,
        "records": queue_rows,
        "prompt_hidden": True, "human_review_required": True,
    }
    key = {
        "protocol": "sealed generated-video prompt assignments v1",
        "queue_sha256": _sha256(_canonical(queue)),
        "evaluation_spec_sha256": _sha256(spec_bytes),
        "assignment_commitment_sha256": commitment,
        "shuffle_seed": seed, "records": key_rows,
    }
    return queue, key


def _decision_rows(
    decision_bytes: list[bytes], queue_sha: str, blind_ids: set[str],
    vocabulary: dict[str, set[str]], media_hashes: dict[str, str],
) -> tuple[list[str], dict[str, list[tuple[str, str, str]]]]:
    reviewers: list[str] = []
    by_id: dict[str, list[tuple[str, str, str]]] = {blind_id: [] for blind_id in blind_ids}
    for index, data in enumerate(decision_bytes, 1):
        decision = _json_object(data, f"reviewer decision {index}")
        if (
            decision.get("protocol") != DECISION_PROTOCOL
            or decision.get("queue_sha256") != queue_sha
        ):
            raise ValueError("reviewer decision does not bind the blind queue")
        reviewer = _nonempty(decision.get("reviewer_id"), "reviewer_id")
        if reviewer in reviewers:
            raise ValueError("reviewer IDs must be unique")
        reviewers.append(reviewer)
        records = decision.get("records")
        if not isinstance(records, list):
            raise ValueError("reviewer decision records must be an array")
        seen: set[str] = set()
        for row in records:
            if not isinstance(row, dict):
                raise ValueError("reviewer decision rows must be objects")
            blind_id = _nonempty(row.get("blind_id"), "decision blind_id")
            if blind_id not in blind_ids or blind_id in seen:
                raise ValueError("reviewer decision blind IDs must be complete and unique")
            seen.add(blind_id)
            factors = tuple(
                _nonempty(row.get(f"predicted_{factor}"), f"predicted_{factor}")
                for factor in ("subject", "action", "scene")
            )
            if any(value not in vocabulary[f"{factor}s"] for factor, value in zip(
                ("subject", "action", "scene"), factors, strict=True,
            )):
                raise ValueError("reviewer prediction is outside the frozen factor vocabulary")
            _nonempty(row.get("notes"), "review notes")
            if row.get("observed_video_sha256") != media_hashes[blind_id]:
                raise ValueError("reviewer decision does not bind the observed blind video")
            by_id[blind_id].append(factors)  # type: ignore[arg-type]
        if seen != blind_ids:
            raise ValueError("each reviewer must cover the complete blind queue")
    if len(reviewers) < 2:
        raise ValueError("blind generation review requires at least two independent raters")
    return reviewers, by_id


def _split_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {
            "status": "missing", "samples": 0, "unanimous_samples": 0,
            "ambiguous_samples": 0, "subject_accuracy": None,
            "action_accuracy": None, "scene_accuracy": None,
            "unanimous_triple_accuracy": None, "triple_accuracy_wilson95": None,
            "shuffled_triple_accuracy": None, "gain_over_shuffled": None,
        }
    total = len(rows)
    unanimous = sum(bool(row["unanimous"]) for row in rows)
    subject = sum(bool(row["subject_correct"]) for row in rows)
    action = sum(bool(row["action_correct"]) for row in rows)
    scene = sum(bool(row["scene_correct"]) for row in rows)
    triples = sum(bool(row["triple_correct"]) for row in rows)
    shuffled = sum(bool(row["shuffled_triple_correct"]) for row in rows)
    return {
        "status": "measured", "samples": total,
        "unanimous_samples": unanimous, "ambiguous_samples": total - unanimous,
        "subject_accuracy": subject / total, "action_accuracy": action / total,
        "scene_accuracy": scene / total,
        "unanimous_triple_accuracy": triples / total,
        "triple_accuracy_wilson95": wilson_interval(triples, total),
        "shuffled_triple_accuracy": shuffled / total,
        "gain_over_shuffled": (triples - shuffled) / total,
        "passes_prompt_control": triples > shuffled,
    }


def reveal_blind_generation_results(
    queue_bytes: bytes,
    key_bytes: bytes,
    decision_bytes: list[bytes],
    media_dir: Path,
) -> dict[str, object]:
    """Verify blinded media and aggregate unanimous human factor judgments."""

    queue = _json_object(queue_bytes, "blind generation queue")
    key = _json_object(key_bytes, "sealed generation key")
    queue_sha = _sha256(queue_bytes)
    queue_records = queue.get("records")
    key_records = key.get("records")
    if (
        queue.get("protocol") != QUEUE_PROTOCOL
        or queue.get("prompt_hidden") is not True
        or key.get("queue_sha256") != queue_sha
        or not isinstance(queue_records, list)
        or not isinstance(key_records, list)
        or len(queue_records) != len(key_records)
        or queue.get("assignment_commitment_sha256")
        != _assignment_commitment(key_records)
        or key.get("assignment_commitment_sha256")
        != queue.get("assignment_commitment_sha256")
    ):
        raise ValueError("sealed key does not bind the blind generation queue")
    queue_by_id: dict[str, dict[str, object]] = {}
    for row in queue_records:
        if not isinstance(row, dict):
            raise ValueError("blind queue records must be objects")
        blind_id = _nonempty(row.get("blind_id"), "queue blind_id")
        if blind_id in queue_by_id:
            raise ValueError("blind queue IDs must be unique")
        queue_by_id[blind_id] = row
    key_by_id: dict[str, dict[str, object]] = {}
    for row in key_records:
        if not isinstance(row, dict):
            raise ValueError("sealed key records must be objects")
        blind_id = _nonempty(row.get("blind_id"), "key blind_id")
        if blind_id in key_by_id:
            raise ValueError("sealed key IDs must be unique")
        key_by_id[blind_id] = row
    if set(queue_by_id) != set(key_by_id):
        raise ValueError("sealed key IDs differ from the blind queue")

    expected_names = {f"{blind_id}.mp4" for blind_id in queue_by_id}
    if media_dir.is_symlink() or not media_dir.is_dir():
        raise ValueError("blind media directory must be a regular directory")
    entries = list(media_dir.iterdir())
    actual_names = {path.name for path in entries}
    if actual_names != expected_names or any(not path.is_file() for path in entries):
        raise ValueError("blind media directory does not exactly match the queue")
    for blind_id, row in queue_by_id.items():
        path = media_dir / f"{blind_id}.mp4"
        if path.is_symlink() or _sha256(path.read_bytes()) != row.get("video_sha256"):
            raise ValueError(f"blind media SHA-256 mismatch: {blind_id}")
        if path.stat().st_size != row.get("bytes"):
            raise ValueError(f"blind media byte size mismatch: {blind_id}")

    vocabulary_value = queue.get("factor_vocabulary")
    if not isinstance(vocabulary_value, dict):
        raise ValueError("blind queue lacks a factor vocabulary")
    vocabulary = {
        field: set(value) if isinstance(value, list) else set()
        for field, value in vocabulary_value.items()
    }
    if set(vocabulary) != {"subjects", "actions", "scenes"} or any(
        not values or any(not isinstance(item, str) or not item for item in values)
        for values in vocabulary.values()
    ):
        raise ValueError("blind queue factor vocabulary is invalid")
    reviewers, predictions = _decision_rows(
        decision_bytes, queue_sha, set(queue_by_id), vocabulary,
        {
            blind_id: _lower_sha(row.get("video_sha256"), "blind video SHA-256")
            for blind_id, row in queue_by_id.items()
        },
    )

    by_seed: dict[int, list[dict[str, object]]] = {}
    for row in key_records:
        seed = row.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("sealed generation seed must be an integer")
        by_seed.setdefault(seed, []).append(row)
    shuffled_expected: dict[str, tuple[str, str, str]] = {}
    for rows in by_seed.values():
        ordered = sorted(rows, key=lambda row: str(row.get("prompt")))
        if len(ordered) < 2:
            raise ValueError("each seed requires at least two prompts for shuffled control")
        replacement_order = _factor_derangement(ordered)
        for row, replacement in zip(ordered, replacement_order, strict=True):
            shuffled_expected[str(row["blind_id"])] = tuple(
                _nonempty(replacement.get(f"expected_{factor}"), f"expected_{factor}")
                for factor in ("subject", "action", "scene")
            )  # type: ignore[assignment]

    revealed: list[dict[str, object]] = []
    for blind_id, assignment in key_by_id.items():
        expected = tuple(
            _nonempty(assignment.get(f"expected_{factor}"), f"expected_{factor}")
            for factor in ("subject", "action", "scene")
        )
        judgments = predictions[blind_id]
        consensus = tuple(
            values[0] if len(set(values)) == 1 else None
            for values in zip(*judgments, strict=True)
        )
        unanimous = all(value is not None for value in consensus)
        subject_correct, action_correct, scene_correct = (
            consensus[index] == expected[index] for index in range(3)
        )
        triple_correct = unanimous and consensus == expected
        shuffled_correct = unanimous and consensus == shuffled_expected[blind_id]
        revealed.append({
            "blind_id": blind_id, "split": assignment.get("split"),
            "seed": assignment.get("seed"), "prompt": assignment.get("prompt"),
            "expected_subject": expected[0], "expected_action": expected[1],
            "expected_scene": expected[2], "consensus_subject": consensus[0],
            "consensus_action": consensus[1], "consensus_scene": consensus[2],
            "unanimous": unanimous, "subject_correct": subject_correct,
            "action_correct": action_correct, "scene_correct": scene_correct,
            "triple_correct": triple_correct,
            "shuffled_triple_correct": shuffled_correct,
        })
    if any(row["split"] not in SPLITS for row in revealed):
        raise ValueError("sealed generation split is invalid")
    return {
        "protocol": "revealed prompt-hidden generated-video factor evaluation v1",
        "queue_sha256": queue_sha, "key_sha256": _sha256(key_bytes),
        "evaluation_spec_sha256": queue.get("evaluation_spec_sha256"),
        "training_manifest_sha256": queue.get("training_manifest_sha256"),
        "split_report_sha256": queue.get("split_report_sha256"),
        "decision_sha256": [_sha256(data) for data in decision_bytes],
        "raters": len(reviewers), "reviewer_ids": reviewers,
        "overall": _split_metrics(revealed),
        **{
            split: _split_metrics([row for row in revealed if row["split"] == split])
            for split in SPLITS
        },
        "records": revealed,
        "limitations": (
            "Human factor judgments are prompt-hidden and disagreements count as incorrect; "
            "nearest-training-video and automated metrics remain separate required evidence."
        ),
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
        delete=False,
    ) as handle:
        handle.write(_canonical(value))
        temporary = Path(handle.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_pair(first_path: Path, first: object, second_path: Path, second: object) -> None:
    if first_path.resolve() == second_path.resolve():
        raise ValueError("blind queue and key paths must differ")
    for path in (first_path, second_path):
        if path.exists():
            raise ValueError("blind queue/key outputs must not already exist")
    _atomic_json(first_path, first)
    try:
        _atomic_json(second_path, second)
    except Exception:
        first_path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("generation_dir", type=Path)
    create.add_argument("evaluation_spec", type=Path)
    create.add_argument("split_manifest", type=Path)
    create.add_argument("split_report", type=Path)
    create.add_argument("media_dir", type=Path)
    create.add_argument("queue", type=Path)
    create.add_argument("key", type=Path)
    create.add_argument("--seed", type=int, required=True)
    reveal = commands.add_parser("reveal")
    reveal.add_argument("queue", type=Path)
    reveal.add_argument("key", type=Path)
    reveal.add_argument("media_dir", type=Path)
    reveal.add_argument("output", type=Path)
    reveal.add_argument("--decisions", type=Path, nargs="+", required=True)
    spec = commands.add_parser("spec")
    spec.add_argument("split_manifest", type=Path)
    spec.add_argument("split_report", type=Path)
    spec.add_argument("output", type=Path)
    spec.add_argument("--checkpoint", type=Path, required=True)
    spec.add_argument("--weights", choices=("ema", "model"), default="ema")
    spec.add_argument("--sampling-steps", type=int, required=True)
    spec.add_argument("--start-timestep", type=int)
    spec.add_argument("--cfg-scale", type=float, required=True)
    spec.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args(argv)
    if args.command == "spec":
        if args.output.resolve() in {
            args.split_manifest.resolve(), args.split_report.resolve(), args.checkpoint.resolve(),
        }:
            raise ValueError("generation evaluation spec output must differ from its inputs")
        contract = checkpoint_generation_contract(
            args.checkpoint, weights=args.weights,
            sampling_steps=args.sampling_steps,
            start_timestep=args.start_timestep, cfg_scale=args.cfg_scale,
        )
        value = build_generation_evaluation_spec(
            args.split_manifest.read_bytes(), args.split_report.read_bytes(),
            generation_contract=contract, seeds=args.seeds,
        )
        _atomic_json(args.output, value)
        return 0
    if args.command == "create":
        inputs = {
            args.evaluation_spec.resolve(), args.split_manifest.resolve(),
            args.split_report.resolve(),
        }
        outputs = {args.media_dir.resolve(), args.queue.resolve(), args.key.resolve()}
        if (
            len(outputs) != 3 or inputs & outputs
            or any(_within(path, args.generation_dir) for path in (args.media_dir, args.queue, args.key))
            or _within(args.queue, args.media_dir) or _within(args.key, args.media_dir)
        ):
            raise ValueError("blind generation outputs must differ from inputs and each other")
        if args.media_dir.exists() or args.queue.exists() or args.key.exists():
            raise ValueError("blind generation outputs must not already exist")
        queue, key = create_blind_generation_pool(
            args.generation_dir, args.evaluation_spec.read_bytes(),
            args.split_manifest.read_bytes(), args.split_report.read_bytes(), args.media_dir,
            seed=args.seed,
        )
        try:
            _atomic_pair(args.queue, queue, args.key, key)
        except Exception:
            shutil.rmtree(args.media_dir, ignore_errors=True)
            raise
        return 0
    inputs = {args.queue.resolve(), args.key.resolve(), *(path.resolve() for path in args.decisions)}
    if args.output.resolve() in inputs or _within(args.output, args.media_dir):
        raise ValueError("blind reveal output must differ from every input")
    report = reveal_blind_generation_results(
        args.queue.read_bytes(), args.key.read_bytes(),
        [path.read_bytes() for path in args.decisions], args.media_dir,
    )
    _atomic_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
