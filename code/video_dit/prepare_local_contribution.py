"""Create an atomic, rights-locked bundle from one contributor-owned video."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

from real_data import (
    RealVideoRecord,
    decode_video_clip,
    load_manifest,
    perceptual_hash,
    write_manifest,
)


_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_SPEC_FIELDS = {
    "schema_version", "contribution_id", "media_path", "subject", "action",
    "scene", "prompt", "clip_start_seconds",
}
_RECEIPT_FIELDS = {
    "schema_version", "contribution_id", "contributor_id", "animal_id",
    "capture_session_id", "rights_holder", "accepted_at", "recorded_at",
    "media_sha256", "license_id", "license_url", "attribution", "grants",
}
_REQUIRED_GRANTS = {
    "model_training", "create_derivatives", "redistribute_normalized_clips",
}
_REVIEW_FIELDS = {
    "schema_version", "contribution_id", "reviewer_id", "reviewed_at",
    "source_manifest_sha256", "rights_receipt_sha256", "clip_sha256",
    "full_contact_sha256", "selected_window_sha256", "subject", "action",
    "scene", "prompt", "full_duration_reviewed", "single_subject_confirmed",
    "continuous_action_confirmed", "scene_confirmed", "no_edit_in_window",
    "review_notes",
}


@dataclass(frozen=True)
class RightsContract:
    contribution_id: str
    contributor_id: str
    animal_id: str
    capture_session_id: str
    rights_holder: str
    accepted_at: str
    recorded_at: str
    media_sha256: str
    license_id: str
    license_url: str
    attribution: str
    receipt_sha256: str
    receipt_bytes: bytes

    @property
    def source_group(self) -> str:
        return "local-contributor:" + hashlib.sha256(
            self.contributor_id.encode()
        ).hexdigest()

    @property
    def session_group(self) -> str:
        identity = "\0".join((
            self.contributor_id, self.animal_id, self.capture_session_id,
        ))
        return "local-session:" + hashlib.sha256(identity.encode()).hexdigest()


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_object(path: Path, *, fields: set[str], name: str) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=_object_without_duplicate_keys)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {name} JSON: {error}") from error
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} must contain exactly {sorted(fields)}")
    return value, raw


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _stable_id(value: object, name: str) -> str:
    result = _text(value, name)
    if not _ID.fullmatch(result):
        raise ValueError(f"{name} must use lowercase letters, digits, dots, dashes, or underscores")
    return result


def _utc(value: object, name: str) -> str:
    text = _text(value, name)
    if not text.endswith("Z") or "T" not in text:
        raise ValueError(f"{name} must be a UTC RFC3339 timestamp")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError:
        raise ValueError(f"{name} must be a UTC RFC3339 timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{name} must be a UTC RFC3339 timestamp")
    return text


def _validate_rights_receipt(
    path: Path,
    *,
    expected_contribution_id: str | None = None,
    expected_media_sha256: str | None = None,
) -> RightsContract:
    receipt, receipt_bytes = _load_object(
        path, fields=_RECEIPT_FIELDS, name="rights receipt",
    )
    if receipt["schema_version"] != 1:
        raise ValueError("rights receipt schema_version must be 1")
    contribution_id = _stable_id(receipt["contribution_id"], "contribution_id")
    media_sha256 = str(receipt["media_sha256"])
    if not _HEX64.fullmatch(media_sha256):
        raise ValueError("rights receipt media_sha256 must be lowercase SHA-256")
    if expected_contribution_id is not None and contribution_id != expected_contribution_id:
        raise ValueError("rights receipt contribution_id does not match the contribution")
    if expected_media_sha256 is not None and media_sha256 != expected_media_sha256:
        raise ValueError("rights receipt media_sha256 does not match the source video")
    grants = receipt["grants"]
    if not isinstance(grants, dict) or set(grants) != _REQUIRED_GRANTS \
            or any(grants[name] is not True for name in _REQUIRED_GRANTS):
        raise ValueError(
            "rights receipt must explicitly grant training, derivatives, and redistribution"
        )
    return RightsContract(
        contribution_id=contribution_id,
        contributor_id=_stable_id(receipt["contributor_id"], "contributor_id"),
        animal_id=_stable_id(receipt["animal_id"], "animal_id"),
        capture_session_id=_stable_id(
            receipt["capture_session_id"], "capture_session_id",
        ),
        rights_holder=_text(receipt["rights_holder"], "rights_holder"),
        accepted_at=_utc(receipt["accepted_at"], "accepted_at"),
        recorded_at=_utc(receipt["recorded_at"], "recorded_at"),
        media_sha256=media_sha256,
        license_id=_text(receipt["license_id"], "license_id"),
        license_url=_text(receipt["license_url"], "license_url"),
        attribution=_text(receipt["attribution"], "attribution"),
        receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        receipt_bytes=receipt_bytes,
    )


def _validate_record_rights(record: RealVideoRecord, rights: RightsContract) -> None:
    expected = {
        "contribution_id": rights.contribution_id,
        "rights_receipt_sha256": rights.receipt_sha256,
        "downloaded_media_sha256": rights.media_sha256,
        "source_id": rights.source_group,
        "duplicate_group": rights.session_group,
        "license": rights.license_id,
        "license_url": rights.license_url,
        "attribution": f"{rights.attribution}; {rights.license_url}",
        "original_media_timestamp": rights.recorded_at,
    }
    for field, value in expected.items():
        if getattr(record, field) != value:
            raise ValueError(f"manifest {field} does not match its rights receipt")


def _probe_duration(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=True, capture_output=True, text=True, timeout=120,
    )
    try:
        duration = float(completed.stdout.strip())
    except ValueError:
        raise ValueError("ffprobe returned an invalid source duration") from None
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("source video duration must be positive and finite")
    return duration


def _normalize(source: Path, destination: Path, start: float) -> None:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(start), "-i", str(source), "-t", "2", "-an",
            "-vf", "fps=8,scale=96:96:force_original_aspect_ratio=increase,crop=96:96",
            "-frames:v", "16", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(destination),
        ],
        check=True, timeout=120,
    )
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,width,height", "-of", "json",
            str(destination),
        ],
        check=True, capture_output=True, text=True, timeout=120,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    if (
        int(stream["nb_read_frames"]) != 16
        or int(stream["width"]) != 96
        or int(stream["height"]) != 96
    ):
        raise ValueError("normalized contribution must be exactly 16 frames at 96x96")


def _write_review_sheets(source: Path, clip: Path, review_dir: Path, duration: float) -> dict[str, str]:
    review_dir.mkdir()
    full = review_dir / "full-contact.jpg"
    selected = review_dir / "selected-window.jpg"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(source), "-vf",
            f"fps=24/{duration},scale=240:-1,tile=4x6", "-frames:v", "1", str(full),
        ],
        check=True, timeout=120,
    )
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(clip), "-vf",
            "scale=240:-1,tile=4x4", "-frames:v", "1", str(selected),
        ],
        check=True, timeout=120,
    )
    return {
        "full_contact_sha256": hashlib.sha256(full.read_bytes()).hexdigest(),
        "selected_window_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
    }


def create_bundle(spec_path: Path, receipt_path: Path, output_dir: Path) -> dict[str, object]:
    spec, _ = _load_object(spec_path, fields=_SPEC_FIELDS, name="contribution spec")
    if spec["schema_version"] != 1:
        raise ValueError("contribution spec schema_version must be 1")
    contribution_id = _stable_id(spec["contribution_id"], "contribution_id")
    relative_media = Path(_text(spec["media_path"], "media_path"))
    if relative_media.is_absolute() or ".." in relative_media.parts:
        raise ValueError("media_path must be a safe path relative to the spec directory")
    media = (spec_path.parent / relative_media).resolve()
    if not media.is_file() or not media.is_relative_to(spec_path.parent.resolve()):
        raise ValueError("media_path must resolve to a file inside the spec directory")
    media_sha256 = hashlib.sha256(media.read_bytes()).hexdigest()
    rights = _validate_rights_receipt(
        receipt_path,
        expected_contribution_id=contribution_id,
        expected_media_sha256=media_sha256,
    )
    for name in ("subject", "action", "scene", "prompt"):
        _text(spec[name], name)
    try:
        start = float(spec["clip_start_seconds"])
    except (TypeError, ValueError):
        raise ValueError("clip_start_seconds must be finite and non-negative") from None
    if not math.isfinite(start) or start < 0:
        raise ValueError("clip_start_seconds must be finite and non-negative")
    duration = _probe_duration(media)
    if start + 2 > duration + 1e-6:
        raise ValueError("selected two-second clip exceeds the source duration")
    if output_dir.exists():
        raise ValueError("contribution output directory must not already exist")
    staging = output_dir.with_name(output_dir.name + ".tmp")
    if staging.exists():
        raise ValueError("contribution staging directory already exists")
    sample_id = f"local-{contribution_id}"
    try:
        (staging / "clips").mkdir(parents=True)
        (staging / "receipts").mkdir()
        (staging / "source-media").mkdir()
        clip_relative = Path("clips") / f"{sample_id}.mp4"
        receipt_relative = Path("receipts") / f"{contribution_id}.json"
        source_media_relative = Path("source-media") / f"{contribution_id}{media.suffix.lower()}"
        _normalize(media, staging / clip_relative, start)
        review_hashes = _write_review_sheets(
            media, staging / clip_relative, staging / "review", duration,
        )
        (staging / receipt_relative).write_bytes(rights.receipt_bytes)
        shutil.copy2(media, staging / source_media_relative)
        clip_sha256 = hashlib.sha256((staging / clip_relative).read_bytes()).hexdigest()
        phash = perceptual_hash(
            decode_video_clip(staging / clip_relative, frames=16, size=96)
        )
        record = RealVideoRecord(
            sample_id=sample_id,
            source_id=rights.source_group,
            clip_path=clip_relative.as_posix(),
            prompt=str(spec["prompt"]).strip(),
            subject=str(spec["subject"]).strip(),
            action=str(spec["action"]).strip(),
            scene=str(spec["scene"]).strip(),
            source_url=f"urn:local-contribution:{contribution_id}",
            license=rights.license_id,
            attribution=f"{rights.attribution}; {rights.license_url}",
            sha256=clip_sha256,
            license_url=rights.license_url,
            original_media_url=f"urn:sha256:{media_sha256}",
            original_media_timestamp=rights.recorded_at,
            downloaded_media_url=f"urn:sha256:{media_sha256}",
            downloaded_media_kind="contributor_original",
            downloaded_media_sha256=media_sha256,
            downloaded_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            contribution_id=contribution_id,
            rights_receipt_path=receipt_relative.as_posix(),
            rights_receipt_sha256=rights.receipt_sha256,
            clip_start_seconds=start,
            clip_frames=16,
            clip_fps=8,
            clip_size=96,
            duplicate_group=rights.session_group,
            perceptual_hash=phash,
        )
        manifest = staging / "source.jsonl"
        write_manifest(manifest, [record])
        evidence = {
            "accepted_at": rights.accepted_at,
            "animal_id_sha256": hashlib.sha256(rights.animal_id.encode()).hexdigest(),
            "capture_session_id_sha256": hashlib.sha256(
                rights.capture_session_id.encode()
            ).hexdigest(),
            "clip_sha256": clip_sha256,
            "contribution_id": contribution_id,
            "contributor_id_sha256": hashlib.sha256(
                rights.contributor_id.encode()
            ).hexdigest(),
            "media_sha256": media_sha256,
            "source_media_path": source_media_relative.as_posix(),
            "source_media_sha256": media_sha256,
            "rights_holder": rights.rights_holder,
            "rights_receipt_sha256": rights.receipt_sha256,
            "review_artifacts": review_hashes,
            "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "training_blocker": "full-duration human factor review is required",
            "training_ready": False,
        }
        (staging / "evidence.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        staging.replace(output_dir)
        return evidence
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _locked_file(root: Path, relative: str, expected_sha256: str, name: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_file() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"{name} must resolve to a file inside its dataset root")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError(f"{name} SHA-256 does not match its manifest")
    return path


def _validate_human_review(
    review_path: Path,
    record: RealVideoRecord,
    manifest_sha256: str,
    full_contact_sha256: str,
    selected_window_sha256: str,
) -> tuple[str, str, bytes]:
    review, review_bytes = _load_object(
        review_path, fields=_REVIEW_FIELDS, name="human review",
    )
    if review["schema_version"] != 1 or review["contribution_id"] != record.contribution_id:
        raise ValueError("human review does not match the contribution")
    expected = {
        "source_manifest_sha256": manifest_sha256,
        "rights_receipt_sha256": record.rights_receipt_sha256,
        "clip_sha256": record.sha256,
        "full_contact_sha256": full_contact_sha256,
        "selected_window_sha256": selected_window_sha256,
        "subject": record.subject,
        "action": record.action,
        "scene": record.scene,
        "prompt": record.prompt,
    }
    for field, value in expected.items():
        if review[field] != value:
            raise ValueError(f"human review {field} does not match the staged contribution")
    for field in (
        "full_duration_reviewed", "single_subject_confirmed",
        "continuous_action_confirmed", "scene_confirmed", "no_edit_in_window",
    ):
        if review[field] is not True:
            raise ValueError(f"human review must explicitly confirm {field}")
    reviewer = _text(review["reviewer_id"], "reviewer_id")
    reviewed_at = _utc(review["reviewed_at"], "reviewed_at")
    _text(review["review_notes"], "review_notes")
    return reviewer, reviewed_at, review_bytes


def approve_bundle(
    bundle_dir: Path, review_path: Path, output_dir: Path,
) -> dict[str, object]:
    bundle_resolved = bundle_dir.resolve()
    output_resolved = output_dir.resolve()
    if output_resolved == bundle_resolved or output_resolved.is_relative_to(bundle_resolved):
        raise ValueError("approved output must be outside the staged contribution bundle")
    if output_dir.exists():
        raise ValueError("approved output directory must not already exist")
    staging = output_dir.with_name(output_dir.name + ".tmp")
    if staging.exists():
        raise ValueError("approved output staging directory already exists")
    manifest_path = bundle_dir / "source.jsonl"
    evidence_path = bundle_dir / "evidence.json"
    if not manifest_path.is_file() or not evidence_path.is_file():
        raise ValueError("staged contribution lacks source.jsonl or evidence.json")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    try:
        evidence = json.loads(
            evidence_path.read_bytes(), object_pairs_hook=_object_without_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid staged evidence JSON: {error}") from error
    if not isinstance(evidence, dict) or evidence.get("training_ready") is not False \
            or evidence.get("source_manifest_sha256") != manifest_sha256:
        raise ValueError("staged evidence must bind a non-training-ready manifest")
    records = load_manifest(manifest_path)
    if len(records) != 1:
        raise ValueError("staged contribution must contain exactly one record")
    record = records[0]
    if record.contribution_id is None or record.rights_receipt_path is None \
            or record.rights_receipt_sha256 is None:
        raise ValueError("staged contribution lacks a complete rights receipt contract")
    _locked_file(bundle_dir, record.clip_path, record.sha256, "clip")
    locked_receipt = _locked_file(
        bundle_dir, record.rights_receipt_path, record.rights_receipt_sha256,
        "rights receipt",
    )
    rights = _validate_rights_receipt(
        locked_receipt,
        expected_contribution_id=record.contribution_id,
        expected_media_sha256=record.downloaded_media_sha256,
    )
    _validate_record_rights(record, rights)
    if evidence.get("rights_receipt_sha256") != rights.receipt_sha256 \
            or evidence.get("media_sha256") != rights.media_sha256 \
            or evidence.get("source_media_sha256") != rights.media_sha256 \
            or evidence.get("rights_holder") != rights.rights_holder \
            or evidence.get("accepted_at") != rights.accepted_at:
        raise ValueError("staged evidence does not match its rights receipt")
    source_media_relative = evidence.get("source_media_path")
    if not isinstance(source_media_relative, str):
        raise ValueError("staged evidence lacks its full source media")
    _locked_file(
        bundle_dir, source_media_relative, rights.media_sha256, "full source media",
    )
    artifacts = evidence.get("review_artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("staged evidence lacks review artifacts")
    full_hash = str(artifacts.get("full_contact_sha256", ""))
    selected_hash = str(artifacts.get("selected_window_sha256", ""))
    _locked_file(bundle_dir, "review/full-contact.jpg", full_hash, "full contact sheet")
    _locked_file(bundle_dir, "review/selected-window.jpg", selected_hash, "selected window sheet")
    reviewer, reviewed_at, review_bytes = _validate_human_review(
        review_path, record, manifest_sha256, full_hash, selected_hash,
    )
    review_sha256 = hashlib.sha256(review_bytes).hexdigest()
    try:
        shutil.copytree(bundle_dir, staging)
        (staging / "reviews").mkdir()
        review_relative = Path("reviews") / f"{record.contribution_id}.json"
        (staging / review_relative).write_bytes(review_bytes)
        approved_evidence = {
            **evidence,
            "human_review_path": review_relative.as_posix(),
            "human_review_sha256": review_sha256,
            "reviewed_at": reviewed_at,
            "reviewer_id_sha256": hashlib.sha256(reviewer.encode()).hexdigest(),
            "training_ready": True,
        }
        approved_evidence.pop("training_blocker", None)
        (staging / "evidence.json").write_text(
            json.dumps(approved_evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output_dir)
        return approved_evidence
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def merge_bundles(
    base_manifest: Path, bundle_dirs: list[Path], output_dir: Path,
) -> dict[str, object]:
    if not bundle_dirs:
        raise ValueError("merge requires at least one contribution bundle")
    if (base_manifest.parent / "merge-evidence.json").exists() \
            or (base_manifest.parent / "contribution-audit").exists():
        raise ValueError(
            "merge base must be the fixed pristine dataset; pass all approved "
            "contribution bundles together instead of incrementally merging"
        )
    input_roots = [base_manifest.parent.resolve(), *(path.resolve() for path in bundle_dirs)]
    output_resolved = output_dir.resolve()
    if any(output_resolved == root or output_resolved.is_relative_to(root) for root in input_roots):
        raise ValueError("merge output must be outside every input dataset tree")
    if output_dir.exists():
        raise ValueError("merged output directory must not already exist")
    staging = output_dir.with_name(output_dir.name + ".tmp")
    if staging.exists():
        raise ValueError("merged output staging directory already exists")
    base_bytes = base_manifest.read_bytes()
    records_with_roots: list[tuple[RealVideoRecord, Path]] = [
        (record, base_manifest.parent) for record in load_manifest(base_manifest)
    ]
    bundle_contracts: list[dict[str, str]] = []
    for bundle in bundle_dirs:
        manifest = bundle / "source.jsonl"
        evidence_path = bundle / "evidence.json"
        if not manifest.is_file() or not evidence_path.is_file():
            raise ValueError("contribution bundle lacks source.jsonl or evidence.json")
        manifest_bytes = manifest.read_bytes()
        try:
            evidence = json.loads(
                evidence_path.read_bytes(), object_pairs_hook=_object_without_duplicate_keys,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid approved evidence JSON: {error}") from error
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if not isinstance(evidence, dict) or evidence.get("training_ready") is not True \
                or evidence.get("source_manifest_sha256") != manifest_sha256:
            raise ValueError("contribution bundle evidence does not bind a training-ready manifest")
        bundle_records = load_manifest(manifest)
        if len(bundle_records) != 1:
            raise ValueError("approved contribution bundle must contain exactly one record")
        if any(record.rights_receipt_path is None for record in bundle_records):
            raise ValueError("contribution bundle records require a rights receipt")
        record = bundle_records[0]
        if record.contribution_id is None or record.rights_receipt_path is None \
                or record.rights_receipt_sha256 is None:
            raise ValueError("approved contribution lacks a complete rights receipt contract")
        locked_receipt = _locked_file(
            bundle, record.rights_receipt_path, record.rights_receipt_sha256,
            "rights receipt",
        )
        rights = _validate_rights_receipt(
            locked_receipt,
            expected_contribution_id=record.contribution_id,
            expected_media_sha256=record.downloaded_media_sha256,
        )
        _validate_record_rights(record, rights)
        if evidence.get("rights_receipt_sha256") != rights.receipt_sha256 \
                or evidence.get("media_sha256") != rights.media_sha256 \
                or evidence.get("source_media_sha256") != rights.media_sha256 \
                or evidence.get("rights_holder") != rights.rights_holder \
                or evidence.get("accepted_at") != rights.accepted_at:
            raise ValueError("approved evidence does not match its rights receipt")
        source_media_relative = evidence.get("source_media_path")
        if not isinstance(source_media_relative, str):
            raise ValueError("approved evidence lacks its full source media")
        _locked_file(
            bundle, source_media_relative, rights.media_sha256, "full source media",
        )
        artifacts = evidence.get("review_artifacts")
        review_relative = evidence.get("human_review_path")
        review_sha256 = str(evidence.get("human_review_sha256", ""))
        if not isinstance(artifacts, dict) or not isinstance(review_relative, str):
            raise ValueError("approved contribution evidence lacks human review artifacts")
        full_hash = str(artifacts.get("full_contact_sha256", ""))
        selected_hash = str(artifacts.get("selected_window_sha256", ""))
        _locked_file(bundle, "review/full-contact.jpg", full_hash, "full contact sheet")
        _locked_file(bundle, "review/selected-window.jpg", selected_hash, "selected window sheet")
        locked_review = _locked_file(bundle, review_relative, review_sha256, "human review")
        reviewer, reviewed_at, _ = _validate_human_review(
            locked_review, record, manifest_sha256, full_hash, selected_hash,
        )
        if evidence.get("reviewed_at") != reviewed_at or evidence.get(
            "reviewer_id_sha256"
        ) != hashlib.sha256(reviewer.encode()).hexdigest():
            raise ValueError("approved evidence does not bind its reviewer contract")
        records_with_roots.extend((record, bundle) for record in bundle_records)
        bundle_contracts.append({
            "bundle": bundle.name,
            "evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            "source_manifest_sha256": manifest_sha256,
        })
    sample_ids = [record.sample_id for record, _ in records_with_roots]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("base and contribution manifests contain duplicate sample_id values")
    try:
        (staging / "clips").mkdir(parents=True)
        (staging / "receipts").mkdir()
        (staging / "contribution-audit").mkdir()
        merged_records: list[RealVideoRecord] = []
        for record, root in records_with_roots:
            source_clip = _locked_file(root, record.clip_path, record.sha256, "clip")
            suffix = source_clip.suffix.lower()
            destination_clip = Path("clips") / (
                hashlib.sha256(record.sample_id.encode()).hexdigest()[:24] + suffix
            )
            if (staging / destination_clip).exists():
                raise ValueError("merged clip destination collision")
            shutil.copy2(source_clip, staging / destination_clip)
            destination_receipt: Path | None = None
            if record.rights_receipt_path is not None:
                assert record.rights_receipt_sha256 is not None
                source_receipt = _locked_file(
                    root, record.rights_receipt_path, record.rights_receipt_sha256,
                    "rights receipt",
                )
                assert record.contribution_id is not None
                destination_receipt = Path("receipts") / (
                    hashlib.sha256(record.contribution_id.encode()).hexdigest()[:24] + ".json"
                )
                if (staging / destination_receipt).exists():
                    raise ValueError("contribution bundles contain duplicate contribution_id values")
                shutil.copy2(source_receipt, staging / destination_receipt)
            merged_records.append(replace(
                record,
                clip_path=destination_clip.as_posix(),
                rights_receipt_path=(
                    destination_receipt.as_posix() if destination_receipt is not None else None
                ),
            ))
        for bundle, contract in zip(bundle_dirs, bundle_contracts, strict=True):
            audit_name = hashlib.sha256(
                contract["source_manifest_sha256"].encode()
            ).hexdigest()[:24]
            audit_destination = staging / "contribution-audit" / audit_name
            shutil.copytree(bundle, audit_destination)
            contract["portable_audit_path"] = audit_destination.relative_to(
                staging
            ).as_posix()
        manifest = staging / "source.jsonl"
        write_manifest(manifest, merged_records)
        report = {
            "base_manifest_sha256": hashlib.sha256(base_bytes).hexdigest(),
            "contribution_bundles": len(bundle_dirs),
            "contribution_contracts": bundle_contracts,
            "records": len(merged_records),
            "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "training_ready": True,
        }
        (staging / "merge-evidence.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        staging.replace(output_dir)
        return report
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("spec", type=Path)
    create.add_argument("receipt", type=Path)
    create.add_argument("output", type=Path)
    approve = subparsers.add_parser("approve")
    approve.add_argument("bundle", type=Path)
    approve.add_argument("review", type=Path)
    approve.add_argument("output", type=Path)
    merge = subparsers.add_parser("merge")
    merge.add_argument("base_manifest", type=Path)
    merge.add_argument("bundles", nargs="+", type=Path)
    merge.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "create":
        create_bundle(args.spec, args.receipt, args.output)
        return 0
    if args.command == "approve":
        approve_bundle(args.bundle, args.review, args.output)
        return 0
    if args.command == "merge":
        merge_bundles(args.base_manifest, args.bundles, args.output)
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    sys.exit(main())
