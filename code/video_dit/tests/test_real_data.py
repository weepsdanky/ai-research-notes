import json
from pathlib import Path
import shutil
import hashlib

import pytest
import torch

from real_data import (
    CompositionalSplit,
    RealVideoDataset,
    RealVideoRecord,
    decode_video_clip,
    perceptual_hash,
    load_manifest,
    split_manifest,
    validate_split,
    write_manifest,
)


def _record(
    sample_id: str,
    subject: str,
    action: str,
    scene: str,
    *,
    source_id: str | None = None,
    duplicate_group: str | None = None,
) -> RealVideoRecord:
    return RealVideoRecord(
        sample_id=sample_id,
        source_id=source_id or sample_id,
        clip_path=f"clips/{sample_id}.mp4",
        prompt=f"a {subject} {action} in {scene}",
        subject=subject,
        action=action,
        scene=scene,
        source_url=f"https://example.test/{sample_id}",
        license="CC-BY-4.0",
        attribution="Example Author",
        sha256=(sample_id.encode().hex() + "0" * 64)[:64],
        duplicate_group=duplicate_group,
        perceptual_hash=hashlib.sha256(sample_id.encode()).hexdigest()[:16],
    )


def _factorial_records() -> list[RealVideoRecord]:
    records = []
    index = 0
    for subject in ("cat", "dog"):
        for action in ("walking", "running"):
            for scene in ("grass", "snow"):
                for repeat in range(3):
                    index += 1
                    records.append(
                        _record(
                            f"v{index}", subject, action, scene,
                            source_id=f"source-{index}",
                        )
                    )
    return records


def test_manifest_round_trip_preserves_provenance_and_rejects_missing_license(tmp_path):
    path = tmp_path / "manifest.jsonl"
    expected = [_record("cat-1", "cat", "walking", "grass")]

    write_manifest(path, expected)

    assert load_manifest(path) == expected
    payload = json.loads(path.read_text().splitlines()[0])
    payload["license"] = ""
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="license"):
        load_manifest(path)


def test_split_holds_out_unseen_triple_but_keeps_all_atoms_and_pairs_in_train():
    records = _factorial_records()
    policy = CompositionalSplit(
        c1_triples={("cat", "running", "snow")},
        c2_pairs=set(),
        iid_fraction=0.2,
        validation_fraction=0.1,
        seed=7,
    )

    assigned = split_manifest(records, policy)
    report = validate_split(assigned, policy)

    assert report["valid"] is True
    assert {
        (r.subject, r.action, r.scene)
        for r in assigned if r.split == "c1"
    } == {("cat", "running", "snow")}
    assert ("cat", "running", "snow") not in {
        (r.subject, r.action, r.scene)
        for r in assigned if r.split == "train"
    }


def test_source_and_near_duplicate_groups_can_never_cross_splits():
    records = _factorial_records()
    records.extend(
        [
            _record(
                "same-source-a", "dog", "walking", "grass",
                source_id="shared-source",
            ),
            _record(
                "same-source-b", "dog", "walking", "grass",
                source_id="shared-source",
            ),
            _record(
                "near-a", "cat", "walking", "grass",
                duplicate_group="duplicate-17",
            ),
            _record(
                "near-b", "cat", "walking", "grass",
                duplicate_group="duplicate-17",
            ),
        ]
    )
    policy = CompositionalSplit(
        c1_triples={("cat", "running", "snow")},
        c2_pairs=set(),
        iid_fraction=0.4,
        validation_fraction=0.2,
        seed=19,
    )

    assigned = split_manifest(records, policy)

    assert len({r.split for r in assigned if r.source_id == "shared-source"}) == 1
    assert len({r.split for r in assigned if r.duplicate_group == "duplicate-17"}) == 1


def test_validator_detects_cross_split_near_duplicate_fingerprints():
    train = _record("train", "cat", "walking", "grass")
    iid = _record("iid", "dog", "running", "snow")
    train = __import__("dataclasses").replace(
        train, split="train", perceptual_hash="0000000000000000"
    )
    iid = __import__("dataclasses").replace(
        iid, split="iid", perceptual_hash="0000000000000001"
    )
    policy = CompositionalSplit(
        c1_triples=set(), c2_pairs=set(), iid_fraction=0, validation_fraction=0,
        near_duplicate_hamming=2,
    )

    with pytest.raises(ValueError, match="near-duplicate"):
        validate_split([train, iid], policy)


def test_validator_fails_when_a_c1_pair_projection_was_not_seen_in_training():
    records = [
        _record("held", "cat", "running", "snow"),
        _record("a", "cat", "walking", "grass"),
        _record("b", "dog", "running", "grass"),
        _record("c", "dog", "walking", "snow"),
    ]
    policy = CompositionalSplit(
        c1_triples={("cat", "running", "snow")},
        c2_pairs=set(),
        iid_fraction=0,
        validation_fraction=0,
        seed=0,
    )

    assigned = split_manifest(records, policy)

    with pytest.raises(ValueError, match="pair projection"):
        validate_split(assigned, policy)


def test_split_rejects_source_group_mixing_held_and_nonheld_triples():
    records = [
        _record("held", "cat", "running", "snow", source_id="mixed"),
        _record("other", "cat", "walking", "snow", source_id="mixed"),
    ]
    policy = CompositionalSplit(
        c1_triples={("cat", "running", "snow")}, c2_pairs=set(),
        iid_fraction=0, validation_fraction=0,
    )

    with pytest.raises(ValueError, match="mixes held-out and ordinary"):
        split_manifest(records, policy)


def test_validator_requires_every_declared_c1_to_exist_in_c1_records():
    records = [
        __import__("dataclasses").replace(
            _record("train", "cat", "walking", "grass"), split="train"
        )
    ]
    policy = CompositionalSplit(
        c1_triples={("cat", "running", "snow")}, c2_pairs=set(),
        iid_fraction=0, validation_fraction=0,
    )

    with pytest.raises(ValueError, match="has no evaluation records"):
        validate_split(records, policy)


def test_validator_requires_fingerprints_for_every_record():
    record = __import__("dataclasses").replace(
        _record("train", "cat", "walking", "grass"),
        split="train", perceptual_hash=None,
    )
    policy = CompositionalSplit(
        c1_triples=set(), c2_pairs=set(), iid_fraction=0, validation_fraction=0,
    )

    with pytest.raises(ValueError, match="perceptual_hash"):
        validate_split([record], policy)


def test_validator_requires_every_declared_c2_pair_in_c2_records():
    records = [
        __import__("dataclasses").replace(
            _record("cat", "cat", "walking", "grass"), split="train"
        ),
        __import__("dataclasses").replace(
            _record("dog", "dog", "running", "snow"), split="train"
        ),
    ]
    policy = CompositionalSplit(
        c1_triples=set(), c2_pairs={("dog", "grass")},
        iid_fraction=0, validation_fraction=0,
    )

    with pytest.raises(ValueError, match="C2 pair has no evaluation records"):
        validate_split(records, policy)


def test_validator_rejects_c2_records_not_matching_any_declared_pair():
    records = [
        __import__("dataclasses").replace(
            _record("cat", "cat", "walking", "grass"), split="train"
        ),
        __import__("dataclasses").replace(
            _record("dog", "dog", "running", "snow"), split="c2"
        ),
    ]
    policy = CompositionalSplit(
        c1_triples=set(), c2_pairs={("dog", "grass")},
        iid_fraction=0, validation_fraction=0,
    )

    with pytest.raises(ValueError, match="no declared pair"):
        validate_split(records, policy)


def test_real_video_dataset_decodes_normalized_temporal_clip(tmp_path):
    source = Path(__file__).parents[1] / "demo_artifacts" / "samples" / (
        "model_person_red_walk_right_snow_seed-42.mp4"
    )
    clip = tmp_path / "person.mp4"
    shutil.copyfile(source, clip)
    import hashlib

    digest = hashlib.sha256(clip.read_bytes()).hexdigest()
    record = RealVideoRecord(
        sample_id="person-1",
        source_id="generated-demo-person",
        clip_path=clip.name,
        prompt="a person walking in snow",
        subject="person",
        action="walking",
        scene="snow",
        source_url="repo://video_dit/demo_artifacts/person",
        license="repository-demo",
        attribution="ai-research-notes",
        sha256=digest,
        split="train",
    )
    manifest = write_manifest(tmp_path / "manifest.jsonl", [record])

    dataset = RealVideoDataset(manifest, split="train", frames=8, size=32)
    sample = dataset[0]

    assert sample["video"].shape == (3, 8, 32, 32)
    assert sample["video"].dtype == torch.float32
    assert sample["video"].min() >= -1 and sample["video"].max() <= 1
    assert not torch.equal(sample["video"][:, 0], sample["video"][:, -1])
    assert sample["prompt"] == "a person walking in snow"
    assert sample["factors"].tolist() == [0, 0, 0]


def test_decoder_fails_closed_when_clip_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        decode_video_clip(tmp_path / "missing.mp4", frames=8, size=32)


def test_perceptual_hash_is_stable_and_changes_for_visually_different_video():
    first = torch.zeros(3, 4, 16, 16)
    second = first.clone()
    second[:, :, :, 8:] = 1

    assert perceptual_hash(first) == perceptual_hash(first.clone())
    assert perceptual_hash(first) != perceptual_hash(second)


def test_dataset_returns_cached_frozen_text_embedding(tmp_path):
    source = Path(__file__).parents[1] / "demo_artifacts" / "samples" / (
        "model_cat_orange_walk_right_snow_seed-42.mp4"
    )
    clip = tmp_path / "cat.mp4"
    shutil.copyfile(source, clip)
    embedding_path = tmp_path / "cat.text.pt"
    torch.save(torch.arange(12, dtype=torch.float32), embedding_path)
    import hashlib

    record = RealVideoRecord(
        sample_id="cat-1",
        source_id="generated-demo-cat",
        clip_path=clip.name,
        prompt="an orange cat walking in snow",
        subject="cat",
        action="walking",
        scene="snow",
        source_url="repo://video_dit/demo_artifacts/cat",
        license="repository-demo",
        attribution="ai-research-notes",
        sha256=hashlib.sha256(clip.read_bytes()).hexdigest(),
        text_embedding_path=embedding_path.name,
        text_encoder_id="openai/clip-vit-base-patch32@main",
        split="train",
    )
    manifest = write_manifest(tmp_path / "manifest.jsonl", [record])

    sample = RealVideoDataset(manifest, split="train", frames=8, size=32)[0]

    assert torch.equal(sample["condition"], torch.arange(12, dtype=torch.float32))
    assert sample["text_encoder_id"] == "openai/clip-vit-base-patch32@main"


def test_dataset_reads_cached_vae_latent_without_decoding_pixels(tmp_path):
    latent_path = tmp_path / "cat.latent.pt"
    expected = torch.randn(4, 8, 12, 12)
    torch.save(expected, latent_path)
    record = RealVideoRecord(
        sample_id="cat-latent",
        source_id="source-cat-latent",
        clip_path="clips/not-needed-at-training-time.mp4",
        prompt="a cat running on grass",
        subject="cat",
        action="running",
        scene="grass",
        source_url="https://example.test/cat-latent",
        license="CC-BY-4.0",
        attribution="Example",
        sha256="a" * 64,
        latent_path=latent_path.name,
        vae_id="stabilityai/sd-vae-ft-mse@revision",
        vae_scaling=0.18215,
        split="train",
    )
    manifest = write_manifest(tmp_path / "manifest.jsonl", [record])

    sample = RealVideoDataset(
        manifest, split="train", frames=8, size=12, space="latent"
    )[0]

    assert torch.equal(sample["video"], expected)
    assert sample["vae_id"] == "stabilityai/sd-vae-ft-mse@revision"
