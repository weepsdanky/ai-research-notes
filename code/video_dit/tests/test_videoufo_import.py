import csv
import hashlib
import io
import json
import tarfile

import pytest

from prepare_videoufo import main
from videoufo_import import (
    VIDEOUFO_REVISION,
    extract_resolved_clip,
    index_tar_shard,
    resolve_candidates,
    scan_metadata,
)


FIELDS = [
    "ID", "Topic", "Brief Caption", "Detailed Caption", "Start Time",
    "End Time", "Aesthetic Quality", "Background Consistency",
    "Dynamic Degree", "Imaging Quality", "Motion Smoothness",
    "Subject Consistency",
]


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(identifier, caption, **overrides):
    row = {
        "ID": identifier,
        "Topic": "animals",
        "Brief Caption": caption,
        "Detailed Caption": caption,
        "Start Time": "0:00:00.000",
        "End Time": "0:00:04.000",
        "Aesthetic Quality": "0.7",
        "Background Consistency": "0.8",
        "Dynamic Degree": "0.6",
        "Imaging Quality": "0.7",
        "Motion Smoothness": "0.9",
        "Subject Consistency": "0.8",
    }
    row.update({key.replace("_", " "): value for key, value in overrides.items()})
    return row


def test_scan_metadata_keeps_only_unambiguous_quality_combinations(tmp_path):
    source = tmp_path / "VideoUFO.csv"
    _write_csv(source, [
        _row("cat-1", "A cat runs across a snowy field."),
        _row("ambiguous", "A cat and dog run through snow."),
        _row("static", "A dog walks on a beach.", Dynamic_Degree="0.05"),
        _row("person-1", "A woman is walking along a beach."),
    ])

    candidates, report = scan_metadata(source, expected_sha256=None)

    assert [item.video_id for item in candidates] == ["cat-1", "person-1"]
    assert candidates[0].subject == "cat"
    assert candidates[0].action == "running"
    assert candidates[0].scene == "snow"
    assert candidates[0].revision == VIDEOUFO_REVISION
    assert candidates[0].review_status == "pending"
    assert report["rows_scanned"] == 4
    assert report["rejections"] == {"ambiguous_factors": 1, "quality": 1}
    assert report["triple_counts"] == {
        "cat|running|snow": 1,
        "person|walking|beach": 1,
    }


def test_scan_metadata_uses_word_boundaries_and_rejects_invalid_time(tmp_path):
    source = tmp_path / "VideoUFO.csv"
    _write_csv(source, [
        _row("catalog", "A catalog is shown while someone talks on grass."),
        _row("bad-time", "A dog runs in snow.", End_Time="not-a-time"),
    ])

    candidates, report = scan_metadata(source, expected_sha256=None)

    assert candidates == []
    assert report["rejections"] == {"ambiguous_factors": 1, "invalid_time": 1}


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "1.1"])
def test_scan_metadata_rejects_non_finite_or_out_of_range_quality(tmp_path, value):
    source = tmp_path / "VideoUFO.csv"
    _write_csv(source, [_row("cat-1", "A cat runs through snow.", Dynamic_Degree=value)])

    candidates, report = scan_metadata(source, expected_sha256=None)

    assert candidates == []
    assert report["rejections"] == {"quality": 1}


def test_scan_metadata_rejects_wrong_schema(tmp_path):
    source = tmp_path / "VideoUFO.csv"
    source.write_text("ID,Caption\ncat-1,A cat runs through snow.\n")

    with pytest.raises(ValueError, match="schema"):
        scan_metadata(source, expected_sha256=None)


def test_scan_metadata_fails_closed_on_source_hash_mismatch(tmp_path):
    source = tmp_path / "VideoUFO.csv"
    _write_csv(source, [_row("cat-1", "A cat runs through snow.")])

    with pytest.raises(ValueError, match="SHA-256"):
        scan_metadata(source, expected_sha256="0" * 64)


def test_sibling_clip_ids_share_one_source_group(tmp_path):
    source = tmp_path / "VideoUFO.csv"
    _write_csv(source, [
        _row("youtubeABC.0", "A cat runs through snow."),
        _row("youtubeABC.12", "A dog walks on grass."),
    ])

    candidates, _ = scan_metadata(source, expected_sha256=None)

    assert {item.source_id for item in candidates} == {"videoufo:youtubeABC"}


def test_cli_writes_atomic_candidates_and_auditable_report(tmp_path):
    source = tmp_path / "VideoUFO.csv"
    _write_csv(source, [_row("dog-1", "A dog jumps through grass.")])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    candidates = tmp_path / "candidates.jsonl"
    report = tmp_path / "coverage.json"

    assert main([
        "scan", str(source), str(candidates), str(report),
        "--expected-sha256", digest,
    ]) == 0

    row = json.loads(candidates.read_text())
    saved_report = json.loads(report.read_text())
    assert row["source_id"] == "videoufo:dog-1"
    assert row["archive_shard"] is None
    assert saved_report["source_sha256"] == digest
    assert saved_report["archive_mapping_status"] == "unresolved"
    assert saved_report["candidates_sha256"] == hashlib.sha256(
        candidates.read_bytes()
    ).hexdigest()


def test_cli_scan_requires_a_locked_source_hash(tmp_path):
    source = tmp_path / "VideoUFO.csv"
    _write_csv(source, [_row("dog-1", "A dog jumps through grass.")])

    with pytest.raises(SystemExit):
        main(["scan", str(source), str(tmp_path / "out"), str(tmp_path / "report")])


def test_index_tar_shard_maps_safe_mp4_members_and_rejects_duplicates(tmp_path):
    archive = tmp_path / "VideoUFO_200.tar"
    with tarfile.open(archive, "w") as handle:
        for name in ("clips/cat-1.mp4", "dog-2.MP4", "notes.txt"):
            payload = b"video" if name.lower().endswith(".mp4") else b"text"
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))

    index = index_tar_shard(archive)

    assert index == {
        "cat-1": "clips/cat-1.mp4",
        "dog-2": "dog-2.MP4",
    }

    duplicate = tmp_path / "duplicate.tar"
    with tarfile.open(duplicate, "w") as handle:
        for name in ("a/same.mp4", "b/same.mp4"):
            info = tarfile.TarInfo(name)
            info.size = 1
            handle.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="duplicate video ID"):
        index_tar_shard(duplicate)


def test_resolve_candidates_locks_shard_hash_and_member(tmp_path):
    source = tmp_path / "VideoUFO.csv"
    _write_csv(source, [_row("cat-1", "A cat runs through snow.")])
    candidates, report = scan_metadata(source, expected_sha256=None)
    archive = tmp_path / "VideoUFO_200.tar"
    with tarfile.open(archive, "w") as handle:
        info = tarfile.TarInfo("cat-1.mp4")
        info.size = 5
        handle.addfile(info, io.BytesIO(b"video"))

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    resolved, updated = resolve_candidates(
        candidates, report, [archive], expected_sha256={archive.name: digest},
    )

    assert resolved[0].archive_shard == "VideoUFO_200.tar"
    assert resolved[0].archive_member == "cat-1.mp4"
    assert updated["archive_mapping_status"] == "complete"
    assert updated["archive_shards"][0]["sha256"] == digest
    with pytest.raises(ValueError, match="SHA-256"):
        resolve_candidates(
            candidates, report, [archive],
            expected_sha256={archive.name: "0" * 64},
        )


def test_cli_resolve_rewrites_only_candidates_present_in_shards(tmp_path):
    source = tmp_path / "VideoUFO.csv"
    _write_csv(source, [
        _row("cat-1", "A cat runs through snow."),
        _row("dog-2", "A dog walks on grass."),
    ])
    candidates = tmp_path / "candidates.jsonl"
    report = tmp_path / "coverage.json"
    main([
        "scan", str(source), str(candidates), str(report),
        "--expected-sha256", hashlib.sha256(source.read_bytes()).hexdigest(),
    ])
    archive = tmp_path / "VideoUFO_200.tar"
    with tarfile.open(archive, "w") as handle:
        info = tarfile.TarInfo("folder/cat-1.mp4")
        info.size = 5
        handle.addfile(info, io.BytesIO(b"video"))
    resolved = tmp_path / "resolved.jsonl"
    resolved_report = tmp_path / "resolved.json"
    checksums = tmp_path / "checksums.json"
    checksums.write_text(json.dumps({
        archive.name: hashlib.sha256(archive.read_bytes()).hexdigest(),
    }))

    assert main([
        "resolve", str(candidates), str(report), str(resolved),
        str(resolved_report), "--shard-checksums", str(checksums), str(archive),
    ]) == 0

    row = json.loads(resolved.read_text())
    saved_report = json.loads(resolved_report.read_text())
    assert row["video_id"] == "cat-1"
    assert saved_report["archive_mapping_status"] == "partial"
    assert saved_report["candidates_unmapped"] == 1

    candidates.write_text(candidates.read_text() + candidates.read_text())
    with pytest.raises(ValueError, match="candidate file"):
        main([
            "resolve", str(candidates), str(report), str(resolved),
            str(resolved_report), "--shard-checksums", str(checksums), str(archive),
        ])


def test_extract_resolved_clip_verifies_archive_and_uses_safe_destination(tmp_path):
    archive = tmp_path / "VideoUFO_200.tar"
    with tarfile.open(archive, "w") as handle:
        info = tarfile.TarInfo("nested/cat-1.mp4")
        info.size = 5
        handle.addfile(info, io.BytesIO(b"video"))
    source = tmp_path / "VideoUFO.csv"
    _write_csv(source, [_row("cat-1", "A cat runs through snow.")])
    candidate = scan_metadata(source, expected_sha256=None)[0][0]
    from dataclasses import replace
    candidate = replace(
        candidate, archive_shard=archive.name, archive_member="nested/cat-1.mp4",
    )
    output = tmp_path / "raw"

    extracted = extract_resolved_clip(
        candidate, archive, output,
        expected_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )

    assert extracted.parent == output
    assert extracted.suffix == ".mp4"
    assert extracted.name != "cat-1.mp4"
    assert extracted.read_bytes() == b"video"
    with pytest.raises(ValueError, match="SHA-256"):
        extract_resolved_clip(
            candidate, archive, output, expected_archive_sha256="0" * 64,
        )
