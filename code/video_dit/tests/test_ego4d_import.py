import hashlib
import json

import pytest

from ego4d_import import scan_ego4d_annotations


def _snapshots():
    metadata = {
        "version": "v2.1",
        "date": "2026-01-01",
        "videos": [
            {
                "video_uid": "video-b", "duration_sec": 20.0,
                "video_source": "university-x", "origin_video_id": "origin-b",
                "fb_participant_id": 7, "physical_setting_name": "home-b",
                "scenarios": ["Household activities"],
            },
            {
                "video_uid": "video-a", "duration_sec": 30.0,
                "video_source": "university-y", "origin_video_id": "origin-a",
                "fb_participant_id": None, "physical_setting_name": "home-a",
                "scenarios": ["Indoor social"],
            },
        ],
    }
    narrations = {
        "video-a": {
            "narration_pass_1": {"narrations": [
                {"timestamp_sec": 4.0, "narration_text": "#C plays with the cat inside the room", "annotation_uid": "a1"},
                {"timestamp_sec": 8.0, "narration_text": "#C opens a drawer", "annotation_uid": "a2"},
            ]},
        },
        "video-b": {
            "narration_pass_2": {"narrations": [
                {"timestamp_sec": 6.0, "narration_text": "#C watches a puppy play outdoors", "annotation_uid": "b1"},
            ]},
        },
    }
    license_bytes = b"Ego4D research data license fixture"
    receipt = {
        "dataset": "ego4d", "accepted_at": "2026-08-14T00:00:00Z",
        "accepted_by": "local-researcher",
        "permits_local_noncommercial_training": True,
        "license_sha256": hashlib.sha256(license_bytes).hexdigest(),
    }
    encode = lambda value: json.dumps(value, sort_keys=True).encode()
    return encode(metadata), encode(narrations), license_bytes, encode(receipt)


def test_scan_freezes_matching_uids_and_provenance_contract():
    metadata, narrations, license_bytes, receipt = _snapshots()
    candidates, report = scan_ego4d_annotations(
        metadata, narrations, license_bytes, receipt, max_video_uids=10,
    )
    assert [row["annotation_uid"] for row in candidates] == ["a1", "b1"]
    assert candidates[0]["duplicate_group"] == "ego4d-origin:university-y:origin-a"
    assert candidates[0]["context_before"] is None
    assert candidates[0]["context_after"]["annotation_uid"] == "a2"
    assert candidates[1]["duplicate_group"] == "ego4d-participant:university-x:7"
    assert all(row["review_status"] == "pending" for row in candidates)
    assert report["selected_video_uids"] == ["video-a", "video-b"]
    assert report["dataset_release"] == "v2_1"
    assert report["uid_file_sha256"] == hashlib.sha256(b"video-a\nvideo-b\n").hexdigest()
    assert report["download_contract"] == {
        "dataset": "video_540ss", "max_bytes": 12_000_000_000,
        "on_budget_exceeded": "stop_without_fallback",
    }
    assert report["training_ready"] is False
    assert report["metadata_sha256"] == hashlib.sha256(metadata).hexdigest()
    assert report["candidate_sha256"] == hashlib.sha256(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidates).encode()
    ).hexdigest()


def test_scan_requires_matching_training_license_receipt():
    metadata, narrations, license_bytes, receipt = _snapshots()
    value = json.loads(receipt); value["license_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="license"):
        scan_ego4d_annotations(
            metadata, narrations, license_bytes, json.dumps(value).encode(),
            max_video_uids=10,
        )
    value = json.loads(receipt); value["accepted_at"] = "yesterday"
    with pytest.raises(ValueError, match="accepted_at"):
        scan_ego4d_annotations(
            metadata, narrations, license_bytes, json.dumps(value).encode(),
            max_video_uids=10,
        )
    value = json.loads(receipt); value["accepted_at"] = "2026-08-14Z"
    with pytest.raises(ValueError, match="accepted_at"):
        scan_ego4d_annotations(
            metadata, narrations, license_bytes, json.dumps(value).encode(),
            max_video_uids=10,
        )
    value = json.loads(receipt); value["permits_local_noncommercial_training"] = False
    with pytest.raises(ValueError, match="training"):
        scan_ego4d_annotations(
            metadata, narrations, license_bytes, json.dumps(value).encode(),
            max_video_uids=10,
        )


def test_scan_rejects_out_of_bounds_and_orphan_narrations():
    metadata, narrations, license_bytes, receipt = _snapshots()
    value = json.loads(narrations); value["video-a"]["narration_pass_1"]["narrations"][0]["timestamp_sec"] = 31
    with pytest.raises(ValueError, match="timestamp"):
        scan_ego4d_annotations(
            metadata, json.dumps(value).encode(), license_bytes, receipt,
            max_video_uids=10,
        )


def test_scan_sorts_temporal_context_and_requires_release_identity():
    metadata, narrations, license_bytes, receipt = _snapshots()
    value = json.loads(narrations)
    value["video-a"]["narration_pass_1"]["narrations"].reverse()
    candidates, _ = scan_ego4d_annotations(
        metadata, json.dumps(value).encode(), license_bytes, receipt,
        max_video_uids=10,
    )
    match = next(row for row in candidates if row["annotation_uid"] == "a1")
    assert match["context_before"] is None
    assert match["context_after"]["annotation_uid"] == "a2"

    value = json.loads(metadata); value.pop("version")
    with pytest.raises(ValueError, match="version"):
        scan_ego4d_annotations(
            json.dumps(value).encode(), narrations, license_bytes, receipt,
            max_video_uids=10,
        )
    value = json.loads(metadata); value["date"] = ""
    with pytest.raises(ValueError, match="date"):
        scan_ego4d_annotations(
            json.dumps(value).encode(), narrations, license_bytes, receipt,
            max_video_uids=10,
        )
    value = json.loads(metadata); value["version"] = "v1"
    with pytest.raises(ValueError, match="v2.1"):
        scan_ego4d_annotations(
            json.dumps(value).encode(), narrations, license_bytes, receipt,
            max_video_uids=10,
        )


def test_scan_prefers_distinct_source_groups_before_filling_uid_budget():
    metadata, narrations, license_bytes, receipt = _snapshots()
    metadata_value = json.loads(metadata)
    metadata_value["videos"].append({
        "video_uid": "video-c", "duration_sec": 20.0,
        "video_source": "university-y", "origin_video_id": "origin-a",
        "fb_participant_id": None, "physical_setting_name": "home-c",
        "scenarios": ["Household activities"],
    })
    narration_value = json.loads(narrations)
    narration_value["video-c"] = {
        "narration_pass_1": {"narrations": [{
            "timestamp_sec": 3.0,
            "narration_text": "#C watches a kitten play inside the kitchen",
            "annotation_uid": "c1",
        }]},
    }
    candidates, report = scan_ego4d_annotations(
        json.dumps(metadata_value).encode(), json.dumps(narration_value).encode(),
        license_bytes, receipt, max_video_uids=2,
    )
    assert report["selected_video_uids"] == ["video-a", "video-b"]
    assert len({row["duplicate_group"] for row in candidates}) == 2
    assert "distinct duplicate groups first" in report["query_contract"]["selection"]
    value = json.loads(narrations); value["orphan"] = value.pop("video-a")
    with pytest.raises(ValueError, match="metadata"):
        scan_ego4d_annotations(
            metadata, json.dumps(value).encode(), license_bytes, receipt,
            max_video_uids=10,
        )
    metadata_value = json.loads(metadata)
    metadata_value["videos"][0]["video_uid"] = "../escape"
    narration_value = json.loads(narrations)
    narration_value["../escape"] = narration_value.pop("video-b")
    with pytest.raises(ValueError, match="video_uid"):
        scan_ego4d_annotations(
            json.dumps(metadata_value).encode(), json.dumps(narration_value).encode(),
            license_bytes, receipt, max_video_uids=10,
        )
