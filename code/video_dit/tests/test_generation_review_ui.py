import hashlib
import json

import pytest

from generation_review_ui import (
    remaining_review_ids, review_queue, save_generation_decision,
)


def _queue():
    digest = hashlib.sha256(b"video").hexdigest()
    return {
        "protocol": "prompt-hidden generated-video factor review queue v1",
        "factor_vocabulary": {
            "subjects": ["cat", "dog", "absent_or_unclear"],
            "actions": ["playing", "walking", "absent_or_unclear"],
            "scenes": ["indoors", "outdoors", "absent_or_unclear"],
        },
        "records": [
            {"blind_id": "generation-0001", "filename": "generation-0001.mp4",
             "bytes": 5, "video_sha256": digest},
            {"blind_id": "generation-0002", "filename": "generation-0002.mp4",
             "bytes": 5, "video_sha256": digest},
        ],
        "prompt_hidden": True,
    }


def _bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def test_save_decision_is_queue_bound_resumable_and_never_exposes_prompt(tmp_path):
    queue_bytes = _bytes(_queue())
    decisions = tmp_path / "rater-a.json"
    media = tmp_path / "media"; media.mkdir()
    for index in (1, 2):
        (media / f"generation-{index:04d}.mp4").write_bytes(b"video")
    remaining = save_generation_decision(
        queue_bytes, media, decisions, reviewer_id="rater-a", blind_id="generation-0001",
        subject="cat", action="playing", scene="indoors", notes="full video watched",
    )
    assert remaining == ["generation-0002"]
    value = json.loads(decisions.read_text())
    assert value["queue_sha256"] == hashlib.sha256(queue_bytes).hexdigest()
    assert value["reviewer_id"] == "rater-a"
    assert value["records"][0]["blind_id"] == "generation-0001"
    assert not any("prompt" in row or "split" in row or "seed" in row for row in value["records"])

    remaining = save_generation_decision(
        queue_bytes, media, decisions, reviewer_id="rater-a", blind_id="generation-0002",
        subject="dog", action="walking", scene="outdoors", notes="full video watched",
    )
    assert remaining == []
    assert remaining_review_ids(
        queue_bytes, decisions, reviewer_id="rater-a",
    ) == []


def test_save_decision_rejects_out_of_vocabulary_and_wrong_queue(tmp_path):
    queue_bytes = _bytes(_queue())
    decisions = tmp_path / "rater.json"
    media = tmp_path / "media"; media.mkdir()
    for index in (1, 2):
        (media / f"generation-{index:04d}.mp4").write_bytes(b"video")
    with pytest.raises(ValueError, match="vocabulary"):
        save_generation_decision(
            queue_bytes, media, decisions, reviewer_id="rater", blind_id="generation-0001",
            subject="person", action="playing", scene="indoors", notes="watched",
        )
    save_generation_decision(
        queue_bytes, media, decisions, reviewer_id="rater", blind_id="generation-0001",
        subject="cat", action="playing", scene="indoors", notes="watched",
    )
    changed = _queue(); changed["records"].reverse()
    with pytest.raises(ValueError, match="queue"):
        save_generation_decision(
            _bytes(changed), media, decisions, reviewer_id="rater", blind_id="generation-0002",
            subject="dog", action="walking", scene="outdoors", notes="watched",
        )


def test_review_queue_validates_media_and_returns_only_blind_fields(tmp_path):
    queue_bytes = _bytes(_queue())
    media = tmp_path / "media"; media.mkdir()
    for index in (1, 2):
        (media / f"generation-{index:04d}.mp4").write_bytes(b"video")
    rows = review_queue(queue_bytes, media)
    assert [row["blind_id"] for row in rows] == ["generation-0001", "generation-0002"]
    assert all(set(row) == {"blind_id", "video_path"} for row in rows)
    (media / "generation-0002.mp4").unlink()
    with pytest.raises(ValueError, match="media"):
        review_queue(queue_bytes, media)
