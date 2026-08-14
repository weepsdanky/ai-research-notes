import hashlib
import json

import pytest

from blind_review_pool import create_blind_pool, reveal_results


def _inputs(count=8):
    candidates = [{"title": f"v{i}", "review_status": "pending", "subject": "cat"} for i in range(count)]
    provenance = [{
        "title": f"v{i}", "original_media_sha1": f"{i:040x}",
        "original_media_timestamp": "t", "poster_sha256": "unused",
        "duration_seconds": 3, "review_preview_url": "https://upload.wikimedia.org/v",
    } for i in range(count)]
    c = "".join(json.dumps(row) + "\n" for row in candidates).encode()
    p = "".join(json.dumps(row) + "\n" for row in provenance).encode()
    report = {
        "metric": "OpenAI CLIP ViT-B/32 single-poster retrieval margin",
        "checkpoint_sha256": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
        "positive_prompts": ["a cat playing indoors"],
        "negative_prompts": ["a cat outdoors"],
        "candidate_sha256": hashlib.sha256(c).hexdigest(),
        "provenance_sha256": hashlib.sha256(p).hexdigest(), "training_ready": False,
        "records": [{
            "title": f"v{i}", "rank": i + 1, "margin": count - i,
            "original_media_sha1": provenance[i]["original_media_sha1"],
            "original_media_timestamp": "t", "poster_sha256": "a" * 64,
        } for i in range(count)],
    }
    return c, p, json.dumps(report).encode()


def test_create_pool_has_disjoint_arms_and_hides_assignments():
    c, p, report = _inputs()
    queue, key = create_blind_pool(
        c, p, report, subject="cat", high_count=2, control_count=3, seed=42,
    )
    assert len(queue["records"]) == 5
    assert not any("arm" in row or "margin" in row for row in queue["records"])
    high = {row["title"] for row in key["records"] if row["arm"] == "high"}
    control = {row["title"] for row in key["records"] if row["arm"] == "random_control"}
    assert high == {"v0", "v1"} and high.isdisjoint(control)
    assert key["queue_sha256"] == hashlib.sha256(
        (json.dumps(queue, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()


def test_create_pool_is_deterministic_and_rejects_report_drift():
    c, p, report = _inputs()
    first = create_blind_pool(c, p, report, subject="cat", high_count=2, control_count=2, seed=7)
    second = create_blind_pool(c, p, report, subject="cat", high_count=2, control_count=2, seed=7)
    assert first == second
    value = json.loads(report); value["candidate_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bind"):
        create_blind_pool(c, p, json.dumps(value).encode(), subject="cat", high_count=2, control_count=2, seed=7)


def test_create_pool_rejects_unlocked_model_prompts_and_bad_poster_hash():
    c, p, report = _inputs()
    value = json.loads(report); value.pop("checkpoint_sha256")
    with pytest.raises(ValueError, match="checkpoint"):
        create_blind_pool(c, p, json.dumps(value).encode(), subject="cat", high_count=2, control_count=2, seed=7)
    value = json.loads(report); value["records"][0]["poster_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="poster SHA"):
        create_blind_pool(c, p, json.dumps(value).encode(), subject="cat", high_count=2, control_count=2, seed=7)
    value = json.loads(report); value["records"].pop()
    with pytest.raises(ValueError, match="complete playable"):
        create_blind_pool(c, p, json.dumps(value).encode(), subject="cat", high_count=2, control_count=2, seed=7)


def test_reveal_requires_all_decisions_and_reports_partial_pool_yield():
    c, p, report = _inputs(4)
    queue, key = create_blind_pool(c, p, report, subject="cat", high_count=1, control_count=1, seed=3)
    queue_bytes = (json.dumps(queue, indent=2, sort_keys=True) + "\n").encode()
    key_bytes = (json.dumps(key, indent=2, sort_keys=True) + "\n").encode()
    selected = {row["title"]: row["arm"] for row in key["records"]}
    decisions = []
    for title, arm in selected.items():
        decisions.append({
            "title": title, "review_status": "approved" if arm == "high" else "rejected",
            "subject": "cat", "action": "playing", "scene": "indoors",
        })
    decision_bytes = "".join(json.dumps(row) + "\n" for row in decisions).encode()
    audit = {
        "candidate_sha256": hashlib.sha256(decision_bytes).hexdigest(),
        "records": [{
            **row,
            "original_media_sha1": next(
                item["original_media_sha1"] for item in queue["records"]
                if item["title"] == row["title"]
            ),
            "original_media_timestamp": "t",
        } for row in decisions],
    }
    media = {
        "queue_sha256": hashlib.sha256(queue_bytes).hexdigest(),
        "training_ready": False,
        "records": [{
            "blind_id": item["blind_id"], "title": item["title"],
            "derivative_path": f"media/{item['blind_id']}.webm", "bytes": 10,
            "derivative_sha256": "d" * 64,
        } for item in queue["records"]],
    }
    audit_bytes = json.dumps(audit).encode(); media_bytes = json.dumps(media).encode()
    result = reveal_results(
        queue_bytes, key_bytes, decision_bytes, audit_bytes, media_bytes,
        target_triple=("cat", "playing", "indoors"),
    )
    assert result["arms"]["high"]["yield"] == 1
    assert result["arms"]["random_control"]["yield"] == 0
    assert result["limitation"].endswith("not full-pool recall")

    tampered_key = json.loads(key_bytes)
    tampered_key["records"][0]["arm"], tampered_key["records"][1]["arm"] = (
        tampered_key["records"][1]["arm"], tampered_key["records"][0]["arm"],
    )
    with pytest.raises(ValueError, match="bind"):
        reveal_results(queue_bytes, json.dumps(tampered_key).encode(), decision_bytes,
                       audit_bytes, media_bytes,
                       target_triple=("cat", "playing", "indoors"))

    wrong_labels = json.loads(audit_bytes)
    wrong_labels["records"][0]["action"] = "sitting"
    with pytest.raises(ValueError, match="decision"):
        reveal_results(queue_bytes, key_bytes, decision_bytes,
                       json.dumps(wrong_labels).encode(), media_bytes,
                       target_triple=("cat", "playing", "indoors"))

    incomplete = (json.dumps(decisions[0]) + "\n").encode()
    incomplete_audit = {**audit, "candidate_sha256": hashlib.sha256(incomplete).hexdigest()}
    with pytest.raises(ValueError, match="incomplete"):
        reveal_results(queue_bytes, key_bytes, incomplete, json.dumps(incomplete_audit).encode(), media_bytes,
                       target_triple=("cat", "playing", "indoors"))


def test_reveal_rejects_duplicate_queue_ids_and_wrong_subject():
    c, p, report = _inputs(4)
    queue, key = create_blind_pool(c, p, report, subject="cat", high_count=1, control_count=1, seed=3)
    queue["records"][1]["blind_id"] = queue["records"][0]["blind_id"]
    queue_bytes = (json.dumps(queue, indent=2, sort_keys=True) + "\n").encode()
    key["queue_sha256"] = hashlib.sha256(queue_bytes).hexdigest()
    key_bytes = (json.dumps(key, indent=2, sort_keys=True) + "\n").encode()
    with pytest.raises(ValueError, match="IDs"):
        reveal_results(queue_bytes, key_bytes, c, b"{}", b"{}",
                       target_triple=("cat", "playing", "indoors"))
