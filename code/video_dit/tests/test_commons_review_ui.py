import json

from commons_review_ui import build_ui


def test_review_ui_has_one_title_change_loader_and_no_then_loader(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps({
        "title": "File:Person.webm", "review_status": "pending",
        "subject": "person", "action": "review_required",
        "scene": "review_required", "prompt": "review person",
        "duration_seconds": 3,
        "review_preview_url": "https://upload.wikimedia.org/person.webm",
    }) + "\n")

    app = build_ui(candidates, status_filter="pending", subject_filter="person")
    dependencies = app.config["dependencies"]

    title_events = [
        dependency for dependency in dependencies
        if any(target[1] == "change" for target in dependency["targets"])
    ]
    assert len(title_events) == 1
    assert not any(
        any(target[1] == "then" for target in dependency["targets"])
        for dependency in dependencies
    )
