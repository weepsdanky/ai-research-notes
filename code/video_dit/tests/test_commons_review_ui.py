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


def test_review_ui_can_use_priority_order_without_changing_candidate_data(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    original = "".join(json.dumps(row) + "\n" for row in [
        {"title": "Crowd timelapse.webm", "description": "many people",
         "review_status": "pending", "subject": "person",
         "duration_seconds": 3, "review_preview_url": "https://upload.wikimedia.org/a.webm"},
        {"title": "Solo runner.webm", "description": "a woman running alone",
         "review_status": "pending", "subject": "person",
         "duration_seconds": 3, "review_preview_url": "https://upload.wikimedia.org/b.webm"},
    ])
    candidates.write_text(original)

    app = build_ui(
        candidates, status_filter="pending", subject_filter="person",
        priority_order=True,
    )

    dropdown = next(component for component in app.config["components"]
                    if component.get("props", {}).get("label") == "Candidate")
    assert dropdown["props"]["value"] == "Solo runner.webm"
    assert candidates.read_text() == original


def test_review_ui_uses_provenance_sidecar_without_rewriting_candidates(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    original = json.dumps({
        "title": "Runner.ogv", "review_status": "pending", "subject": "person",
        "action": "review_required", "scene": "review_required", "prompt": "review",
    }) + "\n"
    candidates.write_text(original)
    provenance = tmp_path / "provenance.jsonl"
    provenance.write_text(json.dumps({
        "title": "Runner.ogv", "duration_seconds": 4,
        "review_preview_url": "https://upload.wikimedia.org/runner.webm",
    }) + "\n")

    app = build_ui(
        candidates, status_filter="pending", subject_filter="person",
        provenance_path=provenance,
    )

    dropdown = next(component for component in app.config["components"]
                    if component.get("props", {}).get("label") == "Candidate")
    assert dropdown["props"]["value"] == "Runner.ogv"
    assert candidates.read_text() == original
