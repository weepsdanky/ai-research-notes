import json

from commons_review import (
    choose_review_derivative,
    fetch_review_media,
    cache_review_preview,
    save_review,
    summarize_reviews,
    enrich_review_metadata,
    fetch_review_media_batch,
    build_contact_sheet,
)


def test_choose_review_derivative_prefers_smallest_usable_video():
    derivatives = [
        {"src": "https://upload.wikimedia.org/original.webm", "width": 1920,
         "height": 1080, "type": "video/webm"},
        {"src": "https://upload.wikimedia.org/240.webm", "width": 426,
         "height": 240, "type": "video/webm"},
        {"src": "https://upload.wikimedia.org/360.mov", "width": 640,
         "height": 360, "type": "video/mp4"},
    ]

    assert choose_review_derivative(derivatives) == "https://upload.wikimedia.org/240.webm"


def test_fetch_review_media_uses_videoinfo_derivatives(monkeypatch):
    payload = {"query": {"pages": {"7": {"title": "File:Cat runs.webm", "videoinfo": [{
        "duration": 12.5,
        "derivatives": [
            {"src": "https://upload.wikimedia.org/240.webm", "width": 426,
             "height": 240, "type": "video/webm"},
        ],
    }]}}}}
    captured = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def read(self): return json.dumps(payload).encode()

    def fake_open(request, timeout):
        captured["url"] = request.full_url
        return Response()

    monkeypatch.setattr("commons_review.urllib.request.urlopen", fake_open)
    media = fetch_review_media("File:Cat runs.webm")
    assert "prop=videoinfo" in captured["url"]
    assert media == {"preview_url": "https://upload.wikimedia.org/240.webm", "duration": 12.5}


def test_cache_review_preview_uses_stable_path_and_user_agent(tmp_path, monkeypatch):
    captured = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def read(self, size=-1): return b"preview-bytes"

    def fake_open(request, timeout):
        captured["headers"] = dict(request.header_items())
        return Response()

    monkeypatch.setattr("commons_review.urllib.request.urlopen", fake_open)
    first = cache_review_preview(
        "File:Cat.webm", "https://upload.wikimedia.org/240.webm", tmp_path,
    )
    second = cache_review_preview(
        "File:Cat.webm", "https://upload.wikimedia.org/240.webm", tmp_path,
    )
    assert first == second
    assert first.read_bytes() == b"preview-bytes"
    assert "User-agent" in captured["headers"]


def test_cache_review_preview_retries_rate_limits(tmp_path, monkeypatch):
    import urllib.error
    calls = []

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def read(self): return b"ok"

    def fake_open(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "rate", {"Retry-After": "0"}, None)
        return Response()

    monkeypatch.setattr("commons_review.urllib.request.urlopen", fake_open)
    path = cache_review_preview(
        "File:Cat.webm", "https://upload.wikimedia.org/cat.webm", tmp_path,
    )
    assert path.read_bytes() == b"ok"
    assert len(calls) == 2


def test_save_review_is_atomic_and_requires_complete_approved_labels(tmp_path):
    path = tmp_path / "candidates.jsonl"
    path.write_text(json.dumps({
        "title": "File:Cat.webm", "review_status": "pending",
        "duration_seconds": 10.0,
        "subject": "cat", "action": "review_required",
        "scene": "review_required", "prompt": "review required",
    }) + "\n")

    import pytest
    with pytest.raises(ValueError, match="complete labels"):
        save_review(path, "File:Cat.webm", status="approved", subject="cat",
                    action="review_required", scene="indoors", prompt="a cat")

    save_review(path, "File:Cat.webm", status="approved", subject="cat",
                action="walking", scene="indoors", prompt="a cat walking indoors",
                start_seconds=1.5, notes="subject visible")
    row = json.loads(path.read_text())
    assert row["review_status"] == "approved"
    assert row["action"] == "walking"
    assert row["start_seconds"] == 1.5


def test_save_review_rejects_segment_past_video_end(tmp_path):
    path = tmp_path / "candidates.jsonl"
    path.write_text(json.dumps({
        "title": "File:short.webm", "duration_seconds": 3.0,
        "review_status": "pending",
    }) + "\n")

    import pytest
    with pytest.raises(ValueError, match="two-second segment exceeds"):
        save_review(
            path, "File:short.webm", status="approved", subject="cat",
            action="walking", scene="indoors", prompt="a cat walking indoors",
            start_seconds=2.0,
        )


def test_save_review_requires_known_positive_duration_for_approval(tmp_path):
    path = tmp_path / "candidates.jsonl"
    path.write_text(json.dumps({
        "title": "File:unknown.webm", "review_status": "pending",
    }) + "\n")

    import pytest
    with pytest.raises(ValueError, match="positive video duration"):
        save_review(
            path, "File:unknown.webm", status="approved", subject="cat",
            action="walking", scene="indoors", prompt="a cat walking indoors",
        )


def test_review_summary_counts_status_and_factor_coverage():
    rows = [
        {"review_status": "approved", "subject": "cat", "action": "walking",
         "scene": "indoors"},
        {"review_status": "approved", "subject": "cat", "action": "walking",
         "scene": "indoors"},
        {"review_status": "rejected", "subject": "dog", "action": "running",
         "scene": "field"},
    ]
    summary = summarize_reviews(rows)
    assert summary["statuses"] == {"approved": 2, "rejected": 1}
    assert summary["approved_triples"] == {"cat|walking|indoors": 2}
    assert summary["subject_statuses"] == {
        "cat": {"approved": 2}, "dog": {"rejected": 1},
    }
    assert summary["pending_placeholders"] == 0


def test_review_summary_exposes_pending_labeling_backlog():
    summary = summarize_reviews([
        {"review_status": "pending", "subject": "cat",
         "action": "review_required", "scene": "review_required"},
    ])

    assert summary["pending_placeholders"] == 1


def test_enrich_review_metadata_persists_duration_without_downloading_video(tmp_path, monkeypatch):
    path = tmp_path / "candidates.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in [
        {"title": "File:A.webm", "review_status": "pending"},
        {"title": "File:B.webm", "review_status": "pending", "duration_seconds": 3.0,
         "review_preview_url": "https://upload.wikimedia.org/b.webm"},
    ]) + "\n")
    calls = []
    monkeypatch.setattr("commons_review.fetch_review_media_batch", lambda titles: {
        title: calls.append(title) or {
            "preview_url": "https://upload.wikimedia.org/240.webm", "duration": 4.5,
        } for title in titles
    })
    enrich_review_metadata(path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert calls == ["File:A.webm"]
    assert rows[0]["duration_seconds"] == 4.5
    assert rows[0]["review_preview_url"].endswith("240.webm")


def test_fetch_review_media_batch_uses_one_request_for_multiple_titles(monkeypatch):
    captured = {}
    payload = {"query": {"pages": {
        "1": {"title": "File:A.webm", "videoinfo": [{"duration": 2,
            "derivatives": [{"src": "https://upload.wikimedia.org/a.webm",
                             "width": 320, "height": 240, "type": "video/webm"}]}]},
        "2": {"title": "File:B.webm", "videoinfo": [{"duration": 3,
            "derivatives": [{"src": "https://upload.wikimedia.org/b.webm",
                             "width": 320, "height": 240, "type": "video/webm"}]}]},
    }}}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def read(self): return json.dumps(payload).encode()

    def fake_open(request, timeout):
        captured["url"] = request.full_url
        return Response()

    monkeypatch.setattr("commons_review.urllib.request.urlopen", fake_open)
    result = fetch_review_media_batch(["File:A.webm", "File:B.webm"])
    assert set(result) == {"File:A.webm", "File:B.webm"}
    assert "%7C" in captured["url"]


def test_batch_records_missing_derivative_without_losing_other_rows(monkeypatch):
    payload = {"query": {"pages": {
        "1": {"title": "File:Good.webm", "videoinfo": [{"duration": 2,
            "derivatives": [{"src": "https://upload.wikimedia.org/good.webm",
                             "width": 320, "height": 240, "type": "video/webm"}]}]},
        "2": {"title": "File:Broken.ogg", "videoinfo": [{"duration": 3,
            "derivatives": []}]},
    }}}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def read(self): return json.dumps(payload).encode()

    monkeypatch.setattr(
        "commons_review.urllib.request.urlopen", lambda request, timeout: Response(),
    )
    result = fetch_review_media_batch(["File:Good.webm", "File:Broken.ogg"])
    assert result["File:Good.webm"]["preview_url"].endswith("good.webm")
    assert "error" in result["File:Broken.ogg"]


def test_build_contact_sheet_renders_three_temporal_frames(tmp_path, monkeypatch):
    video = tmp_path / "clip.webm"
    video.write_bytes(b"video")
    output = tmp_path / "sheet.jpg"
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        from PIL import Image
        Image.new("RGB", (96, 96), "red").save(command[-1])
        return type("Done", (), {})()

    monkeypatch.setattr("commons_review.subprocess.run", fake_run)
    build_contact_sheet(video, output, duration=9, title="Cat", cells=3)
    from PIL import Image
    assert Image.open(output).size == (288, 124)
    assert len(commands) == 3
