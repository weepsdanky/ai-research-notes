import json

import pytest

from commons_review import (
    choose_review_derivative,
    fetch_review_media,
    cache_review_preview,
    save_review,
    summarize_reviews,
    enrich_review_metadata,
    fetch_review_media_batch,
    build_contact_sheet,
    review_queue,
    next_review_title,
    enrich_provenance_sidecar,
    verify_provenance_sidecar,
    prioritize_review_rows,
    merge_provenance_rows,
    completed_review_audit,
    prioritize_pair_deficit_rows,
    main,
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
    payload = {"query": {"pages": {"7": {"title": "File:Cat runs.webm",
        "revisions": [{"revid": 123}], "videoinfo": [{
        "url": "https://upload.wikimedia.org/original.webm", "sha1": "abc",
        "timestamp": "2026-08-14T00:00:00Z",
        "duration": 12.5,
        "derivatives": [
            {"src": "https://upload.wikimedia.org/240.webm", "width": 426,
             "height": 240, "type": "video/webm"},
        ],
    }], "imageinfo": [{
        "descriptionurl": "https://commons.wikimedia.org/wiki/File:Cat_runs.webm",
        "extmetadata": {"Artist": {"value": "Author"},
                        "LicenseShortName": {"value": "CC BY 4.0"},
                        "LicenseUrl": {"value": "https://creativecommons.org/licenses/by/4.0"}},
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
    assert media == {"preview_url": "https://upload.wikimedia.org/240.webm", "duration": 12.5,
                     "source_revision": "123",
                     "original_media_url": "https://upload.wikimedia.org/original.webm",
                     "original_media_sha1": "abc",
                     "original_media_timestamp": "2026-08-14T00:00:00Z",
                     "description_url": "https://commons.wikimedia.org/wiki/File:Cat_runs.webm",
                     "attribution": "Author", "license": "CC BY 4.0",
                     "license_url": "https://creativecommons.org/licenses/by/4.0"}


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

    with pytest.raises(ValueError, match="source/identity group"):
        save_review(path, "File:Cat.webm", status="approved", subject="cat",
                    action="walking", scene="indoors", prompt="a cat walking indoors")

    save_review(path, "File:Cat.webm", status="approved", subject="cat",
                action="walking", scene="indoors", prompt="a cat walking indoors",
                start_seconds=1.5, notes="subject visible",
                duplicate_group="commons-audited-unique:cat")
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
            start_seconds=2.0, duplicate_group="commons-audited-unique:short",
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
            duplicate_group="commons-audited-unique:unknown",
        )


def test_save_review_can_validate_approval_with_locked_sidecar_duration(tmp_path):
    path = tmp_path / "candidates.jsonl"
    original = {
        "title": "File:Runner.ogv", "review_status": "pending",
        "subject": "person", "action": "review_required",
        "scene": "review_required", "prompt": "review required",
    }
    path.write_text(json.dumps(original) + "\n")
    provenance = tmp_path / "provenance.jsonl"
    provenance.write_text(json.dumps({
        "title": "File:Runner.ogv", "duration_seconds": 4.0,
        "review_preview_url": "https://upload.wikimedia.org/runner.webm",
    }) + "\n")

    save_review(
        path, "File:Runner.ogv", status="approved", subject="person",
        action="running", scene="forest", prompt="a person running in a forest",
        start_seconds=1.0, provenance_path=provenance,
        duplicate_group="commons-audited-unique:runner",
    )

    saved = json.loads(path.read_text())
    assert saved["review_status"] == "approved"
    assert "duration_seconds" not in saved


def test_save_review_persists_manual_duplicate_group(tmp_path):
    path = tmp_path / "candidates.jsonl"
    path.write_text(json.dumps({
        "title": "File:Cat series 01.webm", "duration_seconds": 5.0,
        "review_status": "pending", "subject": "cat",
        "action": "review_required", "scene": "review_required",
        "prompt": "review required",
    }) + "\n")

    save_review(
        path, "File:Cat series 01.webm", status="approved", subject="cat",
        action="playing", scene="indoors", prompt="a cat playing indoors",
        duplicate_group="commons-series:cat-series", notes="same filming series",
    )

    assert json.loads(path.read_text())["duplicate_group"] == (
        "commons-series:cat-series"
    )


def test_merge_provenance_rejects_fields_that_can_override_human_review():
    candidates = [{
        "title": "File:Runner.ogv", "review_status": "pending",
        "subject": "person", "action": "review_required",
    }]
    sidecar_rows = [{
        "title": "File:Runner.ogv", "duration_seconds": 4.0,
        "review_status": "approved", "action": "running",
    }]

    with pytest.raises(ValueError, match="unsupported fields"):
        merge_provenance_rows(candidates, sidecar_rows=sidecar_rows)


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


def test_review_queue_filters_subject_status_and_unplayable_rows():
    rows = [
        {"title": "dog", "review_status": "pending", "subject": "dog",
         "duration_seconds": 4, "review_preview_url": "https://upload/dog.webm"},
        {"title": "person-b", "review_status": "pending", "subject": "person",
         "duration_seconds": 3, "review_preview_url": "https://upload/b.webm"},
        {"title": "person-a", "review_status": "pending", "subject": "person",
         "duration_seconds": 8, "review_preview_url": "https://upload/a.webm"},
        {"title": "too-short", "review_status": "pending", "subject": "person",
         "duration_seconds": 1, "review_preview_url": "https://upload/short.webm"},
        {"title": "approved", "review_status": "approved", "subject": "person",
         "duration_seconds": 3, "review_preview_url": "https://upload/ok.webm"},
    ]

    assert review_queue(rows, status="pending", subject="person") == [
        "person-a", "person-b",
    ]
    assert review_queue(rows, status="all", subject="all") == [
        "approved", "dog", "person-a", "person-b",
    ]


def test_review_priority_orders_likely_single_subject_before_obvious_bad_fits():
    rows = [
        {"title": "Station crowd timelapse.webm", "description": "many people",
         "review_status": "pending", "subject": "person",
         "duration_seconds": 4, "review_preview_url": "https://upload/a.webm"},
        {"title": "Solo athlete trail running.webm",
         "description": "A woman running alone on a forest trail",
         "review_status": "pending", "subject": "person",
         "duration_seconds": 4, "review_preview_url": "https://upload/b.webm"},
        {"title": "First-person city walking tour.webm", "description": "POV vlog",
         "review_status": "pending", "subject": "person",
         "duration_seconds": 4, "review_preview_url": "https://upload/c.webm"},
    ]

    ranked = prioritize_review_rows(rows, status="pending", subject="person")

    assert [row["title"] for row in ranked] == [
        "Solo athlete trail running.webm",
        "First-person city walking tour.webm",
        "Station crowd timelapse.webm",
    ]
    assert ranked[0]["priority_score"] > ranked[-1]["priority_score"]
    assert "single-subject:solo" in ranked[0]["priority_reasons"]
    assert "exclude:timelapse" in ranked[-1]["priority_reasons"]
    assert all(row["review_status"] == "pending" for row in rows)


def test_priority_cli_writes_auditable_jsonl_without_modifying_candidates(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    original = json.dumps({
        "title": "Solo runner.webm", "description": "a woman running alone",
        "review_status": "pending", "subject": "person", "duration_seconds": 4,
        "review_preview_url": "https://upload.wikimedia.org/runner.webm",
    }) + "\n"
    candidates.write_text(original)
    output = tmp_path / "priority.jsonl"

    assert main([
        "priority", str(candidates), str(output), "--subject", "person",
    ]) == 0

    report = json.loads(output.read_text())
    assert report["title"] == "Solo runner.webm"
    assert report["priority_score"] > 0
    assert candidates.read_text() == original


def test_priority_cli_reads_playability_from_provenance_sidecar(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    original = json.dumps({
        "title": "Runner.ogv", "description": "one runner",
        "review_status": "pending", "subject": "person",
    }) + "\n"
    candidates.write_text(original)
    provenance = tmp_path / "provenance.jsonl"
    provenance.write_text(json.dumps({
        "title": "Runner.ogv", "duration_seconds": 4,
        "review_preview_url": "https://upload.wikimedia.org/runner.webm",
    }) + "\n")
    output = tmp_path / "priority.jsonl"

    assert main([
        "priority", str(candidates), str(output), "--subject", "person",
        "--provenance", str(provenance),
    ]) == 0

    assert json.loads(output.read_text())["title"] == "Runner.ogv"
    assert candidates.read_text() == original


@pytest.mark.parametrize("command", ["priority", "provenance"])
def test_report_commands_reject_candidate_path_as_output(tmp_path, command):
    candidates = tmp_path / "candidates.jsonl"
    original = json.dumps({
        "title": "Solo runner.webm", "review_status": "pending",
    }) + "\n"
    candidates.write_text(original)
    alias = tmp_path / "candidate-alias.jsonl"
    alias.symlink_to(candidates)

    with pytest.raises(ValueError, match="must differ from candidates"):
        main([command, str(candidates), str(alias)])

    assert candidates.read_text() == original


def test_summary_cli_atomically_writes_optional_json_report(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps({
        "title": "Person.webm", "review_status": "pending", "subject": "person",
        "action": "review_required", "scene": "review_required",
    }) + "\n")
    output = tmp_path / "review-summary.json"

    assert main(["summary", str(candidates), str(output)]) == 0

    report = json.loads(output.read_text())
    assert report["records"] == 1
    assert report["statuses"] == {"pending": 1}
    assert report["pending_placeholders"] == 1


def test_next_review_title_advances_without_looping_on_saved_row():
    assert next_review_title("person-a", ["person-b", "person-c"]) == "person-b"
    assert next_review_title("person-b", ["person-a", "person-b", "person-c"]) == "person-c"
    assert next_review_title("person-c", ["person-a", "person-b", "person-c"]) == "person-a"
    assert next_review_title("person-z", ["person-b", "person-c"]) == "person-b"
    assert next_review_title("person-a", []) is None


def test_enrich_review_metadata_persists_duration_without_downloading_video(tmp_path, monkeypatch):
    path = tmp_path / "candidates.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in [
        {"title": "File:A.webm", "review_status": "pending"},
        {"title": "File:B.webm", "review_status": "pending", "duration_seconds": 3.0,
         "review_preview_url": "https://upload.wikimedia.org/b.webm", "source_revision": "9"},
    ]) + "\n")
    calls = []
    monkeypatch.setattr("commons_review.fetch_review_media_batch", lambda titles: {
        title: calls.append(title) or {
            "preview_url": "https://upload.wikimedia.org/240.webm", "duration": 4.5,
            "source_revision": "123",
            "original_media_url": "https://upload.wikimedia.org/original.webm",
            "original_media_sha1": "abc", "original_media_timestamp": "now",
        } for title in titles
    })
    enrich_review_metadata(path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert calls == ["File:A.webm"]
    assert rows[0]["duration_seconds"] == 4.5
    assert rows[0]["review_preview_url"].endswith("240.webm")
    assert rows[0]["source_revision"] == "123"
    assert rows[0]["original_media_sha1"] == "abc"


def test_provenance_sidecar_is_resumable_and_never_rewrites_candidates(tmp_path, monkeypatch):
    candidates = tmp_path / "candidates.jsonl"
    original = "".join(json.dumps({"title": title, "review_status": "pending"}) + "\n"
                       for title in ("File:A.webm", "File:B.webm"))
    candidates.write_text(original)
    sidecar = tmp_path / "provenance.jsonl"
    sidecar.write_text(json.dumps({
        "title": "File:A.webm", "source_revision": "1",
        "original_media_url": "https://upload.wikimedia.org/a.webm",
        "original_media_sha1": "aaa", "original_media_timestamp": "old",
        "review_preview_url": "https://upload.wikimedia.org/a-240.webm",
        "duration_seconds": 3.0,
        "description_url": "https://commons.wikimedia.org/wiki/File:A.webm",
        "attribution": "Author", "license": "CC0", "license_url": "https://cc0",
    }) + "\n")
    calls = []
    monkeypatch.setattr("commons_review.fetch_review_media_batch", lambda titles: {
        title: calls.append(title) or {
            "source_revision": "2", "original_media_url": "https://upload.wikimedia.org/b.webm",
            "original_media_sha1": "bbb", "original_media_timestamp": "now",
            "preview_url": "https://upload.wikimedia.org/b-240.webm", "duration": 4.0,
            "description_url": "https://commons.wikimedia.org/wiki/File:B.webm",
            "attribution": "Author", "license": "CC0", "license_url": "https://cc0",
        } for title in titles
    })

    report = enrich_provenance_sidecar(candidates, sidecar)

    assert candidates.read_text() == original
    assert calls == ["File:B.webm"]
    rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert [row["title"] for row in rows] == ["File:A.webm", "File:B.webm"]
    assert report == {"candidates": 2, "enriched": 2, "remaining": 0}


def test_provenance_sidecar_rejects_unknown_or_duplicate_titles(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps({"title": "File:A.webm"}) + "\n")
    sidecar = tmp_path / "provenance.jsonl"
    duplicate = {"title": "File:B.webm"}
    sidecar.write_text(json.dumps(duplicate) + "\n" + json.dumps(duplicate) + "\n")

    with pytest.raises(ValueError, match="duplicate|unknown"):
        enrich_provenance_sidecar(candidates, sidecar)


def test_provenance_sidecar_rejects_incomplete_api_provenance(tmp_path, monkeypatch):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps({"title": "File:A.webm"}) + "\n")
    monkeypatch.setattr("commons_review.fetch_review_media_batch", lambda titles: {
        "File:A.webm": {
            "duration": 3, "preview_url": "https://upload.wikimedia.org/a.webm",
            "source_revision": "1", "original_media_url": "https://upload.wikimedia.org/a.webm",
            "original_media_sha1": "", "original_media_timestamp": "now",
            "description_url": "https://commons.wikimedia.org/wiki/File:A.webm",
            "attribution": "Author", "license": "CC0", "license_url": "https://cc0",
        }
    })

    with pytest.raises(ValueError, match="incomplete provenance"):
        enrich_provenance_sidecar(candidates, tmp_path / "provenance.jsonl")


def test_verify_provenance_sidecar_detects_media_replacement(tmp_path, monkeypatch):
    sidecar = tmp_path / "provenance.jsonl"
    sidecar.write_text(json.dumps({
        "title": "File:A.webm", "source_revision": "1",
        "original_media_url": "https://upload.wikimedia.org/a.webm",
        "original_media_sha1": "old", "original_media_timestamp": "yesterday",
    }) + "\n")
    monkeypatch.setattr("commons_review.fetch_review_media_batch", lambda titles: {
        "File:A.webm": {
            "duration": 3, "preview_url": "https://upload.wikimedia.org/a-240.webm",
            "source_revision": "2", "original_media_url": "https://upload.wikimedia.org/a.webm",
            "original_media_sha1": "new", "original_media_timestamp": "today",
        }
    })

    with pytest.raises(ValueError, match="media revision changed"):
        verify_provenance_sidecar(sidecar, ["File:A.webm"])


def test_fetch_review_media_batch_uses_one_request_for_multiple_titles(monkeypatch):
    captured = {}
    imageinfo = [{"descriptionurl": "https://commons.wikimedia.org/wiki/File:X.webm",
        "extmetadata": {"Artist": {"value": "Author"},
        "LicenseShortName": {"value": "CC0"}, "LicenseUrl": {"value": "https://cc0"}}}]
    payload = {"query": {"pages": {
        "1": {"title": "File:A.webm", "revisions": [{"revid": 1}], "videoinfo": [{"duration": 2,
            "derivatives": [{"src": "https://upload.wikimedia.org/a.webm",
                             "width": 320, "height": 240, "type": "video/webm"}]}], "imageinfo": imageinfo},
        "2": {"title": "File:B.webm", "revisions": [{"revid": 2}], "videoinfo": [{"duration": 3,
            "derivatives": [{"src": "https://upload.wikimedia.org/b.webm",
                             "width": 320, "height": 240, "type": "video/webm"}]}], "imageinfo": imageinfo},
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
    imageinfo = [{"descriptionurl": "https://commons.wikimedia.org/wiki/File:X.webm",
        "extmetadata": {"Artist": {"value": "Author"},
        "LicenseShortName": {"value": "CC0"}, "LicenseUrl": {"value": "https://cc0"}}}]
    payload = {"query": {"pages": {
        "1": {"title": "File:Good.webm", "revisions": [{"revid": 1}], "videoinfo": [{"duration": 2,
            "derivatives": [{"src": "https://upload.wikimedia.org/good.webm",
                             "width": 320, "height": 240, "type": "video/webm"}]}], "imageinfo": imageinfo},
        "2": {"title": "File:Broken.ogg", "revisions": [{"revid": 2}], "videoinfo": [{"duration": 3,
            "derivatives": []}], "imageinfo": imageinfo},
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


def test_completed_review_audit_binds_snapshot_and_provenance(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps({
            "title": "File:A.webm", "review_status": "approved",
            "subject": "cat", "action": "playing", "scene": "indoors",
            "prompt": "a cat playing indoors", "start_seconds": 1,
            "duplicate_group": "series:a", "review_notes": "single cat",
        }) + "\n" + json.dumps({
            "title": "File:B.webm", "review_status": "rejected",
            "review_notes": "two cats",
        }) + "\n"
    )
    provenance = tmp_path / "provenance.jsonl"
    provenance.write_text(
        json.dumps({"title": "File:A.webm", "original_media_sha1": "aaa"}) + "\n"
        + json.dumps({"title": "File:B.webm", "original_media_sha1": "bbb"}) + "\n"
    )

    report = completed_review_audit(
        candidates, ["File:B.webm", "File:A.webm"], provenance_path=provenance,
    )

    assert report["candidate_sha256"] == __import__("hashlib").sha256(
        candidates.read_bytes()
    ).hexdigest()
    assert report["statuses"] == {"approved": 1, "rejected": 1}
    assert [row["title"] for row in report["records"]] == [
        "File:A.webm", "File:B.webm",
    ]
    assert report["records"][0]["original_media_sha1"] == "aaa"


def test_completed_review_audit_rejects_pending_or_unknown_titles(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps({
        "title": "File:A.webm", "review_status": "pending",
    }) + "\n")

    with pytest.raises(ValueError, match="not complete"):
        completed_review_audit(candidates, ["File:A.webm"])
    with pytest.raises(ValueError, match="exactly once"):
        completed_review_audit(candidates, ["File:Missing.webm"])

    candidates.write_text(json.dumps({
        "title": "File:A.webm", "review_status": "rejected",
    }) + "\n")
    with pytest.raises(ValueError, match="media SHA-1"):
        completed_review_audit(candidates, ["File:A.webm"])


def test_audit_cli_does_not_overwrite_any_input(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text("{}\n")
    titles = tmp_path / "titles.json"
    titles.write_text("[]\n")
    provenance = tmp_path / "provenance.jsonl"
    provenance.write_text("{}\n")

    for output in (titles, provenance):
        before = output.read_bytes()
        with pytest.raises(ValueError, match="differ from audit inputs"):
            main([
                "audit", str(candidates), str(output), "--titles", str(titles),
                "--provenance", str(provenance),
            ])
        assert output.read_bytes() == before


def test_pair_deficit_priority_ranks_only_playable_pending_candidates():
    rows = [{
        "title": "File:Cat pouncing on a mouse in a garden.webm",
        "description": "One cat playing outside on grass",
        "subject": "cat", "review_status": "pending",
        "duration_seconds": 4, "review_preview_url": "https://upload.wikimedia.org/a.webm",
    }, {
        "title": "File:Puppy with plush toy at home.webm",
        "description": "A dog playing in a room",
        "subject": "dog", "review_status": "pending",
        "duration_seconds": 4, "review_preview_url": "https://upload.wikimedia.org/b.webm",
    }, {
        "title": "File:Cat software demo.webm", "description": "playing with software",
        "subject": "cat", "review_status": "pending", "duration_seconds": 4,
        "review_preview_url": "https://upload.wikimedia.org/c.webm",
    }, {
        "title": "File:Approved dog.webm", "description": "dog playing at home",
        "subject": "dog", "review_status": "approved", "duration_seconds": 4,
        "review_preview_url": "https://upload.wikimedia.org/d.webm",
    }, {
        "title": "File:Cat toy at home.webm", "description": "cat playing indoors",
        "subject": "cat", "review_status": "pending", "duration_seconds": 4,
        "review_preview_url": "https://upload.wikimedia.org/e.webm",
    }]

    ranked = prioritize_pair_deficit_rows(
        rows, target_pairs=[("cat", "playing"), ("playing", "indoors")],
    )

    assert [row["title"] for row in ranked] == [
        "File:Puppy with plush toy at home.webm",
        "File:Cat pouncing on a mouse in a garden.webm",
    ]
    assert ranked[0]["target_pairs"] == ["playing|indoors"]
    assert ranked[1]["target_pairs"] == ["cat|playing"]
    assert all(row["title"] != "File:Cat toy at home.webm" for row in ranked)


def test_pair_deficit_priority_rejects_unsupported_pairs():
    with pytest.raises(ValueError, match="supported target pairs"):
        prioritize_pair_deficit_rows([], target_pairs=[("cat", "indoors")])


def test_pair_priority_cli_writes_read_only_report(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps({
        "title": "File:Puppy toy at home.webm", "description": "dog playing in a room",
        "subject": "dog", "review_status": "pending", "duration_seconds": 3,
        "review_preview_url": "https://upload.wikimedia.org/a.webm",
    }) + "\n")
    output = tmp_path / "priority.jsonl"

    assert main([
        "pair-priority", str(candidates), str(output),
        "--target-pair", "cat|playing", "--target-pair", "playing|indoors",
    ]) == 0

    report = json.loads(output.read_text())
    assert report["target_pairs"] == ["playing|indoors"]
    assert json.loads(candidates.read_text())["review_status"] == "pending"


def test_pair_priority_cli_does_not_overwrite_provenance(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text("{}\n")
    provenance = tmp_path / "provenance.jsonl"
    provenance.write_text("{}\n")
    before = provenance.read_bytes()

    with pytest.raises(ValueError, match="differ from report inputs"):
        main([
            "pair-priority", str(candidates), str(provenance),
            "--provenance", str(provenance), "--target-pair", "cat|playing",
        ])
    assert provenance.read_bytes() == before
