import json

import pytest

from commons_import import CommonsCandidate, candidate_sample_id
from real_data import RealVideoRecord
from prepare_commons import main


def test_search_writes_pending_review_candidates(tmp_path, monkeypatch):
    specs = tmp_path / "specs.json"
    specs.write_text(json.dumps([{
        "query": "cat running", "subject": "cat", "action": "running",
        "scene": "grass", "prompt": "a cat running on grass", "limit": 2,
    }]))
    output = tmp_path / "candidates.jsonl"
    candidate = CommonsCandidate(
        title="File:Cat runs.webm", media_url="https://upload/cat.webm",
        description_url="https://commons/File:Cat_runs.webm",
        description="Cat running", attribution="Author", license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0",
        subject="cat", action="running", scene="grass",
        prompt="a cat running on grass",
    )
    monkeypatch.setattr("prepare_commons.search_candidates", lambda *a, **k: [candidate])

    assert main(["search", str(specs), str(output)]) == 0
    row = json.loads(output.read_text())
    assert row["review_status"] == "pending"
    assert row["title"] == candidate.title
    assert row["query"] == "cat running"


def test_category_persists_batches_and_resumes_traversal_state(tmp_path, monkeypatch):
    output = tmp_path / "candidates.jsonl"
    state = tmp_path / "category-state.json"
    candidate = CommonsCandidate(
        title="File:Runner.webm", media_url="https://upload/runner.webm",
        description_url="https://commons/runner", description="Runner",
        attribution="Author", license="CC0", license_url="https://cc0",
        subject="person", action="review_required", scene="review_required",
        prompt="review the person action and scene",
    )

    def interrupted_category(*args, **kwargs):
        kwargs["on_batch"]([candidate], {"queue": [["Category:Next", 1]]})
        raise RuntimeError("rate limited")

    monkeypatch.setattr("prepare_commons.category_candidates", interrupted_category)
    import pytest
    with pytest.raises(RuntimeError, match="rate limited"):
        main(["category", "People running", "person", str(output),
              "--max-depth", "2", "--state", str(state)])
    assert json.loads(output.read_text())["title"] == candidate.title
    assert json.loads(state.read_text())["traversal"] == {
        "queue": [["Category:Next", 1]],
    }

    captured = []
    def resumed_category(*args, **kwargs):
        captured.append(kwargs["state"])
        kwargs["on_batch"]([], {"queue": []})
        return []

    monkeypatch.setattr("prepare_commons.category_candidates", resumed_category)
    main(["category", "People running", "person", str(output),
          "--max-depth", "2", "--state", str(state)])
    assert captured == [{"queue": [["Category:Next", 1]]}]
    saved = json.loads(state.read_text())
    assert saved["completed"] is True
    assert saved["traversal"] == {"queue": []}


def test_category_rejects_state_from_a_different_traversal_contract(tmp_path):
    output = tmp_path / "candidates.jsonl"
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "completed": False,
        "contract": {"category": "Cats", "subject": "cat", "max_depth": 1,
                     "limit": 100},
        "traversal": {},
    }))
    import pytest
    with pytest.raises(ValueError, match="category state contract"):
        main(["category", "People running", "person", str(output),
              "--max-depth", "2", "--limit", "100", "--state", str(state)])


def test_category_continues_when_global_duplicates_leave_limit_unfilled(
    tmp_path, monkeypatch,
):
    output = tmp_path / "candidates.jsonl"
    output.write_text(json.dumps({
        "title": "File:Duplicate.webm", "query": "category:Old",
    }) + "\n")
    duplicate = CommonsCandidate(
        title="File:Duplicate.webm", media_url="https://upload/duplicate.webm",
        description_url="https://commons/duplicate", description="Duplicate",
        attribution="Author", license="CC0", license_url="https://cc0",
        subject="person", action="review_required", scene="review_required",
        prompt="review the person action and scene",
    )
    fresh = CommonsCandidate(
        title="File:Fresh.webm", media_url="https://upload/fresh.webm",
        description_url="https://commons/fresh", description="Fresh",
        attribution="Author", license="CC0", license_url="https://cc0",
        subject="person", action="review_required", scene="review_required",
        prompt="review the person action and scene",
    )
    calls = []

    def category(*args, **kwargs):
        calls.append(kwargs["limit"])
        if len(calls) == 1:
            kwargs["on_batch"]([duplicate], {"queue": [["Category:Next", 1]]})
        else:
            kwargs["on_batch"]([fresh], {"queue": []})
        return []

    monkeypatch.setattr("prepare_commons.category_candidates", category)
    main(["category", "People", "person", str(output), "--limit", "1"])

    assert calls == [1, 1]
    assert [json.loads(line)["title"] for line in output.read_text().splitlines()] == [
        "File:Duplicate.webm", "File:Fresh.webm",
    ]


def test_category_default_state_is_scoped_to_the_traversal(tmp_path, monkeypatch):
    output = tmp_path / "candidates.jsonl"

    def empty_category(*args, **kwargs):
        kwargs["on_batch"]([], {"queue": []})
        return []

    monkeypatch.setattr("prepare_commons.category_candidates", empty_category)
    main(["category", "Cats", "cat", str(output)])
    main(["category", "People running", "person", str(output)])

    states = sorted(tmp_path.glob("candidates.jsonl.category-*.state.json"))
    assert len(states) == 2
    assert {json.loads(path.read_text())["contract"]["category"] for path in states} == {
        "Cats", "People running",
    }


def test_category_does_not_count_same_category_rows_from_another_contract(
    tmp_path, monkeypatch,
):
    output = tmp_path / "candidates.jsonl"
    output.write_text(json.dumps({
        "title": "File:Old.webm", "query": "category:Shared",
        "category_traversal_id": "different-contract",
    }) + "\n")
    calls = []

    def empty_category(*args, **kwargs):
        calls.append(kwargs["limit"])
        kwargs["on_batch"]([], {"queue": []})
        return []

    monkeypatch.setattr("prepare_commons.category_candidates", empty_category)
    main(["category", "Shared", "person", str(output), "--limit", "1"])

    assert calls == [1]


def test_search_resumes_completed_queries(tmp_path, monkeypatch):
    specs = tmp_path / "specs.json"
    specs.write_text(json.dumps([
        {"query": query, "subject": "cat", "action": "running",
         "scene": "grass", "prompt": "a cat running", "limit": 1}
        for query in ("done", "new")
    ]))
    output = tmp_path / "candidates.jsonl"
    output.write_text(json.dumps({"title": "File:Done.webm", "query": "done"}) + "\n")
    calls = []

    def fake_search(query, **kwargs):
        calls.append(query)
        return []

    monkeypatch.setattr("prepare_commons.search_candidates", fake_search)
    assert main(["search", str(specs), str(output)]) == 0
    assert calls == ["new"]


def test_search_remembers_zero_result_queries(tmp_path, monkeypatch):
    specs = tmp_path / "specs.json"
    specs.write_text(json.dumps([{
        "query": "empty", "subject": "cat", "action": "running",
        "scene": "grass", "prompt": "a cat running", "limit": 1,
    }]))
    output = tmp_path / "candidates.jsonl"
    calls = []
    monkeypatch.setattr(
        "prepare_commons.search_candidates",
        lambda query, **kwargs: calls.append(query) or [],
    )
    main(["search", str(specs), str(output)])
    main(["search", str(specs), str(output)])
    assert calls == ["empty"]


def test_search_persists_each_page_and_resumes_its_continuation(tmp_path, monkeypatch):
    specs = tmp_path / "specs.json"
    specs.write_text(json.dumps([{
        "query": "many cats", "subject": "cat", "action": "review_required",
        "scene": "review_required", "prompt": "review cat", "limit": 2,
    }]))
    output = tmp_path / "candidates.jsonl"
    state = tmp_path / "state.json"
    candidate = CommonsCandidate(
        title="File:First.webm", media_url="https://upload/first.webm",
        description_url="https://commons/File:First.webm", description="Cat",
        attribution="Author", license="CC0", license_url="https://cc0",
        subject="cat", action="review_required", scene="review_required",
        prompt="review cat",
    )

    def interrupted_search(query, **kwargs):
        kwargs["on_page"]([candidate], {"gsroffset": 50})
        raise RuntimeError("rate limited")

    monkeypatch.setattr("prepare_commons.search_candidates", interrupted_search)
    with __import__("pytest").raises(RuntimeError, match="rate limited"):
        main(["search", str(specs), str(output), "--state", str(state)])
    assert json.loads(output.read_text())["title"] == candidate.title
    active = json.loads(state.read_text())["active_queries"]["many cats"]
    assert active == {"accepted": 1, "continuation": {"gsroffset": 50}}

    seen = []
    def resumed_search(query, **kwargs):
        seen.append((kwargs["limit"], kwargs["continuation"]))
        kwargs["on_page"]([], None)
        return []

    monkeypatch.setattr("prepare_commons.search_candidates", resumed_search)
    main(["search", str(specs), str(output), "--state", str(state)])
    assert seen == [(1, {"gsroffset": 50})]
    saved = json.loads(state.read_text())
    assert "many cats" in saved["completed_queries"]
    assert "many cats" not in saved["active_queries"]


def test_search_marks_an_already_satisfied_active_query_complete(tmp_path, monkeypatch):
    specs = tmp_path / "specs.json"
    specs.write_text(json.dumps([{
        "query": "enough cats", "subject": "cat", "action": "review_required",
        "scene": "review_required", "prompt": "review cat", "limit": 1,
    }]))
    output = tmp_path / "candidates.jsonl"
    output.write_text(json.dumps({"title": "File:Cat.webm", "query": "enough cats"}) + "\n")
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "completed_queries": [],
        "active_queries": {
            # Simulate a crash after the candidate file was atomically replaced
            # but before the matching state replacement completed.
            "enough cats": {"accepted": 0, "continuation": {"gsroffset": 50}},
        },
    }))
    calls = []
    monkeypatch.setattr(
        "prepare_commons.search_candidates",
        lambda *args, **kwargs: calls.append((args, kwargs)) or [],
    )

    main(["search", str(specs), str(output), "--state", str(state)])

    assert calls == []
    saved = json.loads(state.read_text())
    assert saved["completed_queries"] == ["enough cats"]
    assert saved["active_queries"] == {}


def test_search_continues_when_cross_query_duplicates_leave_target_unfilled(
    tmp_path, monkeypatch,
):
    specs = tmp_path / "specs.json"
    specs.write_text(json.dumps([{
        "query": "more cats", "subject": "cat", "action": "review_required",
        "scene": "review_required", "prompt": "review cat", "limit": 2,
    }]))
    output = tmp_path / "candidates.jsonl"
    output.write_text(json.dumps({"title": "File:Duplicate.webm", "query": "old query"}) + "\n")
    state = tmp_path / "state.json"
    calls = []

    def candidate(title):
        return CommonsCandidate(
            title=title, media_url=f"https://upload/{title}",
            description_url=f"https://commons/{title}", description="Cat",
            attribution="Author", license="CC0", license_url="https://cc0",
            subject="cat", action="review_required", scene="review_required",
            prompt="review cat",
        )

    def overlapping_search(query, **kwargs):
        calls.append((kwargs["limit"], kwargs["continuation"]))
        if len(calls) == 1:
            kwargs["on_page"](
                [candidate("File:Duplicate.webm"), candidate("File:First.webm")],
                {"gsroffset": 50},
            )
        else:
            kwargs["on_page"]([candidate("File:Second.webm")], None)
        return []

    monkeypatch.setattr("prepare_commons.search_candidates", overlapping_search)
    main(["search", str(specs), str(output), "--state", str(state)])

    assert calls == [(2, {}), (1, {"gsroffset": 50})]
    assert [json.loads(line)["title"] for line in output.read_text().splitlines()] == [
        "File:Duplicate.webm", "File:First.webm", "File:Second.webm",
    ]
    saved = json.loads(state.read_text())
    assert saved["completed_queries"] == ["more cats", "old query"]
    assert saved["active_queries"] == {}


def test_search_writes_initial_active_state_before_the_first_network_page(
    tmp_path, monkeypatch,
):
    specs = tmp_path / "specs.json"
    specs.write_text(json.dumps([{
        "query": "new cats", "subject": "cat", "action": "review_required",
        "scene": "review_required", "prompt": "review cat", "limit": 2,
    }]))
    output = tmp_path / "candidates.jsonl"
    state = tmp_path / "state.json"

    def assert_wal_exists(query, **kwargs):
        saved = json.loads(state.read_text())
        assert saved["active_queries"]["new cats"] == {
            "accepted": 0, "continuation": {},
        }
        kwargs["on_page"]([], None)
        return []

    monkeypatch.setattr("prepare_commons.search_candidates", assert_wal_exists)
    main(["search", str(specs), str(output), "--state", str(state)])

    saved = json.loads(state.read_text())
    assert saved["completed_queries"] == ["new cats"]
    assert saved["active_queries"] == {}


def test_download_normalizes_only_explicitly_approved_rows(tmp_path, monkeypatch):
    rows = [
        {
            "title": f"File:{name}.webm",
            "media_url": f"https://upload.wikimedia.org/{name}.webm",
            "description_url": f"https://commons.wikimedia.org/{name}",
            "description": name,
            "attribution": "Author", "license": "CC0",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0",
            "subject": "cat", "action": "running", "scene": "grass",
            "prompt": "a cat running on grass", "start_seconds": 0,
                "review_status": status, "query": "cat running", "source_revision": "1",
                "original_media_sha1": "abc", "original_media_timestamp": "now",
        }
        for name, status in (("approved", "approved"), ("pending", "pending"))
    ]
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text("".join(json.dumps(row) + "\n" for row in rows))
    calls = []

    def fake_normalize(candidate, output_dir, **kwargs):
        calls.append(candidate.title)
        return {"sample_id": candidate.title}

    monkeypatch.setattr("prepare_commons.normalize_clip", fake_normalize)
    monkeypatch.setattr("prepare_commons.asdict", lambda value: value)
    manifest = tmp_path / "records.jsonl"

    assert main(["download", str(candidates), str(tmp_path / "clips"), str(manifest)]) == 0
    assert calls == ["File:approved.webm"]
    assert json.loads(manifest.read_text())["sample_id"] == "File:approved.webm"


def test_download_persists_each_success_and_resumes(tmp_path, monkeypatch):
    rows = []
    for name in ("one", "two"):
        rows.append({
            "title": f"File:{name}.webm",
            "media_url": f"https://upload.wikimedia.org/{name}.webm",
            "description_url": f"https://commons.wikimedia.org/{name}",
            "description": name, "attribution": "Author", "license": "CC0",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0",
            "subject": "cat", "action": "running", "scene": "grass",
            "prompt": "a cat running", "start_seconds": 0,
            "review_status": "approved", "query": "cats", "source_revision": "1",
            "original_media_sha1": "abc", "original_media_timestamp": "now",
        })
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = tmp_path / "manifest.jsonl"
    calls = []

    def first_pass(candidate, *args, **kwargs):
        calls.append(candidate.title)
        if candidate.title.endswith("two.webm"):
            raise RuntimeError("interrupted")
        return {"sample_id": candidate_sample_id(candidate.title)}

    monkeypatch.setattr("prepare_commons.normalize_clip", first_pass)
    monkeypatch.setattr("prepare_commons.asdict", lambda value: value)
    import pytest
    with pytest.raises(RuntimeError, match="interrupted"):
        main(["download", str(candidates), str(tmp_path / "clips"), str(manifest)])
    assert json.loads(manifest.read_text())["sample_id"] == candidate_sample_id("File:one.webm")

    monkeypatch.setattr(
        "prepare_commons.normalize_clip",
        lambda candidate, *args, **kwargs: calls.append(candidate.title)
        or {"sample_id": candidate_sample_id(candidate.title)},
    )
    main(["download", str(candidates), str(tmp_path / "clips"), str(manifest)])
    assert calls.count("File:one.webm") == 1


def test_download_uses_reviewed_derivative_and_accepts_enriched_fields(tmp_path, monkeypatch):
    row = {
        "title": "File:Cat.webm",
        "media_url": "https://upload.wikimedia.org/original.webm",
        "review_preview_url": "https://upload.wikimedia.org/240p.webm",
        "duration_seconds": 12.0,
        "description_url": "https://commons.wikimedia.org/wiki/File:Cat.webm",
        "description": "Cat", "attribution": "Author", "license": "CC0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0",
        "subject": "cat", "action": "walking", "scene": "indoors",
        "prompt": "a cat walking indoors", "start_seconds": 2,
        "review_status": "approved", "review_notes": "watched", "query": "cats",
        "source_revision": "1", "original_media_sha1": "abc",
        "original_media_timestamp": "now",
    }
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps(row) + "\n")
    captured = {}

    def fake_normalize(candidate, *args, **kwargs):
        captured["candidate"] = candidate
        return {"sample_id": candidate_sample_id(candidate.title)}

    monkeypatch.setattr("prepare_commons.normalize_clip", fake_normalize)
    monkeypatch.setattr("prepare_commons.asdict", lambda value: value)
    main(["download", str(candidates), str(tmp_path / "clips"), str(tmp_path / "out.jsonl")])
    assert captured["candidate"].media_url.endswith("240p.webm")


def test_download_writes_clip_path_relative_to_manifest(tmp_path, monkeypatch):
    row = {
        "title": "File:Cat.webm", "media_url": "https://upload.wikimedia.org/cat.webm",
        "description_url": "https://commons.wikimedia.org/wiki/File:Cat.webm",
        "description": "Cat", "attribution": "Author", "license": "CC0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0",
        "subject": "cat", "action": "walking", "scene": "indoors",
        "prompt": "a cat walking indoors", "start_seconds": 0,
        "review_status": "approved", "source_revision": "1",
        "original_media_url": "https://upload.wikimedia.org/cat.webm",
        "original_media_sha1": "abc", "original_media_timestamp": "now",
    }
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps(row) + "\n")
    clips = tmp_path / "dataset" / "clips"
    manifest = tmp_path / "dataset" / "source.jsonl"
    clip = clips / "cat.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"video")
    record = RealVideoRecord(
        sample_id="cat", source_id="commons:cat", clip_path=str(clip),
        prompt="a cat walking indoors", subject="cat", action="walking",
        scene="indoors", source_url=row["description_url"], license="CC0",
        attribution="Author", sha256="a" * 64, perceptual_hash="0" * 16,
        license_url=row["license_url"], original_media_url=row["media_url"],
        page_revision="1", original_media_sha1="abc", original_media_timestamp="now",
        downloaded_media_url=row["media_url"], downloaded_media_kind="original",
        downloaded_media_sha256="b" * 64, downloaded_at="now",
    )
    monkeypatch.setattr("prepare_commons.normalize_clip", lambda *args, **kwargs: record)

    main(["download", str(candidates), str(clips), str(manifest)])

    assert json.loads(manifest.read_text())["clip_path"] == "clips/cat.mp4"


def test_download_rejects_clip_outside_manifest_tree(tmp_path, monkeypatch):
    row = {
        "title": "File:Cat.webm", "media_url": "https://upload.wikimedia.org/cat.webm",
        "description_url": "https://commons.wikimedia.org/wiki/File:Cat.webm",
        "description": "Cat", "attribution": "Author", "license": "CC0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0",
        "subject": "cat", "action": "walking", "scene": "indoors",
        "prompt": "a cat walking indoors", "start_seconds": 0,
        "review_status": "approved", "source_revision": "1",
        "original_media_url": "https://upload.wikimedia.org/cat.webm",
        "original_media_sha1": "abc", "original_media_timestamp": "now",
    }
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps(row) + "\n")
    outside = tmp_path / "outside.mp4"
    record = RealVideoRecord(
        sample_id="cat", source_id="commons:cat", clip_path=str(outside),
        prompt="a cat walking indoors", subject="cat", action="walking",
        scene="indoors", source_url=row["description_url"], license="CC0",
        attribution="Author", sha256="a" * 64, perceptual_hash="0" * 16,
        license_url=row["license_url"], original_media_url=row["media_url"],
        page_revision="1", original_media_sha1="abc", original_media_timestamp="now",
        downloaded_media_url=row["media_url"], downloaded_media_kind="original",
        downloaded_media_sha256="b" * 64, downloaded_at="now",
    )
    monkeypatch.setattr("prepare_commons.normalize_clip", lambda *args, **kwargs: record)

    import pytest
    with pytest.raises(ValueError, match="inside the manifest directory"):
        main([
            "download", str(candidates), str(tmp_path / "clips"),
            str(tmp_path / "dataset" / "source.jsonl"),
        ])


def test_download_rejects_unreviewed_placeholder_labels(tmp_path):
    row = {
        "title": "File:Cat.webm", "media_url": "https://upload.wikimedia.org/cat.webm",
        "description_url": "https://commons.wikimedia.org/wiki/File:Cat.webm",
        "description": "Cat", "attribution": "Author", "license": "CC0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0",
        "subject": "cat", "action": "review_required", "scene": "review_required",
        "prompt": "review the cat action and scene", "start_seconds": 0,
        "review_status": "approved", "query": "cats", "source_revision": "1",
        "original_media_sha1": "abc", "original_media_timestamp": "now",
    }
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps(row) + "\n")

    import pytest
    with pytest.raises(ValueError, match="placeholder labels"):
        main(["download", str(candidates), str(tmp_path / "clips"), str(tmp_path / "out.jsonl")])


def test_download_rejects_approved_row_without_source_revision(tmp_path):
    row = {
        "title": "File:Cat.webm", "media_url": "https://upload.wikimedia.org/cat.webm",
        "description_url": "https://commons.wikimedia.org/wiki/File:Cat.webm",
        "description": "Cat", "attribution": "Author", "license": "CC0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0",
        "subject": "cat", "action": "walking", "scene": "indoors",
        "prompt": "a cat walking indoors", "start_seconds": 0,
        "review_status": "approved",
    }
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="source revision"):
        main(["download", str(candidates), str(tmp_path / "clips"), str(tmp_path / "out.jsonl")])


def test_download_rejects_appending_to_legacy_manifest(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text("")
    manifest = tmp_path / "source.jsonl"
    manifest.write_text(json.dumps({
        "sample_id": "legacy", "source_id": "commons:legacy", "clip_path": "clips/a.mp4",
        "prompt": "a cat", "subject": "cat", "action": "walking", "scene": "grass",
        "source_url": "https://commons.wikimedia.org/wiki/File:A.webm", "license": "CC0",
        "attribution": "Author", "sha256": "a" * 64,
    }) + "\n")

    with pytest.raises(ValueError, match="new output manifest"):
        main(["download", str(candidates), str(tmp_path / "clips"), str(manifest)])


def test_download_joins_provenance_sidecar_without_rewriting_candidates(tmp_path, monkeypatch):
    candidate = {
        "title": "File:Cat.webm", "media_url": "https://upload.wikimedia.org/cat.webm",
        "description_url": "https://commons.wikimedia.org/wiki/File:Cat.webm",
        "description": "Cat", "attribution": "Author", "license": "CC0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0",
        "subject": "cat", "action": "walking", "scene": "indoors",
        "prompt": "a cat walking indoors", "start_seconds": 0, "review_status": "approved",
    }
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps(candidate) + "\n")
    sidecar = tmp_path / "provenance.jsonl"
    sidecar.write_text(json.dumps({
        "title": "File:Cat.webm", "source_revision": "1",
        "original_media_url": candidate["media_url"], "original_media_sha1": "abc",
        "original_media_timestamp": "now",
        "review_preview_url": "https://upload.wikimedia.org/cat-240.webm",
        "duration_seconds": 4.0,
        "description_url": candidate["description_url"], "attribution": "Current Author",
        "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0",
    }) + "\n")
    captured = {}
    monkeypatch.setattr("prepare_commons.verify_provenance_sidecar", lambda *args: None)
    monkeypatch.setattr("prepare_commons.normalize_clip", lambda row, *args, **kwargs:
                        captured.setdefault("candidate", row) or {"sample_id": "unused"})
    monkeypatch.setattr("prepare_commons.asdict", lambda value: {"sample_id": "cat"})

    main(["download", str(candidates), str(tmp_path / "clips"), str(tmp_path / "out.jsonl"),
          "--provenance", str(sidecar)])

    assert candidates.read_text() == json.dumps(candidate) + "\n"
    assert captured["candidate"].source_revision == "1"
    assert captured["candidate"].media_url.endswith("cat-240.webm")
    assert captured["candidate"].attribution == "Current Author"
    assert captured["candidate"].license == "CC BY 4.0"


def test_download_rejects_incomplete_provenance_sidecar(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps({"title": "File:A.webm", "review_status": "approved"}) + "\n")
    sidecar = tmp_path / "provenance.jsonl"
    sidecar.write_text("")

    with pytest.raises(ValueError, match="provenance sidecar"):
        main(["download", str(candidates), str(tmp_path / "clips"), str(tmp_path / "out.jsonl"),
              "--provenance", str(sidecar)])


def test_download_rejects_sidecar_titles_not_in_candidate_snapshot(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps({"title": "File:A.webm", "review_status": "pending"}) + "\n")
    sidecar = tmp_path / "provenance.jsonl"
    sidecar.write_text(json.dumps({"title": "File:B.webm"}) + "\n")

    with pytest.raises(ValueError, match="unknown candidate titles"):
        main(["download", str(candidates), str(tmp_path / "clips"), str(tmp_path / "out.jsonl"),
              "--provenance", str(sidecar)])


@pytest.mark.parametrize("field", ["license_url", "attribution", "original_media_url"])
def test_download_rejects_incomplete_approved_provenance(tmp_path, field):
    row = {
        "title": "File:Cat.webm", "media_url": "https://upload.wikimedia.org/cat.webm",
        "original_media_url": "https://upload.wikimedia.org/cat.webm",
        "description_url": "https://commons.wikimedia.org/wiki/File:Cat.webm",
        "description": "Cat", "attribution": "Author", "license": "CC0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0",
        "subject": "cat", "action": "walking", "scene": "indoors",
        "prompt": "a cat walking indoors", "start_seconds": 0,
        "review_status": "approved", "source_revision": "1",
        "original_media_sha1": "abc", "original_media_timestamp": "now",
    }
    row[field] = ""
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="provenance"):
        main(["download", str(candidates), str(tmp_path / "clips"), str(tmp_path / "out.jsonl")])


def test_download_rejects_non_commons_media_hosts(tmp_path):
    row = {
        "title": "File:Cat.webm", "media_url": "file:///private/etc/passwd",
        "description_url": "https://commons.wikimedia.org/wiki/File:Cat.webm",
        "description": "Cat", "attribution": "Author", "license": "CC0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0",
        "subject": "cat", "action": "walking", "scene": "indoors",
        "prompt": "a cat walking indoors", "start_seconds": 0,
        "review_status": "approved", "query": "cats", "source_revision": "1",
        "original_media_url": "file:///private/etc/passwd",
        "original_media_sha1": "abc", "original_media_timestamp": "now",
    }
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps(row) + "\n")

    import pytest
    with pytest.raises(ValueError, match="Wikimedia upload URL"):
        main(["download", str(candidates), str(tmp_path / "clips"), str(tmp_path / "out.jsonl")])
