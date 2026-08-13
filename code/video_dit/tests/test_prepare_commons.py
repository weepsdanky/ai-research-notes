import json

from commons_import import CommonsCandidate, candidate_sample_id
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
            "review_status": status, "query": "cat running",
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
            "review_status": "approved", "query": "cats",
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


def test_download_rejects_unreviewed_placeholder_labels(tmp_path):
    row = {
        "title": "File:Cat.webm", "media_url": "https://upload.wikimedia.org/cat.webm",
        "description_url": "https://commons.wikimedia.org/wiki/File:Cat.webm",
        "description": "Cat", "attribution": "Author", "license": "CC0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0",
        "subject": "cat", "action": "review_required", "scene": "review_required",
        "prompt": "review the cat action and scene", "start_seconds": 0,
        "review_status": "approved", "query": "cats",
    }
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps(row) + "\n")

    import pytest
    with pytest.raises(ValueError, match="placeholder labels"):
        main(["download", str(candidates), str(tmp_path / "clips"), str(tmp_path / "out.jsonl")])


def test_download_rejects_non_commons_media_hosts(tmp_path):
    row = {
        "title": "File:Cat.webm", "media_url": "file:///private/etc/passwd",
        "description_url": "https://commons.wikimedia.org/wiki/File:Cat.webm",
        "description": "Cat", "attribution": "Author", "license": "CC0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0",
        "subject": "cat", "action": "walking", "scene": "indoors",
        "prompt": "a cat walking indoors", "start_seconds": 0,
        "review_status": "approved", "query": "cats",
    }
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps(row) + "\n")

    import pytest
    with pytest.raises(ValueError, match="Wikimedia upload URL"):
        main(["download", str(candidates), str(tmp_path / "clips"), str(tmp_path / "out.jsonl")])
