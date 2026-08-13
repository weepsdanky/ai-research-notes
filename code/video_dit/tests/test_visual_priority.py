import hashlib
import json
from pathlib import Path

import pytest
import torch
from PIL import Image
import urllib.error

from visual_priority import (
    CandidatePoster,
    fetch_locked_posters,
    load_input_snapshot,
    rank_visual_candidates,
    write_visual_priority_report,
)


def _row(title="File:Dog.webm", sha1="a" * 40, status="pending", subject="dog"):
    return {
        "title": title, "review_status": status, "subject": subject,
        "original_media_sha1": sha1,
        "original_media_timestamp": "2026-01-01T00:00:00Z",
    }


def test_fetch_locked_posters_rejects_media_revision_drift(tmp_path, monkeypatch):
    rows = [_row()]
    monkeypatch.setattr("visual_priority.fetch_poster_metadata", lambda titles: {
        titles[0]: {
            "original_media_sha1": "b" * 40,
            "original_media_timestamp": "2026-01-01T00:00:00Z",
            "poster_url": "https://upload.wikimedia.org/dog.jpg",
        },
    })

    with pytest.raises(ValueError, match="revision changed"):
        fetch_locked_posters(rows, tmp_path)
    assert not list(tmp_path.iterdir())


def test_download_poster_retries_rate_limit_and_writes_atomically(tmp_path, monkeypatch):
    from visual_priority import download_poster
    calls = []
    image = tmp_path / "source.jpg"
    Image.new("RGB", (8, 8), "blue").save(image)
    data = image.read_bytes()

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def read(self): return data

    def fake_open(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                request.full_url, 429, "rate", {"Retry-After": "0"}, None,
            )
        return Response()

    monkeypatch.setattr("visual_priority.urllib.request.urlopen", fake_open)
    output = tmp_path / "poster.jpg"
    download_poster("https://upload.wikimedia.org/poster.jpg", output)

    assert len(calls) == 2
    assert output.read_bytes() == data
    assert not output.with_suffix(".jpg.tmp").exists()


def test_fetch_locked_posters_downloads_only_pending_target_subjects(tmp_path, monkeypatch):
    rows = [
        _row(), _row("File:Cat.webm", "b" * 40, subject="cat"),
        _row("File:Done.webm", "c" * 40, status="approved"),
    ]
    monkeypatch.setattr("visual_priority.fetch_poster_metadata", lambda titles: {
        title: {
            "original_media_sha1": rows[index]["original_media_sha1"],
            "original_media_timestamp": rows[index]["original_media_timestamp"],
            "poster_url": f"https://upload.wikimedia.org/{index}.jpg",
        } for index, title in enumerate(titles)
    })
    def fake_download(url, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), "red").save(path, format="JPEG")
    monkeypatch.setattr("visual_priority.download_poster", fake_download)

    posters = fetch_locked_posters(rows, tmp_path, subjects={"dog"})

    assert len(posters) == 1
    assert posters[0].title == "File:Dog.webm"
    assert posters[0].poster_sha256 == hashlib.sha256(
        (tmp_path / Path(posters[0].poster_path).name).read_bytes()
    ).hexdigest()


def test_fetch_locked_posters_rechecks_revision_after_download(tmp_path, monkeypatch):
    row = _row()
    calls = []
    def metadata(titles):
        calls.append(titles)
        return {titles[0]: {
            "original_media_sha1": ("a" if len(calls) == 1 else "b") * 40,
            "original_media_timestamp": row["original_media_timestamp"],
            "poster_url": "https://upload.wikimedia.org/dog.jpg",
        }}
    def fake_download(url, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), "red").save(path)
    monkeypatch.setattr("visual_priority.fetch_poster_metadata", metadata)
    monkeypatch.setattr("visual_priority.download_poster", fake_download)

    with pytest.raises(ValueError, match="changed during"):
        fetch_locked_posters([row], tmp_path)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0])
def test_rank_rejects_nonfinite_or_zero_features(bad):
    posters = [CandidatePoster("a", "a" * 40, "t", "u", "p", "h")]
    images = torch.tensor([[bad, 0.0]])
    texts = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="finite non-zero"):
        rank_visual_candidates(
            posters, images, texts, positive_prompts=["positive"],
            negative_prompts=["negative"],
        )


def test_rank_visual_candidates_uses_explicit_positive_minus_negative_margin():
    posters = [
        CandidatePoster("a", "a" * 40, "t", "u", "p", "h"),
        CandidatePoster("b", "b" * 40, "t", "u", "p", "h"),
    ]
    image_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    text_features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    rows = rank_visual_candidates(
        posters, image_features, text_features,
        positive_prompts=["single dog playing indoors"],
        negative_prompts=["multiple dogs", "outdoor dog"],
    )

    assert [row["title"] for row in rows] == ["a", "b"]
    assert rows[0]["margin"] > rows[1]["margin"]
    assert rows[0]["rank"] == 1


def test_rank_accepts_tensors_created_in_inference_mode():
    posters = [CandidatePoster("a", "a" * 40, "t", "u", "p", "h")]
    with torch.inference_mode():
        images = torch.tensor([[1.0, 0.0]])
        texts = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    rows = rank_visual_candidates(
        posters, images, texts, positive_prompts=["positive"],
        negative_prompts=["negative"],
    )

    assert rows[0]["margin"] == 1.0


def test_report_binds_all_inputs_and_rejects_output_alias(tmp_path):
    candidates, provenance, checkpoint = (
        tmp_path / "candidates.jsonl", tmp_path / "provenance.jsonl", tmp_path / "clip.pt",
    )
    candidates.write_text("candidate\n")
    provenance.write_text("provenance\n")
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "report.json"

    candidate_snapshot = load_input_snapshot(candidates)
    provenance_snapshot = load_input_snapshot(provenance)
    checkpoint_snapshot = load_input_snapshot(checkpoint)
    candidates.write_text("replacement candidate\n")
    provenance.write_text("replacement provenance\n")
    checkpoint.write_bytes(b"replacement checkpoint")

    report = write_visual_priority_report(
        output, rows=[{"title": "a", "rank": 1}], candidates=candidates,
        provenance=provenance, checkpoint=checkpoint,
        candidate_sha256=candidate_snapshot.sha256,
        provenance_sha256=provenance_snapshot.sha256,
        checkpoint_sha256=checkpoint_snapshot.sha256,
        positive_prompts=["positive"], negative_prompts=["negative"],
    )

    saved = json.loads(output.read_text())
    assert saved == report
    assert saved["candidate_sha256"] == hashlib.sha256(b"candidate\n").hexdigest()
    assert saved["training_ready"] is False
    with pytest.raises(ValueError, match="differ"):
        write_visual_priority_report(
            candidates, rows=[], candidates=candidates, provenance=provenance,
            checkpoint=checkpoint, positive_prompts=[], negative_prompts=[],
            candidate_sha256=candidate_snapshot.sha256,
            provenance_sha256=provenance_snapshot.sha256,
            checkpoint_sha256=checkpoint_snapshot.sha256,
        )


def test_main_uses_frozen_input_and_checkpoint_snapshots(tmp_path, monkeypatch):
    import visual_priority

    candidates = tmp_path / "candidates.jsonl"
    provenance = tmp_path / "provenance.jsonl"
    checkpoint = tmp_path / "clip.pt"
    posters_dir = tmp_path / "posters"
    output = tmp_path / "report.json"
    candidate_data = (json.dumps(_row()) + "\n").encode()
    provenance_data = (json.dumps(_row()) + "\n").encode()
    checkpoint_data = b"verified checkpoint"
    candidates.write_bytes(candidate_data)
    provenance.write_bytes(provenance_data)
    checkpoint.write_bytes(checkpoint_data)
    monkeypatch.setattr(
        visual_priority, "OFFICIAL_VIT_B32_SHA256",
        hashlib.sha256(checkpoint_data).hexdigest(),
    )

    poster = CandidatePoster(
        "File:Dog.webm", "a" * 40, "2026-01-01T00:00:00Z",
        "https://upload.wikimedia.org/dog.jpg", "poster.jpg", "p" * 64,
    )
    def replace_inputs(*args, **kwargs):
        candidates.write_text("replacement candidate\n")
        provenance.write_text("replacement provenance\n")
        checkpoint.write_bytes(b"replacement checkpoint")
        return [poster]
    monkeypatch.setattr(visual_priority, "fetch_locked_posters", replace_inputs)

    def encode(posters, prompts, frozen_checkpoint, **kwargs):
        assert frozen_checkpoint.read_bytes() == checkpoint_data
        return torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    monkeypatch.setattr(visual_priority, "encode_clip", encode)

    assert visual_priority.main([
        str(candidates), str(provenance), str(posters_dir), str(output),
        "--checkpoint", str(checkpoint), "--positive", "positive",
        "--negative", "negative",
    ]) == 0
    report = json.loads(output.read_text())
    assert report["candidate_sha256"] == hashlib.sha256(candidate_data).hexdigest()
    assert report["provenance_sha256"] == hashlib.sha256(provenance_data).hexdigest()
    assert report["checkpoint_sha256"] == hashlib.sha256(checkpoint_data).hexdigest()
