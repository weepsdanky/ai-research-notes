import hashlib
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

import temporal_priority as target


def _snap(value: bytes) -> target.Snapshot:
    return target.Snapshot(value, hashlib.sha256(value).hexdigest())


def _inputs():
    candidate = {
        "title": "File:Dog.webm", "review_status": "pending", "subject": "dog",
    }
    provenance = {
        "title": "File:Dog.webm", "duration_seconds": 4.0,
        "original_media_sha1": "a" * 40,
        "original_media_timestamp": "2026-01-01T00:00:00Z",
        "review_preview_url": "https://upload.wikimedia.org/dog.webm",
    }
    c = _snap((json.dumps(candidate) + "\n").encode())
    p = _snap((json.dumps(provenance) + "\n").encode())
    upstream = _snap(json.dumps({
        "candidate_sha256": c.sha256, "provenance_sha256": p.sha256,
        "checkpoint_sha256": target.OFFICIAL_VIT_B32_SHA256,
        "metric": target.UPSTREAM_METRIC, "training_ready": False,
        "records": [{
            "title": candidate["title"], "rank": 1, "margin": 0.2,
            "original_media_sha1": provenance["original_media_sha1"],
            "original_media_timestamp": provenance["original_media_timestamp"],
            "poster_sha256": "b" * 64,
        }],
    }).encode())
    return c, p, upstream


def test_shortlist_requires_upstream_snapshot_binding():
    candidate, provenance, upstream = _inputs()
    broken = _snap(json.dumps({
        "candidate_sha256": "0" * 64,
        "provenance_sha256": provenance.sha256, "records": [],
    }).encode())
    with pytest.raises(ValueError, match="candidate snapshot"):
        target.lock_shortlist(candidate, provenance, broken, limit=1, subjects={"dog"})


def test_shortlist_rejects_arbitrary_or_tampered_upstream_report():
    candidate, provenance, upstream = _inputs()
    value = json.loads(upstream.data)
    value["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="visual-priority"):
        target.lock_shortlist(
            candidate, provenance, _snap(json.dumps(value).encode()),
            limit=1, subjects={"dog"},
        )
    value = json.loads(upstream.data)
    value["records"][0]["margin"] = float("nan")
    with pytest.raises(ValueError, match="margins finite"):
        target.lock_shortlist(
            candidate, provenance, _snap(json.dumps(value).encode()),
            limit=1, subjects={"dog"},
        )


def test_shortlist_locks_pending_row_and_provenance():
    candidate, provenance, upstream = _inputs()
    rows = target.lock_shortlist(
        candidate, provenance, upstream, limit=1, subjects={"dog"},
    )
    assert rows[0]["poster_rank"] == 1
    assert rows[0]["original_media_sha1"] == "a" * 40


def test_freeze_model_requires_official_weight_and_eight_frames(tmp_path, monkeypatch):
    source, frozen = tmp_path / "source", tmp_path / "frozen"
    source.mkdir()
    for name in target.MODEL_FILES:
        (source / name).write_bytes(
            b'{"vision_config":{"num_frames":8}}' if name == "config.json" else b"x"
        )
    monkeypatch.setattr(target, "MODEL_FILE_SHA256", {
        name: hashlib.sha256(
            b'{"vision_config":{"num_frames":8}}' if name == "config.json" else b"x"
        ).hexdigest()
        for name in target.MODEL_FILES
    })
    hashes = target.freeze_model(source, frozen)
    assert hashes["model.safetensors"] == hashlib.sha256(b"x").hexdigest()
    assert (frozen / "model.safetensors").read_bytes() == b"x"

    (source / "tokenizer.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="tokenizer.json"):
        target.freeze_model(source, tmp_path / "tampered")
    (source / "tokenizer.json").write_bytes(b"x")

    (source / "config.json").write_text('{"vision_config":{"num_frames":32}}')
    with pytest.raises(ValueError, match="config.json"):
        target.freeze_model(source, tmp_path / "other")


def test_xclip_scores_runs_forward_and_reverse_order():
    class Processor:
        def __call__(self, *, text, videos, **kwargs):
            return {"pixel_values": torch.tensor([[image.getpixel((0, 0))[0] for image in videos[0]]])}
    class Output:
        def __init__(self, logits): self.logits_per_video = logits
    class Model:
        def __call__(self, pixel_values):
            first = pixel_values[0, 0].float()
            return Output(torch.stack((first, -first)).reshape(1, 2))
    frames = [Image.new("RGB", (1, 1), (value, 0, 0)) for value in (1, 2, 3)]
    forward, reverse = target.xclip_scores(
        Model(), Processor(), frames, ["positive", "negative"], "cpu",
    )
    assert forward == [1.0, -1.0]
    assert reverse == [3.0, -3.0]


def test_rerank_records_order_sensitivity_and_stable_rank():
    def video(title):
        return target.LockedVideo(title, 2, "a" * 40, "t", "u", "p", "h")
    rows = target.rerank(
        [video("b"), video("a")], [[2, 0], [2, 0]], [[1, 0], [2, 0]],
        positive_count=1,
    )
    assert [row["title"] for row in rows] == ["a", "b"]
    assert rows[1]["order_sensitivity"] == 1.0


def test_ranking_metrics_reports_recall_and_average_precision():
    rows = [{"title": title} for title in ("a", "b", "c", "d")]
    metrics = target.ranking_metrics(
        rows, {"a": True, "b": False, "c": True, "d": False}, cutoffs=[1, 2, 4],
    )
    assert metrics["recall_at_k"] == {"1": 0.5, "2": 0.5, "4": 1.0}
    assert metrics["average_precision"] == pytest.approx((1 + 2 / 3) / 2)


def test_ranking_metrics_rejects_single_class_or_title_mismatch():
    with pytest.raises(ValueError, match="positive and negative"):
        target.ranking_metrics([{"title": "a"}], {"a": True}, cutoffs=[1])
    with pytest.raises(ValueError, match="identical titles"):
        target.ranking_metrics(
            [{"title": "a"}, {"title": "b"}], {"a": True, "c": False}, cutoffs=[1],
        )


def test_temporal_window_starts_are_bounded_and_preregistered():
    assert target.temporal_window_starts(1.5) == (0.0,)
    assert target.temporal_window_starts(4) == (0.0, 1.5)
    starts = target.temporal_window_starts(100, max_windows=8)
    assert len(starts) == 8
    assert starts[0] == 0 and starts[-1] == 97.5


def test_score_windows_selects_best_margin_and_preserves_every_window(monkeypatch):
    windows = [
        target.TemporalWindow(
            i, i + 2, (i + .5,), (str(i),), (Image.new("RGB", (1, 1)),),
        )
        for i in (0.0, 2.0)
    ]
    values = iter([([1, 0], [1, 0]), ([3, 0], [2, 0])])
    monkeypatch.setattr(target, "xclip_scores", lambda *args, **kwargs: next(values))
    forward, reverse, scores = target.score_temporal_windows(
        object(), object(), windows, ["positive", "negative"],
        positive_count=1, device="cpu",
    )
    assert forward == [3, 0] and reverse == [2, 0]
    assert [row["start_seconds"] for row in scores] == [0.0, 2.0]


def test_report_binds_snapshots_and_is_never_training_ready(tmp_path):
    candidate, provenance, upstream = _inputs()
    report = target.write_report(
        tmp_path / "report.json", rows=[], candidate_snapshot=candidate,
        provenance_snapshot=provenance, upstream_snapshot=upstream,
        model_hashes={"model.safetensors": target.MODEL_FILE_SHA256["model.safetensors"]},
        positive=["positive"], negative=["negative"], device="cpu",
        window_seconds=3.5, max_windows=4,
    )
    assert report["candidate_sha256"] == candidate.sha256
    assert report["upstream_report_sha256"] == upstream.sha256
    assert report["training_ready"] is False
    assert report["window_seconds"] == 3.5 and report["max_windows"] == 4


def test_main_rejects_output_inside_model_or_cache_tree(tmp_path):
    candidates, provenance, upstream = (tmp_path / name for name in ("c", "p", "u"))
    for path in (candidates, provenance, upstream):
        path.write_text("x")
    model, cache = tmp_path / "model", tmp_path / "cache"
    model.mkdir(); cache.mkdir()
    for output in (model / "config.json", cache / "nested" / "report.json"):
        with pytest.raises(ValueError, match="differ"):
            target.main([
                str(candidates), str(provenance), str(upstream), str(cache), str(output),
                "--model-dir", str(model), "--positive", "p", "--negative", "n",
            ])


def test_lock_video_rejects_revision_drift_before_download(tmp_path, monkeypatch):
    row = {
        "title": "File:Dog.webm", "duration_seconds": 2,
        "original_media_sha1": "a" * 40,
        "original_media_timestamp": "t",
        "review_preview_url": "https://upload.wikimedia.org/dog.webm",
    }
    monkeypatch.setattr(target, "fetch_review_media", lambda title: {
        "original_media_sha1": "b" * 40, "original_media_timestamp": "t",
        "preview_url": row["review_preview_url"],
    })
    with pytest.raises(ValueError, match="changed"):
        target.lock_video(row, tmp_path, request_delay=0)
    assert not list(tmp_path.iterdir())
