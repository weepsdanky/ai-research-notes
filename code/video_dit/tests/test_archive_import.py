import json
import hashlib
import subprocess
from types import SimpleNamespace

import pytest

from archive_import import (
    ArchiveItemCandidate,
    parse_archive_item,
    write_archive_candidates,
)
from prepare_archive import main
from prepare_archive_media import main as media_main


def _item(**metadata):
    base = {
        "identifier": "oreo", "title": "A dog playing indoors",
        "mediatype": "movies", "creator": "Creator Name",
        "uploader": "creator@example.com",
        "licenseurl": "https://creativecommons.org/licenses/by/4.0/",
    }
    base.update(metadata)
    return {
        "metadata": base,
        "files": [{
            "name": "source.mov", "source": "original", "format": "QuickTime",
            "sha1": "a" * 40, "md5": "b" * 32, "mtime": "123", "size": "456",
        }],
    }


def test_parse_archive_item_locks_original_file_and_keeps_pending():
    candidate = parse_archive_item(
        _item(), subject="dog", query="dog playing indoors",
    )

    assert candidate == ArchiveItemCandidate(
        identifier="oreo", title="A dog playing indoors", creator="Creator Name",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        original_file="source.mov", original_sha1="a" * 40,
        original_md5="b" * 32, original_mtime=123, original_size=456,
        source_url="https://archive.org/details/oreo",
        media_url="https://archive.org/download/oreo/source.mov",
        subject="dog", query="dog playing indoors",
    )
    assert candidate.review_status == "pending"


@pytest.mark.parametrize("change,match", [
    ({"creator": ""}, "creator"),
    ({"creator": "person@example.com"}, "email"),
    ({"licenseurl": "https://creativecommons.org/licenses/by-nc/4.0/"}, "license"),
    ({"licenseurl": "https://creativecommons.org/foo/licenses/by/4.0/"}, "license"),
    ({"licenseurl": "https://creativecommons.org/licenses/by/4.0/anything"}, "license"),
    ({"licenseurl": "https://creativecommons.org/publicdomain/mark/1.0/"}, "license"),
    ({"licenseurl": ""}, "license"),
    ({"mediatype": "texts"}, "movies"),
])
def test_parse_archive_item_rejects_incomplete_or_incompatible_metadata(change, match):
    with pytest.raises(ValueError, match=match):
        parse_archive_item(_item(**change), subject="dog", query="dog playing indoors")


def test_parse_archive_item_rejects_ambiguous_or_unlocked_originals():
    payload = _item()
    payload["files"].append({**payload["files"][0], "name": "other.mp4"})
    with pytest.raises(ValueError, match="exactly one"):
        parse_archive_item(payload, subject="dog", query="dog playing indoors")


def test_parse_archive_item_rejects_ambiguous_scalar_metadata():
    with pytest.raises(ValueError, match="creator must be a string"):
        parse_archive_item(
            _item(creator=["one", "two"]),
            subject="dog", query="dog playing indoors",
        )

    payload = _item()
    del payload["files"][0]["sha1"]
    with pytest.raises(ValueError, match="digest"):
        parse_archive_item(payload, subject="dog", query="dog playing indoors")


def test_write_archive_candidates_binds_query_and_deduplicates(tmp_path):
    candidate = parse_archive_item(_item(), subject="dog", query="dog playing indoors")
    output = tmp_path / "candidates.jsonl"
    report = tmp_path / "report.json"

    write_archive_candidates([candidate, candidate], output, report)

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["source_id"].startswith("internet-archive-creator:")
    assert rows[0]["duplicate_group"].startswith("internet-archive-creator:")
    assert "creator@example.com" not in output.read_text()
    summary = json.loads(report.read_text())
    assert summary["records"] == 1
    assert summary["candidate_sha256"]
    assert summary["training_ready"] is False


def test_prepare_archive_cli_records_passes_and_fail_closed_rejections(tmp_path, monkeypatch):
    specs = tmp_path / "specs.json"
    specs.write_text(json.dumps([
        {"identifier": "good", "subject": "dog", "query": "dog playing indoors"},
        {"identifier": "bad", "subject": "cat", "query": "cat playing outdoors"},
    ]))
    output, report = tmp_path / "candidates.jsonl", tmp_path / "report.json"

    def fake_fetch(identifier):
        return _item(identifier=identifier) if identifier == "good" else _item(
            identifier=identifier,
            licenseurl="https://creativecommons.org/licenses/by-nc-nd/4.0/",
        )

    monkeypatch.setattr("prepare_archive.fetch_archive_metadata", fake_fetch)
    assert main([str(specs), str(output), str(report)]) == 0

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["identifier"] for row in rows] == ["good"]
    summary = json.loads(report.read_text())
    assert summary["specs"] == 2
    assert summary["records"] == 1
    assert summary["rejections"] == {"license": 1}


def test_prepare_archive_cli_rejects_duplicate_or_unsafe_spec(tmp_path):
    output, report = tmp_path / "out.jsonl", tmp_path / "report.json"
    for specs_value, match in [
        ([{"identifier": "same", "subject": "cat", "query": "q"}] * 2, "unique"),
        ([{"identifier": "../bad", "subject": "cat", "query": "q"}], "identifier"),
        ([{"identifier": "ok", "subject": "person", "query": "q"}], "subject"),
    ]:
        specs = tmp_path / "specs.json"
        specs.write_text(json.dumps(specs_value))
        with pytest.raises(ValueError, match=match):
            main([str(specs), str(output), str(report)])


def test_archive_media_cli_verifies_download_and_writes_locked_manifest(tmp_path, monkeypatch):
    payload = b"real-video-bytes"
    item = _item()
    item["files"] = [{
        "name": "source.mov", "source": "original", "format": "QuickTime",
        "sha1": hashlib.sha1(payload).hexdigest(),
        "md5": hashlib.md5(payload).hexdigest(),
        "mtime": "123", "size": str(len(payload)),
    }]
    candidate = parse_archive_item(
        item,
        subject="dog", query="dog playing indoors",
    ).as_json()
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps(candidate, sort_keys=True) + "\n")

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def read(self, size=-1):
            if getattr(self, "done", False):
                return b""
            self.done = True
            return payload

    monkeypatch.setattr(
        "prepare_archive_media.urllib.request.urlopen",
        lambda request, timeout: Response(),
    )
    monkeypatch.setattr(
        "prepare_archive_media.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=json.dumps({
                "streams": [{
                    "codec_type": "video", "codec_name": "h264",
                    "width": 96, "height": 96, "avg_frame_rate": "8/1",
                    "nb_frames": "16",
                }],
                "format": {"duration": "2.0"},
            }),
        ),
    )
    media_dir = tmp_path / "media"
    manifest = tmp_path / "media-manifest.json"

    assert media_main([
        str(candidates), str(media_dir), str(manifest),
        "--max-total-bytes", "1000",
    ]) == 0

    report = json.loads(manifest.read_text())
    assert report["candidate_sha256"] == hashlib.sha256(candidates.read_bytes()).hexdigest()
    assert report["records"] == 1
    assert report["total_bytes"] == len(payload)
    assert report["training_ready"] is False
    row = report["media"][0]
    downloaded = manifest.parent / row["local_path"]
    assert downloaded.read_bytes() == payload
    assert row["sha256"] == hashlib.sha256(payload).hexdigest()
    assert row["duration_seconds"] == 2.0
    assert row["codec"] == "h264"
    assert row["frames"] == 16


def test_archive_media_cli_rejects_budget_before_network_or_outputs(tmp_path, monkeypatch):
    candidate = parse_archive_item(
        _item(), subject="dog", query="dog playing indoors",
    ).as_json()
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps(candidate, sort_keys=True) + "\n")
    monkeypatch.setattr(
        "prepare_archive_media.urllib.request.urlopen",
        lambda *_args, **_kwargs: pytest.fail("budget rejection must precede network access"),
    )
    media_dir = tmp_path / "media"
    manifest = tmp_path / "media-manifest.json"

    with pytest.raises(ValueError, match="hard cap"):
        media_main([
            str(candidates), str(media_dir), str(manifest),
            "--max-total-bytes", "455",
        ])

    assert not media_dir.exists()
    assert not manifest.exists()


def test_archive_media_cli_revalidates_license_and_creator_group(tmp_path, monkeypatch):
    candidate = parse_archive_item(
        _item(), subject="dog", query="dog playing indoors",
    ).as_json()
    candidates = tmp_path / "candidates.jsonl"
    media_dir = tmp_path / "media"
    manifest = tmp_path / "media-manifest.json"
    monkeypatch.setattr(
        "prepare_archive_media.urllib.request.urlopen",
        lambda *_args, **_kwargs: pytest.fail("contract rejection must precede network access"),
    )

    for field, value, match in [
        ("license_url", "https://creativecommons.org/licenses/by-nc/4.0/", "license"),
        ("source_id", "internet-archive-creator:wrong", "creator group"),
    ]:
        changed = {**candidate, field: value}
        candidates.write_text(json.dumps(changed, sort_keys=True) + "\n")
        with pytest.raises(ValueError, match=match):
            media_main([
                str(candidates), str(media_dir), str(manifest),
                "--max-total-bytes", "1000",
            ])


def test_archive_media_cli_cleans_staging_after_digest_failure(tmp_path, monkeypatch):
    payload = b"substituted-video"
    candidate = parse_archive_item(
        _item(), subject="cat", query="cat playing indoors",
    ).as_json()
    candidate["original_size"] = len(payload)
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps(candidate, sort_keys=True) + "\n")

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def read(self, _size=-1):
            if getattr(self, "done", False):
                return b""
            self.done = True
            return payload

    monkeypatch.setattr(
        "prepare_archive_media.urllib.request.urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    media_dir = tmp_path / "media"
    manifest = tmp_path / "media-manifest.json"

    with pytest.raises(ValueError, match="digest mismatch"):
        media_main([
            str(candidates), str(media_dir), str(manifest),
            "--max-total-bytes", "1000",
        ])

    assert not media_dir.exists()
    assert not media_dir.with_name("media.tmp").exists()
    assert not manifest.exists()
    assert not manifest.with_suffix(".json.tmp").exists()


def test_archive_media_cli_cleans_staging_after_probe_timeout(tmp_path, monkeypatch):
    payload = b"valid-video-bytes"
    item = _item()
    item["files"] = [{
        "name": "source.mov", "source": "original", "format": "QuickTime",
        "sha1": hashlib.sha1(payload).hexdigest(),
        "md5": hashlib.md5(payload).hexdigest(),
        "mtime": "123", "size": str(len(payload)),
    }]
    candidate = parse_archive_item(
        item, subject="dog", query="dog playing indoors",
    ).as_json()
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps(candidate, sort_keys=True) + "\n")

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def read(self, _size=-1):
            if getattr(self, "done", False):
                return b""
            self.done = True
            return payload

    monkeypatch.setattr(
        "prepare_archive_media.urllib.request.urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    monkeypatch.setattr(
        "prepare_archive_media.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(args[0], kwargs["timeout"])
        ),
    )
    media_dir = tmp_path / "media"
    manifest = tmp_path / "media-manifest.json"

    with pytest.raises(ValueError, match="ffprobe timed out"):
        media_main([
            str(candidates), str(media_dir), str(manifest),
            "--max-total-bytes", "1000",
        ])

    assert not media_dir.exists()
    assert not media_dir.with_name("media.tmp").exists()
    assert not manifest.exists()
