from pathlib import Path

from commons_import import (
    CommonsCandidate,
    candidate_from_api,
    fetch_candidate,
    normalize_clip,
)


def test_candidate_from_api_extracts_required_license_provenance():
    payload = {
        "query": {
            "pages": {
                "7": {
                    "title": "File:Cat jump.webm",
                    "imageinfo": [
                        {
                            "url": "https://upload.wikimedia.org/cat.webm",
                            "descriptionurl": "https://commons.wikimedia.org/wiki/File:Cat_jump.webm",
                            "extmetadata": {
                                "Artist": {"value": "Example Author"},
                                "LicenseShortName": {"value": "CC BY 4.0"},
                                "LicenseUrl": {"value": "https://creativecommons.org/licenses/by/4.0"},
                                "ImageDescription": {"value": "A cat jumps on grass"},
                            },
                        }
                    ],
                }
            }
        }
    }

    candidate = candidate_from_api(
        payload,
        subject="cat",
        action="jumping",
        scene="grass",
        prompt="a cat jumping on grass",
        start_seconds=2.5,
    )

    assert candidate.title == "File:Cat jump.webm"
    assert candidate.license == "CC BY 4.0"
    assert candidate.license_url.endswith("/4.0")
    assert candidate.attribution == "Example Author"
    assert candidate.start_seconds == 2.5


def test_normalize_clip_calls_ffmpeg_and_hashes_result(tmp_path, monkeypatch):
    candidate = CommonsCandidate(
        title="File:Cat jump.webm",
        media_url="https://upload.wikimedia.org/cat.webm",
        description_url="https://commons.wikimedia.org/wiki/File:Cat_jump.webm",
        description="A cat jumps on grass",
        attribution="Example Author",
        license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0",
        subject="cat",
        action="jumping",
        scene="grass",
        prompt="a cat jumping on grass",
        start_seconds=1.0,
    )

    class Completed:
        stdout = '{"streams":[{"nb_read_frames":"16","width":96,"height":96}]}'

    def fake_run(command, **kwargs):
        if "-o" in command:
            Path(command[command.index("-o") + 1]).write_bytes(b"source")
        elif command[0].endswith("ffmpeg"):
            Path(command[-1]).write_bytes(b"normalized-mp4")
        return Completed()

    monkeypatch.setattr("commons_import.subprocess.run", fake_run)
    monkeypatch.setattr(
        "commons_import.decode_video_clip", lambda *args, **kwargs: __import__("torch").zeros(3, 16, 96, 96)
    )
    record = normalize_clip(candidate, tmp_path, frames=16, fps=8, size=96)

    assert Path(record.clip_path).is_file()
    assert len(record.sha256) == 64
    assert record.source_id == "commons:File:Cat jump.webm"
    assert record.license == "CC BY 4.0"
    assert record.perceptual_hash is not None


def test_fetch_candidate_uses_exact_commons_title(monkeypatch):
    captured = {}
    payload = {
        "query": {
            "pages": {
                "7": {
                    "title": "File:Cat jump.webm",
                    "imageinfo": [
                        {
                            "url": "https://upload.wikimedia.org/cat.webm",
                            "descriptionurl": "https://commons.wikimedia.org/wiki/File:Cat_jump.webm",
                            "extmetadata": {
                                "Artist": {"value": "Example Author"},
                                "LicenseShortName": {"value": "CC BY 4.0"},
                                "LicenseUrl": {"value": "https://creativecommons.org/licenses/by/4.0"},
                            },
                        }
                    ],
                }
            }
        }
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self):
            import json
            return json.dumps(payload).encode()

    def fake_open(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("commons_import.urllib.request.urlopen", fake_open)

    candidate = fetch_candidate(
        "File:Cat jump.webm",
        subject="cat",
        action="jumping",
        scene="grass",
        prompt="a cat jumping on grass",
    )

    assert "titles=File%3ACat+jump.webm" in captured["url"]
    assert candidate.attribution == "Example Author"
