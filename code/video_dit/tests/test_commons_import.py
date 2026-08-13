from pathlib import Path

from commons_import import (
    CommonsCandidate,
    candidate_from_api,
    fetch_candidate,
    normalize_clip,
    search_candidates,
    category_candidates,
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


def test_search_candidates_keeps_only_video_files_and_paginates(monkeypatch):
    responses = [
        {
            "continue": {"gsroffset": 2, "continue": "gsroffset||"},
            "query": {"pages": {
                "1": {
                    "title": "File:Cat runs.webm",
                    "imageinfo": [{
                        "url": "https://upload.wikimedia.org/cat.webm",
                        "descriptionurl": "https://commons.wikimedia.org/wiki/File:Cat_runs.webm",
                        "mime": "video/webm",
                        "extmetadata": {
                            "Artist": {"value": "Cat Author"},
                            "LicenseShortName": {"value": "CC BY 4.0"},
                            "LicenseUrl": {"value": "https://creativecommons.org/licenses/by/4.0"},
                        },
                    }],
                },
                "2": {
                    "title": "File:Cat.jpg",
                    "imageinfo": [{
                        "url": "https://upload.wikimedia.org/cat.jpg",
                        "descriptionurl": "https://commons.wikimedia.org/wiki/File:Cat.jpg",
                        "mime": "image/jpeg",
                        "extmetadata": {},
                    }],
                },
            }},
        },
        {"query": {"pages": {
            "3": {
                "title": "File:Second cat.ogv",
                "imageinfo": [{
                    "url": "https://upload.wikimedia.org/cat.ogv",
                    "descriptionurl": "https://commons.wikimedia.org/wiki/File:Second_cat.ogv",
                    "mime": "video/ogg",
                    "extmetadata": {
                        "Attribution": {"value": "Other Author"},
                        "LicenseShortName": {"value": "CC0"},
                        "LicenseUrl": {"value": "https://creativecommons.org/publicdomain/zero/1.0"},
                    },
                }],
            },
        }}},
    ]
    requested_urls = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self):
            import json
            return json.dumps(self.payload).encode()

    def fake_open(request, timeout):
        requested_urls.append(request.full_url)
        return Response(responses.pop(0))

    monkeypatch.setattr("commons_import.urllib.request.urlopen", fake_open)

    candidates = search_candidates(
        "cat running", subject="cat", action="running", scene="grass",
        prompt="a cat running on grass", limit=2,
    )

    assert [candidate.title for candidate in candidates] == [
        "File:Cat runs.webm", "File:Second cat.ogv",
    ]
    assert "gsroffset=2" in requested_urls[1]


def test_category_candidates_reads_files_from_subcategories(monkeypatch):
    calls = []

    def fake_json(parameters):
        calls.append(parameters)
        if parameters.get("list") == "categorymembers" and parameters["cmtitle"] == "Category:Videos of cats":
            return {"query": {"categorymembers": [
                {"ns": 14, "title": "Category:Videos of jumping cats"},
                {"ns": 6, "title": "File:Cat walking.webm"},
            ]}}
        if parameters.get("list") == "categorymembers":
            return {"query": {"categorymembers": [
                {"ns": 6, "title": "File:Cat jumping.webm"},
            ]}}
        titles = parameters["titles"].split("|")
        return {"query": {"pages": {str(index): {
                "title": title, "imageinfo": [{
                "url": "https://upload.wikimedia.org/cat.webm",
                "descriptionurl": "https://commons.wikimedia.org/cat",
                "mime": "video/webm",
                "extmetadata": {
                    "Artist": {"value": "Author"},
                    "LicenseShortName": {"value": "CC0"},
                    "LicenseUrl": {"value": "https://creativecommons.org/publicdomain/zero/1.0"},
                },
                }],
        } for index, title in enumerate(titles)}}}

    monkeypatch.setattr("commons_import._api_json", fake_json)
    rows = category_candidates(
        "Videos of cats", subject="cat", max_depth=1, limit=10,
    )
    assert {row.title for row in rows} == {
        "File:Cat walking.webm", "File:Cat jumping.webm",
    }
    assert any(call.get("cmtitle") == "Category:Videos of jumping cats" for call in calls)
    metadata_calls = [call for call in calls if call.get("prop") == "imageinfo"]
    assert len(metadata_calls) == 2
    queried_titles = "|".join(call["titles"] for call in metadata_calls)
    assert "File:Cat walking.webm" in queried_titles
    assert "File:Cat jumping.webm" in queried_titles


def test_category_limit_counts_videos_after_mime_filter(monkeypatch):
    pages = iter([
        {"query": {"categorymembers": [
            {"ns": 6, "title": "File:Photo.jpg"},
            {"ns": 6, "title": "File:First.webm"},
        ]}, "continue": {"cmcontinue": "next"}},
        {"query": {"categorymembers": [
            {"ns": 6, "title": "File:Second.webm"},
        ]}},
    ])

    def fake_json(parameters):
        if parameters.get("list") == "categorymembers":
            return next(pages)
        result = {}
        for index, title in enumerate(parameters["titles"].split("|")):
            mime = "image/jpeg" if title.endswith(".jpg") else "video/webm"
            result[str(index)] = {"title": title, "imageinfo": [{
                "url": "https://upload.wikimedia.org/media", "mime": mime,
                "descriptionurl": "https://commons.wikimedia.org/media",
                "extmetadata": {
                    "Artist": {"value": "Author"},
                    "LicenseShortName": {"value": "CC0"},
                    "LicenseUrl": {"value": "https://creativecommons.org/publicdomain/zero/1.0"},
                },
            }]}
        return {"query": {"pages": result}}

    monkeypatch.setattr("commons_import._api_json", fake_json)
    rows = category_candidates("Cats", subject="cat", max_depth=0, limit=2)
    assert [row.title for row in rows] == ["File:First.webm", "File:Second.webm"]
