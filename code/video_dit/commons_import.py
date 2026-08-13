"""Import license-auditable short clips from Wikimedia Commons."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
from pathlib import Path
import re
import shutil
import subprocess
import json
import urllib.parse
import urllib.error
import urllib.request
import uuid
import time
from datetime import datetime, timezone
from typing import Callable

from real_data import RealVideoRecord, decode_video_clip, perceptual_hash


@dataclass(frozen=True)
class CommonsCandidate:
    title: str
    media_url: str
    description_url: str
    description: str
    attribution: str
    license: str
    license_url: str
    subject: str
    action: str
    scene: str
    prompt: str
    start_seconds: float = 0.0
    source_revision: str = ""
    original_media_url: str = ""
    original_media_sha1: str = ""
    original_media_timestamp: str = ""


def _api_json(parameters: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(parameters),
        headers={"User-Agent": "ai-research-notes-video-dit/0.1 research"},
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 5:
                raise
            retry_after = error.headers.get("Retry-After") if error.headers else None
            time.sleep(float(retry_after) if retry_after else 2 ** attempt)
    raise RuntimeError("unreachable")


def _plain(value: str) -> str:
    return re.sub(r"<[^>]+>", "", html.unescape(value)).strip()


def _is_video_file(title: str, mime: str) -> bool:
    """Accept Commons video MIME values, including legacy Ogg video metadata."""

    return mime.startswith("video/") or (
        mime == "application/ogg" and title.casefold().endswith(".ogv")
    )


def candidate_from_api(
    payload: dict[str, object],
    *,
    subject: str,
    action: str,
    scene: str,
    prompt: str,
    start_seconds: float = 0.0,
) -> CommonsCandidate:
    pages = payload["query"]["pages"]  # type: ignore[index]
    if len(pages) != 1:  # type: ignore[arg-type]
        raise ValueError("Commons response must contain exactly one page")
    page = next(iter(pages.values()))  # type: ignore[union-attr]
    info = page["imageinfo"][0]
    metadata = info["extmetadata"]

    def field(name: str) -> str:
        value = metadata.get(name, {}).get("value", "")
        return _plain(value)

    required = {
        "attribution": field("Attribution") or field("Artist"),
        "license": field("LicenseShortName"),
        "license_url": field("LicenseUrl"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Commons metadata lacks {', '.join(missing)}")
    return CommonsCandidate(
        title=page["title"],
        media_url=info["url"],
        description_url=info["descriptionurl"],
        description=field("ImageDescription"),
        attribution=required["attribution"],
        license=required["license"],
        license_url=required["license_url"],
        subject=subject,
        action=action,
        scene=scene,
        prompt=prompt,
        start_seconds=start_seconds,
        source_revision=str((page.get("revisions") or [{}])[0].get("revid", "")),
        original_media_url=info["url"],
        original_media_sha1=str(info.get("sha1", "")),
        original_media_timestamp=str(info.get("timestamp", "")),
    )


def fetch_candidate(
    title: str,
    *,
    subject: str,
    action: str,
    scene: str,
    prompt: str,
    start_seconds: float = 0.0,
) -> CommonsCandidate:
    """Fetch one exact file page; search results never enter training directly."""

    parameters = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "prop": "imageinfo|revisions",
            "iiprop": "url|sha1|timestamp|extmetadata",
            "rvprop": "ids",
            "titles": title,
        }
    )
    payload = _api_json(dict(urllib.parse.parse_qsl(parameters)))
    return candidate_from_api(
        payload,
        subject=subject,
        action=action,
        scene=scene,
        prompt=prompt,
        start_seconds=start_seconds,
    )


def search_candidates(
    query: str,
    *,
    subject: str,
    action: str,
    scene: str,
    prompt: str,
    limit: int = 100,
    continuation: dict[str, object] | None = None,
    on_page: Callable[
        [list[CommonsCandidate], dict[str, object] | None], None
    ] | None = None,
) -> list[CommonsCandidate]:
    """Search Commons metadata, returning candidates for human review only."""

    if not query.strip():
        raise ValueError("query must be non-empty")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    candidates: list[CommonsCandidate] = []
    next_page: dict[str, object] = dict(continuation or {})
    while len(candidates) < limit:
        parameters: dict[str, object] = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,
            "gsrlimit": min(50, limit - len(candidates)),
            "prop": "imageinfo|revisions",
            "iiprop": "url|mime|sha1|timestamp|extmetadata",
            "rvprop": "ids",
            **next_page,
        }
        payload = _api_json(parameters)
        pages = payload.get("query", {}).get("pages", {})
        page_candidates: list[CommonsCandidate] = []
        for page in pages.values():
            info = page.get("imageinfo", [{}])[0]
            if not _is_video_file(str(page["title"]), str(info.get("mime", ""))):
                continue
            try:
                candidate = candidate_from_api(
                    {"query": {"pages": {str(page.get("pageid", page["title"])): page}}},
                    subject=subject,
                    action=action,
                    scene=scene,
                    prompt=prompt,
                )
            except ValueError:
                continue
            candidates.append(candidate)
            page_candidates.append(candidate)
            if len(candidates) == limit:
                break
        continuation_value = payload.get("continue")
        next_page = dict(continuation_value) if isinstance(continuation_value, dict) else {}
        if on_page is not None:
            on_page(page_candidates, next_page or None)
        if not next_page:
            break
    return candidates


def category_candidates(
    category: str,
    *,
    subject: str,
    max_depth: int = 1,
    limit: int = 500,
    state: dict[str, object] | None = None,
    on_batch: Callable[
        [list[CommonsCandidate], dict[str, object]], None
    ] | None = None,
) -> list[CommonsCandidate]:
    """Enumerate video files from a Commons category tree for human review."""

    if max_depth < 0 or limit < 1:
        raise ValueError("max_depth must be non-negative and limit positive")
    initial = f"Category:{category.removeprefix('Category:')}"
    if state is None:
        queue = [(initial, 0)]
        visited_categories: set[str] = set()
        visited_titles: set[str] = set()
        pending_titles: list[str] = []
        current: tuple[str, int, dict[str, object]] | None = None
    else:
        queue = [(str(item[0]), int(item[1])) for item in state.get("queue", [])]
        visited_categories = {str(item) for item in state.get("visited_categories", [])}
        visited_titles = {str(item) for item in state.get("visited_titles", [])}
        pending_titles = [str(item) for item in state.get("pending_titles", [])]
        saved_current = state.get("current")
        current = None if not saved_current else (
            str(saved_current["category"]), int(saved_current["depth"]),
            dict(saved_current.get("continuation", {})),
        )

    def snapshot() -> dict[str, object]:
        return {
            "queue": [[name, depth] for name, depth in queue],
            "visited_categories": sorted(visited_categories),
            "visited_titles": sorted(visited_titles),
            "pending_titles": pending_titles,
            "current": None if current is None else {
                "category": current[0], "depth": current[1],
                "continuation": current[2],
            },
        }

    candidates: list[CommonsCandidate] = []
    while (pending_titles or current is not None or queue) and len(candidates) < limit:
        if pending_titles:
            batch = pending_titles[:50]
            metadata = _api_json({
                "action": "query", "format": "json", "prop": "imageinfo|revisions",
                "iiprop": "url|mime|sha1|timestamp|extmetadata", "rvprop": "ids",
                "titles": "|".join(batch),
            })
            pages_by_title = {
                str(page["title"]): (str(page_id), page)
                for page_id, page in metadata.get("query", {}).get("pages", {}).items()
            }
            batch_candidates: list[CommonsCandidate] = []
            consumed = 0
            for title in batch:
                consumed += 1
                visited_titles.add(title)
                entry = pages_by_title.get(title)
                if entry is None:
                    continue
                page_id, page = entry
                info = page.get("imageinfo", [{}])[0]
                if not _is_video_file(title, str(info.get("mime", ""))):
                    continue
                try:
                    candidate = candidate_from_api(
                        {"query": {"pages": {page_id: page}}}, subject=subject,
                        action="review_required", scene="review_required",
                        prompt=f"review the {subject} action and scene",
                    )
                except ValueError:
                    continue
                candidates.append(candidate)
                batch_candidates.append(candidate)
                if len(candidates) == limit:
                    break
            del pending_titles[:consumed]
            if on_batch is not None:
                on_batch(batch_candidates, snapshot())
            continue
        if current is None:
            name, depth = queue.pop(0)
            if name in visited_categories:
                continue
            visited_categories.add(name)
            current = (name, depth, {})
        name, depth, continuation = current
        while len(candidates) < limit and current is not None:
            payload = _api_json({
                "action": "query", "format": "json", "list": "categorymembers",
                "cmtitle": name, "cmtype": "subcat|file", "cmlimit": 500,
                **continuation,
            })
            for member in payload.get("query", {}).get("categorymembers", []):
                if member.get("ns") == 6:
                    title = str(member["title"])
                    if title not in visited_titles and title not in pending_titles:
                        pending_titles.append(title)
                elif member.get("ns") == 14 and depth < max_depth:
                    queue.append((str(member["title"]), depth + 1))
            continuation_value = payload.get("continue")
            if isinstance(continuation_value, dict) and continuation_value:
                continuation = dict(continuation_value)
                current = (name, depth, continuation)
            else:
                current = None
            if on_batch is not None:
                on_batch([], snapshot())
    return candidates


def candidate_sample_id(value: str) -> str:
    value = value.removeprefix("File:").rsplit(".", 1)[0].lower()
    readable = re.sub(r"[^a-z0-9]+", "-", value).strip("-") or "commons-video"
    suffix = hashlib.sha256(value.encode()).hexdigest()[:10]
    return f"{readable}-{suffix}"


def normalize_clip(
    candidate: CommonsCandidate,
    output_dir: Path,
    *,
    frames: int = 16,
    fps: int = 8,
    size: int = 96,
) -> RealVideoRecord:
    """Download and normalize one two-second clip, preserving provenance."""

    if frames < 2 or fps <= 0 or size <= 0:
        raise ValueError("frames, fps, and size must be positive")
    if not candidate.source_revision.strip():
        raise ValueError("Commons candidate requires a source revision")
    if not candidate.original_media_sha1.strip() or not candidate.original_media_timestamp.strip():
        raise ValueError("Commons candidate requires a locked original media revision")
    duration = frames / fps
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_id = candidate_sample_id(candidate.title)
    destination = output_dir / f"{sample_id}.mp4"
    temporary = output_dir / f".{sample_id}.{uuid.uuid4().hex}.tmp.mp4"
    source = output_dir / f".{sample_id}.source.partial"
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required for resumable Commons downloads")
    download = [
        curl, "--fail", "--silent", "--show-error", "--retry", "4",
        "--retry-all-errors", "--user-agent",
        "ai-research-notes-video-dit/0.1 research", "--proto", "=https",
        "--continue-at", "-", "-o", str(source),
        candidate.media_url,
    ]
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(candidate.start_seconds), "-i", str(source),
        "-t", str(duration), "-an",
        "-vf",
        (
            f"fps={fps},scale={size}:{size}:force_original_aspect_ratio=increase,"
            f"crop={size}:{size}"
        ),
        "-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(temporary),
    ]
    try:
        subprocess.run(download, check=True)
        downloaded_media_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        downloaded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        subprocess.run(command, check=True)
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_frames,width,height", "-of", "json",
                str(temporary),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        if (
            int(stream["nb_read_frames"]) != frames
            or int(stream["width"]) != size
            or int(stream["height"]) != size
        ):
            raise ValueError(f"normalized clip {candidate.title} failed media validation")
        temporary.replace(destination)
        source.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    fingerprint = perceptual_hash(
        decode_video_clip(destination, frames=frames, size=size)
    )
    return RealVideoRecord(
        sample_id=sample_id,
        source_id=f"commons:{candidate.title}",
        clip_path=str(destination),
        prompt=candidate.prompt,
        subject=candidate.subject,
        action=candidate.action,
        scene=candidate.scene,
        source_url=candidate.description_url,
        license=candidate.license,
        attribution=f"{candidate.attribution}; {candidate.license_url}",
        sha256=digest,
        license_url=candidate.license_url,
        original_media_url=candidate.original_media_url or candidate.media_url,
        page_revision=candidate.source_revision,
        original_media_sha1=candidate.original_media_sha1,
        original_media_timestamp=candidate.original_media_timestamp,
        downloaded_media_url=candidate.media_url,
        downloaded_media_kind=(
            "original" if candidate.media_url == (candidate.original_media_url or candidate.media_url)
            else "review_derivative"
        ),
        downloaded_media_sha256=downloaded_media_digest,
        downloaded_at=downloaded_at,
        perceptual_hash=fingerprint,
    )
