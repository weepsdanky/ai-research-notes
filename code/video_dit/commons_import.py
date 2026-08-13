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
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
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
) -> list[CommonsCandidate]:
    """Search Commons metadata, returning candidates for human review only."""

    if not query.strip():
        raise ValueError("query must be non-empty")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    candidates: list[CommonsCandidate] = []
    continuation: dict[str, object] = {}
    while len(candidates) < limit:
        parameters: dict[str, object] = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,
            "gsrlimit": min(50, limit - len(candidates)),
            "prop": "imageinfo",
            "iiprop": "url|mime|extmetadata",
            **continuation,
        }
        payload = _api_json(parameters)
        pages = payload.get("query", {}).get("pages", {})
        for page in pages.values():
            info = page.get("imageinfo", [{}])[0]
            if not str(info.get("mime", "")).startswith("video/"):
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
            if len(candidates) == limit:
                break
        continuation = payload.get("continue", {})
        if not continuation:
            break
    return candidates


def category_candidates(
    category: str,
    *,
    subject: str,
    max_depth: int = 1,
    limit: int = 500,
) -> list[CommonsCandidate]:
    """Enumerate video files from a Commons category tree for human review."""

    if max_depth < 0 or limit < 1:
        raise ValueError("max_depth must be non-negative and limit positive")
    queue = [(f"Category:{category.removeprefix('Category:')}", 0)]
    visited_categories: set[str] = set()
    visited_titles: set[str] = set()
    candidates: list[CommonsCandidate] = []
    while queue and len(candidates) < limit:
        current, depth = queue.pop(0)
        if current in visited_categories:
            continue
        visited_categories.add(current)
        continuation: dict[str, object] = {}
        while len(candidates) < limit:
            payload = _api_json({
                "action": "query", "format": "json", "list": "categorymembers",
                "cmtitle": current, "cmtype": "subcat|file", "cmlimit": 500,
                **continuation,
            })
            page_titles: list[str] = []
            for member in payload.get("query", {}).get("categorymembers", []):
                if member.get("ns") == 6:
                    title = str(member["title"])
                    if title not in visited_titles:
                        visited_titles.add(title)
                        page_titles.append(title)
                elif member.get("ns") == 14 and depth < max_depth:
                    queue.append((str(member["title"]), depth + 1))
            for offset in range(0, len(page_titles), 50):
                batch = page_titles[offset:offset + 50]
                metadata = _api_json({
                    "action": "query", "format": "json", "prop": "imageinfo",
                    "iiprop": "url|mime|extmetadata", "titles": "|".join(batch),
                })
                for page_id, page in metadata.get("query", {}).get("pages", {}).items():
                    info = page.get("imageinfo", [{}])[0]
                    if not str(info.get("mime", "")).startswith("video/"):
                        continue
                    try:
                        candidates.append(candidate_from_api(
                            {"query": {"pages": {str(page_id): page}}},
                            subject=subject, action="review_required",
                            scene="review_required",
                            prompt=f"review the {subject} action and scene",
                        ))
                    except ValueError:
                        continue
                    if len(candidates) == limit:
                        break
                if len(candidates) == limit:
                    break
            continuation = payload.get("continue", {})
            if not continuation:
                break
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
        perceptual_hash=fingerprint,
    )
