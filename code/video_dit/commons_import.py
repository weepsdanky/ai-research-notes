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
import urllib.request
import uuid

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
    request = urllib.request.Request(
        f"https://commons.wikimedia.org/w/api.php?{parameters}",
        headers={"User-Agent": "ai-research-notes-video-dit/0.1 research"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read())
    return candidate_from_api(
        payload,
        subject=subject,
        action=action,
        scene=scene,
        prompt=prompt,
        start_seconds=start_seconds,
    )


def _slug(value: str) -> str:
    value = value.removeprefix("File:").rsplit(".", 1)[0].lower()
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-") or "commons-video"


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
    sample_id = _slug(candidate.title)
    destination = output_dir / f"{sample_id}.mp4"
    temporary = output_dir / f".{sample_id}.{uuid.uuid4().hex}.tmp.mp4"
    source = output_dir / f".{sample_id}.{uuid.uuid4().hex}.source"
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required for resumable Commons downloads")
    download = [
        curl, "-L", "--fail", "--silent", "--show-error", "--retry", "4",
        "--retry-all-errors", "--user-agent",
        "ai-research-notes-video-dit/0.1 research", "-o", str(source),
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
    finally:
        temporary.unlink(missing_ok=True)
        source.unlink(missing_ok=True)
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
