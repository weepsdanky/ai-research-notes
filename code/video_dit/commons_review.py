"""Human-review helpers for Wikimedia Commons video candidates."""

from __future__ import annotations

from collections import Counter
import html
import hashlib
import json
from pathlib import Path
from typing import Iterable
import urllib.parse
import urllib.request
import argparse
import urllib.error
import time
import shutil
import subprocess
import tempfile
import re
from PIL import Image, ImageDraw


REVIEW_STATUSES = {"pending", "approved", "rejected"}

_REVIEW_PRIORITY_TERMS = {
    "single-subject:solo": (4, ("solo", "alone", "single person")),
    "single-subject:person": (2, ("a man", "a woman", "athlete", "runner", "hiker")),
    "target-action": (2, ("walking", "running", "jogging", "hiking")),
    "natural-scene": (1, ("forest", "trail", "field", "snow", "beach", "park")),
    "exclude:timelapse": (-7, ("timelapse", "time-lapse", "time lapse", "hyperlapse")),
    "exclude:crowd": (-4, ("crowd", "many people", "participants", "march")),
    "exclude:first-person": (-4, ("first-person", "first person", "pov")),
    "exclude:vlog": (-2, ("vlog", "walking tour", "city tour")),
    "exclude:animation": (-6, ("animation", "stop motion", "stop-motion")),
}


def _plain(value: str) -> str:
    return re.sub(r"<[^>]+>", "", html.unescape(value)).strip()


def choose_review_derivative(derivatives: Iterable[dict[str, object]]) -> str:
    """Pick the smallest HTTP video derivative suitable for visual review."""

    usable = [
        derivative for derivative in derivatives
        if str(derivative.get("src", "")).startswith("https://upload.wikimedia.org/")
        and str(derivative.get("type", "")).startswith("video/")
        and int(derivative.get("height", 0)) >= 180
    ]
    if not usable:
        raise ValueError("Commons video has no usable HTTPS review derivative")
    return str(min(usable, key=lambda item: (
        int(item.get("height", 0)), int(item.get("width", 0)),
    ))["src"])


def fetch_review_media(title: str) -> dict[str, object]:
    return fetch_review_media_batch([title])[title]


def fetch_review_media_batch(titles: list[str]) -> dict[str, dict[str, object]]:
    if not 1 <= len(titles) <= 50:
        raise ValueError("videoinfo batch requires 1 to 50 titles")
    parameters = urllib.parse.urlencode({
        "action": "query", "format": "json", "prop": "videoinfo|imageinfo|revisions",
        "viprop": "derivatives|dimensions|url|timestamp|sha1", "rvprop": "ids",
        "iiprop": "url|extmetadata",
        "titles": "|".join(titles),
    })
    request = urllib.request.Request(
        f"https://commons.wikimedia.org/w/api.php?{parameters}",
        headers={"User-Agent": "ai-research-notes-video-dit/0.1 research"},
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read())
            break
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 5:
                raise
            retry_after = error.headers.get("Retry-After") if error.headers else None
            time.sleep(float(retry_after) if retry_after else 2 ** attempt)
    pages = payload.get("query", {}).get("pages", {})
    result: dict[str, dict[str, object]] = {}
    for page in pages.values():
        info = page.get("videoinfo", [{}])[0]
        title = str(page["title"])
        revision = str((page.get("revisions") or [{}])[0].get("revid", ""))
        if not revision:
            raise ValueError(f"Commons response lacks source revision for: {title}")
        image_info = (page.get("imageinfo") or [{}])[0]
        metadata = image_info.get("extmetadata", {})

        def metadata_field(name: str) -> str:
            return _plain(str(metadata.get(name, {}).get("value", "")))

        license_fields = {
            "description_url": str(image_info.get("descriptionurl", "")),
            "attribution": metadata_field("Attribution") or metadata_field("Artist"),
            "license": metadata_field("LicenseShortName"),
            "license_url": metadata_field("LicenseUrl"),
        }
        if not all(license_fields.values()):
            raise ValueError(f"Commons response lacks license provenance for: {title}")
        try:
            preview_url = choose_review_derivative(info.get("derivatives", []))
        except ValueError as error:
            result[title] = {
                "duration": float(info.get("duration", 0)), "error": str(error),
                "source_revision": revision,
                "original_media_url": str(info.get("url", "")),
                "original_media_sha1": str(info.get("sha1", "")),
                "original_media_timestamp": str(info.get("timestamp", "")),
                **license_fields,
            }
        else:
            result[title] = {
                "preview_url": preview_url, "duration": float(info.get("duration", 0)),
                "source_revision": revision,
                "original_media_url": str(info.get("url", "")),
                "original_media_sha1": str(info.get("sha1", "")),
                "original_media_timestamp": str(info.get("timestamp", "")),
                **license_fields,
            }
    missing = set(titles) - set(result)
    if missing:
        raise ValueError(f"Commons response lacks videoinfo for: {sorted(missing)[:3]}")
    return result


def cache_review_preview(title: str, url: str, cache_dir: Path) -> Path:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "upload.wikimedia.org":
        raise ValueError("review preview requires an HTTPS Wikimedia upload URL")
    suffix = Path(parsed.path).suffix or ".webm"
    digest = hashlib.sha256(title.encode()).hexdigest()[:16]
    destination = cache_dir / f"{digest}{suffix}"
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    request = urllib.request.Request(
        url, headers={"User-Agent": "ai-research-notes-video-dit/0.1 research"},
    )
    try:
        for attempt in range(8):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    temporary.write_bytes(response.read())
                break
            except urllib.error.HTTPError as error:
                if error.code != 429 or attempt == 7:
                    raise
                retry_after = error.headers.get("Retry-After") if error.headers else None
                time.sleep(float(retry_after) if retry_after else min(60, 2 ** attempt))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def build_contact_sheet(
    video: Path, output: Path, *, duration: float, title: str, cells: int = 5,
) -> Path:
    if cells < 2 or duration <= 0:
        raise ValueError("contact sheet requires at least two cells and positive duration")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")
    with tempfile.TemporaryDirectory(prefix="commons-contact-") as directory:
        images = []
        for index in range(cells):
            timestamp = duration * (index + 0.5) / cells
            frame = Path(directory) / f"{index}.png"
            subprocess.run([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss",
                str(max(0, timestamp)), "-i", str(video), "-frames:v", "1",
                "-vf", "scale=96:96:force_original_aspect_ratio=increase,crop=96:96",
                str(frame),
            ], check=True)
            images.append(Image.open(frame).convert("RGB"))
        label_height = 28
        sheet = Image.new("RGB", (96 * cells, 96 + label_height), "white")
        for index, image in enumerate(images):
            sheet.paste(image, (96 * index, 0))
        ImageDraw.Draw(sheet).text((4, 100), title[:80], fill="black")
        output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output, quality=88)
    return output


def load_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def review_queue(
    rows: Iterable[dict[str, object]], *, status: str = "pending", subject: str = "all",
) -> list[str]:
    """Return a deterministic queue containing only videos safe to review."""

    if status not in REVIEW_STATUSES | {"all"}:
        raise ValueError(f"unknown review status: {status}")
    materialized = list(rows)
    subjects = {str(row.get("subject", "")) for row in materialized} | {"all"}
    if subject not in subjects:
        raise ValueError(f"unknown review subject: {subject}")
    return sorted(
        str(row["title"])
        for row in materialized
        if (status == "all" or row.get("review_status") == status)
        and (subject == "all" or row.get("subject") == subject)
        and float(row.get("duration_seconds", 0) or 0) >= 2
        and bool(row.get("review_preview_url"))
    )


def prioritize_review_rows(
    rows: Iterable[dict[str, object]], *, status: str = "pending", subject: str = "all",
) -> list[dict[str, object]]:
    """Rank the manual-review queue without changing candidate labels or decisions."""

    materialized = list(rows)
    titles = set(review_queue(materialized, status=status, subject=subject))
    ranked = []
    for row in materialized:
        title = str(row.get("title", ""))
        if title not in titles:
            continue
        searchable = " ".join((title, str(row.get("description", "")))).casefold()
        score = 0
        reasons = []
        for reason, (weight, terms) in _REVIEW_PRIORITY_TERMS.items():
            if any(term in searchable for term in terms):
                score += weight
                reasons.append(reason)
        ranked.append({
            "title": title,
            "review_status": str(row.get("review_status", "")),
            "subject": str(row.get("subject", "")),
            "priority_score": score,
            "priority_reasons": reasons,
        })
    return sorted(ranked, key=lambda row: (-int(row["priority_score"]), str(row["title"])))


def next_review_title(current: str, remaining: list[str]) -> str | None:
    """Choose the next remaining review item after the current row was saved."""

    if not remaining:
        return None
    if current in remaining:
        index = remaining.index(current)
        return remaining[(index + 1) % len(remaining)]
    return remaining[0]


def enrich_review_metadata(path: Path) -> dict[str, object]:
    """Persist lightweight duration/derivative metadata, resuming row by row."""

    rows = load_rows(path)
    pending = [
        row for row in rows
        if "duration_seconds" not in row or not row.get("review_preview_url")
        or not row.get("source_revision")
    ]
    for offset in range(0, len(pending), 50):
        batch = pending[offset:offset + 50]
        media_by_title = fetch_review_media_batch([str(row["title"]) for row in batch])
        for row in batch:
            media = media_by_title[str(row["title"])]
            row["duration_seconds"] = float(media["duration"])
            row["source_revision"] = str(media["source_revision"])
            row["original_media_url"] = str(media["original_media_url"])
            row["original_media_sha1"] = str(media["original_media_sha1"])
            row["original_media_timestamp"] = str(media["original_media_timestamp"])
            if media.get("preview_url"):
                row["review_preview_url"] = str(media["preview_url"])
                row.pop("review_media_error", None)
            else:
                row["review_media_error"] = str(media["error"])
        write_rows(path, rows)
    return summarize_reviews(rows)


def enrich_provenance_sidecar(candidates_path: Path, sidecar_path: Path) -> dict[str, int]:
    """Enrich an immutable candidate snapshot into a resumable provenance sidecar."""

    candidates = load_rows(candidates_path)
    titles = [str(row.get("title", "")) for row in candidates]
    if any(not title for title in titles) or len(titles) != len(set(titles)):
        raise ValueError("candidate titles must be non-empty and unique")
    existing = load_rows(sidecar_path) if sidecar_path.is_file() else []
    existing_titles = [str(row.get("title", "")) for row in existing]
    if len(existing_titles) != len(set(existing_titles)):
        raise ValueError("provenance sidecar contains duplicate titles")
    if not set(existing_titles) <= set(titles):
        raise ValueError("provenance sidecar contains unknown titles")
    rows_by_title = {str(row["title"]): row for row in existing}
    required_fields = (
        "source_revision", "original_media_url", "original_media_sha1",
        "original_media_timestamp", "description_url", "attribution", "license",
        "license_url", "duration_seconds",
    )
    remaining = [
        title for title in titles
        if title not in rows_by_title
        or not all(str(rows_by_title[title].get(name, "")).strip() for name in required_fields)
    ]
    for offset in range(0, len(remaining), 50):
        batch = remaining[offset:offset + 50]
        media_by_title = fetch_review_media_batch(batch)
        for title in batch:
            media = media_by_title[title]
            row = {
                "title": title,
                "duration_seconds": float(media["duration"]),
                "source_revision": str(media["source_revision"]),
                "original_media_url": str(media["original_media_url"]),
                "original_media_sha1": str(media["original_media_sha1"]),
                "original_media_timestamp": str(media["original_media_timestamp"]),
                "description_url": str(media["description_url"]),
                "attribution": str(media["attribution"]),
                "license": str(media["license"]),
                "license_url": str(media["license_url"]),
            }
            required = (
                "source_revision", "original_media_url", "original_media_sha1",
                "original_media_timestamp", "description_url", "attribution",
                "license", "license_url",
            )
            if not all(str(row[name]).strip() for name in required):
                raise ValueError(f"Commons API returned incomplete provenance for: {title}")
            if media.get("preview_url"):
                row["review_preview_url"] = str(media["preview_url"])
            else:
                row["review_media_error"] = str(media["error"])
            rows_by_title[title] = row
        write_rows(sidecar_path, [rows_by_title[title] for title in titles if title in rows_by_title])
    return {
        "candidates": len(titles),
        "enriched": len(rows_by_title),
        "remaining": len(titles) - len(rows_by_title),
    }


def verify_provenance_sidecar(sidecar_path: Path, titles: list[str]) -> None:
    """Fail if a Commons media revision changed since enrichment."""

    rows = {str(row["title"]): row for row in load_rows(sidecar_path)}
    for offset in range(0, len(titles), 50):
        batch = titles[offset:offset + 50]
        current = fetch_review_media_batch(batch)
        for title in batch:
            locked = rows.get(title)
            if locked is None:
                raise ValueError(f"provenance sidecar lacks title: {title}")
            fields = ("original_media_url", "original_media_sha1", "original_media_timestamp")
            if any(str(locked.get(field, "")) != str(current[title].get(field, ""))
                   for field in fields):
                raise ValueError(f"Commons media revision changed since enrichment: {title}")


def save_review(
    path: Path,
    title: str,
    *,
    status: str,
    subject: str,
    action: str,
    scene: str,
    prompt: str,
    start_seconds: float = 0.0,
    notes: str = "",
) -> dict[str, object]:
    if status not in REVIEW_STATUSES:
        raise ValueError(f"status must be one of {sorted(REVIEW_STATUSES)}")
    if start_seconds < 0:
        raise ValueError("start_seconds must be non-negative")
    if status == "approved" and (
        not all(value.strip() for value in (subject, action, scene, prompt))
        or "review_required" in (subject, action, scene)
    ):
        raise ValueError("approved review requires complete labels and prompt")
    rows = load_rows(path)
    matches = [row for row in rows if row.get("title") == title]
    if len(matches) != 1:
        raise ValueError(f"candidate title must match exactly one row: {title}")
    reviewed = matches[0]
    duration = float(reviewed.get("duration_seconds", 0))
    if status == "approved":
        if duration <= 0:
            raise ValueError("approved review requires a known positive video duration")
        if start_seconds + 2 > duration:
            raise ValueError("approved two-second segment exceeds video duration")
    reviewed.update({
        "review_status": status, "subject": subject, "action": action,
        "scene": scene, "prompt": prompt, "start_seconds": start_seconds,
        "review_notes": notes,
    })
    write_rows(path, rows)
    return reviewed


def summarize_reviews(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    materialized = list(rows)
    statuses = Counter(str(row.get("review_status", "missing")) for row in materialized)
    triples = Counter(
        "|".join(str(row[field]) for field in ("subject", "action", "scene"))
        for row in materialized if row.get("review_status") == "approved"
    )
    subject_statuses: dict[str, Counter[str]] = {}
    for row in materialized:
        subject = str(row.get("subject", "missing"))
        subject_statuses.setdefault(subject, Counter())[str(
            row.get("review_status", "missing")
        )] += 1
    pending_placeholders = sum(
        row.get("review_status") == "pending"
        and "review_required" in (row.get("action"), row.get("scene"))
        for row in materialized
    )
    return {
        "records": len(materialized),
        "statuses": dict(sorted(statuses.items())),
        "approved_triples": dict(sorted(triples.items())),
        "subject_statuses": {
            subject: dict(sorted(counts.items()))
            for subject, counts in sorted(subject_statuses.items())
        },
        "pending_placeholders": pending_placeholders,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("enrich", "provenance", "summary", "priority"))
    parser.add_argument("candidates", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--status", choices=sorted(REVIEW_STATUSES | {"all"}),
                        default="pending")
    parser.add_argument("--subject", default="all")
    args = parser.parse_args(argv)
    if args.output is not None and args.command in {"provenance", "priority"}:
        if args.output.resolve() == args.candidates.resolve():
            raise ValueError("report output must differ from candidates")
    if args.command == "enrich":
        report = enrich_review_metadata(args.candidates)
    elif args.command == "provenance":
        if args.output is None:
            parser.error("provenance requires an output sidecar path")
        report = enrich_provenance_sidecar(args.candidates, args.output)
    elif args.command == "summary":
        report = summarize_reviews(load_rows(args.candidates))
        if args.output is not None:
            if args.output.resolve() == args.candidates.resolve():
                raise ValueError("report output must differ from candidates")
            write_json(args.output, report)
    else:
        if args.output is None:
            parser.error("priority requires an output JSONL path")
        report = prioritize_review_rows(
            load_rows(args.candidates), status=args.status, subject=args.subject,
        )
        write_rows(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
