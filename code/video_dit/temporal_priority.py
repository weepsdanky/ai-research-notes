"""Rerank a frozen poster shortlist with locked eight-frame X-CLIP inference."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image
import torch

from commons_review import fetch_review_media, fetch_review_media_batch
from visual_priority import OFFICIAL_VIT_B32_SHA256


MODEL_REPO = "microsoft/xclip-base-patch32"
MODEL_REVISION = "a2e27a78a2b5d802e894b8a1ef14f3a8ce490963"
MODEL_FILE_SHA256 = {
    "config.json": "13bb919d1ef16f3b80b03cdf5c575d688dfafd21d12566849cc557b00e058ce3",
    "merges.txt": "f526393189112391ce6f9795d4695f704121ce452c3aad1f5335cc41337eba85",
    "model.safetensors": "abf286e8cdd0612761c3e42d3a55eca998382dfa67a04a0f3fdcdfa4f150cdbb",
    "preprocessor_config.json": "c14b2b5c8f26a754df62235ba79d1ca63cfdd9b3de76ee688e4a30ea1e5c6986",
    "special_tokens_map.json": "c4864a9376a8401918425bed71fc14fc0e81f9b59ec45c1cf96cccb2df508eac",
    "tokenizer.json": "a75dc79c6ec004a7e2d346c20e0af8d29aa2b251ea356964718aef8b8f052e80",
    "tokenizer_config.json": "7e28810d55aafb02f6b216f05fa1208a4aac62abf794796288a70314eed5ddf3",
    "vocab.json": "5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363",
}
MODEL_FILES = tuple(MODEL_FILE_SHA256)
USER_AGENT = (
    "ai-research-notes-video-dit/0.1 "
    "(https://github.com/weepsdanky/ai-research-notes; temporal review audit)"
)
UPSTREAM_METRIC = "OpenAI CLIP ViT-B/32 single-poster retrieval margin"


@dataclass(frozen=True)
class Snapshot:
    data: bytes
    sha256: str


@dataclass(frozen=True)
class LockedVideo:
    title: str
    duration_seconds: float
    original_media_sha1: str
    original_media_timestamp: str
    derivative_url: str
    derivative_path: str
    derivative_sha256: str
    source_duration_seconds: float | None = None


@dataclass(frozen=True)
class TemporalWindow:
    start_seconds: float
    end_seconds: float
    frame_timestamps: tuple[float, ...]
    frame_sha256: tuple[str, ...]
    frames: tuple[Image.Image, ...]


def snapshot(path: Path) -> Snapshot:
    data = path.read_bytes()
    return Snapshot(data, hashlib.sha256(data).hexdigest())


def parse_jsonl(value: Snapshot) -> list[dict[str, object]]:
    return [json.loads(line) for line in value.data.decode().splitlines() if line.strip()]


def parse_json(value: Snapshot) -> dict[str, object]:
    result = json.loads(value.data)
    if not isinstance(result, dict):
        raise ValueError("JSON snapshot must be an object")
    return result


def lock_shortlist(
    candidate_snapshot: Snapshot, provenance_snapshot: Snapshot,
    upstream_snapshot: Snapshot, *, limit: int, subjects: set[str] | None,
) -> list[dict[str, object]]:
    if limit < 1:
        raise ValueError("limit must be positive")
    candidates = parse_jsonl(candidate_snapshot)
    provenance = parse_jsonl(provenance_snapshot)
    upstream = parse_json(upstream_snapshot)
    if upstream.get("candidate_sha256") != candidate_snapshot.sha256:
        raise ValueError("upstream report does not bind the candidate snapshot")
    if upstream.get("provenance_sha256") != provenance_snapshot.sha256:
        raise ValueError("upstream report does not bind the provenance snapshot")
    if (
        upstream.get("metric") != UPSTREAM_METRIC
        or upstream.get("checkpoint_sha256") != OFFICIAL_VIT_B32_SHA256
        or upstream.get("training_ready") is not False
    ):
        raise ValueError("upstream report is not a locked visual-priority report")
    candidate_by_title = _unique_by_title(candidates, "candidate")
    provenance_by_title = _unique_by_title(provenance, "provenance")
    if set(candidate_by_title) != set(provenance_by_title):
        raise ValueError("candidate and provenance title sets differ")
    records = upstream.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("upstream report lacks records")
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for expected_rank, record in enumerate(records, 1):
        if not isinstance(record, dict) or not isinstance(record.get("title"), str):
            raise ValueError("upstream record lacks a title")
        title = str(record["title"])
        if title in seen:
            raise ValueError("upstream record titles must be unique")
        seen.add(title)
        candidate = candidate_by_title.get(title)
        source = provenance_by_title.get(title)
        if candidate is None or source is None:
            raise ValueError(f"upstream title is absent from locked inputs: {title}")
        margin = record.get("margin")
        if (
            record.get("rank") != expected_rank
            or not isinstance(margin, (int, float))
            or isinstance(margin, bool)
            or not np.isfinite(float(margin))
        ):
            raise ValueError("upstream ranks must be contiguous and margins finite")
        for field in ("original_media_sha1", "original_media_timestamp"):
            if record.get(field) != source.get(field):
                raise ValueError(f"upstream poster revision differs from provenance: {title}")
        poster_sha = record.get("poster_sha256")
        if (
            not isinstance(poster_sha, str) or len(poster_sha) != 64
            or any(character not in "0123456789abcdef" for character in poster_sha)
        ):
            raise ValueError(f"upstream poster SHA-256 is invalid: {title}")
        if candidate.get("review_status") != "pending":
            continue
        if subjects and str(candidate.get("subject", "")) not in subjects:
            continue
        duration = float(source.get("duration_seconds", 0))
        if duration <= 0 or not np.isfinite(duration):
            raise ValueError(f"locked positive duration is required: {title}")
        for field in (
            "original_media_sha1", "original_media_timestamp", "review_preview_url",
        ):
            if not isinstance(source.get(field), str) or not str(source[field]).strip():
                raise ValueError(f"provenance lacks {field}: {title}")
        selected.append({
            **candidate,
            "duration_seconds": duration,
            "original_media_sha1": source["original_media_sha1"],
            "original_media_timestamp": source["original_media_timestamp"],
            "review_preview_url": source["review_preview_url"],
            "poster_rank": record.get("rank"),
            "poster_margin": record.get("margin"),
        })
        if len(selected) == limit:
            break
    if not selected:
        raise ValueError("shortlist is empty")
    return selected


def _unique_by_title(
    rows: list[dict[str, object]], label: str,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        title = row.get("title")
        if not isinstance(title, str) or not title or title in result:
            raise ValueError(f"{label} titles must be non-empty and unique")
        result[title] = row
    return result


def download_derivative(url: str, destination: Path) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "upload.wikimedia.org":
        raise ValueError("derivative URL must be HTTPS upload.wikimedia.org")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        for attempt in range(8):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    with temporary.open("wb") as handle:
                        shutil.copyfileobj(response, handle)
                break
            except urllib.error.HTTPError as error:
                if error.code != 429 or attempt == 7:
                    raise
                retry = error.headers.get("Retry-After") if error.headers else None
                time.sleep(float(retry) if retry else min(60, 2 ** attempt))
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise ValueError("derivative download was empty")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def probe_video_duration(video: Path) -> float:
    """Return the local derivative's decodable container duration."""

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required")
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video),
        ],
        capture_output=True,
        text=True,
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError as error:
        raise RuntimeError("ffprobe did not return a video duration") from error
    if result.returncode != 0 or not np.isfinite(duration) or duration <= 0:
        raise RuntimeError("ffprobe did not return a finite positive video duration")
    return duration


def extract_uniform_frames(
    video: Path, *, duration: float, count: int = 8, start_seconds: float = 0.0,
) -> tuple[list[Image.Image], tuple[float, ...], tuple[str, ...]]:
    if (
        count < 2 or duration <= 0 or start_seconds < 0
        or not np.isfinite(duration) or not np.isfinite(start_seconds)
    ):
        raise ValueError("frame count and duration must be valid")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")
    timestamps = tuple(
        start_seconds + duration * (index + 0.5) / count for index in range(count)
    )
    frames: list[Image.Image] = []
    hashes: list[str] = []
    with tempfile.TemporaryDirectory(prefix="xclip-frames-") as directory:
        for index, timestamp in enumerate(timestamps):
            path = Path(directory) / f"{index:02d}.png"
            result = subprocess.run([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss",
                f"{timestamp:.9f}", "-i", str(video), "-frames:v", "1", str(path),
            ])
            if result.returncode != 0 or not path.is_file():
                raise RuntimeError(
                    f"ffmpeg could not decode frame {index} at {timestamp:.6f}s"
                )
            data = path.read_bytes()
            image = Image.open(io.BytesIO(data)).convert("RGB")
            image.load()
            frames.append(image)
            hashes.append(hashlib.sha256(data).hexdigest())
    return frames, timestamps, tuple(hashes)


def temporal_window_starts(
    duration: float, *, window_seconds: float = 2.0, max_windows: int = 8,
) -> tuple[float, ...]:
    if (
        duration <= 0 or window_seconds <= 0 or max_windows < 1
        or not np.isfinite(duration) or not np.isfinite(window_seconds)
    ):
        raise ValueError("temporal window parameters must be finite and positive")
    if duration <= window_seconds:
        return (0.0,)
    count = min(max_windows, max(1, int(np.ceil(duration / window_seconds))))
    # Container duration may exceed the last decodable presentation timestamp slightly.
    # Leave a small deterministic tail guard instead of sampling exactly at EOF.
    tail = max(0.0, duration - window_seconds - min(0.5, window_seconds / 4))
    return tuple(float(value) for value in np.linspace(0, tail, count))


def extract_temporal_windows(
    video: Path, *, duration: float, window_seconds: float = 2.0,
    max_windows: int = 8,
) -> list[TemporalWindow]:
    windows = []
    for start in temporal_window_starts(
        duration, window_seconds=window_seconds, max_windows=max_windows,
    ):
        length = min(window_seconds, duration - start)
        frames, timestamps, hashes = extract_uniform_frames(
            video, duration=length, count=8, start_seconds=start,
        )
        windows.append(TemporalWindow(
            start_seconds=start, end_seconds=start + length,
            frame_timestamps=timestamps, frame_sha256=hashes, frames=tuple(frames),
        ))
    return windows


def lock_video(
    row: dict[str, object], cache_dir: Path, *, request_delay: float,
    window_seconds: float = 2.0, max_windows: int = 8,
) -> tuple[LockedVideo, list[TemporalWindow]]:
    title = str(row["title"])
    current = fetch_review_media(title)
    if (
        current.get("original_media_sha1") != row["original_media_sha1"]
        or current.get("original_media_timestamp") != row["original_media_timestamp"]
        or current.get("preview_url") != row["review_preview_url"]
    ):
        raise ValueError(f"Commons media or derivative changed: {title}")
    url = str(row["review_preview_url"])
    key = hashlib.sha256(
        "\0".join((title, str(row["original_media_sha1"]), url)).encode()
    ).hexdigest()[:24]
    suffix = Path(urllib.parse.urlparse(url).path).suffix or ".webm"
    path = cache_dir / f"{key}{suffix}"
    if not path.is_file():
        download_derivative(url, path)
        time.sleep(request_delay)
    derivative_data = path.read_bytes()
    if not derivative_data:
        raise ValueError(f"cached derivative is empty: {title}")
    source_duration = float(row["duration_seconds"])
    derivative_duration = probe_video_duration(path)
    windows = extract_temporal_windows(
        path, duration=derivative_duration,
        window_seconds=window_seconds, max_windows=max_windows,
    )
    return LockedVideo(
        title=title, duration_seconds=derivative_duration,
        original_media_sha1=str(row["original_media_sha1"]),
        original_media_timestamp=str(row["original_media_timestamp"]),
        derivative_url=url, derivative_path=str(path),
        derivative_sha256=hashlib.sha256(derivative_data).hexdigest(),
        source_duration_seconds=source_duration,
    ), windows


def freeze_model(model_dir: Path, destination: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in MODEL_FILES:
        source = model_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"X-CLIP model file is required: {name}")
        value = snapshot(source)
        (destination / name).parent.mkdir(parents=True, exist_ok=True)
        (destination / name).write_bytes(value.data)
        hashes[name] = value.sha256
    mismatches = [
        name for name, expected in MODEL_FILE_SHA256.items()
        if hashes.get(name) != expected
    ]
    if mismatches:
        raise ValueError(f"X-CLIP fixed-revision file SHA-256 mismatch: {mismatches}")
    config = json.loads((destination / "config.json").read_text())
    if config.get("vision_config", {}).get("num_frames") != 8:
        raise ValueError("X-CLIP checkpoint must declare exactly 8 frames")
    return hashes


def load_xclip(model_dir: Path, device: str):
    from transformers import XCLIPModel, XCLIPProcessor, __version__

    processor = XCLIPProcessor.from_pretrained(model_dir, local_files_only=True)
    model = XCLIPModel.from_pretrained(model_dir, local_files_only=True).eval().to(device)
    return model, processor, __version__


def xclip_scores(
    model, processor, frames: list[Image.Image], prompts: list[str], device: str,
) -> tuple[list[float], list[float]]:
    def run(video: list[Image.Image]) -> list[float]:
        inputs = processor(
            text=prompts, images=video, return_tensors="pt", padding=True,
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits_per_video[0].float().cpu()
        if logits.shape != (len(prompts),) or not torch.isfinite(logits).all():
            raise ValueError("X-CLIP logits must be finite and match prompts")
        return [float(value) for value in logits]
    return run(frames), run(list(reversed(frames)))


def score_temporal_windows(
    model, processor, windows: list[TemporalWindow], prompts: list[str],
    *, positive_count: int, device: str,
) -> tuple[list[float], list[float], list[dict[str, object]]]:
    if positive_count < 1 or positive_count >= len(prompts) or not windows:
        raise ValueError("window scoring requires positive and negative prompts")
    scored = []
    for window in windows:
        forward, reverse = xclip_scores(
            model, processor, list(window.frames), prompts, device,
        )
        margin = float(np.mean(forward[:positive_count]) - max(forward[positive_count:]))
        reverse_margin = float(
            np.mean(reverse[:positive_count]) - max(reverse[positive_count:])
        )
        if not np.isfinite(margin) or not np.isfinite(reverse_margin):
            raise ValueError("window margins must be finite")
        scored.append({
            "start_seconds": window.start_seconds,
            "end_seconds": window.end_seconds,
            "frame_timestamps": window.frame_timestamps,
            "frame_sha256": window.frame_sha256,
            "logits": forward, "reverse_logits": reverse,
            "margin": margin, "reverse_margin": reverse_margin,
            "order_sensitivity": margin - reverse_margin,
        })
    best = max(scored, key=lambda row: (float(row["margin"]), -float(row["start_seconds"])))
    return list(best["logits"]), list(best["reverse_logits"]), scored


def rerank(
    videos: list[LockedVideo], logits: list[list[float]],
    reverse_logits: list[list[float]], *, positive_count: int,
) -> list[dict[str, object]]:
    if positive_count < 1 or not logits or len(logits) != len(videos):
        raise ValueError("X-CLIP ranking inputs are invalid")
    rows: list[dict[str, object]] = []
    for video, forward, reverse in zip(videos, logits, reverse_logits, strict=True):
        if len(forward) <= positive_count or len(reverse) != len(forward):
            raise ValueError("X-CLIP prompt score shape is invalid")
        margin = float(np.mean(forward[:positive_count]) - max(forward[positive_count:]))
        reverse_margin = float(
            np.mean(reverse[:positive_count]) - max(reverse[positive_count:])
        )
        if not np.isfinite(margin) or not np.isfinite(reverse_margin):
            raise ValueError("X-CLIP margins must be finite")
        rows.append({
            **video.__dict__, "logits": forward, "reverse_logits": reverse,
            "margin": margin, "reverse_margin": reverse_margin,
            "order_sensitivity": margin - reverse_margin,
        })
    rows.sort(key=lambda row: (-float(row["margin"]), str(row["title"])))
    for index, row in enumerate(rows, 1):
        row["rank"] = index
    return rows


def ranking_metrics(
    rows: list[dict[str, object]], labels: dict[str, bool], *, cutoffs: list[int],
) -> dict[str, object]:
    if not rows or not labels or not cutoffs or any(value < 1 for value in cutoffs):
        raise ValueError("ranking metrics require rows, labels, and positive cutoffs")
    titles = [str(row.get("title", "")) for row in rows]
    if any(not title for title in titles) or len(titles) != len(set(titles)):
        raise ValueError("ranking rows require unique non-empty titles")
    if set(titles) != set(labels):
        raise ValueError("ranking rows and labels must have identical titles")
    positives = sum(labels.values())
    if positives < 1 or positives == len(labels):
        raise ValueError("ranking evaluation requires positive and negative labels")
    metrics: dict[str, object] = {
        "records": len(rows), "positives": positives,
        "average_precision": 0.0, "recall_at_k": {},
    }
    hits = 0
    precision_sum = 0.0
    for index, row in enumerate(rows, 1):
        if labels[str(row["title"])]:
            hits += 1
            precision_sum += hits / index
    metrics["average_precision"] = precision_sum / positives
    metrics["recall_at_k"] = {
        str(cutoff): sum(
            labels[str(row["title"])] for row in rows[:min(cutoff, len(rows))]
        ) / positives
        for cutoff in sorted(set(cutoffs))
    }
    return metrics


def write_report(
    output: Path, *, rows: list[dict[str, object]], candidate_snapshot: Snapshot,
    provenance_snapshot: Snapshot, upstream_snapshot: Snapshot,
    model_hashes: dict[str, str], positive: list[str], negative: list[str],
    device: str, window_seconds: float, max_windows: int,
    transformers_version: str,
) -> dict[str, object]:
    report = {
        "metric": "X-CLIP eight-frame forward margin with reverse-order diagnostic",
        "limitation": (
            "read-only ranking for full temporal human review; scores cannot approve "
            "subject, action, scene, source independence, or training readiness"
        ),
        "candidate_sha256": candidate_snapshot.sha256,
        "provenance_sha256": provenance_snapshot.sha256,
        "upstream_report_sha256": upstream_snapshot.sha256,
        "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
        "model_file_sha256": model_hashes, "frame_count": 8,
        "sampling": "evenly spaced fixed windows; eight midpoint frames per window",
        "window_seconds": window_seconds, "max_windows": max_windows,
        "positive_prompts": positive, "negative_prompts": negative,
        "device": device, "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "records": rows, "training_ready": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("provenance", type=Path)
    parser.add_argument("upstream_report", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--positive", action="append", required=True)
    parser.add_argument("--negative", action="append", required=True)
    parser.add_argument("--subject", action="append", choices=("cat", "dog", "person"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--max-windows", type=int, default=8)
    args = parser.parse_args(argv)
    if args.request_delay < 0:
        raise ValueError("request_delay must be non-negative")
    paths = (args.candidates, args.provenance, args.upstream_report)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("candidate, provenance, and upstream report are required")
    output_resolved = args.output.resolve()
    protected_directories = (args.model_dir.resolve(), args.cache.resolve())
    if (
        any(output_resolved == path.resolve() for path in paths)
        or any(output_resolved == directory or output_resolved.is_relative_to(directory)
                for directory in protected_directories)
    ):
        raise ValueError("output must differ from every input")
    candidate_value, provenance_value, upstream_value = map(snapshot, paths)
    selected = lock_shortlist(
        candidate_value, provenance_value, upstream_value, limit=args.limit,
        subjects=set(args.subject) if args.subject else None,
    )
    with tempfile.TemporaryDirectory(prefix="xclip-model-") as directory:
        frozen_model = Path(directory)
        model_hashes = freeze_model(args.model_dir, frozen_model)
        model, processor, transformers_version = load_xclip(frozen_model, args.device)
        videos: list[LockedVideo] = []
        forward: list[list[float]] = []
        reverse: list[list[float]] = []
        windows_by_title: dict[str, list[dict[str, object]]] = {}
        prompts = args.positive + args.negative
        for row in selected:
            locked, windows = lock_video(
                row, args.cache, request_delay=args.request_delay,
                window_seconds=args.window_seconds, max_windows=args.max_windows,
            )
            scores, reverse_scores, window_scores = score_temporal_windows(
                model, processor, windows, prompts,
                positive_count=len(args.positive), device=args.device,
            )
            videos.append(locked)
            forward.append(scores)
            reverse.append(reverse_scores)
            windows_by_title[locked.title] = window_scores
    current = fetch_review_media_batch([str(row["title"]) for row in selected])
    for row in selected:
        title = str(row["title"])
        if (
            current[title].get("original_media_sha1") != row["original_media_sha1"]
            or current[title].get("original_media_timestamp")
            != row["original_media_timestamp"]
            or current[title].get("preview_url") != row["review_preview_url"]
        ):
            raise ValueError(f"Commons media changed during X-CLIP ranking: {title}")
    ranked = rerank(
        videos, forward, reverse, positive_count=len(args.positive),
    )
    for row in ranked:
        row["windows"] = windows_by_title[str(row["title"])]
    report = write_report(
        args.output, rows=ranked, candidate_snapshot=candidate_value,
        provenance_snapshot=provenance_value, upstream_snapshot=upstream_value,
        model_hashes=model_hashes, positive=args.positive, negative=args.negative,
        device=args.device, window_seconds=args.window_seconds,
        max_windows=args.max_windows, transformers_version=transformers_version,
    )
    print(json.dumps({
        "records": len(ranked), "output": str(args.output),
        "checkpoint_sha256": report["model_file_sha256"]["model.safetensors"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
