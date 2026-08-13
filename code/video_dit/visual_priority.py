"""Rank locked Commons posters for human video review; never approve labels."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image
import torch


OFFICIAL_VIT_B32_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
USER_AGENT = (
    "ai-research-notes-video-dit/0.1 "
    "(https://github.com/weepsdanky/ai-research-notes; research poster audit)"
)


@dataclass(frozen=True)
class CandidatePoster:
    title: str
    original_media_sha1: str
    original_media_timestamp: str
    poster_url: str
    poster_path: str
    poster_sha256: str


@dataclass(frozen=True)
class InputSnapshot:
    data: bytes
    sha256: str


def load_input_snapshot(path: Path) -> InputSnapshot:
    data = path.read_bytes()
    return InputSnapshot(data=data, sha256=hashlib.sha256(data).hexdigest())


def _api_json(parameters: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(parameters),
        headers={"User-Agent": USER_AGENT},
    )
    for attempt in range(7):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise ValueError("Commons poster response must be an object")
            return payload
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 6:
                raise
            retry = error.headers.get("Retry-After") if error.headers else None
            time.sleep(float(retry) if retry else min(60, 2 ** attempt))
    raise RuntimeError("unreachable")


def fetch_poster_metadata(titles: list[str]) -> dict[str, dict[str, str]]:
    """Fetch one locked current-media poster URL for each exact title."""

    result: dict[str, dict[str, str]] = {}
    for offset in range(0, len(titles), 50):
        batch = titles[offset:offset + 50]
        payload = _api_json({
            "action": "query", "format": "json", "formatversion": 2,
            "prop": "imageinfo", "iiprop": "url|sha1|timestamp",
            "iiurlwidth": 320, "titles": "|".join(batch),
        })
        pages = payload.get("query", {}).get("pages", [])  # type: ignore[union-attr]
        if not isinstance(pages, list):
            raise ValueError("Commons poster response lacks pages")
        for page in pages:
            if not isinstance(page, dict) or not isinstance(page.get("title"), str):
                continue
            infos = page.get("imageinfo")
            if not isinstance(infos, list) or len(infos) != 1 or not isinstance(infos[0], dict):
                continue
            info = infos[0]
            result[str(page["title"])] = {
                "original_media_sha1": str(info.get("sha1", "")),
                "original_media_timestamp": str(info.get("timestamp", "")),
                "poster_url": str(info.get("thumburl", "")),
            }
    missing = set(titles) - set(result)
    if missing:
        raise ValueError(f"Commons response lacks poster metadata: {sorted(missing)[:3]}")
    return result


def download_poster(url: str, path: Path) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "upload.wikimedia.org":
        raise ValueError("poster URL must be HTTPS upload.wikimedia.org")
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT},
    )
    for attempt in range(8):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            break
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 7:
                raise
            retry = error.headers.get("Retry-After") if error.headers else None
            time.sleep(float(retry) if retry else min(60, 10 * 2 ** attempt))
    else:
        raise RuntimeError("unreachable")
    if not data:
        raise ValueError("Commons poster download was empty")
    image = Image.open(io.BytesIO(data))
    image.verify()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fetch_locked_posters(
    rows: list[dict[str, object]], output: Path, *, subjects: set[str] | None = None,
    request_delay: float = 0.0,
) -> list[CandidatePoster]:
    if request_delay < 0:
        raise ValueError("request_delay must be non-negative")
    selected = [
        row for row in rows
        if row.get("review_status") == "pending"
        and (subjects is None or str(row.get("subject", "")) in subjects)
    ]
    titles = [str(row.get("title", "")) for row in selected]
    if any(not title for title in titles) or len(titles) != len(set(titles)):
        raise ValueError("selected pending candidate titles must be non-empty and unique")
    metadata = fetch_poster_metadata(titles)
    posters: list[CandidatePoster] = []
    for row in selected:
        title = str(row["title"])
        locked = metadata[title]
        if (
            str(row.get("original_media_sha1", "")) != locked["original_media_sha1"]
            or str(row.get("original_media_timestamp", ""))
            != locked["original_media_timestamp"]
        ):
            raise ValueError(f"Commons media revision changed before poster ranking: {title}")
        url = locked["poster_url"]
        if not url:
            raise ValueError(f"Commons poster URL is empty: {title}")
        cache_key = "\0".join((title, locked["original_media_sha1"], url))
        path = output / f"{hashlib.sha256(cache_key.encode()).hexdigest()[:24]}.jpg"
        if not path.is_file():
            download_poster(url, path)
            time.sleep(request_delay)
        data = path.read_bytes()
        Image.open(io.BytesIO(data)).verify()
        posters.append(CandidatePoster(
            title=title,
            original_media_sha1=locked["original_media_sha1"],
            original_media_timestamp=locked["original_media_timestamp"],
            poster_url=url, poster_path=str(path),
            poster_sha256=hashlib.sha256(data).hexdigest(),
        ))
    after = fetch_poster_metadata(titles)
    for row in selected:
        title = str(row["title"])
        if any(
            after[title][field] != metadata[title][field]
            for field in ("original_media_sha1", "original_media_timestamp")
        ):
            raise ValueError(f"Commons media revision changed during poster ranking: {title}")
    return posters


def rank_visual_candidates(
    posters: list[CandidatePoster], image_features: torch.Tensor,
    text_features: torch.Tensor, *, positive_prompts: list[str],
    negative_prompts: list[str],
) -> list[dict[str, object]]:
    if not positive_prompts or not negative_prompts:
        raise ValueError("visual ranking requires positive and negative prompts")
    if image_features.ndim != 2 or image_features.shape[0] != len(posters):
        raise ValueError("image feature shape does not match posters")
    if text_features.ndim != 2 or text_features.shape[0] != (
        len(positive_prompts) + len(negative_prompts)
    ):
        raise ValueError("text feature shape does not match prompts")
    image_features = image_features.float()
    text_features = text_features.float()
    if (
        not torch.isfinite(image_features).all()
        or not torch.isfinite(text_features).all()
        or (image_features.norm(dim=-1) == 0).any()
        or (text_features.norm(dim=-1) == 0).any()
    ):
        raise ValueError("CLIP features must be finite non-zero vectors")
    image_features = image_features / image_features.norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)
    text_features = text_features / text_features.norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)
    similarities = image_features @ text_features.T
    positive = similarities[:, :len(positive_prompts)].mean(dim=1)
    negative = similarities[:, len(positive_prompts):].max(dim=1).values
    margins = positive - negative
    order = sorted(range(len(posters)), key=lambda index: (-float(margins[index]), posters[index].title))
    rows = []
    for rank, index in enumerate(order, 1):
        rows.append({
            **asdict(posters[index]), "rank": rank,
            "positive_score": float(positive[index]),
            "negative_score": float(negative[index]),
            "margin": float(margins[index]),
        })
    return rows


def write_visual_priority_report(
    output: Path, *, rows: list[dict[str, object]], candidates: Path,
    provenance: Path, checkpoint: Path, positive_prompts: list[str],
    negative_prompts: list[str], candidate_sha256: str,
    provenance_sha256: str, checkpoint_sha256: str,
) -> dict[str, object]:
    inputs = (candidates, provenance, checkpoint)
    if any(output.resolve() == path.resolve() for path in inputs):
        raise ValueError("visual report output must differ from every input")
    report = {
        "metric": "OpenAI CLIP ViT-B/32 single-poster retrieval margin",
        "limitation": (
            "poster ranking only selects videos for full temporal human review; "
            "it cannot establish action, scene, identity, or approval"
        ),
        "candidate_sha256": candidate_sha256,
        "provenance_sha256": provenance_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "positive_prompts": positive_prompts,
        "negative_prompts": negative_prompts,
        "records": rows,
        "training_ready": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    return report


def encode_clip(
    posters: list[CandidatePoster], prompts: list[str], checkpoint: Path,
    *, batch_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    import clip

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model, preprocess = clip.load(str(checkpoint), device="cpu", jit=False)
    model = model.float().eval()
    image_batches = []
    with torch.inference_mode():
        for offset in range(0, len(posters), batch_size):
            images = torch.stack([
                preprocess(Image.open(poster.poster_path).convert("RGB"))
                for poster in posters[offset:offset + batch_size]
            ])
            image_batches.append(model.encode_image(images).float())
        text = model.encode_text(clip.tokenize(prompts, truncate=True)).float()
    images = torch.cat(image_batches) if image_batches else torch.empty((0, text.shape[-1]))
    return images, text


def _load_snapshot_rows(snapshot: InputSnapshot) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in snapshot.data.decode().splitlines() if line.strip()
    ]


def _merge_locked_rows(
    candidates: list[dict[str, object]], provenance: list[dict[str, object]],
) -> list[dict[str, object]]:
    candidate_titles = [str(row.get("title", "")) for row in candidates]
    provenance_titles = [str(row.get("title", "")) for row in provenance]
    if (
        any(not title for title in candidate_titles + provenance_titles)
        or len(candidate_titles) != len(set(candidate_titles))
        or len(provenance_titles) != len(set(provenance_titles))
        or set(candidate_titles) != set(provenance_titles)
    ):
        raise ValueError("candidate and provenance titles must be unique and identical")
    by_title = {str(row["title"]): row for row in provenance}
    locked_fields = ("original_media_sha1", "original_media_timestamp")
    merged = []
    for candidate in candidates:
        title = str(candidate["title"])
        source = by_title[title]
        for field in locked_fields:
            if not isinstance(source.get(field), str) or not str(source[field]).strip():
                raise ValueError(f"provenance lacks {field}: {title}")
        merged.append({**candidate, **{field: source[field] for field in locked_fields}})
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("provenance", type=Path)
    parser.add_argument("posters", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--subject", action="append", choices=("cat", "dog", "person"))
    parser.add_argument("--positive", action="append", required=True)
    parser.add_argument("--negative", action="append", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--request-delay", type=float, default=1.0)
    args = parser.parse_args(argv)
    paths = (args.candidates, args.provenance, args.checkpoint)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("candidate, provenance, and checkpoint files are required")
    candidate_snapshot = load_input_snapshot(args.candidates)
    provenance_snapshot = load_input_snapshot(args.provenance)
    checkpoint_snapshot = load_input_snapshot(args.checkpoint)
    if checkpoint_snapshot.sha256 != OFFICIAL_VIT_B32_SHA256:
        raise ValueError(
            "OpenAI ViT-B/32 checkpoint SHA mismatch: "
            f"{checkpoint_snapshot.sha256}"
        )
    rows = _merge_locked_rows(
        _load_snapshot_rows(candidate_snapshot),
        _load_snapshot_rows(provenance_snapshot),
    )
    posters = fetch_locked_posters(
        rows, args.posters, subjects=set(args.subject) if args.subject else None,
        request_delay=args.request_delay,
    )
    with tempfile.NamedTemporaryFile(suffix=".pt") as frozen_checkpoint:
        frozen_checkpoint.write(checkpoint_snapshot.data)
        frozen_checkpoint.flush()
        images, text = encode_clip(
            posters, args.positive + args.negative,
            Path(frozen_checkpoint.name), batch_size=args.batch_size,
        )
    ranked = rank_visual_candidates(
        posters, images, text, positive_prompts=args.positive,
        negative_prompts=args.negative,
    )
    report = write_visual_priority_report(
        args.output, rows=ranked, candidates=args.candidates,
        provenance=args.provenance, checkpoint=args.checkpoint,
        candidate_sha256=candidate_snapshot.sha256,
        provenance_sha256=provenance_snapshot.sha256,
        checkpoint_sha256=checkpoint_snapshot.sha256,
        positive_prompts=args.positive, negative_prompts=args.negative,
    )
    print(json.dumps({
        "records": len(ranked), "output": str(args.output),
        "checkpoint_sha256": report["checkpoint_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
