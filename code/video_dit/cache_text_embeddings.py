"""Cache frozen CLIP text embeddings beside an auditable video manifest."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Callable

import torch

from real_data import load_manifest, write_manifest


Encode = Callable[[list[str]], torch.Tensor]


def cache_manifest_embeddings(
    source: Path,
    output: Path,
    *,
    encoder_id: str,
    encode: Encode,
    batch_size: int = 32,
) -> Path:
    if not encoder_id or "@" not in encoder_id:
        raise ValueError("encoder_id must include a pinned revision after @")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    records = load_manifest(source)
    embedding_dir = output.parent / f"{output.stem}_text_embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    updated = []
    for offset in range(0, len(records), batch_size):
        batch = records[offset:offset + batch_size]
        embeddings = encode([record.prompt for record in batch])
        if embeddings.ndim != 2 or embeddings.shape[0] != len(batch):
            raise ValueError("encoder must return [batch, embedding_size]")
        if not torch.isfinite(embeddings).all():
            raise ValueError("text encoder returned non-finite embeddings")
        for record, embedding in zip(batch, embeddings, strict=True):
            safe_id = hashlib.sha256(record.sample_id.encode()).hexdigest()
            path = embedding_dir / f"{safe_id}.pt"
            torch.save(embedding.detach().float().cpu(), path)
            updated.append(
                replace(
                    record,
                    text_embedding_path=str(path.relative_to(output.parent)),
                    text_encoder_id=encoder_id,
                )
            )
    return write_manifest(output, updated)


def openai_clip_encoder(checkpoint: Path) -> tuple[str, Encode]:
    """Load the official OpenAI JIT checkpoint and expose its frozen text tower."""

    import hashlib
    import clip

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    scripted = torch.jit.load(str(checkpoint), map_location="cpu")
    model = clip.model.build_model(scripted.state_dict()).float().eval()

    def encode(prompts: list[str]) -> torch.Tensor:
        tokenized = clip.tokenize(prompts, truncate=True)
        with torch.inference_mode():
            embeddings = model.encode_text(tokenized).float()
        return embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    return f"openai/CLIP-ViT-B-32@sha256:{digest}", encode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    encoder_id, encode = openai_clip_encoder(args.checkpoint)
    cache_manifest_embeddings(
        args.source,
        args.output,
        encoder_id=encoder_id,
        encode=encode,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
