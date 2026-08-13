from pathlib import Path

import torch

from cache_text_embeddings import cache_manifest_embeddings
from real_data import RealVideoRecord, load_manifest, write_manifest


def test_cache_embeddings_updates_manifest_with_pinned_encoder(tmp_path):
    source = write_manifest(
        tmp_path / "source.jsonl",
        [
            RealVideoRecord(
                sample_id="cat-1",
                source_id="source-cat",
                clip_path="clips/cat.mp4",
                prompt="a cat jumping on grass",
                subject="cat",
                action="jumping",
                scene="grass",
                source_url="https://example.test/cat",
                license="CC0",
                attribution="Example",
                sha256="1" * 64,
            )
        ],
    )
    output = tmp_path / "embedded.jsonl"

    cache_manifest_embeddings(
        source,
        output,
        encoder_id="test-encoder@revision",
        encode=lambda prompts: torch.tensor([[1.0, 2.0, 3.0]]),
    )

    record = load_manifest(output)[0]
    assert record.text_encoder_id == "test-encoder@revision"
    embedding_path = output.parent / record.text_embedding_path
    assert torch.equal(
        torch.load(embedding_path, weights_only=True),
        torch.tensor([1.0, 2.0, 3.0]),
    )


def test_cache_embedding_filename_does_not_trust_sample_id_as_path(tmp_path):
    source = write_manifest(
        tmp_path / "source.jsonl",
        [
            RealVideoRecord(
                sample_id="../../escape",
                source_id="source-cat", clip_path="clips/cat.mp4",
                prompt="a cat", subject="cat", action="walking", scene="grass",
                source_url="https://example.test/cat", license="CC0",
                attribution="Example", sha256="1" * 64,
            )
        ],
    )
    output = tmp_path / "embedded.jsonl"

    cache_manifest_embeddings(
        source, output, encoder_id="test@revision",
        encode=lambda prompts: torch.ones(1, 3),
    )

    record = load_manifest(output)[0]
    embedding = (output.parent / record.text_embedding_path).resolve()
    assert embedding.is_relative_to((tmp_path / "embedded_text_embeddings").resolve())
    assert not (tmp_path.parent / "escape.pt").exists()
