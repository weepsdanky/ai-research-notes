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


def test_cache_token_embeddings_preserves_sequence_and_contract(tmp_path):
    source = write_manifest(tmp_path / "source.jsonl", [RealVideoRecord(
        sample_id="cat", source_id="cat", clip_path="clips/cat.mp4",
        prompt="a cat", subject="cat", action="walking", scene="grass",
        source_url="https://example.test/cat", license="CC0",
        attribution="Example", sha256="1" * 64,
    )])
    output = tmp_path / "tokens.jsonl"

    cache_manifest_embeddings(
        source, output, encoder_id="test@revision#tokens-v2",
        encode=lambda prompts: torch.ones(1, 7, 5), embedding_kind="tokens",
    )

    record = load_manifest(output)[0]
    embedding = torch.load(output.parent / record.text_embedding_path, weights_only=True)
    assert embedding.shape == (7, 5)
    assert record.text_embedding_kind == "tokens"


def test_token_cache_rejects_old_mask_contract_and_empty_sequence(tmp_path):
    source = write_manifest(tmp_path / "source.jsonl", [RealVideoRecord(
        sample_id="cat", source_id="cat", clip_path="clips/cat.mp4",
        prompt="a cat", subject="cat", action="walking", scene="grass",
        source_url="https://example.test/cat", license="CC0",
        attribution="Example", sha256="1" * 64,
    )])
    with __import__("pytest").raises(ValueError, match="tokens-v2"):
        cache_manifest_embeddings(
            source, tmp_path / "old.jsonl", encoder_id="clip@rev#tokens-v1",
            encode=lambda prompts: torch.ones(1, 7, 5), embedding_kind="tokens",
        )
    with __import__("pytest").raises(ValueError, match="valid token"):
        cache_manifest_embeddings(
            source, tmp_path / "empty.jsonl", encoder_id="clip@rev#tokens-v2",
            encode=lambda prompts: torch.zeros(1, 7, 5), embedding_kind="tokens",
        )


def test_openai_token_encoder_zeroes_padding_hidden_states(tmp_path, monkeypatch):
    checkpoint = tmp_path / "clip.pt"
    checkpoint.write_bytes(b"weights")

    class Model:
        dtype = torch.float32
        positional_embedding = torch.zeros(7, 5)
        token_embedding = lambda self, ids: torch.ones(ids.shape[0], ids.shape[1], 5)
        transformer = lambda self, value: value
        ln_final = lambda self, value: value
        float = lambda self: self
        eval = lambda self: self

    monkeypatch.setattr("cache_text_embeddings.torch.jit.load", lambda *a, **k: type("J", (), {"state_dict": lambda self: {}})())
    import clip
    monkeypatch.setattr(clip.model, "build_model", lambda state: Model())
    monkeypatch.setattr(
        clip, "tokenize", lambda prompts, truncate: torch.tensor([[1, 2, 3, 0, 0, 0, 0]]),
    )
    from cache_text_embeddings import openai_clip_encoder
    encoder_id, encode = openai_clip_encoder(checkpoint, embedding_kind="tokens")

    tokens = encode(["a cat"])

    assert torch.count_nonzero(tokens[:, :3]) > 0
    assert torch.count_nonzero(tokens[:, 3:]) == 0
    assert encoder_id.endswith("#tokens-v2")
