from dataclasses import replace
import hashlib
import json

import torch

from real_data import RealVideoRecord, write_manifest
from vae_latents import cache_manifest_latents, decode_video_latents


def _record(tmp_path):
    clip = tmp_path / "clips" / "cat.mp4"
    clip.parent.mkdir()
    clip.write_bytes(b"video")
    return RealVideoRecord(
        sample_id="cat", source_id="commons:cat", clip_path="clips/cat.mp4",
        prompt="a cat walking", subject="cat", action="walking", scene="grass",
        source_url="https://commons.wikimedia.org/cat", license="CC0",
        attribution="Author", sha256=hashlib.sha256(b"video").hexdigest(),
        perceptual_hash="0" * 16,
        split="train",
    )


def test_cache_manifest_latents_is_deterministic_and_writes_vae_contract(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "latent.jsonl"
    write_manifest(source, [_record(tmp_path)])
    video = torch.linspace(-1, 1, 3 * 4 * 16 * 16).reshape(3, 4, 16, 16)
    monkeypatch.setattr("vae_latents.decode_video_clip", lambda *args, **kwargs: video)

    class Encoder:
        vae_id = "test/vae@sha256:abc"
        scaling_factor = 0.25
        latent_size = 2
        latent_channels = 4

        def encode(self, frames):
            assert frames.shape == (4, 3, 16, 16)
            return torch.ones(4, 4, 2, 2)

    cache_manifest_latents(source, output, encoder=Encoder(), frames=4, size=16)

    row = json.loads(output.read_text())
    latent = torch.load(tmp_path / row["latent_path"], weights_only=True)
    assert latent.shape == (4, 4, 2, 2)
    assert torch.equal(latent, torch.full_like(latent, 0.25))
    assert row["vae_id"] == "test/vae@sha256:abc"
    assert row["vae_scaling"] == 0.25


def test_decode_video_latents_unscales_and_restores_channel_first_video():
    seen = {}

    class Decoder:
        scaling_factor = 0.5

        def decode(self, latents):
            seen["latents"] = latents
            return torch.zeros(latents.shape[0], 3, 16, 16)

    video = decode_video_latents(torch.ones(4, 3, 2, 2), Decoder())
    assert torch.equal(seen["latents"], torch.full((3, 4, 2, 2), 2.0))
    assert video.shape == (3, 3, 16, 16)


def test_cache_manifest_latents_rejects_source_sha_mismatch(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "latent.jsonl"
    write_manifest(source, [replace(_record(tmp_path), sha256="a" * 64)])

    class Encoder:
        vae_id = "test/vae@revision"
        scaling_factor = 0.25
        latent_size = 2
        latent_channels = 4
        downsample_factor = 8
        def encode(self, frames): return torch.ones(4, 4, 2, 2)

    import pytest
    with pytest.raises(ValueError, match="sha256 mismatch"):
        cache_manifest_latents(source, output, encoder=Encoder(), frames=4, size=16)


def test_decode_rejects_invalid_scaling_and_nonfinite_output():
    class Decoder:
        vae_id = "test"
        scaling_factor = 0.0
        def decode(self, latents): return torch.full((3, 3, 16, 16), float("nan"))

    import pytest
    with pytest.raises(ValueError, match="positive and finite"):
        decode_video_latents(torch.ones(4, 3, 2, 2), Decoder())
