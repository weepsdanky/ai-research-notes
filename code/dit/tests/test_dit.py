import pytest
import torch

from dit import (
    DiT,
    DiTBlock,
    DiTConfig,
    LabelEmbedder,
    get_2d_sincos_pos_embed,
    timestep_embedding,
)


def tiny_config(**overrides):
    values = dict(
        image_size=8,
        patch_size=2,
        in_channels=1,
        hidden_size=16,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        num_classes=10,
        class_dropout_prob=0.1,
    )
    values.update(overrides)
    return DiTConfig(**values)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"image_size": 7}, "divisible"),
        ({"hidden_size": 18}, "num_heads"),
        ({"class_dropout_prob": 1.1}, "dropout"),
        ({"depth": 0}, "positive"),
    ],
)
def test_config_rejects_invalid_architectures(overrides, message):
    with pytest.raises(ValueError, match=message):
        tiny_config(**overrides)


def test_patch_embedding_and_output_shapes():
    model = DiT(tiny_config())
    x = torch.randn(3, 1, 8, 8)
    tokens = model.patch_embed(x)
    output = model(x, torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]))
    assert tokens.shape == (3, 16, 16)
    assert output.shape == x.shape


def test_learned_sigma_doubles_output_channels():
    model = DiT(tiny_config(learn_sigma=True))
    output = model(
        torch.randn(2, 1, 8, 8),
        torch.tensor([0, 1]),
        torch.tensor([1, 2]),
    )
    assert output.shape == (2, 2, 8, 8)


def test_zero_initialized_dit_starts_as_zero_function():
    model = DiT(tiny_config())
    output = model(
        torch.randn(2, 1, 8, 8),
        torch.tensor([5, 9]),
        torch.tensor([1, 4]),
    )
    assert torch.equal(output, torch.zeros_like(output))


def test_zero_initialized_adaln_block_is_identity():
    block = DiTBlock(hidden_size=16, num_heads=4, mlp_ratio=2.0)
    torch.nn.init.zeros_(block.adaLN_modulation[-1].weight)
    torch.nn.init.zeros_(block.adaLN_modulation[-1].bias)
    x = torch.randn(2, 7, 16)
    condition = torch.randn(2, 16)
    assert torch.equal(block(x, condition), x)


def test_fixed_position_embedding_is_deterministic_and_unique():
    first = get_2d_sincos_pos_embed(16, 4)
    second = get_2d_sincos_pos_embed(16, 4)
    assert first.shape == (1, 16, 16)
    assert torch.equal(first, second)
    assert not torch.equal(first[:, 0], first[:, -1])


def test_timestep_embedding_supports_odd_dimensions():
    embedding = timestep_embedding(torch.tensor([0, 1, 2]), dim=7)
    assert embedding.shape == (3, 7)
    assert torch.isfinite(embedding).all()


def test_forced_label_dropout_uses_null_embedding():
    embedder = LabelEmbedder(num_classes=3, hidden_size=4, dropout_prob=0.0)
    labels = torch.tensor([0, 1, 2])
    force_drop = torch.tensor([False, True, False])
    result = embedder(labels, train=False, force_drop_ids=force_drop)
    assert torch.equal(result[0], embedder.embedding.weight[0])
    assert torch.equal(result[1], embedder.embedding.weight[3])
    assert torch.equal(result[2], embedder.embedding.weight[2])


def test_classifier_free_guidance_combines_conditional_and_null_predictions(monkeypatch):
    model = DiT(tiny_config())
    x = torch.randn(4, 1, 8, 8)
    t = torch.tensor([3, 4, 9, 9])
    labels = torch.tensor([1, 2, 1, 2])

    def fake_forward(x, t, labels, force_drop_ids=None):
        values = torch.where(force_drop_ids, 2.0, 5.0)
        return values[:, None, None, None].expand_as(x)

    monkeypatch.setattr(model, "forward", fake_forward)
    output = model.forward_with_cfg(x, t, labels, cfg_scale=3.0)
    # 2 + 3 * (5 - 2) = 11
    assert torch.equal(output, torch.full_like(output, 11.0))


def test_classifier_free_guidance_rejects_odd_batches():
    model = DiT(tiny_config())
    with pytest.raises(ValueError, match="even batch"):
        model.forward_with_cfg(
            torch.randn(3, 1, 8, 8),
            torch.zeros(3, dtype=torch.long),
            torch.zeros(3, dtype=torch.long),
            cfg_scale=2.0,
        )


def test_forward_validates_input_shapes():
    model = DiT(tiny_config())
    with pytest.raises(ValueError, match="x must have shape"):
        model(
            torch.randn(2, 1, 7, 8),
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
        )

