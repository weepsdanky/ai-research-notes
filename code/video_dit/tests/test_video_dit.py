import pytest
import torch

from video_dit import VideoDiT, VideoDiTConfig


def tiny_config(**overrides):
    values = dict(
        frames=4,
        image_size=32,
        patch_size=8,
        in_channels=3,
        hidden_size=32,
        num_heads=4,
        num_pairs=2,
        condition_dropout_prob=0.1,
    )
    values.update(overrides)
    return VideoDiTConfig(**values)


def test_variant_one_alternates_full_spatial_and_temporal_blocks():
    model = VideoDiT(tiny_config(num_pairs=3))

    assert [block.axis for block in model.blocks] == [
        "spatial",
        "temporal",
        "spatial",
        "temporal",
        "spatial",
        "temporal",
    ]


def test_cross_attention_adds_no_state_to_legacy_condition_modes():
    for mode in ("structured", "text"):
        config = VideoDiTConfig(
            frames=4, image_size=16, patch_size=4, hidden_size=32,
            num_heads=4, num_pairs=1, condition_mode=mode,
            text_embedding_size=12 if mode == "text" else None,
        )
        keys = VideoDiT(config).state_dict()

        assert not any("cross_" in key for key in keys)


@pytest.mark.parametrize("learn_sigma, expected_channels", [(False, 3), (True, 6)])
def test_video_dit_maps_a_five_dimensional_video_to_same_grid(
    learn_sigma, expected_channels
):
    model = VideoDiT(tiny_config(learn_sigma=learn_sigma))
    video = torch.randn(2, 3, 4, 32, 32)
    timesteps = torch.tensor([1, 9])
    conditions = torch.tensor([[0, 0, 0, 0], [2, 2, 1, 3]])

    prediction = model(video, timesteps, conditions)

    assert prediction.shape == (2, expected_channels, 4, 32, 32)
    # DiT/Latte zero the output projection so the network starts as the zero map.
    assert torch.count_nonzero(prediction) == 0


def test_cfg_returns_paired_guided_predictions_and_preserves_input_shape():
    model = VideoDiT(tiny_config())
    video = torch.randn(4, 3, 4, 32, 32)
    timesteps = torch.tensor([2, 3, 2, 3])
    conditions = torch.tensor(
        [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]]
    )

    guided = model.forward_with_cfg(video, timesteps, conditions, cfg_scale=2.5)

    assert guided.shape == video.shape
    assert torch.equal(guided[:2], guided[2:])


def test_cfg_uses_unconditional_plus_scale_times_conditional_delta():
    model = VideoDiT(tiny_config())
    with torch.no_grad():
        model.final_layer.linear.weight.normal_()
        model.final_layer.adaLN_modulation[-1].weight.normal_()
    video = torch.randn(4, 3, 4, 32, 32)
    timesteps = torch.tensor([2, 3, 2, 3])
    conditions = torch.tensor(
        [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]]
    )
    paired_video = torch.cat([video[:2], video[:2]])
    paired_timesteps = torch.cat([timesteps[:2], timesteps[:2]])
    force_drop = torch.tensor([False, False, True, True])
    direct = model(paired_video, paired_timesteps, conditions, force_drop)
    conditional, unconditional = direct.chunk(2)

    guided = model.forward_with_cfg(video, timesteps, conditions, cfg_scale=2.5)
    expected = unconditional + 2.5 * (conditional - unconditional)

    assert torch.allclose(guided[:2], expected, atol=1e-5, rtol=1e-5)


def test_model_rejects_wrong_temporal_extent_instead_of_silently_reshaping():
    model = VideoDiT(tiny_config())

    with pytest.raises(ValueError, match="x must have shape"):
        model(
            torch.randn(1, 3, 5, 32, 32),
            torch.tensor([1]),
            torch.tensor([[0, 0, 0, 0]]),
        )


def test_text_condition_mode_accepts_precomputed_frozen_embeddings():
    config = tiny_config(
        image_size=16,
        patch_size=4,
        num_pairs=1,
        condition_mode="text",
        text_embedding_size=24,
    )
    model = VideoDiT(config)
    video = torch.randn(2, 3, 4, 16, 16)
    timesteps = torch.tensor([2, 7])
    text_embeddings = torch.randn(2, 24)

    output = model(video, timesteps, text_embeddings)

    assert output.shape == video.shape
    assert model.condition_embedder.projection.in_features == 24


def test_text_condition_mode_rejects_discrete_factor_ids():
    model = VideoDiT(
        tiny_config(
            image_size=16,
            patch_size=4,
            num_pairs=1,
            condition_mode="text",
            text_embedding_size=24,
        )
    )

    with pytest.raises(ValueError, match="text embeddings"):
        model(
            torch.randn(1, 3, 4, 16, 16),
            torch.tensor([1]),
            torch.zeros(1, 4, dtype=torch.long),
        )


def test_text_token_mode_cross_attends_to_sequence_and_changes_prediction():
    config = tiny_config(
        image_size=16, patch_size=4, num_pairs=1,
        condition_mode="text_tokens", text_embedding_size=24,
    )
    model = VideoDiT(config)
    with torch.no_grad():
        model.final_layer.linear.weight.normal_()
        model.final_layer.adaLN_modulation[-1].weight.normal_()
        for block in model.blocks:
            block.cross_gate.fill_(1)
    video = torch.randn(1, 3, 4, 16, 16)
    timesteps = torch.tensor([5])

    first_tokens = torch.zeros(1, 7, 24)
    first_tokens[:, 0] = 1
    second_tokens = torch.zeros(1, 7, 24)
    second_tokens[:, :2] = -1
    first = model(video, timesteps, first_tokens)
    second = model(video, timesteps, second_tokens)

    assert first.shape == video.shape
    assert not torch.allclose(first, second)


def test_text_token_padding_is_masked_and_cfg_null_context_stays_finite():
    model = VideoDiT(tiny_config(
        image_size=16, patch_size=4, num_pairs=1,
        condition_mode="text_tokens", text_embedding_size=24,
    ))
    video = torch.randn(2, 3, 4, 16, 16)
    timesteps = torch.tensor([5, 5])
    embeddings = torch.zeros(2, 7, 24)
    embeddings[:, :3] = torch.randn(2, 3, 24)

    output = model(
        video, timesteps, embeddings,
        force_drop_ids=torch.tensor([False, True]),
    )

    assert torch.isfinite(output).all()
