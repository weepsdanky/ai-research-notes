import pytest
import torch

from dataset import (
    CanonicalVideoDataset,
    PromptSpec,
    SyntheticVideoDataset,
    parse_prompt,
    render_video,
)


def test_symbolic_prompt_controls_subject_clothing_scene_and_motion():
    prompt = "a person in a red coat walking in the snow to the right"
    spec = parse_prompt(prompt)

    assert spec == PromptSpec(
        subject="person",
        color="red",
        motion="walk_right",
        scene="snow",
    )


def test_renderer_makes_a_real_temporal_video_and_is_seed_reproducible():
    spec = PromptSpec(subject="cat", color="orange", motion="walk_right", scene="grass")
    first = render_video(spec, frames=8, size=32, seed=17)
    second = render_video(spec, frames=8, size=32, seed=17)

    assert first.shape == (3, 8, 32, 32)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)
    assert first.min() >= -1 and first.max() <= 1
    assert not torch.equal(first[:, 0], first[:, -1])


def test_walk_right_has_smooth_positive_foreground_motion():
    spec = PromptSpec(subject="person", color="red", motion="walk_right", scene="studio")
    video = render_video(spec, frames=8, size=32, seed=5)
    # Studio background is nearly static; frame differences isolate the moving subject.
    background = video[:, :, :8, :8].median(dim=2, keepdim=True).values.median(
        dim=3, keepdim=True
    ).values
    foreground = (video - background).abs().mean(dim=0) > 0.12
    x_coordinates = torch.arange(32, dtype=torch.float32)
    centroids = []
    for frame in range(8):
        column_mass = foreground[frame].float().sum(dim=0)
        centroids.append((column_mass * x_coordinates).sum() / column_mass.sum())
    trajectory = torch.stack(centroids)

    assert trajectory[-1] - trajectory[0] > 8
    assert torch.all(trajectory[1:] >= trajectory[:-1] - 1.0)


def test_dataset_returns_condition_ids_and_canonical_prompt():
    dataset = SyntheticVideoDataset(length=4, frames=6, size=32, base_seed=100)
    sample = dataset[2]

    assert sample["video"].shape == (3, 6, 32, 32)
    assert sample["condition"].shape == (4,)
    assert sample["condition"].dtype == torch.long
    assert "prompt" in sample and isinstance(sample["prompt"], str)


def test_unknown_subject_is_rejected_instead_of_silently_becoming_a_cat():
    with pytest.raises(ValueError, match="supported subject"):
        parse_prompt("a spaceship flying over the ocean")


def test_canonical_overfit_curriculum_cycles_three_fixed_prompt_clips():
    dataset = CanonicalVideoDataset(length=7, frames=4, size=32, base_seed=12)

    samples = [dataset[index] for index in range(7)]

    assert [sample["prompt"] for sample in samples[:3]] == [
        "an orange cat walking to the right in the snow",
        "a brown dog walking to the left on grass",
        "a person in a red coat walking to the right in the snow",
    ]
    assert [sample["condition"][0].item() for sample in samples] == [
        0,
        1,
        2,
        0,
        1,
        2,
        0,
    ]
    assert torch.equal(samples[0]["video"], samples[3]["video"])
    assert torch.equal(samples[1]["condition"], samples[4]["condition"])
