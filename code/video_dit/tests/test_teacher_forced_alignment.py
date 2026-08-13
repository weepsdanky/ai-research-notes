import torch

from teacher_forced_alignment import teacher_forced_retrieval


class ConditionOracle(torch.nn.Module):
    def forward(self, noisy, timesteps, condition):
        values = condition[:, 0].reshape(-1, 1, 1, 1, 1)
        return values.expand_as(noisy)


class InvalidPrediction(torch.nn.Module):
    def forward(self, noisy, timesteps, condition):
        return torch.full_like(noisy, float("nan"))


def test_teacher_forced_retrieval_finds_each_clean_video_condition():
    clean = torch.zeros(3, 1, 2, 2, 2)
    conditions = torch.tensor([[0.0], [1.0], [2.0]])
    noises = torch.stack([torch.zeros_like(clean[0]), torch.ones_like(clean[0]),
                          torch.full_like(clean[0], 2.0)])

    report = teacher_forced_retrieval(
        ConditionOracle(), clean, conditions, ["zero", "one", "two"],
        noisy_batches=[noises], timesteps=[torch.ones(3, dtype=torch.long)],
        target_batches=[noises], channels=1,
    )

    assert report["prompt_top1_accuracy"] == 1.0
    assert report["predicted_prompt_ids"] == ["zero", "one", "two"]
    assert report["error_margin_mean"] > 0


def test_teacher_forced_retrieval_rejects_non_unique_prompts():
    with __import__("pytest").raises(ValueError, match="unique"):
        teacher_forced_retrieval(
            ConditionOracle(), torch.zeros(2, 1, 2, 2, 2), torch.zeros(2, 1),
            ["same", "same"], noisy_batches=[torch.zeros(2, 1, 2, 2, 2)],
            timesteps=[torch.ones(2, dtype=torch.long)],
            target_batches=[torch.zeros(2, 1, 2, 2, 2)], channels=1,
        )


def test_teacher_forced_retrieval_groups_repeated_sample_prompts():
    clean = torch.zeros(3, 1, 2, 2, 2)
    conditions = torch.tensor([[0.0], [1.0]])
    noises = torch.stack([
        torch.zeros_like(clean[0]), torch.zeros_like(clean[0]),
        torch.ones_like(clean[0]),
    ])

    report = teacher_forced_retrieval(
        ConditionOracle(), clean, conditions, ["cat", "dog"],
        sample_prompt_ids=["cat", "cat", "dog"],
        noisy_batches=[noises], timesteps=[torch.ones(3, dtype=torch.long)],
        target_batches=[noises], channels=1,
    )

    assert report["samples"] == 3
    assert report["conditions"] == 2
    assert report["prompt_top1_accuracy"] == 1.0


def test_condition_batching_is_numerically_equivalent():
    clean = torch.zeros(3, 1, 2, 2, 2)
    conditions = torch.tensor([[0.0], [1.0], [2.0]])
    noise = torch.stack([torch.zeros_like(clean[0]), torch.ones_like(clean[0]),
                         torch.full_like(clean[0], 2.0)])
    kwargs = dict(
        model=ConditionOracle(), clean=clean, conditions=conditions,
        prompt_ids=["zero", "one", "two"], noisy_batches=[noise],
        timesteps=[torch.ones(3, dtype=torch.long)], target_batches=[noise],
        channels=1,
    )

    serial = teacher_forced_retrieval(**kwargs, condition_batch_size=1)
    batched = teacher_forced_retrieval(**kwargs, condition_batch_size=2)

    assert serial["mse_matrix"] == batched["mse_matrix"]


def test_teacher_forced_retrieval_rejects_non_finite_predictions():
    with __import__("pytest").raises(ValueError, match="finite"):
        teacher_forced_retrieval(
            InvalidPrediction(), torch.zeros(2, 1, 2, 2, 2),
            torch.tensor([[0.0], [1.0]]), ["a", "b"],
            noisy_batches=[torch.zeros(2, 1, 2, 2, 2)],
            timesteps=[torch.ones(2, dtype=torch.long)],
            target_batches=[torch.zeros(2, 1, 2, 2, 2)], channels=1,
        )


def test_teacher_forced_retrieval_counts_tied_minimum_as_ambiguous():
    report = teacher_forced_retrieval(
        ConditionOracle(), torch.zeros(2, 1, 2, 2, 2),
        torch.tensor([[0.0], [0.0]]), ["a", "b"],
        noisy_batches=[torch.zeros(2, 1, 2, 2, 2)],
        timesteps=[torch.ones(2, dtype=torch.long)],
        target_batches=[torch.zeros(2, 1, 2, 2, 2)], channels=1,
    )

    assert report["prompt_top1_accuracy"] == 0.0
    assert report["ambiguous_tie_rate"] == 1.0
