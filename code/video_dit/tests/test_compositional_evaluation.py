import torch

from compositional_evaluation import (
    FactorPrediction,
    evaluate_compositional_generations,
    prompt_retrieval_report,
)


def test_report_separates_id_c1_c2_and_scores_all_three_factors():
    rows = [
        FactorPrediction("id-1", "iid", "cat", "running", "grass", "cat", "running", "grass", 0.8),
        FactorPrediction("c1-1", "c1", "cat", "running", "snow", "cat", "running", "snow", 0.7),
        FactorPrediction("c1-2", "c1", "dog", "walking", "grass", "dog", "running", "grass", 0.6),
        FactorPrediction("c2-1", "c2", "dog", "walking", "snow", "cat", "walking", "snow", 0.5),
    ]

    report = evaluate_compositional_generations(rows)

    assert report["iid"]["triple_accuracy"] == 1.0
    assert report["c1"]["triple_accuracy"] == 0.5
    assert report["c1"]["subject_accuracy"] == 1.0
    assert report["c1"]["action_accuracy"] == 0.5
    assert report["c2"]["scene_accuracy"] == 1.0
    assert report["c2"]["nearest_train_similarity_max"] == 0.5


def test_prompt_shuffle_control_must_be_worse_than_real_prompts_to_pass():
    aligned = [
        FactorPrediction("a", "c1", "cat", "running", "snow", "cat", "running", "snow", 0.4),
        FactorPrediction("b", "c1", "dog", "walking", "grass", "dog", "walking", "grass", 0.4),
    ]
    shuffled = [
        FactorPrediction("a", "c1", "cat", "running", "snow", "dog", "walking", "grass", 0.4),
        FactorPrediction("b", "c1", "dog", "walking", "grass", "cat", "running", "snow", 0.4),
    ]

    report = evaluate_compositional_generations(aligned, shuffled=shuffled)

    assert report["c1"]["triple_accuracy_gain_over_shuffled"] == 1.0
    assert report["c1"]["passes_prompt_control"] is True


def test_empty_split_is_reported_as_missing_not_as_zero_accuracy():
    report = evaluate_compositional_generations([])

    assert report["c1"]["status"] == "missing"
    assert report["c1"]["triple_accuracy"] is None


def test_prompt_retrieval_scores_alignment_margin_and_train_neighbor():
    video = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    prompts = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    train = torch.tensor([[0.8, 0.2], [0.1, 0.9]])

    report = prompt_retrieval_report(
        video, prompts, prompt_ids=["cat", "dog"],
        train_embeddings=train, train_ids=["train-cat", "train-dog"],
    )

    assert report["prompt_top1_accuracy"] == 1.0
    assert report["shuffled_prompt_top1_accuracy"] == 0.0
    assert report["prompt_top1_gain_over_shuffled"] == 1.0
    assert report["prompt_margin_mean"] == 1.0
    assert report["predicted_prompt_ids"] == ["cat", "dog"]
    assert report["prompt_margins"] == [1.0, 1.0]
    assert report["nearest_train_ids"] == ["train-cat", "train-dog"]
    assert 0.9 < report["nearest_train_similarity_max"] < 1.0


def test_prompt_retrieval_exposes_a_shuffled_or_ignored_condition():
    video = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    prompts = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    report = prompt_retrieval_report(video, prompts, prompt_ids=["cat", "dog"])

    assert report["prompt_top1_accuracy"] == 0.0
    assert report["shuffled_prompt_top1_accuracy"] == 1.0
    assert report["prompt_margin_mean"] == -1.0
    assert report["predicted_prompt_ids"] == ["dog", "cat"]
    assert report["nearest_train_similarity_max"] is None


def test_prompt_retrieval_rejects_single_prompt_as_uninformative():
    with __import__("pytest").raises(ValueError, match="at least two"):
        prompt_retrieval_report(
            torch.ones(1, 3), torch.ones(1, 3), prompt_ids=["cat"],
        )
