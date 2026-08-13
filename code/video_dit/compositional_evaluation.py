"""Aggregate factor correctness, prompt controls, and memorization evidence."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

import torch


@dataclass(frozen=True)
class FactorPrediction:
    sample_id: str
    split: str
    expected_subject: str
    expected_action: str
    expected_scene: str
    predicted_subject: str
    predicted_action: str
    predicted_scene: str
    nearest_train_similarity: float


def _metrics(rows: list[FactorPrediction]) -> dict[str, object]:
    if not rows:
        return {
            "status": "missing",
            "samples": 0,
            "subject_accuracy": None,
            "action_accuracy": None,
            "scene_accuracy": None,
            "triple_accuracy": None,
            "nearest_train_similarity_mean": None,
            "nearest_train_similarity_max": None,
        }
    subject = [row.expected_subject == row.predicted_subject for row in rows]
    action = [row.expected_action == row.predicted_action for row in rows]
    scene = [row.expected_scene == row.predicted_scene for row in rows]
    triple = [all(values) for values in zip(subject, action, scene, strict=True)]
    neighbors = [row.nearest_train_similarity for row in rows]
    return {
        "status": "measured",
        "samples": len(rows),
        "subject_accuracy": mean(subject),
        "action_accuracy": mean(action),
        "scene_accuracy": mean(scene),
        "triple_accuracy": mean(triple),
        "nearest_train_similarity_mean": mean(neighbors),
        "nearest_train_similarity_max": max(neighbors),
    }


def evaluate_compositional_generations(
    rows: list[FactorPrediction],
    *,
    shuffled: list[FactorPrediction] | None = None,
) -> dict[str, dict[str, object]]:
    """Never collapse ID/C1/C2 into one flattering aggregate number."""

    result: dict[str, dict[str, object]] = {}
    for split in ("iid", "c1", "c2"):
        metrics = _metrics([row for row in rows if row.split == split])
        if shuffled is not None:
            control = _metrics([row for row in shuffled if row.split == split])
            actual = metrics["triple_accuracy"]
            baseline = control["triple_accuracy"]
            gain = (
                float(actual) - float(baseline)
                if actual is not None and baseline is not None
                else None
            )
            metrics["shuffled_triple_accuracy"] = baseline
            metrics["triple_accuracy_gain_over_shuffled"] = gain
            metrics["passes_prompt_control"] = gain is not None and gain > 0
        result[split] = metrics
    return result


def _normalized_rows(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim != 2 or value.shape[0] == 0:
        raise ValueError(f"{name} must have shape [samples, dimensions]")
    value = value.detach().float().cpu()
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    norms = value.norm(dim=-1, keepdim=True)
    if torch.any(norms <= 1e-12):
        raise ValueError(f"{name} must not contain zero vectors")
    return value / norms


def prompt_retrieval_report(
    video_embeddings: torch.Tensor,
    prompt_embeddings: torch.Tensor,
    *,
    prompt_ids: list[str],
    train_embeddings: torch.Tensor | None = None,
    train_ids: list[str] | None = None,
) -> dict[str, object]:
    """Score paired video/prompt retrieval and optional training-set neighbors.

    Inputs are deliberately encoder-agnostic so the metric can be tested without
    downloading a model.  Callers should record the frozen encoder identity next
    to this report; the scores are diagnostics, not proof of visual correctness.
    """

    videos = _normalized_rows(video_embeddings, "video_embeddings")
    prompts = _normalized_rows(prompt_embeddings, "prompt_embeddings")
    if videos.shape != prompts.shape:
        raise ValueError("video and prompt embeddings must have the same shape")
    if videos.shape[0] < 2:
        raise ValueError("prompt retrieval requires at least two paired samples")
    if len(prompt_ids) != videos.shape[0] or len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("prompt_ids must be unique and match the sample count")

    similarities = videos @ prompts.T
    expected = torch.arange(videos.shape[0])
    correct = similarities.diagonal()
    alternatives = similarities.masked_fill(
        torch.eye(videos.shape[0], dtype=torch.bool), float("-inf")
    ).max(dim=1).values
    top1 = similarities.argmax(dim=1)
    # A deterministic cyclic mismatch is an explicit shuffled-prompt control.
    # Evaluate it against the same similarity matrix so noise and video content
    # are held fixed; only the claimed video/prompt pairing changes.
    shuffled_expected = expected.roll(1)

    nearest_ids: list[str] | None = None
    nearest_values: list[float] = []
    if train_embeddings is not None:
        if train_ids is None:
            raise ValueError("train_ids are required with train_embeddings")
        train = _normalized_rows(train_embeddings, "train_embeddings")
        if train.shape[1] != videos.shape[1] or len(train_ids) != train.shape[0]:
            raise ValueError("training embeddings and ids do not match video embeddings")
        values, indices = (videos @ train.T).max(dim=1)
        nearest_values = [float(value) for value in values]
        nearest_ids = [train_ids[int(index)] for index in indices]

    return {
        "samples": videos.shape[0],
        "prompt_ids": prompt_ids,
        "prompt_top1_accuracy": float((top1 == expected).float().mean()),
        "shuffled_prompt_top1_accuracy": float(
            (top1 == shuffled_expected).float().mean()
        ),
        "prompt_top1_gain_over_shuffled": float(
            (top1 == expected).float().mean()
            - (top1 == shuffled_expected).float().mean()
        ),
        "prompt_margin_mean": float((correct - alternatives).mean()),
        "predicted_prompt_ids": [prompt_ids[int(index)] for index in top1],
        "prompt_margins": [float(value) for value in correct - alternatives],
        "paired_prompt_similarities": [float(value) for value in correct],
        "nearest_train_ids": nearest_ids,
        "nearest_train_similarities": nearest_values or None,
        "nearest_train_similarity_mean": (
            mean(nearest_values) if nearest_values else None
        ),
        "nearest_train_similarity_max": max(nearest_values) if nearest_values else None,
    }
