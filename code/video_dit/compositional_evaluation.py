"""Aggregate factor correctness, prompt controls, and memorization evidence."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


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
