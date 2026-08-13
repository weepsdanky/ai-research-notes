import json

import torch

from evaluate_video import save_metrics, temporal_metrics


def _moving_square(direction: int = 1) -> torch.Tensor:
    video = torch.zeros(3, 8, 16, 16)
    for frame in range(8):
        left = 1 + (frame if direction > 0 else 7 - frame)
        video[:, frame, 5:9, left : left + 4] = 1
    return video


def test_temporal_metrics_detect_direction_and_smoothness():
    right = temporal_metrics(_moving_square(1))
    left = temporal_metrics(_moving_square(-1))

    assert right.displacement > 5
    assert left.displacement < -5
    assert 0 < right.adjacent_mse < right.endpoint_mse
    assert right.smoothness_ratio < 0.5


def test_temporal_metrics_returns_finite_center_for_a_static_clip():
    metrics = temporal_metrics(torch.zeros(3, 8, 16, 16))

    assert metrics.centroid_path == (7.5,) * 8
    assert metrics.displacement == 0
    assert metrics.adjacent_mse == 0
    assert metrics.endpoint_mse == 0
    assert metrics.smoothness_ratio == 0


def test_metric_json_is_complete_and_atomically_replaced(tmp_path):
    destination = tmp_path / "metrics.json"
    save_metrics(_moving_square(1), destination)

    assert json.loads(destination.read_text())["displacement"] > 5
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]
