"""Measure whether a generated clip changes smoothly in the requested direction."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import uuid

import torch


@dataclass(frozen=True)
class TemporalMetrics:
    centroid_path: tuple[float, ...]
    displacement: float
    adjacent_mse: float
    endpoint_mse: float
    smoothness_ratio: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def temporal_metrics(video: torch.Tensor) -> TemporalMetrics:
    """Compute simple content/motion diagnostics for ``[C,T,H,W]`` video."""

    if video.ndim != 4 or video.shape[1] < 2:
        raise ValueError("video must have shape [C,T,H,W] with at least two frames")
    value = video.detach().float().cpu()
    temporal_median = value.median(dim=1, keepdim=True).values
    motion_weight = (value - temporal_median).abs().mean(dim=0)
    x_coordinates = torch.arange(value.shape[-1], dtype=torch.float32)
    centers = []
    fallback_center = (value.shape[-1] - 1) / 2
    for frame in motion_weight:
        column_mass = frame.sum(dim=0)
        total_mass = column_mass.sum()
        centers.append(
            fallback_center
            if float(total_mass) <= 1e-12
            else float((column_mass * x_coordinates).sum() / total_mass)
        )
    adjacent = float((value[:, 1:] - value[:, :-1]).square().mean())
    endpoint = float((value[:, -1] - value[:, 0]).square().mean())
    return TemporalMetrics(
        centroid_path=tuple(centers),
        displacement=centers[-1] - centers[0],
        adjacent_mse=adjacent,
        endpoint_mse=endpoint,
        smoothness_ratio=0.0 if endpoint <= 1e-12 else adjacent / endpoint,
    )


def save_metrics(video: torch.Tensor, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(temporal_metrics(video).to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
