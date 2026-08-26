"""Conditional Flow Matching for the existing five-dimensional Video DiT.

Convention used throughout this module:

* ``t=0`` is data and ``t=1`` is standard Gaussian noise;
* ``z_t = (1-t) * z_0 + t * epsilon``;
* the training target is the forward-path velocity ``epsilon - z_0``;
* generation solves the same ODE backwards, from ``t=1`` to ``t=0``.

There is deliberately no DDPM posterior, beta schedule, or stochastic reverse
step in this file.  Euler and Heun integrate a deterministic probability-flow
ODE, and their negative ``dt`` is what reverses the forward probability path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


GENERATIVE_PROCESS = "linear_conditional_flow_matching"
TIME_CONVENTION = "data_at_0_noise_at_1"
PREDICTION_TYPE = "flow_velocity"
INFERENCE_TIME_GRID = "shifted_linear_1_to_0"


def shift_times(times: torch.Tensor, shift: float) -> torch.Tensor:
    """Apply a monotone endpoint-preserving rational time transform."""

    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("time shift must be positive and finite")
    if torch.any((times < 0) | (times > 1)):
        raise ValueError("flow times must lie in [0, 1]")
    return shift * times / (1.0 + (shift - 1.0) * times)


def _broadcast_time(times: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if times.ndim != 1 or times.shape[0] != value.shape[0]:
        raise ValueError("times must have shape [batch]")
    if not torch.is_floating_point(times):
        raise TypeError("flow times must be floating point")
    return times.reshape(times.shape[0], *((1,) * (value.ndim - 1)))


@dataclass(frozen=True)
class FlowMatchingSchedule:
    """Linear conditional path plus explicit time-sampling conventions.

    ``time_shift > 1`` maps sampled times toward the noisy endpoint.  This is a
    training-distribution choice, not an ODE solver step-size schedule.
    ``model_time_scale`` preserves the current DiT time-embedding convention.
    """

    model_time_scale: float = 1_000.0
    time_sampling: str = "uniform"
    time_shift: float = 1.0
    logit_mean: float = 0.0
    logit_std: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.model_time_scale) or self.model_time_scale <= 0:
            raise ValueError("model_time_scale must be positive and finite")
        if self.time_sampling not in {"uniform", "logit_normal"}:
            raise ValueError("time_sampling must be 'uniform' or 'logit_normal'")
        if not math.isfinite(self.time_shift) or self.time_shift <= 0:
            raise ValueError("time_shift must be positive and finite")
        if not math.isfinite(self.logit_mean):
            raise ValueError("logit_mean must be finite")
        if not math.isfinite(self.logit_std) or self.logit_std <= 0:
            raise ValueError("logit_std must be positive and finite")

    def model_times(self, times: torch.Tensor) -> torch.Tensor:
        """Map normalized flow time to the existing DiT embedding scale."""

        return times * self.model_time_scale

    def shift_times(self, times: torch.Tensor) -> torch.Tensor:
        """Apply ``s*t / (1 + (s-1)*t)`` while preserving both endpoints."""

        return shift_times(times, self.time_shift)

    def sample_times(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.time_sampling == "uniform":
            raw = torch.rand(batch_size, device=device, generator=generator)
        else:
            logits = torch.randn(batch_size, device=device, generator=generator)
            raw = torch.sigmoid(self.logit_mean + self.logit_std * logits)
        return self.shift_times(raw)

    def interpolate(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        if clean.ndim != 5:
            raise ValueError("clean video must have shape [B,C,T,H,W]")
        if noise.shape != clean.shape:
            raise ValueError("noise must have the same shape as clean")
        if torch.any((times < 0) | (times > 1)):
            raise ValueError("flow times must lie in [0, 1]")
        time = _broadcast_time(times, clean)
        return (1.0 - time) * clean + time * noise

    @staticmethod
    def velocity_target(clean: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if noise.shape != clean.shape:
            raise ValueError("noise must have the same shape as clean")
        return noise - clean

    @staticmethod
    def clean_and_noise_from_velocity(
        state: torch.Tensor,
        velocity: torch.Tensor,
        times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Invert the linear path without the endpoint singularities of epsilon-only conversion."""

        if velocity.shape != state.shape:
            raise ValueError("velocity must have the same shape as state")
        time = _broadcast_time(times, state)
        clean = state - time * velocity
        noise = state + (1.0 - time) * velocity
        return clean, noise


def _model_velocity(
    model: torch.nn.Module,
    state: torch.Tensor,
    model_times: torch.Tensor,
    condition_ids: torch.Tensor,
    cfg_scale: float,
) -> torch.Tensor:
    if cfg_scale == 1.0:
        prediction = model(state, model_times, condition_ids)
    else:
        if state.shape[0] % 2:
            raise ValueError("guided sampling requires an even batch size")
        prediction = model.forward_with_cfg(
            state, model_times, condition_ids, cfg_scale
        )
    if prediction.shape != state.shape:
        raise ValueError(
            "Flow Matching requires exactly C velocity channels; disable learned sigma"
        )
    return prediction


def flow_matching_loss(
    model: torch.nn.Module,
    schedule: FlowMatchingSchedule,
    clean: torch.Tensor,
    times: torch.Tensor,
    condition_ids: torch.Tensor,
    noise: torch.Tensor | None = None,
    *,
    return_metrics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """Conditional Flow Matching MSE on the linear-path velocity field."""

    if noise is None:
        noise = torch.randn_like(clean)
    state = schedule.interpolate(clean, noise, times)
    target = schedule.velocity_target(clean, noise)
    prediction = model(state, schedule.model_times(times), condition_ids)
    if prediction.shape != clean.shape:
        raise ValueError(
            "Flow Matching requires exactly C velocity channels; disable learned sigma"
        )
    loss = F.mse_loss(prediction, target)
    metrics = {
        "loss": loss.item(),
        "mse": loss.item(),
        "time_mean": times.mean().item(),
        "time_std": times.std(unbiased=False).item(),
    }
    return (loss, metrics) if return_metrics else loss


@torch.no_grad()
def flow_sample_loop(
    model: torch.nn.Module,
    schedule: FlowMatchingSchedule,
    shape: tuple[int, ...],
    condition_ids: torch.Tensor,
    device: torch.device | str,
    *,
    sampling_steps: int = 20,
    solver: str = "heun",
    cfg_scale: float = 1.0,
    initial_noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    trajectory_frames: int = 0,
    sampling_time_shift: float = 1.0,
) -> tuple[torch.Tensor, list[tuple[float, torch.Tensor]]]:
    """Integrate the learned velocity field from noise ``t=1`` to data ``t=0``."""

    if len(shape) != 5:
        raise ValueError("shape must be [B,C,T,H,W]")
    if sampling_steps <= 0:
        raise ValueError("sampling_steps must be positive")
    if solver not in {"euler", "heun"}:
        raise ValueError("solver must be 'euler' or 'heun'")
    if trajectory_frames < 0:
        raise ValueError("trajectory_frames must be non-negative")
    if condition_ids.shape[0] != shape[0]:
        raise ValueError("condition batch must match sample shape")

    state = (
        torch.randn(shape, device=device, generator=generator)
        if initial_noise is None
        else initial_noise.to(device)
    )
    if tuple(state.shape) != shape:
        raise ValueError("initial_noise must match shape")
    grid = shift_times(
        torch.linspace(1.0, 0.0, sampling_steps + 1, device=device),
        sampling_time_shift,
    )
    capture = set()
    if trajectory_frames:
        capture = set(
            torch.linspace(0, sampling_steps, min(trajectory_frames, sampling_steps + 1))
            .round()
            .long()
            .tolist()
        )
    trajectory: list[tuple[float, torch.Tensor]] = []
    was_training = model.training
    model.eval()
    try:
        if 0 in capture:
            trajectory.append((1.0, state.detach().cpu()))
        for index in range(sampling_steps):
            current = grid[index]
            following = grid[index + 1]
            dt = following - current  # negative: generation runs noise -> data
            times = torch.full(
                (shape[0],), current.item(), device=device, dtype=state.dtype
            )
            velocity = _model_velocity(
                model,
                state,
                schedule.model_times(times),
                condition_ids,
                cfg_scale,
            )
            proposal = state + dt * velocity
            if solver == "heun":
                next_times = torch.full(
                    (shape[0],), following.item(), device=device, dtype=state.dtype
                )
                next_velocity = _model_velocity(
                    model,
                    proposal,
                    schedule.model_times(next_times),
                    condition_ids,
                    cfg_scale,
                )
                state = state + dt * 0.5 * (velocity + next_velocity)
            else:
                state = proposal
            if index + 1 in capture:
                trajectory.append((following.item(), state.detach().cpu()))
    finally:
        model.train(was_training)
    return state, trajectory
