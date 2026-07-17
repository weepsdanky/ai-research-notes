"""DDPM equations used to train and sample the teaching DiT."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _extract(values: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
    if t.dtype != torch.long:
        raise TypeError("timesteps must use torch.long")
    return values.gather(0, t).reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


@dataclass
class DiffusionSchedule:
    """Precomputed coefficients for a discrete Gaussian diffusion process."""

    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    def __post_init__(self) -> None:
        if self.timesteps < 2:
            raise ValueError("timesteps must be at least 2")
        if not 0 < self.beta_start < self.beta_end < 1:
            raise ValueError("betas must satisfy 0 < beta_start < beta_end < 1")
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.alpha_bar_prev = torch.cat([torch.ones(1), self.alpha_bar[:-1]])
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.posterior_variance = (
            self.beta * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )
        self.posterior_mean_coef_x0 = (
            self.beta * torch.sqrt(self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )
        self.posterior_mean_coef_xt = (
            (1.0 - self.alpha_bar_prev)
            * torch.sqrt(self.alpha)
            / (1.0 - self.alpha_bar)
        )

    def to(self, device: torch.device | str) -> "DiffusionSchedule":
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        if noise.shape != x0.shape:
            raise ValueError("noise must have the same shape as x0")
        xt = (
            _extract(self.sqrt_alpha_bar, t, x0.shape) * x0
            + _extract(self.sqrt_one_minus_alpha_bar, t, x0.shape) * noise
        )
        return xt, noise

    def predict_x0_from_eps(
        self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        sqrt_alpha_bar = _extract(self.sqrt_alpha_bar, t, xt.shape)
        return (xt - _extract(self.sqrt_one_minus_alpha_bar, t, xt.shape) * eps) / sqrt_alpha_bar

    def q_posterior(
        self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = (
            _extract(self.posterior_mean_coef_x0, t, xt.shape) * x0
            + _extract(self.posterior_mean_coef_xt, t, xt.shape) * xt
        )
        variance = _extract(self.posterior_variance, t, xt.shape)
        return mean, variance


def diffusion_loss(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    x0: torch.Tensor,
    t: torch.Tensor,
    labels: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    xt, target_noise = schedule.q_sample(x0, t, noise)
    predicted_noise = model(xt, t, labels)
    return F.mse_loss(predicted_noise, target_noise)


@torch.no_grad()
def p_sample(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    xt: torch.Tensor,
    t: torch.Tensor,
    labels: torch.Tensor,
    cfg_scale: float = 1.0,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if cfg_scale == 1.0:
        predicted_noise = model(xt, t, labels)
    else:
        if xt.shape[0] % 2:
            raise ValueError("guided sampling requires an even batch size")
        predicted_noise = model.forward_with_cfg(xt, t, labels, cfg_scale)
    predicted_x0 = schedule.predict_x0_from_eps(xt, t, predicted_noise).clamp(-1, 1)
    posterior_mean, posterior_variance = schedule.q_posterior(predicted_x0, xt, t)
    if noise is None:
        noise = torch.randn_like(xt)
    nonzero_mask = (t != 0).float().reshape(t.shape[0], *((1,) * (xt.ndim - 1)))
    return posterior_mean + nonzero_mask * torch.sqrt(posterior_variance) * noise


@torch.no_grad()
def sample_loop(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    shape: tuple[int, int, int, int],
    labels: torch.Tensor,
    device: torch.device | str,
    cfg_scale: float = 1.0,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    x = initial_noise.to(device) if initial_noise is not None else torch.randn(shape, device=device)
    if tuple(x.shape) != shape:
        raise ValueError("initial_noise must match shape")
    try:
        for step in reversed(range(schedule.timesteps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            x = p_sample(model, schedule, x, t, labels, cfg_scale)
    finally:
        model.train(was_training)
    return x

