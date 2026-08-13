"""DDPM equations used to train and sample the teaching DiT."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


def _randn_like(
    value: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    if generator is None:
        return torch.randn_like(value)
    return torch.randn(
        value.shape,
        device=generator.device,
        dtype=value.dtype,
        generator=generator,
    ).to(value.device)


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
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]]).clamp(min=1e-20)
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

    def model_variance(
        self, variance_values: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpolate log variance between the posterior and beta (LEARNED_RANGE)."""

        minimum = _extract(self.posterior_log_variance_clipped, t, variance_values.shape)
        maximum = _extract(torch.log(self.beta), t, variance_values.shape)
        fraction = (variance_values + 1) / 2
        log_variance = fraction * maximum + (1 - fraction) * minimum
        return torch.exp(log_variance), log_variance


def _normal_kl(
    mean1: torch.Tensor,
    logvar1: torch.Tensor,
    mean2: torch.Tensor,
    logvar2: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (
        -1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
        + (mean1 - mean2).square() * torch.exp(-logvar2)
    )


def _approx_standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (
        1.0
        + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3)))
    )


def _discretized_gaussian_log_likelihood(
    x: torch.Tensor, means: torch.Tensor, log_scales: torch.Tensor
) -> torch.Tensor:
    """Log likelihood for images scaled to [-1, 1], as used by improved DDPM."""

    centered = x - means
    inverse_std = torch.exp(-log_scales)
    plus = inverse_std * (centered + 1.0 / 255.0)
    minus = inverse_std * (centered - 1.0 / 255.0)
    cdf_plus = _approx_standard_normal_cdf(plus)
    cdf_minus = _approx_standard_normal_cdf(minus)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_minus = torch.log((1.0 - cdf_minus).clamp(min=1e-12))
    log_cdf_delta = torch.log((cdf_plus - cdf_minus).clamp(min=1e-12))
    return torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_minus, log_cdf_delta),
    )


def diffusion_loss(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    x0: torch.Tensor,
    t: torch.Tensor,
    labels: torch.Tensor,
    noise: torch.Tensor | None = None,
    vb_weight: float | None = None,
    return_metrics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    xt, target_noise = schedule.q_sample(x0, t, noise)
    prediction = model(xt, t, labels)
    if prediction.shape[1] == x0.shape[1]:
        loss = F.mse_loss(prediction, target_noise)
        metrics = {"loss": loss.item(), "mse": loss.item(), "vb": 0.0}
    elif prediction.shape[1] == 2 * x0.shape[1]:
        predicted_noise, variance_values = prediction.chunk(2, dim=1)
        mse = F.mse_loss(predicted_noise, target_noise)
        # Stop the VB term from changing the mean prediction, matching improved DDPM.
        predicted_x0 = schedule.predict_x0_from_eps(xt, t, predicted_noise.detach()).clamp(-1, 1)
        model_mean, _ = schedule.q_posterior(predicted_x0, xt, t)
        _, model_log_variance = schedule.model_variance(variance_values, t)
        true_mean, true_variance = schedule.q_posterior(x0, xt, t)
        true_log_variance = torch.log(true_variance.clamp(min=1e-20))
        kl = _normal_kl(
            true_mean, true_log_variance, model_mean, model_log_variance
        ).flatten(1).mean(1) / math.log(2.0)
        decoder_nll = -_discretized_gaussian_log_likelihood(
            x0, model_mean, 0.5 * model_log_variance
        ).flatten(1).mean(1) / math.log(2.0)
        vb = torch.where(t == 0, decoder_nll, kl).mean()
        # RESCALED_MSE from improved-diffusion / the official DiT training stack.
        weight = schedule.timesteps / 1000.0 if vb_weight is None else vb_weight
        loss = mse + weight * vb
        metrics = {"loss": loss.item(), "mse": mse.item(), "vb": vb.item()}
    else:
        raise ValueError("model must predict C noise channels or 2C noise/variance channels")
    return (loss, metrics) if return_metrics else loss


@torch.no_grad()
def p_sample(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    xt: torch.Tensor,
    t: torch.Tensor,
    labels: torch.Tensor,
    cfg_scale: float = 1.0,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if cfg_scale == 1.0:
        prediction = model(xt, t, labels)
    else:
        if xt.shape[0] % 2:
            raise ValueError("guided sampling requires an even batch size")
        prediction = model.forward_with_cfg(xt, t, labels, cfg_scale)
    if prediction.shape[1] == 2 * xt.shape[1]:
        predicted_noise, variance_values = prediction.chunk(2, dim=1)
        posterior_variance, _ = schedule.model_variance(variance_values, t)
    else:
        predicted_noise = prediction
        posterior_variance = None
    predicted_x0 = schedule.predict_x0_from_eps(xt, t, predicted_noise).clamp(-1, 1)
    posterior_mean, fixed_variance = schedule.q_posterior(predicted_x0, xt, t)
    if posterior_variance is None:
        posterior_variance = fixed_variance
    if noise is None:
        noise = _randn_like(xt, generator)
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
    trajectory_frames: int = 0,
    generator: torch.Generator | None = None,
) -> torch.Tensor | tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
    was_training = model.training
    model.eval()
    if initial_noise is not None:
        x = initial_noise.to(device)
    elif generator is None:
        x = torch.randn(shape, device=device)
    else:
        x = torch.randn(shape, device=generator.device, generator=generator).to(device)
    if tuple(x.shape) != shape:
        raise ValueError("initial_noise must match shape")
    trajectory: list[tuple[int, torch.Tensor]] = []
    capture = set()
    if trajectory_frames:
        if not 2 <= trajectory_frames <= schedule.timesteps + 1:
            raise ValueError(
                "trajectory_frames must be zero or between 2 and timesteps + 1"
            )
        trajectory.append((schedule.timesteps, x.detach().cpu()))
        remaining = trajectory_frames - 1
        capture = {0} if remaining == 1 else {
            round(i * (schedule.timesteps - 1) / (remaining - 1))
            for i in range(remaining)
        }
    try:
        for step in reversed(range(schedule.timesteps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            x = p_sample(model, schedule, x, t, labels, cfg_scale, generator=generator)
            if step in capture:
                trajectory.append((step, x.detach().cpu()))
    finally:
        model.train(was_training)
    if trajectory_frames:
        return x, trajectory
    return x


@torch.no_grad()
def ddim_sample_loop(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    shape: tuple[int, int, int, int],
    labels: torch.Tensor,
    device: torch.device | str,
    cfg_scale: float = 1.0,
    initial_noise: torch.Tensor | None = None,
    sampling_steps: int = 50,
    eta: float = 0.0,
    trajectory_frames: int = 0,
    generator: torch.Generator | None = None,
) -> torch.Tensor | tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
    """Sample on a respaced schedule using DDIM; eta=0 is deterministic."""

    if not 1 <= sampling_steps <= schedule.timesteps:
        raise ValueError("sampling_steps must be between 1 and diffusion timesteps")
    if eta < 0:
        raise ValueError("eta must be non-negative")
    if trajectory_frames and not 2 <= trajectory_frames <= sampling_steps + 1:
        raise ValueError("trajectory_frames must be zero or between 2 and sampling_steps + 1")
    was_training = model.training
    model.eval()
    if initial_noise is not None:
        x = initial_noise.to(device)
    elif generator is None:
        x = torch.randn(shape, device=device)
    else:
        x = torch.randn(shape, device=generator.device, generator=generator).to(device)
    if tuple(x.shape) != shape:
        raise ValueError("initial_noise must match shape")
    steps = torch.linspace(
        schedule.timesteps - 1, 0, sampling_steps, dtype=torch.float64
    ).round().long().tolist()
    trajectory: list[tuple[int, torch.Tensor]] = []
    capture_indices = set()
    if trajectory_frames:
        trajectory.append((schedule.timesteps, x.detach().cpu()))
        remaining = trajectory_frames - 1
        capture_indices = {sampling_steps - 1} if remaining == 1 else {
            round(i * (sampling_steps - 1) / (remaining - 1))
            for i in range(remaining)
        }
    try:
        for index, step in enumerate(steps):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            if cfg_scale == 1.0:
                prediction = model(x, t, labels)
            else:
                prediction = model.forward_with_cfg(x, t, labels, cfg_scale)
            predicted_noise = prediction[:, : x.shape[1]]
            alpha_bar = _extract(schedule.alpha_bar, t, x.shape)
            predicted_x0 = (
                (x - torch.sqrt(1 - alpha_bar) * predicted_noise)
                / torch.sqrt(alpha_bar)
            ).clamp(-1, 1)
            if index + 1 < len(steps):
                previous_t = torch.full(
                    (shape[0],), steps[index + 1], device=device, dtype=torch.long
                )
                alpha_bar_previous = _extract(schedule.alpha_bar, previous_t, x.shape)
            else:
                alpha_bar_previous = torch.ones_like(alpha_bar)
            sigma = eta * torch.sqrt(
                ((1 - alpha_bar_previous) / (1 - alpha_bar))
                * (1 - alpha_bar / alpha_bar_previous)
            ).clamp(min=0)
            direction = torch.sqrt(
                (1 - alpha_bar_previous - sigma.square()).clamp(min=0)
            ) * predicted_noise
            noise = torch.zeros_like(x) if eta == 0 else _randn_like(x, generator)
            x = torch.sqrt(alpha_bar_previous) * predicted_x0 + direction + sigma * noise
            if index in capture_indices:
                trajectory.append((step, x.detach().cpu()))
    finally:
        model.train(was_training)
    if trajectory_frames:
        return x, trajectory
    return x
