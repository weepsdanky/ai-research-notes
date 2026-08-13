"""DDPM training equations and DDIM sampling for five-dimensional videos."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


def _extract(
    values: torch.Tensor, timesteps: torch.Tensor, shape: tuple[int, ...]
) -> torch.Tensor:
    """Gather one coefficient per sample and broadcast over C,T,H,W."""

    if timesteps.dtype != torch.long:
        raise TypeError("timesteps must use torch.long")
    if timesteps.ndim != 1 or timesteps.shape[0] != shape[0]:
        raise ValueError("timesteps must have shape [batch]")
    return values.gather(0, timesteps).reshape(
        timesteps.shape[0], *((1,) * (len(shape) - 1))
    )


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


@dataclass
class DiffusionSchedule:
    """Linear-beta Gaussian diffusion coefficients used by the Latte baseline."""

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
        self.alpha_bar_previous = torch.cat(
            [torch.ones(1), self.alpha_bar[:-1]]
        )
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.posterior_variance = (
            self.beta
            * (1.0 - self.alpha_bar_previous)
            / (1.0 - self.alpha_bar)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat(
                [self.posterior_variance[1:2], self.posterior_variance[1:]]
            ).clamp(min=1e-20)
        )
        self.posterior_mean_coef_x0 = (
            self.beta
            * torch.sqrt(self.alpha_bar_previous)
            / (1.0 - self.alpha_bar)
        )
        self.posterior_mean_coef_xt = (
            (1.0 - self.alpha_bar_previous)
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
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if clean.ndim != 5:
            raise ValueError("clean video must have shape [B,C,T,H,W]")
        if noise is None:
            noise = torch.randn_like(clean)
        if noise.shape != clean.shape:
            raise ValueError("noise must have the same shape as clean")
        noisy = (
            _extract(self.sqrt_alpha_bar, timesteps, clean.shape) * clean
            + _extract(self.sqrt_one_minus_alpha_bar, timesteps, clean.shape)
            * noise
        )
        return noisy, noise

    def predict_x0_from_epsilon(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        return (
            noisy
            - _extract(self.sqrt_one_minus_alpha_bar, timesteps, noisy.shape)
            * epsilon
        ) / _extract(self.sqrt_alpha_bar, timesteps, noisy.shape)

    def velocity_target(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Return the v-prediction target used by modern diffusion training."""

        alpha = _extract(self.sqrt_alpha_bar, timesteps, clean.shape)
        sigma = _extract(self.sqrt_one_minus_alpha_bar, timesteps, clean.shape)
        return alpha * noise - sigma * clean

    def clean_and_epsilon_from_velocity(
        self,
        noisy: torch.Tensor,
        velocity: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Invert ``x_t = alpha*x0 + sigma*eps`` and ``v = alpha*eps-sigma*x0``."""

        alpha = _extract(self.sqrt_alpha_bar, timesteps, noisy.shape)
        sigma = _extract(self.sqrt_one_minus_alpha_bar, timesteps, noisy.shape)
        clean = alpha * noisy - sigma * velocity
        epsilon = sigma * noisy + alpha * velocity
        return clean, epsilon

    def q_posterior(
        self,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = (
            _extract(self.posterior_mean_coef_x0, timesteps, noisy.shape) * clean
            + _extract(self.posterior_mean_coef_xt, timesteps, noisy.shape) * noisy
        )
        variance = _extract(self.posterior_variance, timesteps, noisy.shape)
        return mean, variance

    def model_variance(
        self, variance_values: torch.Tensor, timesteps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Improved-DDPM LEARNED_RANGE interpolation used by official DiT."""

        minimum = _extract(
            self.posterior_log_variance_clipped,
            timesteps,
            variance_values.shape,
        )
        maximum = _extract(torch.log(self.beta), timesteps, variance_values.shape)
        fraction = (variance_values + 1) / 2
        log_variance = fraction * maximum + (1 - fraction) * minimum
        return torch.exp(log_variance), log_variance


def _normal_kl(
    mean1: torch.Tensor,
    log_variance1: torch.Tensor,
    mean2: torch.Tensor,
    log_variance2: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (
        -1
        + log_variance2
        - log_variance1
        + torch.exp(log_variance1 - log_variance2)
        + (mean1 - mean2).square() * torch.exp(-log_variance2)
    )


def _standard_normal_cdf(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * (
        1
        + torch.tanh(
            math.sqrt(2.0 / math.pi)
            * (value + 0.044715 * value.pow(3))
        )
    )


def _discretized_gaussian_log_likelihood(
    value: torch.Tensor,
    means: torch.Tensor,
    log_scales: torch.Tensor,
) -> torch.Tensor:
    centered = value - means
    inverse_std = torch.exp(-log_scales)
    cdf_plus = _standard_normal_cdf(
        inverse_std * (centered + 1.0 / 255.0)
    )
    cdf_minus = _standard_normal_cdf(
        inverse_std * (centered - 1.0 / 255.0)
    )
    return torch.where(
        value < -0.999,
        torch.log(cdf_plus.clamp(min=1e-12)),
        torch.where(
            value > 0.999,
            torch.log((1 - cdf_minus).clamp(min=1e-12)),
            torch.log((cdf_plus - cdf_minus).clamp(min=1e-12)),
        ),
    )


def diffusion_loss(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    clean: torch.Tensor,
    timesteps: torch.Tensor,
    condition_ids: torch.Tensor,
    noise: torch.Tensor | None = None,
    vb_weight: float | None = None,
    prediction_type: str = "epsilon",
    return_metrics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """Epsilon/v MSE plus optional detached-mean learned-variance VLB."""

    if prediction_type not in {"epsilon", "velocity"}:
        raise ValueError("prediction_type must be 'epsilon' or 'velocity'")

    noisy, target_noise = schedule.q_sample(clean, timesteps, noise)
    prediction = model(noisy, timesteps, condition_ids)
    channels = clean.shape[1]
    target = (
        target_noise
        if prediction_type == "epsilon"
        else schedule.velocity_target(clean, target_noise, timesteps)
    )
    if prediction.shape == clean.shape:
        loss = F.mse_loss(prediction, target)
        metrics = {"loss": loss.item(), "mse": loss.item(), "vb": 0.0}
    elif prediction.shape == (
        clean.shape[0],
        2 * channels,
        *clean.shape[2:],
    ):
        predicted_mean, variance_values = prediction.chunk(2, dim=1)
        mse = F.mse_loss(predicted_mean, target)
        if prediction_type == "velocity":
            predicted_clean, predicted_noise = schedule.clean_and_epsilon_from_velocity(
                noisy, predicted_mean.detach(), timesteps
            )
        else:
            predicted_noise = predicted_mean
            predicted_clean = schedule.predict_x0_from_epsilon(
                noisy, timesteps, predicted_noise.detach()
            )
        # Detaching epsilon here is the RESCALED_MSE trick: VLB trains variance only.
        predicted_clean = predicted_clean.clamp(-1, 1)
        model_mean, _ = schedule.q_posterior(
            predicted_clean, noisy, timesteps
        )
        _, model_log_variance = schedule.model_variance(
            variance_values, timesteps
        )
        true_mean, true_variance = schedule.q_posterior(
            clean, noisy, timesteps
        )
        true_log_variance = torch.log(true_variance.clamp(min=1e-20))
        kl = _normal_kl(
            true_mean,
            true_log_variance,
            model_mean,
            model_log_variance,
        ).flatten(1).mean(1) / math.log(2.0)
        decoder_nll = -_discretized_gaussian_log_likelihood(
            clean, model_mean, 0.5 * model_log_variance
        ).flatten(1).mean(1) / math.log(2.0)
        vb = torch.where(timesteps == 0, decoder_nll, kl).mean()
        weight = schedule.timesteps / 1000.0 if vb_weight is None else vb_weight
        loss = mse + weight * vb
        metrics = {"loss": loss.item(), "mse": mse.item(), "vb": vb.item()}
    else:
        raise ValueError("model must predict C epsilon or 2C epsilon/variance channels")
    return (loss, metrics) if return_metrics else loss


def _model_prediction(
    model: torch.nn.Module,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    condition_ids: torch.Tensor,
    cfg_scale: float,
) -> torch.Tensor:
    if cfg_scale == 1.0:
        return model(noisy, timesteps, condition_ids)
    if noisy.shape[0] % 2:
        raise ValueError("guided sampling requires an even batch size")
    return model.forward_with_cfg(
        noisy, timesteps, condition_ids, cfg_scale
    )


@torch.no_grad()
def p_sample(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    condition_ids: torch.Tensor,
    cfg_scale: float = 1.0,
    generator: torch.Generator | None = None,
    prediction_type: str = "epsilon",
    clip_denoised: bool = True,
) -> torch.Tensor:
    prediction = _model_prediction(
        model, noisy, timesteps, condition_ids, cfg_scale
    )
    channels = noisy.shape[1]
    if prediction.shape[1] == 2 * channels:
        predicted_mean, variance_values = prediction.chunk(2, dim=1)
        posterior_variance, _ = schedule.model_variance(
            variance_values, timesteps
        )
    else:
        predicted_mean = prediction
        posterior_variance = None
    if prediction_type == "velocity":
        predicted_clean, _ = schedule.clean_and_epsilon_from_velocity(
            noisy, predicted_mean, timesteps
        )
    elif prediction_type == "epsilon":
        predicted_clean = schedule.predict_x0_from_epsilon(
            noisy, timesteps, predicted_mean
        )
    else:
        raise ValueError("prediction_type must be 'epsilon' or 'velocity'")
    if clip_denoised:
        predicted_clean = predicted_clean.clamp(-1, 1)
    posterior_mean, fixed_variance = schedule.q_posterior(
        predicted_clean, noisy, timesteps
    )
    variance = fixed_variance if posterior_variance is None else posterior_variance
    nonzero = (timesteps != 0).float().reshape(
        timesteps.shape[0], *((1,) * (noisy.ndim - 1))
    )
    return posterior_mean + nonzero * torch.sqrt(variance) * _randn_like(
        noisy, generator
    )


@torch.no_grad()
def sample_loop(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    shape: tuple[int, int, int, int, int],
    condition_ids: torch.Tensor,
    device: torch.device | str,
    cfg_scale: float = 1.0,
    initial_noise: torch.Tensor | None = None,
    trajectory_frames: int = 0,
    generator: torch.Generator | None = None,
    prediction_type: str = "epsilon",
    clip_denoised: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
    """Reference ancestral DDPM sampler."""

    if trajectory_frames and not 2 <= trajectory_frames <= schedule.timesteps + 1:
        raise ValueError("invalid trajectory_frames")
    was_training = model.training
    model.eval()
    if initial_noise is not None:
        video = initial_noise.to(device)
    elif generator is None:
        video = torch.randn(shape, device=device)
    else:
        video = torch.randn(
            shape, device=generator.device, generator=generator
        ).to(device)
    if tuple(video.shape) != shape:
        raise ValueError("initial_noise must match shape")
    trajectory: list[tuple[int, torch.Tensor]] = []
    captures: set[int] = set()
    if trajectory_frames:
        trajectory.append((schedule.timesteps, video.detach().cpu()))
        remaining = trajectory_frames - 1
        captures = (
            {0}
            if remaining == 1
            else {
                round(index * (schedule.timesteps - 1) / (remaining - 1))
                for index in range(remaining)
            }
        )
    try:
        for step in reversed(range(schedule.timesteps)):
            timesteps = torch.full(
                (shape[0],), step, device=device, dtype=torch.long
            )
            video = p_sample(
                model,
                schedule,
                video,
                timesteps,
                condition_ids,
                cfg_scale,
                generator,
                prediction_type,
                clip_denoised,
            )
            if step in captures:
                trajectory.append((step, video.detach().cpu()))
    finally:
        model.train(was_training)
    return (video, trajectory) if trajectory_frames else video


@torch.no_grad()
def ddim_sample_loop(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    shape: tuple[int, int, int, int, int],
    condition_ids: torch.Tensor,
    device: torch.device | str,
    cfg_scale: float = 1.0,
    initial_noise: torch.Tensor | None = None,
    sampling_steps: int = 50,
    eta: float = 0.0,
    trajectory_frames: int = 0,
    generator: torch.Generator | None = None,
    prediction_type: str = "epsilon",
    clip_denoised: bool = True,
    start_timestep: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
    """Respaced DDIM; ``eta=0`` is deterministic for fixed initial noise."""

    start = schedule.timesteps - 1 if start_timestep is None else start_timestep
    if not 0 <= start < schedule.timesteps:
        raise ValueError("start_timestep must be inside the diffusion schedule")
    if not 1 <= sampling_steps <= start + 1:
        raise ValueError("sampling_steps must be between 1 and start_timestep + 1")
    if eta < 0:
        raise ValueError("eta must be non-negative")
    if trajectory_frames and not 2 <= trajectory_frames <= sampling_steps + 1:
        raise ValueError("invalid trajectory_frames")
    was_training = model.training
    model.eval()
    if initial_noise is not None:
        video = initial_noise.to(device)
    elif generator is None:
        video = torch.randn(shape, device=device)
    else:
        video = torch.randn(
            shape, device=generator.device, generator=generator
        ).to(device)
    if tuple(video.shape) != shape:
        raise ValueError("initial_noise must match shape")
    steps = torch.linspace(
        start,
        0,
        sampling_steps,
        dtype=torch.float64,
    ).round().long().tolist()
    trajectory: list[tuple[int, torch.Tensor]] = []
    capture_indices: set[int] = set()
    if trajectory_frames:
        trajectory.append((start, video.detach().cpu()))
        remaining = trajectory_frames - 1
        capture_indices = (
            {sampling_steps - 1}
            if remaining == 1
            else {
                round(index * (sampling_steps - 1) / (remaining - 1))
                for index in range(remaining)
            }
        )
    try:
        for index, step in enumerate(steps):
            timesteps = torch.full(
                (shape[0],), step, device=device, dtype=torch.long
            )
            prediction = _model_prediction(
                model, video, timesteps, condition_ids, cfg_scale
            )
            predicted_mean = prediction[:, : video.shape[1]]
            alpha_bar = _extract(schedule.alpha_bar, timesteps, video.shape)
            if prediction_type == "velocity":
                predicted_clean, predicted_noise = (
                    schedule.clean_and_epsilon_from_velocity(
                        video, predicted_mean, timesteps
                    )
                )
            elif prediction_type == "epsilon":
                predicted_noise = predicted_mean
                predicted_clean = (
                    video - torch.sqrt(1 - alpha_bar) * predicted_noise
                ) / torch.sqrt(alpha_bar)
            else:
                raise ValueError("prediction_type must be 'epsilon' or 'velocity'")
            if clip_denoised:
                predicted_clean = predicted_clean.clamp(-1, 1)
            if index + 1 < len(steps):
                previous_timesteps = torch.full(
                    (shape[0],),
                    steps[index + 1],
                    device=device,
                    dtype=torch.long,
                )
                previous_alpha_bar = _extract(
                    schedule.alpha_bar, previous_timesteps, video.shape
                )
            else:
                previous_alpha_bar = torch.ones_like(alpha_bar)
            sigma = eta * torch.sqrt(
                ((1 - previous_alpha_bar) / (1 - alpha_bar))
                * (1 - alpha_bar / previous_alpha_bar)
            ).clamp(min=0)
            direction = torch.sqrt(
                (1 - previous_alpha_bar - sigma.square()).clamp(min=0)
            ) * predicted_noise
            noise = (
                torch.zeros_like(video)
                if eta == 0
                else _randn_like(video, generator)
            )
            video = (
                torch.sqrt(previous_alpha_bar) * predicted_clean
                + direction
                + sigma * noise
            )
            if index in capture_indices:
                state_timestep = steps[index + 1] if index + 1 < len(steps) else -1
                trajectory.append((state_timestep, video.detach().cpu()))
    finally:
        model.train(was_training)
    return (video, trajectory) if trajectory_frames else video
