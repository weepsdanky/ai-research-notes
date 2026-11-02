import torch

from diffusion import DiffusionSchedule
from mosaic_debug import direct_x0_probe, partial_denoising_probe


class ExactVelocity(torch.nn.Module):
    def __init__(self, schedule: DiffusionSchedule, clean: torch.Tensor) -> None:
        super().__init__()
        self.schedule = schedule
        self.clean = clean

    def forward(self, noisy, timesteps, conditions):
        del conditions
        alpha = self.schedule.sqrt_alpha_bar.gather(0, timesteps).reshape(-1, 1, 1, 1, 1)
        sigma = self.schedule.sqrt_one_minus_alpha_bar.gather(0, timesteps).reshape(-1, 1, 1, 1, 1)
        epsilon = (noisy - alpha * self.clean) / sigma.clamp_min(1e-12)
        return alpha * epsilon - sigma * self.clean


def test_exact_velocity_has_zero_direct_and_partial_reconstruction_error():
    schedule = DiffusionSchedule(timesteps=20)
    clean = torch.randn(1, 3, 2, 4, 4)
    condition = torch.zeros(1, 1)
    model = ExactVelocity(schedule, clean)

    direct = direct_x0_probe(
        model, schedule, clean, condition,
        timesteps=[5, 19], prediction_type="velocity", seed=7,
    )
    partial = partial_denoising_probe(
        model, schedule, clean, condition,
        timesteps=[5, 19], prediction_type="velocity", sampling_steps=20, seed=7,
    )

    assert direct["5"]["x0_mse"] < 1e-10
    assert direct["19"]["x0_mse"] < 1e-10
    assert partial["5"]["reconstruction_mse"] < 1e-9
    assert partial["19"]["reconstruction_mse"] < 1e-9


def test_partial_probe_rejects_timestep_outside_schedule():
    schedule = DiffusionSchedule(timesteps=20)
    clean = torch.randn(1, 3, 2, 4, 4)
    model = ExactVelocity(schedule, clean)

    with __import__("pytest").raises(ValueError, match="inside"):
        partial_denoising_probe(
            model, schedule, clean, torch.zeros(1, 1),
            timesteps=[20], prediction_type="velocity", sampling_steps=20, seed=7,
        )
