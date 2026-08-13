import torch

from diffusion import DiffusionSchedule, ddim_sample_loop, diffusion_loss, p_sample
from video_dit import VideoDiT, VideoDiTConfig


def tiny_model(learn_sigma: bool = False) -> VideoDiT:
    return VideoDiT(
        VideoDiTConfig(
            frames=4,
            image_size=32,
            patch_size=8,
            hidden_size=32,
            num_heads=4,
            num_pairs=1,
            learn_sigma=learn_sigma,
        )
    )


def test_forward_diffusion_broadcasts_coefficients_over_all_five_dimensions():
    schedule = DiffusionSchedule(timesteps=8)
    clean = torch.ones(2, 3, 4, 8, 8)
    noise = torch.zeros_like(clean)
    timesteps = torch.tensor([0, 7], dtype=torch.long)

    noisy, returned_noise = schedule.q_sample(clean, timesteps, noise)

    expected = schedule.sqrt_alpha_bar[timesteps].reshape(2, 1, 1, 1, 1)
    assert torch.allclose(noisy, expected.expand_as(clean))
    assert torch.equal(returned_noise, noise)


def test_epsilon_loss_backpropagates_through_a_real_video_dit():
    model = tiny_model()
    schedule = DiffusionSchedule(timesteps=8)
    clean = torch.randn(2, 3, 4, 32, 32)
    timesteps = torch.tensor([2, 5])
    conditions = torch.tensor([[0, 0, 0, 0], [2, 2, 1, 3]])

    loss, metrics = diffusion_loss(
        model, schedule, clean, timesteps, conditions, return_metrics=True
    )
    loss.backward()

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert metrics["mse"] > 0 and metrics["vb"] == 0
    assert model.final_layer.linear.weight.grad is not None
    assert torch.count_nonzero(model.final_layer.linear.weight.grad) > 0


def test_learned_range_variance_adds_a_finite_vlb_term():
    model = tiny_model(learn_sigma=True)
    schedule = DiffusionSchedule(timesteps=8)
    clean = torch.rand(2, 3, 4, 32, 32) * 2 - 1
    timesteps = torch.tensor([0, 5])
    conditions = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]])

    loss, metrics = diffusion_loss(
        model, schedule, clean, timesteps, conditions, return_metrics=True
    )

    assert torch.isfinite(loss)
    assert metrics["vb"] > 0


def test_ddim_is_deterministic_at_eta_zero_and_returns_denoising_diagnostics():
    model = tiny_model()
    model.train()
    schedule = DiffusionSchedule(timesteps=8)
    initial_noise = torch.randn(1, 3, 4, 32, 32)
    conditions = torch.tensor([[0, 0, 0, 0]])
    kwargs = dict(
        model=model,
        schedule=schedule,
        shape=(1, 3, 4, 32, 32),
        condition_ids=conditions,
        device="cpu",
        initial_noise=initial_noise,
        sampling_steps=4,
        eta=0.0,
        trajectory_frames=3,
    )

    first, trajectory = ddim_sample_loop(**kwargs)
    second, _ = ddim_sample_loop(**kwargs)

    assert first.shape == initial_noise.shape
    assert torch.equal(first, second)
    assert len(trajectory) == 3
    assert trajectory[0][1].shape == initial_noise.shape
    assert model.training is True


class _ZeroPrediction(torch.nn.Module):
    def forward(self, noisy, timesteps, condition_ids):
        return torch.zeros_like(noisy)


def test_samplers_can_preserve_unbounded_latent_predictions():
    model = _ZeroPrediction()
    schedule = DiffusionSchedule(timesteps=8)
    initial = torch.full((1, 3, 4, 8, 8), 2.0)
    conditions = torch.zeros(1, 4, dtype=torch.long)
    timesteps = torch.zeros(1, dtype=torch.long)

    clipped = p_sample(
        model, schedule, initial, timesteps, conditions, clip_denoised=True,
    )
    latent = p_sample(
        model, schedule, initial, timesteps, conditions, clip_denoised=False,
    )
    ddim_clipped = ddim_sample_loop(
        model, schedule, initial.shape, conditions, "cpu", initial_noise=initial,
        sampling_steps=1, clip_denoised=True,
    )
    ddim_latent = ddim_sample_loop(
        model, schedule, initial.shape, conditions, "cpu", initial_noise=initial,
        sampling_steps=1, clip_denoised=False,
    )

    assert clipped.abs().max() <= 1
    assert latent.abs().max() > 1
    assert ddim_clipped.abs().max() <= 1
    assert ddim_latent.abs().max() > 1


def test_ddim_can_start_from_an_explicit_earlier_timestep():
    model = _ZeroPrediction()
    schedule = DiffusionSchedule(timesteps=8)
    initial = torch.randn(1, 3, 4, 8, 8)
    conditions = torch.zeros(1, 4, dtype=torch.long)

    _, trajectory = ddim_sample_loop(
        model, schedule, initial.shape, conditions, "cpu",
        initial_noise=initial, sampling_steps=3, start_timestep=5,
        trajectory_frames=2,
    )

    assert trajectory[0][0] == 5
    assert trajectory[-1][0] == -1


def test_velocity_parameterization_recovers_clean_video_and_noise():
    schedule = DiffusionSchedule(timesteps=8)
    clean = torch.randn(2, 3, 4, 8, 8)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([2, 7])
    noisy, _ = schedule.q_sample(clean, timesteps, noise)
    velocity = schedule.velocity_target(clean, noise, timesteps)

    recovered_clean, recovered_noise = schedule.clean_and_epsilon_from_velocity(
        noisy, velocity, timesteps
    )

    assert torch.allclose(recovered_clean, clean, atol=1e-5)
    assert torch.allclose(recovered_noise, noise, atol=1e-5)


def test_velocity_loss_backpropagates_through_video_dit():
    model = tiny_model()
    schedule = DiffusionSchedule(timesteps=8)
    clean = torch.randn(2, 3, 4, 32, 32)
    timesteps = torch.tensor([2, 7])
    conditions = torch.tensor([[0, 0, 0, 0], [2, 2, 1, 3]])

    loss, metrics = diffusion_loss(
        model,
        schedule,
        clean,
        timesteps,
        conditions,
        prediction_type="velocity",
        return_metrics=True,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["mse"] > 0
    assert torch.count_nonzero(model.final_layer.linear.weight.grad) > 0
