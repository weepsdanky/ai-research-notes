import pytest
import torch

from diffusion import DiffusionSchedule, ddim_sample_loop, diffusion_loss, p_sample, sample_loop
from dit import DiT, DiTConfig


def model_and_schedule(timesteps=4):
    model = DiT(
        DiTConfig(
            image_size=8,
            patch_size=2,
            in_channels=1,
            hidden_size=16,
            depth=1,
            num_heads=4,
        )
    )
    return model, DiffusionSchedule(timesteps=timesteps)


def test_schedule_coefficients_satisfy_ddpm_identities():
    schedule = DiffusionSchedule(timesteps=10)
    assert torch.allclose(schedule.alpha, 1 - schedule.beta)
    assert torch.all(schedule.alpha_bar[1:] < schedule.alpha_bar[:-1])
    assert schedule.posterior_variance[0].item() == 0.0
    assert torch.all(schedule.posterior_variance[1:] > 0)
    assert torch.all(schedule.posterior_variance[1:] <= schedule.beta[1:])


def test_q_sample_matches_closed_form_for_supplied_noise():
    schedule = DiffusionSchedule(timesteps=10)
    x0 = torch.ones(2, 1, 8, 8)
    noise = torch.full_like(x0, 2.0)
    t = torch.tensor([0, 9])
    xt, returned_noise = schedule.q_sample(x0, t, noise)
    expected = (
        schedule.sqrt_alpha_bar[t, None, None, None] * x0
        + schedule.sqrt_one_minus_alpha_bar[t, None, None, None] * noise
    )
    assert torch.equal(returned_noise, noise)
    assert torch.allclose(xt, expected)


def test_noise_prediction_inverts_forward_process():
    schedule = DiffusionSchedule(timesteps=10)
    x0 = torch.randn(2, 1, 8, 8)
    noise = torch.randn_like(x0)
    t = torch.tensor([2, 8])
    xt, _ = schedule.q_sample(x0, t, noise)
    reconstructed = schedule.predict_x0_from_eps(xt, t, noise)
    assert torch.allclose(reconstructed, x0, atol=1e-5)


def test_learned_range_variance_does_not_clip_model_output():
    schedule = DiffusionSchedule(timesteps=10)
    t = torch.tensor([5])
    values = torch.full((1, 1, 1, 1), 2.0)
    _, actual_log_variance = schedule.model_variance(values, t)
    minimum = schedule.posterior_log_variance_clipped[5]
    maximum = torch.log(schedule.beta[5])
    expected = 1.5 * maximum - 0.5 * minimum
    assert torch.allclose(actual_log_variance.squeeze(), expected)


def test_diffusion_loss_is_finite_and_backpropagates():
    model, schedule = model_and_schedule()
    x0 = torch.randn(2, 1, 8, 8)
    labels = torch.tensor([1, 2])
    t = torch.tensor([0, 3])
    loss = diffusion_loss(model, schedule, x0, t, labels, noise=torch.ones_like(x0))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.final_layer.linear.weight.grad is not None
    assert model.final_layer.linear.weight.grad.abs().sum() > 0


def test_learned_sigma_hybrid_loss_is_finite_and_backpropagates():
    model = DiT(
        DiTConfig(
            image_size=8,
            patch_size=2,
            in_channels=1,
            hidden_size=16,
            depth=1,
            num_heads=4,
            learn_sigma=True,
        )
    )
    schedule = DiffusionSchedule(timesteps=4)
    x0 = torch.randn(2, 1, 8, 8)
    loss, metrics = diffusion_loss(
        model,
        schedule,
        x0,
        torch.tensor([1, 3]),
        torch.tensor([2, 4]),
        noise=torch.ones_like(x0),
        return_metrics=True,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert set(metrics) == {"loss", "mse", "vb"}
    assert metrics["vb"] >= 0
    assert model.final_layer.linear.weight.grad is not None


def test_final_reverse_step_adds_no_noise():
    model, schedule = model_and_schedule()
    model.eval()
    xt = torch.randn(2, 1, 8, 8)
    t = torch.zeros(2, dtype=torch.long)
    labels = torch.tensor([1, 2])
    first = p_sample(model, schedule, xt, t, labels, noise=torch.randn_like(xt))
    second = p_sample(model, schedule, xt, t, labels, noise=torch.randn_like(xt))
    assert torch.equal(first, second)


def test_sample_loop_returns_finite_images_and_restores_train_mode():
    model, schedule = model_and_schedule(timesteps=2)
    model.train()
    initial_noise = torch.randn(2, 1, 8, 8)
    samples = sample_loop(
        model,
        schedule,
        shape=(2, 1, 8, 8),
        labels=torch.tensor([1, 2]),
        device="cpu",
        initial_noise=initial_noise,
    )
    assert samples.shape == initial_noise.shape
    assert torch.isfinite(samples).all()
    assert model.training


def test_sample_loop_can_return_evenly_spaced_denoising_trajectory():
    model, schedule = model_and_schedule(timesteps=6)
    final, trajectory = sample_loop(
        model,
        schedule,
        shape=(2, 1, 8, 8),
        labels=torch.tensor([1, 2]),
        device="cpu",
        initial_noise=torch.zeros(2, 1, 8, 8),
        trajectory_frames=4,
    )
    assert final.shape == (2, 1, 8, 8)
    assert len(trajectory) == 4
    assert trajectory[0][0] == 6
    assert trajectory[-1][0] == 0
    assert torch.equal(trajectory[-1][1], final)


def test_ddim_eta_zero_is_deterministic_and_supports_fewer_steps():
    model, schedule = model_and_schedule(timesteps=10)
    initial_noise = torch.randn(2, 1, 8, 8)
    kwargs = dict(
        model=model,
        schedule=schedule,
        shape=tuple(initial_noise.shape),
        labels=torch.tensor([1, 2]),
        device="cpu",
        initial_noise=initial_noise,
        sampling_steps=4,
        eta=0.0,
        trajectory_frames=3,
    )
    first, first_path = ddim_sample_loop(**kwargs)
    second, second_path = ddim_sample_loop(**kwargs)
    assert torch.equal(first, second)
    assert len(first_path) == 3
    assert first_path[0][0] == 10
    assert first_path[-1][0] == 0


def test_one_step_ddim_trajectory_contains_initial_noise_and_final_image():
    model, schedule = model_and_schedule(timesteps=10)
    final, trajectory = ddim_sample_loop(
        model,
        schedule,
        shape=(2, 1, 8, 8),
        labels=torch.tensor([1, 2]),
        device="cpu",
        initial_noise=torch.randn(2, 1, 8, 8),
        sampling_steps=1,
        trajectory_frames=2,
    )
    assert [step for step, _ in trajectory] == [10, 9]
    assert torch.equal(trajectory[-1][1], final)


def test_guided_sample_loop_runs_conditional_and_null_pairs():
    model, schedule = model_and_schedule(timesteps=2)
    initial_noise = torch.randn(2, 1, 8, 8)
    samples = sample_loop(
        model,
        schedule,
        shape=(2, 1, 8, 8),
        labels=torch.tensor([3, 3]),
        device="cpu",
        cfg_scale=2.0,
        initial_noise=initial_noise,
    )
    assert samples.shape == initial_noise.shape
    assert torch.isfinite(samples).all()


def test_schedule_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="at least 2"):
        DiffusionSchedule(timesteps=1)
    with pytest.raises(ValueError, match="betas"):
        DiffusionSchedule(beta_start=0.5, beta_end=0.1)


def test_timestep_dtype_is_checked():
    schedule = DiffusionSchedule(timesteps=4)
    with pytest.raises(TypeError, match="torch.long"):
        schedule.q_sample(torch.randn(2, 1, 8, 8), torch.tensor([0.0, 1.0]))
