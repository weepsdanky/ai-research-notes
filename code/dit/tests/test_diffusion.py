import pytest
import torch

from diffusion import DiffusionSchedule, diffusion_loss, p_sample, sample_loop
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
