import pytest
import torch

from diffusion import (
    DiffusionSchedule,
    ddim_sample_loop,
    diffusion_loss,
    p_sample,
    respaced_schedule,
    sample_loop,
    space_timesteps,
)
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


def test_uniform_respacing_matches_official_fractional_stride_indices():
    assert space_timesteps(10, 4) == (0, 3, 6, 9)


def test_respaced_schedule_preserves_selected_cumulative_alphas():
    original = DiffusionSchedule(timesteps=10)
    respaced = respaced_schedule(original, 4)
    selected = torch.tensor([0, 3, 6, 9])

    assert respaced.timesteps == 4
    assert tuple(respaced.timestep_map.tolist()) == (0, 3, 6, 9)
    assert torch.allclose(respaced.alpha_bar, original.alpha_bar[selected], atol=1e-7)


class _RecordingNoiseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.timesteps = []

    def forward(self, x, t, labels):
        self.timesteps.append(int(t[0]))
        return torch.zeros_like(x)


def test_respaced_ancestral_sampler_calls_model_with_original_timesteps():
    model = _RecordingNoiseModel()
    schedule = DiffusionSchedule(timesteps=10)
    sample_loop(
        model,
        schedule,
        shape=(1, 1, 2, 2),
        labels=torch.tensor([0]),
        device="cpu",
        initial_noise=torch.zeros(1, 1, 2, 2),
        sampling_steps=4,
    )
    assert model.timesteps == [9, 6, 3, 0]


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


def test_velocity_target_inverts_forward_process_to_x0_and_noise():
    schedule = DiffusionSchedule(timesteps=10)
    x0 = torch.tensor([[[[-0.75, 0.25], [0.5, 1.0]]]])
    noise = torch.tensor([[[[0.5, -1.0], [0.25, 0.75]]]])
    t = torch.tensor([7])
    xt, _ = schedule.q_sample(x0, t, noise)

    velocity = schedule.velocity_target(x0, t, noise)

    assert torch.allclose(schedule.predict_x0_from_v(xt, t, velocity), x0, atol=1e-6)
    assert torch.allclose(schedule.predict_eps_from_v(xt, t, velocity), noise, atol=1e-6)


class _ConstantPredictionModel(torch.nn.Module):
    def __init__(self, prediction):
        super().__init__()
        self.prediction = torch.nn.Parameter(prediction.clone())

    def forward(self, x, t, labels):
        return self.prediction.expand_as(x)


def test_velocity_diffusion_loss_trains_against_v_target():
    schedule = DiffusionSchedule(timesteps=10)
    x0 = torch.full((1, 1, 2, 2), 0.25)
    noise = torch.full_like(x0, -0.5)
    t = torch.tensor([6])
    target = schedule.velocity_target(x0, t, noise)
    model = _ConstantPredictionModel(target)

    loss = diffusion_loss(
        model, schedule, x0, t, torch.tensor([0]), noise=noise,
        prediction_type="velocity",
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-12)


class _EquivalentVelocityModel(torch.nn.Module):
    def __init__(self, schedule, epsilon):
        super().__init__()
        self.schedule = schedule
        self.epsilon = epsilon

    def forward(self, x, t, labels):
        alpha = self.schedule.sqrt_alpha_bar[t, None, None, None]
        sigma = self.schedule.sqrt_one_minus_alpha_bar[t, None, None, None]
        return (self.epsilon - sigma * x) / alpha


def test_velocity_and_epsilon_parameterizations_give_same_ddpm_step():
    schedule = DiffusionSchedule(timesteps=10)
    xt = torch.tensor([[[[0.2, -0.4], [0.8, 0.1]]]])
    epsilon = torch.full_like(xt, 0.3)
    labels = torch.tensor([0])
    t = torch.tensor([5])
    reverse_noise = torch.full_like(xt, -0.2)

    epsilon_result = p_sample(
        _ConstantPredictionModel(epsilon), schedule, xt, t, labels,
        noise=reverse_noise, clip_denoised=False,
    )
    velocity_result = p_sample(
        _EquivalentVelocityModel(schedule, epsilon), schedule, xt, t, labels,
        noise=reverse_noise, clip_denoised=False, prediction_type="velocity",
    )

    assert torch.allclose(velocity_result, epsilon_result, atol=1e-6)


def test_velocity_and_epsilon_parameterizations_give_same_ddim_path():
    schedule = DiffusionSchedule(timesteps=10)
    initial = torch.tensor([[[[0.2, -0.4], [0.8, 0.1]]]])
    epsilon = torch.full_like(initial, 0.3)
    common = dict(
        schedule=schedule, shape=tuple(initial.shape), labels=torch.tensor([0]),
        device="cpu", initial_noise=initial, sampling_steps=4,
        clip_denoised=False,
    )

    epsilon_result = ddim_sample_loop(
        _ConstantPredictionModel(epsilon), **common
    )
    velocity_result = ddim_sample_loop(
        _EquivalentVelocityModel(schedule, epsilon), **common,
        prediction_type="velocity",
    )

    assert torch.allclose(velocity_result, epsilon_result, atol=1e-5)


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


class _FixedLearnedVarianceModel(torch.nn.Module):
    def __init__(self, epsilon: float):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x, t, labels):
        epsilon = torch.full_like(x, self.epsilon)
        variance = torch.zeros_like(x)
        return torch.cat([epsilon, variance], dim=1)


def test_learned_variance_loss_can_disable_x0_clipping_for_paper_recipe():
    schedule = DiffusionSchedule(timesteps=10)
    model = _FixedLearnedVarianceModel(epsilon=-1000.0)
    x0 = torch.zeros(1, 1, 2, 2)
    common = dict(
        model=model,
        schedule=schedule,
        x0=x0,
        t=torch.tensor([5]),
        labels=torch.tensor([0]),
        noise=torch.zeros_like(x0),
        vb_mode="raw",
        return_metrics=True,
    )
    _, clipped = diffusion_loss(**common, clip_denoised=True)
    _, faithful = diffusion_loss(**common, clip_denoised=False)

    assert faithful["vb"] > clipped["vb"]


def test_raw_and_rescaled_vb_modes_have_explicit_official_weights():
    schedule = DiffusionSchedule(timesteps=10)
    model = _FixedLearnedVarianceModel(epsilon=0.0)
    x0 = torch.full((1, 1, 2, 2), 0.25)
    common = dict(
        model=model,
        schedule=schedule,
        x0=x0,
        t=torch.tensor([5]),
        labels=torch.tensor([0]),
        noise=torch.zeros_like(x0),
        clip_denoised=False,
        return_metrics=True,
    )
    raw_loss, raw = diffusion_loss(**common, vb_mode="raw")
    rescaled_loss, rescaled = diffusion_loss(**common, vb_mode="rescaled")

    assert raw_loss.item() - raw["mse"] == pytest.approx(raw["vb"])
    assert rescaled_loss.item() - rescaled["mse"] == pytest.approx(
        schedule.timesteps / 1000.0 * rescaled["vb"]
    )


def test_final_reverse_step_adds_no_noise():
    model, schedule = model_and_schedule()
    model.eval()
    xt = torch.randn(2, 1, 8, 8)
    t = torch.zeros(2, dtype=torch.long)
    labels = torch.tensor([1, 2])
    first = p_sample(model, schedule, xt, t, labels, noise=torch.randn_like(xt))
    second = p_sample(model, schedule, xt, t, labels, noise=torch.randn_like(xt))
    assert torch.equal(first, second)


def test_ancestral_sampling_clipping_is_explicit():
    schedule = DiffusionSchedule(timesteps=10)
    model = _FixedLearnedVarianceModel(epsilon=-1000.0)
    xt = torch.zeros(1, 1, 2, 2)
    t = torch.zeros(1, dtype=torch.long)
    labels = torch.zeros(1, dtype=torch.long)

    clipped = p_sample(model, schedule, xt, t, labels, clip_denoised=True)
    faithful = p_sample(model, schedule, xt, t, labels, clip_denoised=False)

    assert clipped.abs().max().item() <= 1.0
    assert faithful.abs().max().item() > 1.0


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


def test_ddim_sampling_clipping_is_explicit():
    schedule = DiffusionSchedule(timesteps=10)
    model = _FixedLearnedVarianceModel(epsilon=-100.0)
    common = dict(
        model=model,
        schedule=schedule,
        shape=(1, 1, 2, 2),
        labels=torch.zeros(1, dtype=torch.long),
        device="cpu",
        initial_noise=torch.zeros(1, 1, 2, 2),
        sampling_steps=1,
    )
    clipped = ddim_sample_loop(**common, clip_denoised=True)
    faithful = ddim_sample_loop(**common, clip_denoised=False)
    assert clipped.abs().max().item() <= 1.0
    assert faithful.abs().max().item() > 1.0


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
