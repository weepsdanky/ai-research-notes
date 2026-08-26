import math

import torch

from flow_matching import (
    FlowMatchingSchedule,
    flow_matching_loss,
    flow_sample_loop,
)
from video_dit import VideoDiT, VideoDiTConfig


def test_linear_probability_path_has_exact_endpoints_and_velocity():
    schedule = FlowMatchingSchedule()
    clean = torch.tensor([[[[[-2.0]]]]])
    noise = torch.tensor([[[[[3.0]]]]])

    endpoints = schedule.interpolate(
        clean.repeat(2, 1, 1, 1, 1),
        noise.repeat(2, 1, 1, 1, 1),
        torch.tensor([0.0, 1.0]),
    )

    assert endpoints[:, 0, 0, 0, 0].tolist() == [-2.0, 3.0]
    assert schedule.velocity_target(clean, noise).item() == 5.0


def test_velocity_recovers_clean_video_and_noise_at_every_time():
    schedule = FlowMatchingSchedule()
    clean = torch.randn(3, 2, 4, 3, 3)
    noise = torch.randn_like(clean)
    times = torch.tensor([0.0, 0.35, 1.0])
    state = schedule.interpolate(clean, noise, times)

    recovered_clean, recovered_noise = schedule.clean_and_noise_from_velocity(
        state, noise - clean, times
    )

    assert torch.allclose(recovered_clean, clean, atol=1e-6)
    assert torch.allclose(recovered_noise, noise, atol=1e-6)


def test_time_shift_preserves_endpoints_and_moves_mass_toward_noise():
    schedule = FlowMatchingSchedule(time_shift=3.0)
    times = torch.tensor([0.0, 0.25, 0.5, 1.0])

    shifted = schedule.shift_times(times)

    assert shifted[[0, -1]].tolist() == [0.0, 1.0]
    assert torch.all(shifted[1:-1] > times[1:-1])


class _ConstantVelocity(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, state, model_times, condition_ids):
        return torch.full_like(state, self.value)


def test_euler_and_heun_reverse_the_path_from_noise_to_data_in_10_to_30_steps():
    initial_noise = torch.full((1, 1, 2, 2, 2), 3.0)
    conditions = torch.zeros(1, 4, dtype=torch.long)
    schedule = FlowMatchingSchedule()

    for solver, steps in (("euler", 10), ("heun", 30)):
        sample, trajectory = flow_sample_loop(
            _ConstantVelocity(5.0),
            schedule,
            tuple(initial_noise.shape),
            conditions,
            "cpu",
            initial_noise=initial_noise,
            sampling_steps=steps,
            solver=solver,
            trajectory_frames=3,
        )
        assert torch.allclose(sample, torch.full_like(sample, -2.0), atol=1e-5)
        assert trajectory[0][0] == 1.0
        assert trajectory[-1][0] == 0.0


class _StateVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, state, model_times, condition_ids):
        self.calls += 1
        return state


def test_heun_has_lower_error_than_euler_on_a_curved_ode():
    initial = torch.ones(1, 1, 2, 2, 2)
    conditions = torch.zeros(1, 4, dtype=torch.long)
    model = _StateVelocity()
    kwargs = dict(
        model=model,
        schedule=FlowMatchingSchedule(),
        shape=tuple(initial.shape),
        condition_ids=conditions,
        device="cpu",
        initial_noise=initial,
        sampling_steps=10,
    )

    euler, _ = flow_sample_loop(**kwargs, solver="euler")
    assert model.calls == 10
    model.calls = 0
    heun, _ = flow_sample_loop(**kwargs, solver="heun")
    exact = math.exp(-1.0)

    assert abs(heun.mean().item() - exact) < abs(euler.mean().item() - exact)
    assert model.calls == 20


def test_inference_time_shift_changes_interior_grid_but_not_endpoints():
    initial = torch.ones(1, 1, 2, 2, 2)
    _, trajectory = flow_sample_loop(
        _ConstantVelocity(0.0),
        FlowMatchingSchedule(),
        tuple(initial.shape),
        torch.zeros(1, 4, dtype=torch.long),
        "cpu",
        initial_noise=initial,
        sampling_steps=2,
        solver="euler",
        sampling_time_shift=3.0,
        trajectory_frames=3,
    )

    assert [time for time, _ in trajectory] == [1.0, 0.75, 0.0]


def test_flow_loss_backpropagates_through_the_existing_video_dit():
    model = VideoDiT(VideoDiTConfig(
        frames=4,
        image_size=16,
        patch_size=8,
        hidden_size=32,
        num_heads=4,
        num_pairs=1,
        condition_dropout_prob=0.0,
        learn_sigma=False,
    ))
    clean = torch.randn(2, 3, 4, 16, 16)
    times = torch.tensor([0.2, 0.8])
    conditions = torch.tensor([[0, 0, 0, 0], [2, 2, 1, 3]])

    loss, metrics = flow_matching_loss(
        model,
        FlowMatchingSchedule(),
        clean,
        times,
        conditions,
        return_metrics=True,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["mse"] > 0
    assert torch.count_nonzero(model.final_layer.linear.weight.grad) > 0


def test_flow_rejects_the_ddpm_learned_variance_output_contract():
    model = VideoDiT(VideoDiTConfig(
        frames=2,
        image_size=8,
        patch_size=4,
        hidden_size=32,
        num_heads=4,
        num_pairs=1,
        learn_sigma=True,
    ))
    clean = torch.randn(1, 3, 2, 8, 8)

    with __import__("pytest").raises(ValueError, match="exactly C velocity channels"):
        flow_matching_loss(
            model,
            FlowMatchingSchedule(),
            clean,
            torch.tensor([0.5]),
            torch.zeros(1, 4, dtype=torch.long),
        )
