from pathlib import Path

import torch

from train_flow import parse_args, train


def test_flow_cpu_dry_run_updates_the_same_video_dit_and_records_flow_contract(tmp_path):
    output_dir = tmp_path / "flow-dry-run"
    summary = train(parse_args([
        "--dry-run",
        "--device", "cpu",
        "--batch-size", "2",
        "--output-dir", str(output_dir),
    ]))

    safe = torch.load(
        summary["inference_checkpoint"], map_location="cpu", weights_only=True
    )
    full = torch.load(summary["checkpoint"], map_location="cpu", weights_only=False)

    assert summary["updates_this_run"] == 1
    assert summary["generative_process"] == "linear_conditional_flow_matching"
    assert safe["time_convention"] == "data_at_0_noise_at_1"
    assert safe["prediction_type"] == "flow_velocity"
    assert safe["inference_defaults"] == {
        "time_grid": "shifted_linear_1_to_0",
        "solver": "heun",
        "sampling_steps": 20,
        "sampling_time_shift": 1.0,
    }
    assert safe["config"]["learn_sigma"] is False
    assert safe["config"]["frames"] == 4
    assert "optimizer" not in safe and "rng_state" not in safe
    assert safe["flow"]["time_sampling"] == "uniform"
    assert {"optimizer", "rng_state"} <= set(full)
    assert Path(summary["inference_checkpoint"]).is_file()


def test_flow_resume_consumes_the_exact_next_batch(tmp_path):
    def args(name, steps, resume=None):
        values = [
            "--device", "cpu", "--max-steps", str(steps),
            "--batch-size", "2", "--frames", "4", "--patch-size", "8",
            "--hidden-size", "32", "--num-heads", "4", "--num-pairs", "1",
            "--output-dir", str(tmp_path / name),
        ]
        if resume is not None:
            values.extend(["--resume", str(resume)])
        return parse_args(values)

    continuous = train(args("continuous", 2))
    first = train(args("first", 1))
    resumed = train(args("resumed", 2, first["checkpoint"]))
    continuous_state = torch.load(
        continuous["checkpoint"], map_location="cpu", weights_only=False
    )["model"]
    resumed_state = torch.load(
        resumed["checkpoint"], map_location="cpu", weights_only=False
    )["model"]

    assert resumed["final_loss"] == continuous["final_loss"]
    for name in continuous_state:
        assert torch.equal(continuous_state[name], resumed_state[name]), name


def test_flow_resume_rejects_a_synthetic_curriculum_change(tmp_path):
    first = train(parse_args([
        "--dry-run", "--device", "cpu", "--batch-size", "2",
        "--curriculum", "canonical", "--output-dir", str(tmp_path / "first"),
    ]))
    changed = parse_args([
        "--dry-run", "--device", "cpu", "--batch-size", "2",
        "--curriculum", "factorial", "--resume", first["checkpoint"],
        "--output-dir", str(tmp_path / "changed"),
    ])

    with __import__("pytest").raises(ValueError, match="contract"):
        train(changed)
