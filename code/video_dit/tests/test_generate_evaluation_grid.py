import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import generate_evaluation_grid as target


def test_cli_commits_exact_preregistered_prompt_seed_grid(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    spec = {
        "protocol": "frozen compositional generation evaluation v1",
        "evaluation_tier": "pilot",
        "confirmatory_gate": {
            "passes": False,
            "required_c1_source_groups": 50,
            "required_train_pair_source_groups": 10,
            "observed_c1_source_groups": {"cat|playing|indoors": 16},
            "observed_train_pair_source_groups": {
                "cat|indoors": 23, "cat|playing": 5, "playing|indoors": 8,
            },
        },
        "training_manifest_sha256": "a" * 64,
        "split_report_sha256": "b" * 64,
        "generation_contract": {
            "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
            "training_step": 10000, "weights": "ema",
            "prediction_type": "velocity", "sampling_steps": 50,
            "start_timestep": 999, "clip_denoised": False,
            "effective_cfg_scale": 2.5, "condition_mode": "text_tokens",
            "text_encoder_id": "clip@sha#tokens-v2",
            "training_manifest_sha256": "a" * 64,
        },
        "seeds": [17, 29],
        "records": [
            {"prompt": "a cat playing indoors", "split": "c1",
             "subject": "cat", "action": "playing", "scene": "indoors"},
            {"prompt": "a dog walking outdoors", "split": "iid",
             "subject": "dog", "action": "walking", "scene": "outdoors"},
        ],
        "training_ready": False,
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    calls = []

    class Engine:
        def generate(self, prompt, output_dir, **kwargs):
            calls.append((prompt, kwargs["seed"]))
            stem = f"sample-{len(calls)}"
            video = output_dir / f"{stem}.mp4"
            metadata = output_dir / f"{stem}.json"
            video.write_bytes(stem.encode())
            metadata.write_text(json.dumps({"prompt": prompt, "seed": kwargs["seed"]}))
            return SimpleNamespace(video_path=video, metadata_path=metadata)

    monkeypatch.setattr(target, "load_engine", lambda *args, **kwargs: Engine())
    output = tmp_path / "grid"
    target.main([
        str(checkpoint), str(spec_path), str(output),
        "--text-checkpoint", str(tmp_path / "clip.pt"),
        "--vae-model", "vae", "--vae-revision", "rev",
    ])

    assert calls == [
        ("a cat playing indoors", 17), ("a dog walking outdoors", 17),
        ("a cat playing indoors", 29), ("a dog walking outdoors", 29),
    ]
    inventory = json.loads((output / "grid-inventory.json").read_text())
    assert inventory["expected_generations"] == 4
    assert inventory["completed_generations"] == 4
    assert inventory["evaluation_spec_sha256"] == hashlib.sha256(
        spec_path.read_bytes()
    ).hexdigest()


def test_cli_removes_partial_grid_when_any_generation_fails(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    spec = {
        "protocol": "frozen compositional generation evaluation v1",
        "evaluation_tier": "pilot",
        "confirmatory_gate": {
            "passes": False,
            "required_c1_source_groups": 50,
            "required_train_pair_source_groups": 10,
            "observed_c1_source_groups": {"cat|playing|indoors": 1},
            "observed_train_pair_source_groups": {
                "cat|indoors": 1, "cat|playing": 1, "playing|indoors": 1,
            },
        },
        "training_manifest_sha256": "a" * 64,
        "split_report_sha256": "b" * 64,
        "generation_contract": {
            "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
            "training_step": 10000, "weights": "ema",
            "prediction_type": "velocity", "sampling_steps": 50,
            "start_timestep": 999, "clip_denoised": False,
            "effective_cfg_scale": 2.5, "condition_mode": "text_tokens",
            "text_encoder_id": "clip@sha#tokens-v2",
            "training_manifest_sha256": "a" * 64,
        },
        "seeds": [1, 2],
        "records": [
            {"prompt": "c1", "split": "c1", "subject": "cat",
             "action": "playing", "scene": "indoors"},
            {"prompt": "iid", "split": "iid", "subject": "dog",
             "action": "walking", "scene": "outdoors"},
        ],
        "training_ready": False,
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))

    class FailingEngine:
        calls = 0

        def generate(self, prompt, output_dir, **kwargs):
            self.calls += 1
            (output_dir / "partial.mp4").write_bytes(b"partial")
            if self.calls == 2:
                raise RuntimeError("sampling failed")
            metadata = output_dir / "partial.json"
            metadata.write_text("{}")
            return SimpleNamespace(
                video_path=output_dir / "partial.mp4", metadata_path=metadata,
            )

    monkeypatch.setattr(target, "load_engine", lambda *args, **kwargs: FailingEngine())
    output = tmp_path / "grid"
    try:
        target.main([
            str(checkpoint), str(spec_path), str(output),
            "--text-checkpoint", str(tmp_path / "clip.pt"),
            "--vae-model", "vae", "--vae-revision", "rev",
        ])
    except RuntimeError as error:
        assert str(error) == "sampling failed"
    else:
        raise AssertionError("expected the simulated sampling failure")

    assert not output.exists()
    assert not list(tmp_path.glob(".grid.*"))
