from pathlib import Path

from web_ui import (
    PREVIEW_MODE,
    build_ui,
    ensure_loopback_proxy_bypass,
    parse_args,
    run_ui_request,
)


def test_ui_dataset_preview_request_returns_honestly_labelled_animation(tmp_path):
    video, diagnostic, parsed, status, metadata = run_ui_request(
        mode=PREVIEW_MODE,
        prompt="a brown dog walking left on grass",
        checkpoint_path="",
        seed=11,
        sampling_steps=3,
        cfg_scale=2.5,
        trajectory_frames=3,
        fps=8,
        output_dir=tmp_path,
        device="cpu",
    )

    assert Path(video).is_file() and Path(video).suffix == ".mp4"
    assert diagnostic is None
    assert parsed == {
        "subject": "dog",
        "color": "brown",
        "motion": "walk_left",
        "scene": "grass",
    }
    assert "not model output" in status
    assert Path(metadata).is_file()


def test_build_ui_constructs_an_app_without_launching_a_server(tmp_path):
    demo = build_ui(
        default_checkpoint=str(tmp_path / "checkpoint.pt"),
        output_dir=tmp_path / "outputs",
        device="cpu",
    )

    assert demo.__class__.__name__ == "Blocks"
    assert demo.title == "Video DiT temporal animation lab"


def test_web_ui_cli_accepts_readme_positional_checkpoint():
    args = parse_args(["runs/faithful/video_dit.pt"])

    assert args.checkpoint == "runs/faithful/video_dit.pt"


def test_ui_defaults_to_the_checkpoint_name_written_by_training():
    args = parse_args([])

    assert args.checkpoint.endswith("demo_artifacts/video_dit_inference.pt")


def test_web_ui_bypasses_system_proxy_for_its_loopback_health_check(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "example.test")
    monkeypatch.delenv("no_proxy", raising=False)

    ensure_loopback_proxy_bypass("127.0.0.1")

    expected = {"example.test", "127.0.0.1", "localhost", "::1"}
    assert set(__import__("os").environ["NO_PROXY"].split(",")) == expected
    assert set(__import__("os").environ["no_proxy"].split(",")) == expected
