from pathlib import Path

import torch

from diffusion import DiffusionSchedule
from dit import DiT, DiTConfig
from inference import DiTInference
from web_ui import DEFAULT_CHECKPOINT, render_page


def test_default_ui_checkpoint_is_the_portable_demo_artifact():
    assert isinstance(DEFAULT_CHECKPOINT, Path)
    assert DEFAULT_CHECKPOINT.name == "image_dit_inference.pt"
    assert DEFAULT_CHECKPOINT.parent.name == "demo_artifacts"


def test_checkpoint_metadata_is_html_escaped_before_rendering():
    config = DiTConfig(
        image_size=8, patch_size=2, in_channels=1,
        hidden_size=16, depth=1, num_heads=4,
    )
    pipeline = DiTInference(
        DiT(config),
        DiffusionSchedule(timesteps=4),
        torch.device("cpu"),
        metadata={"dataset": '<img src=x onerror="alert(1)">'},
        class_names=('<script>alert("x")</script>',) + tuple(str(i) for i in range(9)),
    )

    page = render_page(pipeline).decode()

    assert '<script>alert("x")</script>' not in page
    assert "<img src=x" not in page
    assert "&lt;script&gt;" in page
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in page
