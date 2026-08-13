"""Export generated videos and sampling trajectories to standard media files."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence
import uuid

import torch
from PIL import Image


def _rgb_frames(video: torch.Tensor) -> torch.Tensor:
    """Convert a normalized ``[C, T, H, W]`` tensor to uint8 ``[T, H, W, C]``."""
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError("video must have shape [3, T, H, W]")
    if video.shape[1] < 1 or video.shape[2] < 1 or video.shape[3] < 1:
        raise ValueError("video must contain at least one non-empty frame")
    return (
        ((video.detach().to(dtype=torch.float32, device="cpu").clamp(-1, 1) + 1) * 127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 3, 0)
        .contiguous()
    )


def save_video_mp4(
    video: torch.Tensor,
    output_path: str | Path,
    *,
    fps: float = 8,
) -> Path:
    """Encode a normalized RGB video tensor as a browser-playable H.264 MP4."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to export MP4 files")

    frames = _rgb_frames(video)
    _, height, width, _ = frames.shape
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}.tmp{destination.suffix}"
    )
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        subprocess.run(command, input=frames.numpy().tobytes(), check=True)
        temporary.replace(destination)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"ffmpeg could not encode {destination}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def save_denoising_gif(
    trajectory: Sequence[torch.Tensor],
    output_path: str | Path,
    *,
    fps: float = 4,
) -> Path:
    """Save diffusion states as GIF frames, each showing the full time axis.

    This intentionally visualizes a different axis from :func:`save_video_mp4`:
    the GIF advances through denoising steps, while each GIF frame is a
    left-to-right contact sheet of every temporal frame in that state.
    """
    if not trajectory:
        raise ValueError("trajectory must contain at least one video state")
    if fps <= 0:
        raise ValueError("fps must be positive")

    previews: list[Image.Image] = []
    expected_shape: tuple[int, ...] | None = None
    for state in trajectory:
        if expected_shape is None:
            expected_shape = tuple(state.shape)
        elif tuple(state.shape) != expected_shape:
            raise ValueError("all trajectory states must have the same shape")
        frames = _rgb_frames(state)
        contact_sheet = torch.cat(list(frames), dim=1)
        previews.append(Image.fromarray(contact_sheet.numpy(), mode="RGB"))

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}.tmp{destination.suffix}"
    )
    duration_ms = max(1, round(1000 / fps))
    try:
        previews[0].save(
            temporary,
            save_all=True,
            append_images=previews[1:],
            duration=duration_ms,
            loop=0,
            format="GIF",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
