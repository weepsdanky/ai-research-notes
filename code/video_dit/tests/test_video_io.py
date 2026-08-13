import json
import subprocess

import torch
from PIL import Image

from video_io import save_denoising_gif, save_video_mp4


def _moving_square_video(frames: int = 5, size: int = 16) -> torch.Tensor:
    video = torch.full((3, frames, size, size), -1.0)
    for frame in range(frames):
        left = 2 * frame
        video[:, frame, 4:8, left : left + 4] = 1.0
    return video


def test_normalized_video_tensor_becomes_a_playable_mp4(tmp_path):
    output = tmp_path / "moving-square.mp4"

    result = save_video_mp4(_moving_square_video(), output, fps=5)

    assert result == output
    assert output.is_file() and output.stat().st_size > 0
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_type,width,height,nb_read_frames",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream == {
        "codec_type": "video",
        "width": 16,
        "height": 16,
        "nb_read_frames": "5",
    }


def test_denoising_states_become_an_animated_temporal_contact_sheet(tmp_path):
    clean = _moving_square_video(frames=4, size=16)
    trajectory = [torch.zeros_like(clean), clean * 0.5, clean]
    output = tmp_path / "denoising.gif"

    result = save_denoising_gif(trajectory, output, fps=4)

    assert result == output
    with Image.open(output) as preview:
        assert preview.is_animated
        assert preview.n_frames == 3
        # Each denoising state displays all four temporal frames side by side.
        assert preview.size == (64, 16)


def test_media_writes_leave_no_partial_temporary_files(tmp_path):
    video = _moving_square_video(frames=4, size=16)

    save_video_mp4(video, tmp_path / "atomic.mp4", fps=4)
    save_denoising_gif([torch.zeros_like(video), video], tmp_path / "atomic.gif")

    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]
