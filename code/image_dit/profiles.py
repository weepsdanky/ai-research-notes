"""Named diffusion recipes that keep teaching defaults separate from paper parity."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DiffusionProfile:
    name: str
    vb_mode: str
    loss_clip_denoised: bool
    sampling_clip_denoised: bool
    cfg_channels: str
    sampler: str
    sampling_steps: int
    max_grad_norm: float | None
    amp: bool

    def metadata(self) -> dict[str, object]:
        return asdict(self)


PROFILES = {
    "pixel_teaching": DiffusionProfile(
        name="pixel_teaching",
        vb_mode="rescaled",
        loss_clip_denoised=True,
        sampling_clip_denoised=True,
        cfg_channels="all",
        sampler="ddim",
        sampling_steps=50,
        max_grad_norm=1.0,
        amp=True,
    ),
    "paper_compatible": DiffusionProfile(
        name="paper_compatible",
        vb_mode="raw",
        loss_clip_denoised=False,
        sampling_clip_denoised=False,
        cfg_channels="first_3",
        sampler="paper_ddpm",
        sampling_steps=250,
        max_grad_norm=None,
        amp=False,
    ),
}


def get_profile(name: str) -> DiffusionProfile:
    try:
        return PROFILES[name]
    except KeyError as error:
        raise ValueError(f"unknown profile {name!r}; choose one of {tuple(PROFILES)}") from error
