"""Procedural prompt/video pairs for the small Video DiT teaching model.

The renderer deliberately uses a closed, inspectable vocabulary.  Every prompt
factor has a visible ground-truth effect, so a laptop-scale model can learn both
appearance and motion without pretending to understand unrestricted language.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


SUBJECTS = ("cat", "dog", "person")
COLORS = ("orange", "brown", "red", "blue", "black")
MOTIONS = ("walk_right", "walk_left", "bounce", "stand")
SCENES = ("snow", "grass", "night", "studio")

VOCABULARIES = (SUBJECTS, COLORS, MOTIONS, SCENES)


@dataclass(frozen=True)
class PromptSpec:
    subject: str
    color: str
    motion: str
    scene: str

    def __post_init__(self) -> None:
        values = (self.subject, self.color, self.motion, self.scene)
        for value, vocabulary in zip(values, VOCABULARIES, strict=True):
            if value not in vocabulary:
                raise ValueError(f"unsupported prompt value {value!r}; choose from {vocabulary}")

    def condition_ids(self) -> torch.Tensor:
        return torch.tensor(
            [vocabulary.index(value) for value, vocabulary in zip(
                (self.subject, self.color, self.motion, self.scene),
                VOCABULARIES,
                strict=True,
            )],
            dtype=torch.long,
        )

    def canonical_prompt(self) -> str:
        if self.subject == "person":
            subject = f"a person in a {self.color} coat"
        else:
            article = "an" if self.color[0] in "aeiou" else "a"
            subject = f"{article} {self.color} {self.subject}"
        motions = {
            "walk_right": "walking to the right",
            "walk_left": "walking to the left",
            "bounce": "bouncing",
            "stand": "standing still",
        }
        scenes = {
            "snow": "in the snow",
            "grass": "on grass",
            "night": "at night",
            "studio": "in a studio",
        }
        return f"{subject} {motions[self.motion]} {scenes[self.scene]}"


def parse_prompt(prompt: str) -> PromptSpec:
    """Map a small English/Chinese prompt grammar to explicit factors.

    Unknown adjectives are ignored, but the subject is required rather than
    silently substituted. This is a symbolic parser, not a pretrained language
    encoder; callers can inspect the returned :class:`PromptSpec` in the UI.
    """

    text = re.sub(r"\s+", " ", prompt.lower().strip())
    aliases = {
        "subject": {
            "cat": ("cat", "kitten", "猫"),
            "dog": ("dog", "puppy", "狗"),
            "person": ("person", "human", "man", "woman", "人物", "人"),
        },
        "color": {
            "orange": ("orange", "橙"),
            "brown": ("brown", "棕"),
            "red": ("red", "红"),
            "blue": ("blue", "蓝"),
            "black": ("black", "黑"),
        },
        "scene": {
            "snow": ("snow", "snowy", "雪"),
            "grass": ("grass", "field", "草"),
            "night": ("night", "moon", "夜"),
            "studio": ("studio", "plain", "影棚", "室内"),
        },
    }

    def choose(group: str, default: str) -> str:
        for value, words in aliases[group].items():
            if any(word in text for word in words):
                return value
        return default

    if any(word in text for word in ("left", "向左", "往左")):
        motion = "walk_left"
    elif any(word in text for word in ("bounce", "jump", "跳", "弹跳")):
        motion = "bounce"
    elif any(word in text for word in ("stand", "still", "静止", "站立")):
        motion = "stand"
    else:
        motion = "walk_right"

    subject = choose("subject", "")
    if not subject:
        raise ValueError(
            "prompt must name a supported subject: cat, dog, or person"
        )
    default_color = "red" if subject == "person" else "orange"
    return PromptSpec(
        subject=subject,
        color=choose("color", default_color),
        motion=motion,
        scene=choose("scene", "snow"),
    )


_RGB = {
    "orange": (232, 132, 40),
    "brown": (132, 82, 46),
    "red": (214, 43, 48),
    "blue": (42, 105, 210),
    "black": (36, 37, 43),
}


def _ellipse(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float], fill, outline=(30, 30, 35)) -> None:
    draw.ellipse(tuple(round(v) for v in box), fill=fill, outline=outline, width=1)


def _draw_cat(draw: ImageDraw.ImageDraw, x: float, y: float, scale: float, color, phase: float) -> None:
    _ellipse(draw, (x - 6*scale, y - 3*scale, x + 5*scale, y + 3*scale), color)
    _ellipse(draw, (x + 3*scale, y - 6*scale, x + 9*scale, y), color)
    draw.polygon([(x+3*scale,y-5*scale),(x+4*scale,y-9*scale),(x+6*scale,y-6*scale)], fill=color)
    draw.polygon([(x+6*scale,y-6*scale),(x+9*scale,y-9*scale),(x+9*scale,y-4*scale)], fill=color)
    eye_y = y - 3.5*scale
    _ellipse(draw, (x+5*scale, eye_y, x+6*scale, eye_y+scale), (245,245,235), outline=None)
    _ellipse(draw, (x+7*scale, eye_y, x+8*scale, eye_y+scale), (245,245,235), outline=None)
    leg = math.sin(phase) * 1.2 * scale
    draw.line((x-3*scale,y+2*scale,x-3*scale+leg,y+7*scale), fill=(35,30,30), width=max(1,round(scale)))
    draw.line((x+2*scale,y+2*scale,x+2*scale-leg,y+7*scale), fill=(35,30,30), width=max(1,round(scale)))
    draw.arc((x-10*scale,y-8*scale,x-3*scale,y+2*scale), 90, 270, fill=color, width=max(1,round(2*scale)))


def _draw_dog(draw: ImageDraw.ImageDraw, x: float, y: float, scale: float, color, phase: float) -> None:
    _ellipse(draw, (x - 7*scale, y - 3*scale, x + 5*scale, y + 3*scale), color)
    _ellipse(draw, (x + 3*scale, y - 6*scale, x + 10*scale, y + scale), color)
    draw.ellipse((round(x+3*scale),round(y-6*scale),round(x+6*scale),round(y-1*scale)), fill=(83,55,42))
    _ellipse(draw, (x+8*scale,y-3*scale,x+11*scale,y), (55,40,36), outline=None)
    _ellipse(draw, (x+6*scale,y-4*scale,x+7*scale,y-3*scale), (245,245,235), outline=None)
    leg = math.sin(phase) * 1.2 * scale
    for base, sign in ((-4,1),(2,-1)):
        draw.line((x+base*scale,y+2*scale,x+base*scale+sign*leg,y+7*scale), fill=(35,30,30), width=max(1,round(scale)))
    draw.line((x-6*scale,y-2*scale,x-10*scale,y-6*scale), fill=color, width=max(1,round(2*scale)))


def _draw_person(draw: ImageDraw.ImageDraw, x: float, y: float, scale: float, color, phase: float) -> None:
    skin = (221, 174, 133)
    _ellipse(draw, (x-2.5*scale,y-10*scale,x+2.5*scale,y-5*scale), skin)
    # The large colored trapezoid is intentionally unmistakable as a coat.
    draw.polygon([
        (x-3*scale,y-5*scale),(x+3*scale,y-5*scale),
        (x+5*scale,y+3*scale),(x-5*scale,y+3*scale),
    ], fill=color, outline=(35,30,35))
    swing = math.sin(phase) * 2.0 * scale
    draw.line((x-3*scale,y-3*scale,x-6*scale+swing,y+2*scale), fill=color, width=max(1,round(2*scale)))
    draw.line((x+3*scale,y-3*scale,x+6*scale-swing,y+2*scale), fill=color, width=max(1,round(2*scale)))
    draw.line((x-2*scale,y+3*scale,x-3*scale+swing,y+9*scale), fill=(39,48,70), width=max(1,round(2*scale)))
    draw.line((x+2*scale,y+3*scale,x+3*scale-swing,y+9*scale), fill=(39,48,70), width=max(1,round(2*scale)))


def _background(draw: ImageDraw.ImageDraw, scene: str, size: int, rng: np.random.Generator, frame: int) -> None:
    if scene == "snow":
        draw.rectangle((0, 0, size, size), fill=(168, 205, 232))
        draw.rectangle((0, round(size * .58), size, size), fill=(241, 246, 250))
        for i in range(max(8, size // 2)):
            x = int((rng.integers(0, size) + frame * (i % 3)) % size)
            y = int((rng.integers(0, size) + frame * (1 + i % 2)) % size)
            draw.point((x, y), fill=(255, 255, 255))
    elif scene == "grass":
        draw.rectangle((0, 0, size, size), fill=(128, 196, 236))
        draw.rectangle((0, round(size * .58), size, size), fill=(90, 166, 69))
    elif scene == "night":
        draw.rectangle((0, 0, size, size), fill=(18, 29, 65))
        draw.rectangle((0, round(size * .65), size, size), fill=(33, 44, 59))
        _ellipse(draw, (size*.72,size*.10,size*.87,size*.25), (244,235,169), outline=None)
    else:
        draw.rectangle((0, 0, size, size), fill=(225, 215, 204))
        draw.line((0, round(size*.72), size, round(size*.72)), fill=(140,125,115), width=1)


def render_video(
    spec: PromptSpec,
    frames: int = 8,
    size: int = 32,
    seed: int = 0,
) -> torch.Tensor:
    """Render a true temporal clip with shape ``[C, T, H, W]`` in ``[-1, 1]``."""

    if frames < 2:
        raise ValueError("frames must be at least 2")
    if size < 24:
        raise ValueError("size must be at least 24 pixels")
    if not isinstance(spec, PromptSpec):
        raise TypeError("spec must be a PromptSpec")

    output: list[np.ndarray] = []
    scale = size / 32.0
    color = _RGB[spec.color]
    for frame in range(frames):
        # A per-frame deterministic stream makes repeated calls bit-identical.
        rng = np.random.default_rng(seed * 10_007 + frame)
        image = Image.new("RGB", (size, size))
        draw = ImageDraw.Draw(image)
        _background(draw, spec.scene, size, rng, frame)

        progress = frame / (frames - 1)
        if spec.motion == "walk_right":
            x = size * (.25 + .48 * progress)
            y = size * .63 + math.sin(progress * math.tau * 2) * scale
        elif spec.motion == "walk_left":
            x = size * (.73 - .48 * progress)
            y = size * .63 + math.sin(progress * math.tau * 2) * scale
        elif spec.motion == "bounce":
            x = size * .5
            y = size * .64 - abs(math.sin(progress * math.tau)) * size * .25
        else:
            x, y = size * .5, size * .63
        phase = progress * math.tau * 2
        {"cat": _draw_cat, "dog": _draw_dog, "person": _draw_person}[spec.subject](
            draw, x, y, scale, color, phase
        )
        output.append(np.asarray(image, dtype=np.uint8).copy())

    array = np.stack(output, axis=0)
    return torch.from_numpy(array).permute(3, 0, 1, 2).float().div(127.5).sub(1.0)


class SyntheticVideoDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Infinite-style deterministic prompt/video curriculum indexed by integer."""

    def __init__(
        self,
        length: int = 50_000,
        frames: int = 8,
        size: int = 32,
        base_seed: int = 0,
    ) -> None:
        if length <= 0:
            raise ValueError("length must be positive")
        self.length = length
        self.frames = frames
        self.size = size
        self.base_seed = base_seed

    def __len__(self) -> int:
        return self.length

    def spec_for_index(self, index: int) -> PromptSpec:
        if not 0 <= index < self.length:
            raise IndexError(index)
        rng = np.random.default_rng(self.base_seed + index)
        return PromptSpec(*(v[int(rng.integers(len(v)))] for v in VOCABULARIES))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        spec = self.spec_for_index(index)
        return {
            "video": render_video(spec, self.frames, self.size, self.base_seed + index),
            "condition": spec.condition_ids(),
            "prompt": spec.canonical_prompt(),
        }


CANONICAL_SPECS = (
    PromptSpec("cat", "orange", "walk_right", "snow"),
    PromptSpec("dog", "brown", "walk_left", "grass"),
    PromptSpec("person", "red", "walk_right", "snow"),
)


class CanonicalVideoDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Three fixed clips for an explicit, fast end-to-end overfit check.

    This curriculum is intentionally not a diversity dataset. Repeating one
    deterministic clip per subject lets a learner establish that training,
    conditioning, temporal generation, and MP4 inference can produce a
    recognizable result before attempting the harder factorial distribution.
    """

    def __init__(
        self,
        length: int = 50_000,
        frames: int = 8,
        size: int = 32,
        base_seed: int = 0,
    ) -> None:
        if length <= 0:
            raise ValueError("length must be positive")
        self.length = length
        self.frames = frames
        self.size = size
        self.base_seed = base_seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        if not 0 <= index < self.length:
            raise IndexError(index)
        canonical_index = index % len(CANONICAL_SPECS)
        spec = CANONICAL_SPECS[canonical_index]
        return {
            "video": render_video(
                spec,
                self.frames,
                self.size,
                self.base_seed + canonical_index,
            ),
            "condition": spec.condition_ids(),
            "prompt": spec.canonical_prompt(),
        }
