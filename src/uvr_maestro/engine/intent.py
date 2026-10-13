"""Explainable model selection by task, source and architecture."""

from __future__ import annotations

import re

from uvr_maestro.models import ModelArch, Stem
from uvr_maestro.registry import Registry
from uvr_maestro.spec import ModelSpec

STEM_KEYWORDS = {
    "vocals": Stem.VOCALS,
    "vocal": Stem.VOCALS,
    "voice": Stem.VOCALS,
    "singer": Stem.VOCALS,
    "acappella": Stem.VOCALS,
    "gesang": Stem.VOCALS,
    "stimme": Stem.VOCALS,
    "instrumental": Stem.INSTRUMENTAL,
    "music": Stem.INSTRUMENTAL,
    "karaoke": Stem.KARAOKE,
    "drums": Stem.DRUMS,
    "drum": Stem.DRUMS,
    "schlagzeug": Stem.DRUMS,
    "bass": Stem.BASS,
    "guitar": Stem.GUITAR,
    "piano": Stem.PIANO,
    "echo": Stem.DE_ECHO,
    "de-echo": Stem.DE_ECHO,
    "deecho": Stem.DE_ECHO,
    "reverb": Stem.DE_REVERB,
    "dereverb": Stem.DE_REVERB,
    "hall": Stem.DE_REVERB,
    "noise": Stem.DE_NOISE,
    "denoise": Stem.DE_NOISE,
    "rauschen": Stem.DE_NOISE,
    "backing": Stem.BACKING_VOCALS,
    "wind": Stem.WIND,
}
ARCH_KEYWORDS = {
    "vr": ModelArch.VR_5,
    "mdx": ModelArch.MDX,
    "kim": ModelArch.MDX,
    "mdxc": ModelArch.MDXC,
    "mdx23c": ModelArch.MDXC,
    "demucs": ModelArch.DEMUCS_V4,
    "htdemucs": ModelArch.DEMUCS_V4,
}


def rank_intent(intent: str, *, registry: Registry) -> list[tuple[ModelSpec, float]]:
    """Rank supported compatible models, refusing unrelated requests."""
    tokens = set(re.findall(r"[\w-]+", intent.casefold()))
    # Specific tasks outrank broad words such as vocal or music.
    targets = {stem for token, stem in STEM_KEYWORDS.items() if token in tokens}
    specific = targets - {Stem.VOCALS, Stem.INSTRUMENTAL}
    target = next(iter(sorted(specific or targets, key=lambda value: value.value)), None)
    arch = next((value for token, value in ARCH_KEYWORDS.items() if token in tokens), None)
    if target is None and arch is None:
        return []
    scored = []
    for row in registry.all():
        if not row.supported:
            continue
        compatible = target is None or (
            row.primary_stem is target
            or row.secondary_stem is target
            or target.value in row.purposes
            or target.value in [stem.casefold().replace(" ", "_") for stem in row.stems]
        )
        if not compatible or (arch is not None and row.arch is not arch):
            continue
        score = (
            2.0 if row.primary_stem is target or (target and target.value in row.purposes) else 1.0
        )
        if row.installed:
            score += 0.25
        if row.verified:
            score += 0.1
        scored.append((row, score))
    return sorted(scored, key=lambda item: (-item[1], item[0].slug))[:10]
