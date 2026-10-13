"""Value types shared by the catalog, executor and worker."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class OSType(str, Enum):
    """Supported host operating systems."""

    WINDOWS = "windows"
    MACOS = "darwin"
    LINUX = "linux"


def detect_os() -> OSType:
    """Return the host OS, allowing a test override."""
    return OSType(os.environ.get("UVR_MAESTRO_OS", platform.system().lower()))


class ModelArch(str, Enum):
    """Engine architecture, independent of a model's display name."""

    VR_5 = "vr_5"
    VR_4 = "vr_4"
    MDX = "mdx"
    MDXC = "mdxc"
    DEMUCS = "demucs"
    DEMUCS_V3 = "demucs_v3"
    DEMUCS_V4 = "demucs_v4"
    LEGACY_DEMUCS = "legacy_demucs"
    ROFORMER = "roformer"
    UNKNOWN = "unknown"


SUPPORTED_ARCHES = {
    ModelArch.VR_5,
    ModelArch.VR_4,
    ModelArch.MDX,
    ModelArch.MDXC,
    ModelArch.DEMUCS_V3,
    ModelArch.DEMUCS_V4,
    ModelArch.ROFORMER,
}


class Stem(str, Enum):
    """Common source and restoration targets."""

    VOCALS = "vocals"
    INSTRUMENTAL = "instrumental"
    DRUMS = "drums"
    BASS = "bass"
    OTHER = "other"
    GUITAR = "guitar"
    PIANO = "piano"
    KARAOKE = "karaoke"
    DE_ECHO = "de_echo"
    DE_REVERB = "de_reverb"
    DE_NOISE = "de_noise"
    WIND = "wind"
    LEAD_VOCALS = "lead_vocals"
    BACKING_VOCALS = "backing_vocals"

    @classmethod
    def family(cls) -> list[str]:
        """Return the standard source families."""
        return [
            cls.VOCALS.value,
            cls.INSTRUMENTAL.value,
            cls.DRUMS.value,
            cls.BASS.value,
            cls.OTHER.value,
        ]


class SaveFormat(str, Enum):
    """Output containers."""

    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"


class ModelStatus(str, Enum):
    """Artifact installation state, distinct from integration verification."""

    INSTALLED = "installed"
    AVAILABLE = "available"
    BROKEN = "broken"
    UNSUPPORTED = "unsupported"


@dataclass
class Artifact:
    """One immutable upstream filename and its managed relative directory."""

    filename: str
    directory: str
    url: str
    role: str = "weights"
    size_bytes: int = 0
    sha256: str | None = None


@dataclass
class ModelSpec:
    """One catalog identity, capabilities and complete artifact set."""

    slug: str
    display_name: str
    arch: ModelArch
    primary_stem: Stem
    secondary_stem: Stem | None
    sample_rate: int
    download_url: str
    size_bytes: int = 0
    sidecar_url: str | None = None
    verified: bool = False
    notes: str | None = None
    path: Path | None = None
    status: ModelStatus = ModelStatus.AVAILABLE
    artifacts: list[Artifact] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    purposes: list[str] = field(default_factory=list)
    stems: list[str] = field(default_factory=list)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    owned: bool = False

    @property
    def installed(self) -> bool:
        """Return whether all required artifacts were resolved."""
        return self.status is ModelStatus.INSTALLED

    @property
    def supported(self) -> bool:
        """Return whether this runtime implements the architecture."""
        if self.arch is ModelArch.ROFORMER:
            return self.metadata.get("roformer_variant") in ("mel_band", "band_split")
        return self.arch in SUPPORTED_ARCHES


ModelRow = ModelSpec


@dataclass
class PipelineSpec:
    """Resolved configuration sent to a single inference worker."""

    model_row: ModelSpec
    settings: dict[str, Any]
    overrides: dict[str, Any]
    output_dir: Path = Path(".")
    run_id: str = ""


@dataclass
class RunManifest:
    """Durable evidence about a completed run."""

    tool: str
    model_slug: str
    model_arch: str
    primary_stem: str
    secondary_stem: str | None
    input_sha256: str
    inputs: list[str]
    outputs: list[str]
    settings: dict[str, Any]
    gpu: str
    runtime_seconds: float
    fallback_to_cpu: bool
    version: str
    fallback_reason: str | None = None
    model_sha256: str | None = None
    engine_revision: str = ""
    audio: list[dict[str, Any]] = field(default_factory=list)
