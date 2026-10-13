"""Client-visible schemas and shared parameter annotations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, model_validator

from uvr_maestro.models import ModelArch, SaveFormat, Stem

READS = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
)
ADDS = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False
)
DESTROYS = ToolAnnotations(
    readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False
)
NETWORK_READS = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True
)
NETWORK_WRITES = ToolAnnotations(
    readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=True
)
ArchArg = ModelArch
StemArg = Stem
SaveFormatArg = SaveFormat
AudioPath = Annotated[str, Field(description="Absolute or caller-relative local audio file path.")]
OutputDir = Annotated[
    str, Field(description="Output root. A unique run subdirectory prevents overwrites.")
]
ModelName = Annotated[
    str, Field(description="Catalog ID, upstream filename or alias from list_models.")
]
Confirm = Annotated[
    bool, Field(description="True executes the stated change. False returns its plan.")
]
Profile = Annotated[
    str | None, Field(description="Profile filename without extension. None uses default.")
]
JobId = Annotated[str, Field(description="Job ID returned by start_separation or an audio tool.")]


class SeparationOptions(BaseModel):
    """Explicit overrides. Omitted values retain the selected profile."""

    model_config = ConfigDict(extra="forbid")
    device: Literal["auto", "cpu", "cuda", "mps"] | None = Field(
        default=None,
        description="Requested backend: auto, cpu, cuda or mps. Actual use is reported.",
    )
    save_format: SaveFormat | None = Field(
        default=None, description="Output container: wav, mp3 or flac."
    )
    wav_subtype: Literal["PCM_16", "PCM_24", "FLOAT"] | None = Field(
        default=None, description="WAV encoding: PCM_16, PCM_24 or FLOAT."
    )
    mp3_bitrate: Literal["96k", "128k", "192k", "256k", "320k"] | None = Field(
        default=None, description="MP3 bitrate: 96k, 128k, 192k, 256k or 320k."
    )
    segment_size: int | None = Field(
        default=None, ge=64, le=4096, description="MDX/MDXC spectrogram frame count."
    )
    overlap: float | None = Field(
        default=None, ge=0, lt=1, description="Demucs segment overlap fraction, 0 through below 1."
    )
    overlap_mdx: float | None = Field(
        default=None, ge=0, lt=1, description="MDX window overlap fraction, 0 through below 1."
    )
    overlap_mdx23: int | None = Field(
        default=None, ge=2, le=32, description="MDXC overlap count, 2 through 32."
    )
    batch_size: int | None = Field(
        default=None, ge=1, le=16, description="VR/MDX/MDXC inference batch size, 1 through 16."
    )
    window_size: int | None = Field(
        default=None, ge=64, le=4096, description="VR inference window size in spectrogram frames."
    )
    aggression: int | None = Field(
        default=None, ge=0, le=100, description="VR mask aggression percent, 0 through 100."
    )
    tta: bool | None = Field(default=None, description="Enable VR test-time augmentation.")
    post_process: bool | None = Field(default=None, description="Enable VR mask post-processing.")
    post_process_threshold: float | None = Field(
        default=None, ge=0, le=1, description="VR post-processing mask threshold, 0 through 1."
    )
    high_end: bool | None = Field(default=None, description="Enable VR high-frequency mirroring.")
    shifts: int | None = Field(
        default=None, ge=0, le=10, description="Demucs random shift count, 0 through 10."
    )
    demucs_segment: float | None = Field(
        default=None, gt=0, le=30, description="Demucs segment length in seconds."
    )
    stem: str | None = Field(
        default=None, description="Demucs/MDXC source label, e.g. Vocals, or All Stems."
    )
    normalize_output: bool | None = Field(
        default=None, description="Apply UVR's peak normalization to each separated output."
    )
    primary_stem_only: bool | None = Field(
        default=None, description="Write only the model's primary output in a dual-stem run."
    )
    secondary_stem_only: bool | None = Field(
        default=None, description="Write only the complementary output in a dual-stem run."
    )
    denoise: bool | None = Field(
        default=None, description="MDX dual-pass noise reduction. Does not invoke a cleanup model."
    )
    is_invert_spec: bool | None = Field(
        default=None, description="Use spectral inversion for complementary MDX/MDXC stems."
    )
    semitone_shift: float | None = Field(
        default=None, ge=-24, le=24, description="Inference pitch offset in semitones."
    )
    is_match_frequency_pitch: bool | None = Field(
        default=None, description="Restore frequency alignment after inference pitch shifting."
    )
    mdxc_model_segment: bool | None = Field(
        default=None, description="Use MDXC's configured model segment length."
    )
    roformer_segment_seconds: float | None = Field(
        default=None, ge=0.1, le=30, description="Classic Roformer chunk duration in seconds."
    )
    roformer_overlap: float | None = Field(
        default=None, ge=0, lt=1, description="Classic Roformer overlap fraction."
    )

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        """Reject contradictory stem selection."""
        if self.primary_stem_only and self.secondary_stem_only:
            raise ValueError("primary-only and secondary-only cannot both be true")
        return self


class AudioContract(BaseModel):
    """Source-relative selection and lossless export guarantees."""

    model_config = ConfigDict(extra="forbid")
    start_seconds: float = Field(default=0, ge=0, description="Source start in seconds.")
    end_seconds: float | None = Field(
        default=None, gt=0, description="Exclusive source end in seconds. None means EOF."
    )
    output_sample_rate: int | None = Field(
        default=None, ge=8000, le=192000, description="Output Hz. None preserves the source rate."
    )
    exact_length: bool = Field(
        default=True, description="Require sample-exact lossless duration. MP3 must set False."
    )

    @model_validator(mode="after")
    def ordered(self) -> Self:
        """Refuse empty, reversed or non-finite selections."""
        import math

        if not math.isfinite(self.start_seconds) or (
            self.end_seconds is not None
            and (not math.isfinite(self.end_seconds) or self.end_seconds <= self.start_seconds)
        ):
            raise ValueError("selection must be finite with end greater than start")
        return self


class LocalModel(BaseModel):
    """Explicit local model artifacts; registration never moves the weights."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, description="Unique local identity without path separators.")
    architecture: ModelArch = Field(description="Supported model architecture.")
    weights_path: str = Field(description="Existing local checkpoint or ONNX path.")
    config_path: str | None = Field(default=None, description="MDX23C YAML configuration path.")
    parameters: dict[str, object] = Field(
        default_factory=dict, description="Explicit UVR parameters for unidentified VR/MDX weights."
    )
    stems: list[str] = Field(min_length=1, description="Actual emitted source labels.")
    primary_stem: Stem = Field(default=Stem.DRUMS, description="Task family for model lookup.")


class EnsembleEntry(BaseModel):
    """One source selected from one model for a compatible ensemble."""

    model_config = ConfigDict(extra="forbid")
    model: str = Field(description="Catalog ID or alias of an installed model.")
    stem: str = Field(default="Vocals", description="Actual output source label, e.g. Vocals.")
    weight: float = Field(default=1.0, gt=0, le=100, description="Positive averaging weight.")
    options: SeparationOptions | None = Field(
        default=None, description="This member's overrides, applied after common options."
    )


class ChainStep(BaseModel):
    """A cleanup or separation step with explicit output routing."""

    model_config = ConfigDict(extra="forbid")
    model: str = Field(description="Catalog ID or alias of an installed model.")
    output_stem: str = Field(
        description="Actual output label passed to the next step, e.g. Vocals or No Echo."
    )
    options: SeparationOptions | None = Field(
        default=None, description="Overrides for this step. Omit to use the selected profile."
    )


class ListModelsIn(BaseModel):
    """Compatibility schema for catalog consumers."""

    arch: ModelArch | None = None
    primary_stem: Stem | None = None
    only_installed: bool = False
    only_available: bool = False
    query: str | None = None
