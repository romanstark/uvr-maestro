"""Validate catalog rows and typed persisted settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from uvr_maestro.models import Artifact, ModelArch, ModelSpec, Stem
from uvr_maestro.storage import safe_name


@dataclass
class SettingSpec:
    """A scalar setting with optional choices and numeric bounds."""

    key: str
    type: str
    default: Any
    enum: list[Any] | None = None
    doc: str | None = None
    scope: str = "session"
    minimum: float | None = None
    maximum: float | None = None


@dataclass
class EndpointSpec:
    """Source URLs and explicit download policy."""

    vr_catalog: str
    mdx_catalog: str
    mdx_model_names: str
    demucs_model_names: str
    mdx23_configs_base: str
    model_repo_release_base: str
    update_patch_repo_base: str
    bulletin: str
    download_checks: str
    mac_arm_installer: str
    mac_x86_installer: str
    windows_setup: str
    linux_source: str
    bulletin_refresh_seconds: int = 21600
    catalog_refresh_seconds: int = 86400
    download_chunk_bytes: int = 65536
    download_timeout_seconds: int = 300


def build_row(entry: dict[str, Any]) -> ModelSpec:
    """Build a row, refusing invalid identifiers and architecture values."""
    try:
        slug = safe_name(entry["slug"])
        arch = ModelArch(entry["architecture"])
        primary = Stem(entry["primary_stem"])
        secondary = Stem(entry["secondary_stem"]) if entry.get("secondary_stem") else None
        rate = int(entry["sample_rate"])
        url = str(entry["download_url"])
        display_name = str(entry["display_name"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid catalog row: {error}") from error
    if rate not in (32000, 44100):
        raise ValueError(f"unsupported model sample rate: {rate}")
    artifacts = []
    for raw in entry.get("artifacts", []):
        artifact = Artifact(**raw)
        safe_name(artifact.filename)
        parts = artifact.directory.replace("\\", "/").split("/")
        if any(part in ("", ".", "..") for part in parts):
            raise ValueError("invalid artifact directory")
        if artifact.size_bytes < 0:
            raise ValueError("negative artifact size")
        artifacts.append(artifact)
    size = int(entry.get("size_bytes", 0))
    if size < 0:
        raise ValueError("negative model size")
    return ModelSpec(
        slug=slug,
        display_name=display_name,
        arch=arch,
        primary_stem=primary,
        secondary_stem=secondary,
        sample_rate=rate,
        download_url=url,
        size_bytes=size,
        sidecar_url=entry.get("sidecar_url"),
        verified=bool(entry.get("verified", False)),
        notes=entry.get("notes"),
        artifacts=artifacts,
        metadata=entry.get("metadata", {}),
        aliases=entry.get("aliases", []),
        purposes=entry.get("purposes", []),
        stems=entry.get("stems", []),
    )


def build_endpoint(entry: dict[str, Any]) -> EndpointSpec:
    """Build the endpoint bundle."""
    return EndpointSpec(**entry["endpoints"], **entry.get("defaults", {}))


def build_setting(entry: dict[str, Any]) -> SettingSpec:
    """Build and validate a setting's default."""
    result = SettingSpec(**entry)
    validate_setting_value(result, result.default)
    return result


def validate_setting_value(spec: SettingSpec, value: Any) -> None:
    """Refuse values outside a setting's type, choices or bounds."""
    types = {"int": int, "float": (float, int), "str": str, "bool": bool}
    if spec.type == "segment":
        if value == "Default":
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 30:
            raise ValueError("segment must be Default or seconds above 0 through 30")
        return
    if spec.type not in (*types, "enum"):
        raise ValueError(f"unknown setting type: {spec.type}")
    if spec.enum is not None and value not in spec.enum:
        raise ValueError(f"setting {spec.key}: expected one of {spec.enum}")
    if spec.type != "enum":
        if not isinstance(value, types[spec.type]):
            raise ValueError(f"setting {spec.key}: expected {spec.type}")
        if spec.type in ("int", "float") and isinstance(value, bool):
            raise ValueError(f"setting {spec.key}: boolean is not numeric")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        import math

        if not math.isfinite(value):
            raise ValueError(f"setting {spec.key}: must be finite")
        if spec.minimum is not None and value < spec.minimum:
            raise ValueError(f"setting {spec.key}: below {spec.minimum}")
        if spec.maximum is not None and value > spec.maximum:
            raise ValueError(f"setting {spec.key}: above {spec.maximum}")
