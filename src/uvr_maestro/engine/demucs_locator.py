"""Resolve local Demucs artifact bags without implicit Hub downloads."""

from pathlib import Path

from uvr_maestro.models import ModelSpec, ModelStatus
from uvr_maestro.registry import default_registry


def demucs_checkpoint_path(slug: str) -> Path | None:
    """Return the resolved YAML bag path for an installed supported model."""
    try:
        return default_registry().get(slug).path
    except KeyError:
        return None


def row_disk_path(row: ModelSpec) -> Path | None:
    """Return a row's already reconciled local bag path."""
    return row.path


def row_status(row: ModelSpec) -> ModelStatus:
    """Return complete-artifact state rather than guessing a Hub filename."""
    return row.status
