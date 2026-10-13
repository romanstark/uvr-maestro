"""Validated profiles stored separately from UVR's desktop settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from uvr_maestro.registry import Registry
from uvr_maestro.spec import validate_setting_value
from uvr_maestro.storage import atomic_json, contained, safe_name


def profile_path(settings_root: Path, name: str | None) -> Path:
    """Resolve a named profile without permitting directory traversal."""
    return contained(settings_root, settings_root / f"{safe_name(name or 'default')}.json")


def default_settings(registry: Registry) -> dict[str, Any]:
    """Return the declared default setting values."""
    return {s.key: s.default for s in registry.settings()}


def validate_map(registry: Registry, values: dict[str, Any]) -> None:
    """Validate every setting name and value."""
    for key, value in values.items():
        try:
            setting = registry.setting(key)
        except KeyError as error:
            raise ValueError(f"unknown setting {key!r}") from error
        validate_setting_value(setting, value)


def load_named(registry: Registry, *, settings_root: Path, name: str | None) -> dict[str, Any]:
    """Resolve defaults overlaid by a named profile, refusing corrupted files."""
    target = profile_path(settings_root, name)
    values = default_settings(registry)
    if target.exists():
        data = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("profile must be a JSON object")
        validate_map(registry, data)
        values.update(data)
    return values


def write_named(
    registry: Registry, *, settings_root: Path, name: str | None, values: dict[str, Any]
) -> Path:
    """Merge changes into a profile and atomically persist them."""
    validate_map(registry, values)
    merged = load_named(registry, settings_root=settings_root, name=name)
    merged.update(values)
    target = profile_path(settings_root, name)
    atomic_json(target, merged)
    return target


def reset_named(*, settings_root: Path, name: str | None) -> Path | None:
    """Remove a confined profile after the executor has authorized it."""
    target = profile_path(settings_root, name)
    if not target.exists():
        return None
    target.unlink()
    return target


def list_saved(settings_root: Path) -> list[str]:
    """List profiles without creating storage."""
    return sorted(p.stem for p in settings_root.glob("*.json") if p.is_file())
