"""Engine / settings: default merge, validation, atomic write."""

from __future__ import annotations

from pathlib import Path

import pytest

from uvr_maestro.catalog import CATALOG_DIR
from uvr_maestro.engine import settings as settings_mod
from uvr_maestro.registry import Registry


@pytest.fixture
def registry() -> Registry:
    return Registry.load_bundled(CATALOG_DIR)


def test_default_settings_includes_known_keys(registry: Registry) -> None:
    defaults = settings_mod.default_settings(registry)
    for key in ("segment_size", "overlap", "save_format", "gpu_conversion"):
        assert key in defaults, f"missing default for {key!r}"


def test_load_named_missing_file_returns_defaults(registry: Registry, tmp_path: Path) -> None:
    values = settings_mod.load_named(registry, settings_root=tmp_path, name="nope")
    assert values == settings_mod.default_settings(registry)


def test_write_then_load_round_trip(registry: Registry, tmp_path: Path) -> None:
    """``write_named`` followed by ``load_named`` returns the same dict."""
    payload = {"segment_size": 512, "save_format": "flac", "gpu_conversion": False}
    written = settings_mod.write_named(
        registry, settings_root=tmp_path, name="round_trip", values=payload
    )
    assert written.exists()
    loaded = settings_mod.load_named(registry, settings_root=tmp_path, name="round_trip")
    for k, v in payload.items():
        assert loaded[k] == v


def test_write_rejects_unknown_key(registry: Registry, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        settings_mod.write_named(
            registry,
            settings_root=tmp_path,
            name="invalid",
            values={"no_such_setting": 1},
        )
