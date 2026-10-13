"""Registry: load, filter, reconcile. No torch, no torch hub, no audio.

These tests run against the bundled YAML and a temp directory. The
contract is that :class:`Registry.load_bundled` returns a registry
where ``len(all()) >= 5`` (current VR/MDX/Demucs catalog sample has at
least that many rows) and that :func:`synchronize_with_disk` correctly
transitions rows across disk state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from uvr_maestro.catalog import CATALOG_DIR
from uvr_maestro.registry import Registry


@pytest.fixture
def bundled_registry() -> Registry:
    """A fresh registry loaded from the shipped YAML files."""
    return Registry.load_bundled(CATALOG_DIR)


def test_load_bundled_returns_rows(bundled_registry: Registry) -> None:
    """``len(all())`` is at least the sample size shipped in 0.1.0."""
    rows = bundled_registry.all()
    assert len(rows) >= 5, f"expected at least 5 catalog rows, got {len(rows)}"


def test_load_bundled_endpoints_present(bundled_registry: Registry) -> None:
    """The endpoint catalog is a single row; it must load without error."""
    assert bundled_registry.endpoints.windows_setup.startswith("https://")
    assert bundled_registry.endpoints.model_repo_release_base.startswith("https://")


def test_filter_by_arch(bundled_registry: Registry) -> None:
    """``filter(arch=ModelArch.VR_5)`` returns only VR-5 rows."""
    from uvr_maestro.models import ModelArch

    rows = bundled_registry.filter(arch=ModelArch.VR_5)
    assert rows
    assert all(m.arch is ModelArch.VR_5 for m in rows)


def test_filter_query_matches_display_name(bundled_registry: Registry) -> None:
    """``filter(query='de-echo')`` matches FoxJoy's UVR-De-Echo rows."""
    rows = bundled_registry.filter(query="de-echo")
    assert rows, "expected at least one De-Echo row"
    assert any("De-Echo" in m.display_name for m in rows)


def test_synchronize_with_disk_marks_missing(tmp_path: Path, bundled_registry: Registry) -> None:
    """Empty dir means every non-Demucs row is Available, not Installed."""
    bundled_registry.synchronize_with_disk(tmp_path)
    supported = [model for model in bundled_registry.all() if model.supported]
    assert supported
    assert all(model.status.value == "available" for model in supported)
    assert not any(model.installed for model in supported)
    assert all(
        model.status.value == "unsupported"
        for model in bundled_registry.all()
        if not model.supported
    )


def test_synchronize_with_disk_marks_installed(tmp_path: Path, bundled_registry: Registry) -> None:
    """A file matching the size declared in the catalog row is Installed."""
    bundled_registry.synchronize_with_disk(tmp_path)
    target = bundled_registry.get("3_HP-Vocal-UVR")
    # Pretend the row is 1024 bytes; write a 1024-byte file with the right
    # extension.
    artifact = target.artifacts[0]
    artifact.size_bytes = 1024
    target_path = tmp_path / artifact.directory / artifact.filename
    target_path.parent.mkdir(parents=True)
    target_path.write_bytes(b"\0" * 1024)
    bundled_registry.synchronize_with_disk(tmp_path)
    assert target.status.value == "installed", target.status


def _ext(url: str) -> str:
    """Mirror :func:`uvr_maestro.engine.disk.extension_for_row` for the test."""
    if url.startswith("pytorch_hub:"):
        return ""
    from pathlib import Path

    return Path(url.split("?", 1)[0]).suffix.lower() or ".pth"


def test_runtime_capabilities_and_curated_config_survive_an_old_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refresh cannot revive an incompatible upstream Roformer checkpoint/config pair."""
    from uvr_maestro.registry import default_registry, reset_default_registry

    monkeypatch.setenv("UVR_MAESTRO_DATA_DIR", str(tmp_path / "data"))
    cached = Registry.load_bundled()
    model = cached.get("deverb_bs_roformer_8_256dim_8depth")
    model.download_url = "https://example.invalid/modified.ckpt"
    model.artifacts[0].url = model.download_url
    model.metadata.pop("roformer_variant", None)
    model.verified = False
    cached._settings = {}
    cache = tmp_path / "cache"
    cached.write_cache(cache / "catalog.json")
    reset_default_registry()
    try:
        current = default_registry(models_dir=tmp_path / "models", cache_dir=cache)
        restored = current.get(model.slug)
        assert restored.supported and restored.verified
        assert "/archive/" in restored.download_url
        assert any(item.key == "roformer_overlap" for item in current.settings())
        assert restored.artifacts[0].sha256
    finally:
        reset_default_registry()
