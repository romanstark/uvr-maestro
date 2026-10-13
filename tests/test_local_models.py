"""Owner registrations survive upstream replacement and never own the original weights."""

from pathlib import Path

import pytest

from uvr_maestro.discovery import Locations
from uvr_maestro.executor import Workbench
from uvr_maestro.models import ModelArch
from uvr_maestro.registry import Registry, reset_default_registry
from uvr_maestro.storage import MaestroError
from uvr_maestro.toolargs import LocalModel


def test_registration_refresh_modified_artifact_and_unregister(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("UVR_MAESTRO_DATA_DIR", str(tmp_path / "data"))
    reset_default_registry()
    wb = Workbench(Locations.discover())
    weights = tmp_path / "custom.ckpt"
    weights.write_bytes(b"fixture checkpoint")
    config = tmp_path / "custom.yaml"
    config.write_text("audio: {}\nmodel: {}\ntraining:\n  instruments: [Kick, Snare]\n")
    spec = LocalModel(
        name="local-drums",
        architecture=ModelArch.MDXC,
        weights_path=str(weights),
        config_path=str(config),
        stems=["Kick", "Snare"],
    )
    try:
        assert wb.register(spec, False)["blocked"]
        assert not wb.locations.cache.exists()
        assert wb.register(spec, True)["success"]
        Registry.load_bundled().write_cache(wb.locations.cache / "catalog.json")
        reset_default_registry()
        assert wb.model("local-drums")["stems"] == ["Kick", "Snare"]
        assert not wb.model("local-drums")["owned"]
        with pytest.raises(MaestroError, match="reused"):
            wb.remove("local-drums", True)
        weights.write_bytes(b"modified checkpoint")
        with pytest.raises(MaestroError, match="registered artifact changed"):
            wb.model("local-drums")
        assert wb.unregister("local-drums", False)["blocked"]
        assert wb.unregister("local-drums", True)["success"]
        assert weights.read_bytes() == b"modified checkpoint" and config.exists()
        with pytest.raises(KeyError):
            wb.registry().get("local-drums")
    finally:
        wb.manager.close()
        reset_default_registry()
