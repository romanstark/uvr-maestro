"""Published MCP behavior, descriptions and refusal before mutations."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from uvr_maestro import server
from uvr_maestro.discovery import Locations
from uvr_maestro.registry import reset_default_registry
from uvr_maestro.toolargs import SeparationOptions
from uvr_maestro.workbench import Workbench


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Keep all operation tests away from an installed UVR application."""
    monkeypatch.setenv("UVR_MAESTRO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("UVR_MAESTRO_UVR_DIR", str(tmp_path / "no-desktop"))
    wb = Workbench(Locations.discover())
    monkeypatch.setattr(server, "_workbench", wb)
    reset_default_registry()
    yield wb
    wb.manager.close()
    reset_default_registry()


def test_status_without_engine_and_catalog_filters(isolated: Workbench, tmp_path: Path) -> None:
    """Catalog discovery remains useful before installing Torch."""
    status = server.uvr_status()
    assert status["success"] and not status["engine"]["ready"]
    rows = server.list_models(arch="mdx")["rows"]
    assert rows and all(row["arch"] == "mdx" for row in rows)
    assert server.list_models(primary_stem="vocals")["rows"]
    assert not list(tmp_path.iterdir())


def test_all_public_parameters_have_descriptions() -> None:
    """Check the actual client schemas, not the private Pydantic classes."""
    tools = asyncio.run(server.mcp.list_tools())
    assert len(tools) >= 28
    for tool in tools:
        assert tool.annotations is not None
        for name, parameter in tool.inputSchema.get("properties", {}).items():
            assert parameter.get("description"), f"{tool.name}.{name}"
    models = next(tool for tool in tools if tool.name == "list_models")
    serialized = json.dumps(models.inputSchema)
    assert "mdxc" in serialized and "demucs_v4" in serialized


def test_reset_and_settings_guard_has_no_side_effects(
    isolated: Workbench,
    tmp_path: Path,
) -> None:
    """Refusals preserve the exact existing profile and show its target."""
    assert server.set_settings({"segment_size": 512})["blocked"]
    assert not list(tmp_path.iterdir())
    result = server.set_settings({"segment_size": 512}, confirm=True)
    target = Path(result["path"])
    before = target.read_bytes()
    assert server.reset_settings()["blocked"]
    assert target.read_bytes() == before
    assert server.set_settings({"device": "cpu"}, confirm=True)["success"]
    assert server.get_settings()["values"]["segment_size"] == 512
    assert server.reset_settings(confirm=True)["success"] and not target.exists()


@pytest.mark.parametrize("name", ["../outside", "/absolute", "C:\\escape", "CON", "trail."])
def test_profile_paths_cannot_escape(isolated: Workbench, name: str, tmp_path: Path) -> None:
    """Malformed names fail before reading, writing or deleting any file."""
    assert not server.set_settings({"device": "cpu"}, name=name, confirm=True)["success"]
    assert not server.reset_settings(name=name, confirm=True)["success"]
    assert not list(tmp_path.iterdir())


def test_invalid_or_unavailable_processing_never_succeeds(
    isolated: Workbench,
    tmp_path: Path,
) -> None:
    """The former placeholders must not return completed audio on nonexistent inputs."""
    output = str(tmp_path / "output")
    calls = [
        server.start_separation("absent.wav", "Kim Vocal 2", output),
        server.pitch_shift("absent.wav", output, 2),
        server.time_stretch("absent.wav", output, 1.05),
        server.align_inputs("absent-a.wav", "absent-b.wav", output),
        server.matchering_inputs("absent-a.wav", "absent-b.wav", output),
        server.create_sample("absent.wav", output),
    ]
    assert all(not result["success"] for result in calls)
    assert not list(tmp_path.iterdir())


def test_invalid_option_combinations() -> None:
    """Reject mutually exclusive source selection and non-finite values."""
    with pytest.raises(ValueError):
        SeparationOptions(primary_stem_only=True, secondary_stem_only=True)
    with pytest.raises(ValueError):
        SeparationOptions.model_validate({"batch_size": 0})
    with pytest.raises(ValueError):
        SeparationOptions(overlap=float("nan"))


@pytest.mark.asyncio
async def test_refresh_works_through_async_mcp_dispatch(
    isolated: Workbench,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SDK calls handlers inside its event loop, unlike ordinary sync tests."""
    from uvr_maestro.engine import fetch

    async def upstream(url: str, **kwargs):
        if url == isolated.registry().endpoints.download_checks:
            return {"vr_download_list": {"VR v5: NetworkFixture": "NetworkFixture.pth"}}
        return {}

    monkeypatch.setattr(fetch, "fetch_json", upstream)
    await server.mcp.call_tool("refresh_catalog", {"confirm": True})
    assert (isolated.locations.cache / "catalog.json").is_file()
    assert isolated.registry().get("NetworkFixture").slug == "NetworkFixture"
    assert isolated.journal()[0]["state"] == "completed"
