"""Application executor ownership, preflight and truthful confirmation plans."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from uvr_maestro.discovery import Locations
from uvr_maestro.executor import Workbench
from uvr_maestro.models import Artifact, ModelArch, ModelSpec, Stem
from uvr_maestro.registry import Registry
from uvr_maestro.storage import MaestroError


@pytest.mark.parametrize("platform,suffix", [("darwin", ""), ("win32", ".exe")])
def test_desktop_codec_discovery_without_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str, suffix: str
) -> None:
    """Reuse platform-native bundled codecs without modifying the installation."""
    from uvr_maestro import executor as module

    desktop = tmp_path / "desktop"
    desktop.mkdir()
    locations = Locations(
        tmp_path, tmp_path / "models", tmp_path / "profiles", tmp_path / "cache",
        tmp_path / "runtime", desktop,
    )
    workbench = Workbench(locations)
    monkeypatch.setattr(module.shutil, "which", lambda _: None)
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform=platform))
    try:
        assert workbench.executables() == {}
        expected = {}
        for name in ("ffmpeg", "rubberband"):
            path = desktop / (name + suffix)
            path.write_bytes(b"owned codec fixture")
            expected[name] = str(path)
        assert workbench.executables() == expected
        assert all(Path(path).read_bytes() == b"owned codec fixture" for path in expected.values())
    finally:
        workbench.manager.close()


def wb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Workbench:
    locations = Locations(
        tmp_path,
        tmp_path / "models",
        tmp_path / "profiles",
        tmp_path / "cache",
        tmp_path / "runtime",
        None,
    )
    result = Workbench(locations)
    registry = Registry.load_bundled()
    monkeypatch.setattr(result, "registry", lambda: registry)
    return result


def test_download_plan_has_actual_url_and_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = wb(tmp_path, monkeypatch)
    try:
        result = executor.download("Kim Vocal 2", False)
        assert result["blocked"] and not result["success"]
        assert result["plan"][0]["path"].endswith("MDX_Net_Models\\Kim_Vocal_2.onnx") or (
            result["plan"][0]["path"].endswith("MDX_Net_Models/Kim_Vocal_2.onnx")
        )
        assert result["plan"][0]["url"].endswith("Kim_Vocal_2.onnx")
        assert not list(tmp_path.iterdir())
    finally:
        executor.manager.close()


def test_reused_model_cannot_be_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = wb(tmp_path, monkeypatch)
    row = executor.registry().get("3_HP-Vocal-UVR")
    artifact = row.artifacts[0]
    reused = tmp_path / "reused"
    path = reused / artifact.directory / artifact.filename
    path.parent.mkdir(parents=True)
    path.write_bytes(b"model")
    executor.registry().synchronize_with_disk(executor.locations.models, [reused])
    try:
        with pytest.raises(MaestroError, match="cannot be removed"):
            executor.remove(row.slug, True)
        assert path.read_bytes() == b"model"
    finally:
        executor.manager.close()


def test_shared_checkpoint_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = wb(tmp_path, monkeypatch)
    endpoints = executor.registry().endpoints
    checkpoint = Artifact("shared.th", "Demucs_Models/v3_v4_repo", "https://example.test/shared")
    bag = Artifact("one.yaml", checkpoint.directory, "https://example.test/one", role="bag")
    one = ModelSpec(
        "one",
        "One",
        ModelArch.DEMUCS_V4,
        Stem.VOCALS,
        Stem.INSTRUMENTAL,
        44100,
        bag.url,
        artifacts=[checkpoint, bag],
    )
    two = ModelSpec(
        "two",
        "Two",
        ModelArch.DEMUCS_V4,
        Stem.VOCALS,
        Stem.INSTRUMENTAL,
        44100,
        bag.url,
        artifacts=[checkpoint],
    )
    registry = Registry([one, two], [], endpoints)
    monkeypatch.setattr(executor, "registry", lambda: registry)
    for artifact in one.artifacts:
        path = executor.locations.models / artifact.directory / artifact.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model")
    registry.synchronize_with_disk(executor.locations.models)
    shared = executor.locations.models / checkpoint.directory / checkpoint.filename
    try:
        assert executor.remove("one", False)["blocked"]
        assert executor.remove("one", True)["success"]
        assert shared.exists()
        assert not (shared.parent / "one.yaml").exists()
    finally:
        executor.manager.close()
