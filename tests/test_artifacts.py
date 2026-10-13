"""Artifact completeness, ownership and failure-safe downloads."""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest

from uvr_maestro.catalog_sync import parse_downloads
from uvr_maestro.engine import fetch
from uvr_maestro.models import Artifact, ModelArch, ModelSpec, Stem
from uvr_maestro.registry import Registry
from uvr_maestro.storage import contained


def model(artifacts: list[Artifact]) -> ModelSpec:
    """Create a small, synthetic public-model fixture."""
    return ModelSpec(
        "fixture",
        "Fixture",
        ModelArch.MDX,
        Stem.VOCALS,
        Stem.INSTRUMENTAL,
        44100,
        "https://example.test/weights.onnx",
        artifacts=artifacts,
    )


def test_real_filenames_and_demucs_bag() -> None:
    """The shipped catalog represents weight bags and sidecars explicitly."""
    registry = Registry.load_bundled()
    assert registry.get("Kim Vocal 2").artifacts[0].filename == "Kim_Vocal_2.onnx"
    bag = registry.get("htdemucs_ft")
    assert len([item for item in bag.artifacts if item.role == "weights"]) == 4
    assert any(item.role == "bag" for item in bag.artifacts)
    mdxc = registry.get("MDX23C-8KFFT-InstVoc_HQ")
    assert mdxc.arch is ModelArch.MDXC
    assert any(item.filename.endswith(".yaml") for item in mdxc.artifacts)


def test_complete_artifact_sets_and_reused_ownership(tmp_path: Path) -> None:
    """A YAML bag alone is not an installed Demucs model."""
    registry = Registry.load_bundled()
    row = registry.get("htdemucs")
    owned, reused = tmp_path / "owned", tmp_path / "reused"
    artifact = next(item for item in row.artifacts if item.role == "bag")
    path = reused / artifact.directory / artifact.filename
    path.parent.mkdir(parents=True)
    path.write_text("models: [955717e8]")
    registry.synchronize_with_disk(owned, [reused])
    assert not row.installed
    for artifact in row.artifacts:
        (reused / artifact.directory / artifact.filename).write_bytes(b"fixture")
    registry.synchronize_with_disk(owned, [reused])
    assert row.installed and not row.owned


def test_download_preserves_old_bytes_on_corrupt_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed replacement cannot destroy a previously valid model file."""
    data = b"correct model bytes"
    artifact = Artifact(
        "weights.onnx",
        "MDX_Net_Models",
        "https://example.test/weights.onnx",
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    row = model([artifact])
    target = tmp_path / artifact.directory / artifact.filename
    target.parent.mkdir()
    target.write_bytes(data)
    original_client = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"corrupt"))
    monkeypatch.setattr(
        fetch.httpx, "Client", lambda **kwargs: original_client(transport=transport)
    )
    with pytest.raises(ValueError):
        fetch.download_one(row, dest_dir=tmp_path, force=True)
    assert target.read_bytes() == data
    assert not list(target.parent.glob(".download-*"))


def test_partial_existing_file_is_repaired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing partial data is not treated as a successfully installed download."""
    data = b"complete weights"
    artifact = Artifact(
        "weights.onnx",
        "MDX_Net_Models",
        "https://example.test/weights.onnx",
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    target = tmp_path / artifact.directory / artifact.filename
    target.parent.mkdir()
    target.write_bytes(b"x")
    original_client = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=data))
    monkeypatch.setattr(
        fetch.httpx, "Client", lambda **kwargs: original_client(transport=transport)
    )
    result = fetch.download_one(model([artifact]), dest_dir=tmp_path)
    assert result["ok"] and target.read_bytes() == data


def test_refresh_commits_new_data_and_preserves_last_valid_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real added row enters the cache, while invalid upstream data never replaces it."""
    registry = Registry.load_bundled()
    endpoints = registry.endpoints
    target = tmp_path / "catalog.json"

    async def valid(url: str, **kwargs):
        if url == endpoints.download_checks:
            return {"vr_download_list": {"VR Arch Single Model v5: Fresh": "Fresh.pth"}}
        return {}

    monkeypatch.setattr(fetch, "fetch_json", valid)
    result = fetch.refresh_catalog(cache_path=target, endpoints=endpoints)
    assert "Fresh" in result["added"]
    loaded = Registry.load_cache(target)
    assert loaded is not None and loaded.get("Fresh").slug == "Fresh"
    before = target.read_bytes()

    async def invalid(url: str, **kwargs):
        return {"unexpected": []}

    monkeypatch.setattr(fetch, "fetch_json", invalid)
    with pytest.raises(TypeError):
        fetch.refresh_catalog(cache_path=target, endpoints=endpoints)
    assert target.read_bytes() == before


def test_structurally_invalid_cache_falls_back(tmp_path: Path) -> None:
    """Syntactically valid but malformed JSON does not break server initialization."""
    cache = tmp_path / "catalog.json"
    for payload in ("[]", '{"schema_version": 1, "models": null}', '{"schema_version": 999}'):
        cache.write_text(payload)
        assert Registry.load_cache(cache) is None


def test_confinement_checks_resolved_symlinks(tmp_path: Path) -> None:
    """Path containment also rejects absolute and parent-relative escapes."""
    with pytest.raises(ValueError):
        contained(tmp_path / "models", tmp_path / "outside.onnx")


def test_new_architecture_is_disclosed_as_unsupported() -> None:
    """New upstream families cannot become executable through a YAML row alone."""
    endpoints = Registry.load_bundled().endpoints
    rows = parse_downloads(
        {
            "vr_download_list": {},
            "other_network_list": {
                "Roformer: future": {"future.ckpt": "https://example.test/future"}
            },
        },
        endpoints,
    )
    assert len(rows) == 1 and not rows[0].supported
