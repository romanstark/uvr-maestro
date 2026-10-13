"""Independent local registrations that remain read-only and survive catalog refresh."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from uvr_maestro.engine.config import classic_roformer_variant, load_config
from uvr_maestro.models import SUPPORTED_ARCHES, ModelArch, ModelSpec, ModelStatus, Stem
from uvr_maestro.storage import MaestroError, atomic_json, safe_name, sha256
from uvr_maestro.toolargs import LocalModel


def records(cache: Path) -> dict[str, dict[str, Any]]:
    """Load owner records; malformed registrations must not silently disappear."""
    path = cache / "local-models.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("models"), dict):
        raise MaestroError("invalid_local_models", "local registrations are malformed")
    return data["models"]


def prepare(spec: LocalModel) -> dict[str, Any]:
    """Validate artifacts/config and fingerprint their actual contents."""
    safe_name(spec.name)
    for stem in spec.stems:
        safe_name(stem)
    if len({stem.casefold() for stem in spec.stems}) != len(spec.stems):
        raise ValueError("stem labels must be unique")
    if spec.architecture not in SUPPORTED_ARCHES:
        raise MaestroError(
            "architecture_unsupported", "registration requires an implemented adapter"
        )
    weights = Path(spec.weights_path).expanduser().resolve()
    if not weights.is_file() or weights.stat().st_size == 0:
        raise ValueError("weights must be an existing nonempty file")
    parameters = dict(spec.parameters)
    paths = {weights.name: str(weights)}
    if spec.architecture in (ModelArch.MDXC, ModelArch.ROFORMER):
        if not spec.config_path:
            raise ValueError("MDX23C requires an explicit YAML config")
        config = Path(spec.config_path).expanduser().resolve()
        data = load_config(config.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not all(
            key in data for key in ("audio", "model", "training")
        ):
            raise ValueError("MDX23C config requires audio, model and training objects")
        instruments = data["training"].get("instruments", [])
        if not instruments or not set(spec.stems).issubset(set(instruments)):
            raise ValueError("declared MDX23C stems must occur in training.instruments")
        parameters.update(config_yaml=config.name)
        if spec.architecture is ModelArch.ROFORMER:
            parameters["roformer_variant"] = classic_roformer_variant(data)
        paths[config.name] = str(config)
    elif spec.architecture in (ModelArch.DEMUCS_V3, ModelArch.DEMUCS_V4):
        raise ValueError("custom Demucs bags require catalog registration with all checkpoints")
    elif not parameters:
        from uvr_maestro.engine.uvr_bridge import metadata_for

        parameters = metadata_for(weights, spec.architecture.value)
    return {
        "name": spec.name,
        "architecture": spec.architecture.value,
        "weights_path": str(weights),
        "parameters": parameters,
        "stems": spec.stems,
        "primary_stem": spec.primary_stem.value,
        "artifact_paths": paths,
        "fingerprints": {name: sha256(Path(path)) for name, path in paths.items()},
    }


def save(cache: Path, models: dict[str, dict[str, Any]]) -> None:
    """Atomically update independent records without modifying model artifacts."""
    atomic_json(cache / "local-models.json", {"schema_version": 1, "models": models})


def row(record: dict[str, Any]) -> ModelSpec:
    """Resolve read-only registered artifacts; changed fingerprints fail at preflight."""
    paths = record["artifact_paths"]
    present = all(Path(path).is_file() for path in paths.values())
    return ModelSpec(
        slug=record["name"],
        display_name=record["name"],
        arch=ModelArch(record["architecture"]),
        primary_stem=Stem(record["primary_stem"]),
        secondary_stem=None,
        sample_rate=44100,
        download_url="",
        path=Path(record["weights_path"]) if present else None,
        status=ModelStatus.INSTALLED if present else ModelStatus.BROKEN,
        metadata={**record["parameters"], "local_fingerprints": record["fingerprints"]},
        stems=list(record["stems"]),
        artifact_paths=dict(paths),
        owned=False,
        notes="Owner-registered local artifacts; read-only; inference has not been verified.",
    )
