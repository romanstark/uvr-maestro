"""Validated offline catalog, atomic cache and live artifact reconciliation."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from uvr_maestro.catalog import CATALOG_DIR
from uvr_maestro.discovery import Locations
from uvr_maestro.models import ModelArch, ModelSpec, Stem
from uvr_maestro.spec import (
    EndpointSpec,
    SettingSpec,
    build_endpoint,
    build_row,
    build_setting,
)
from uvr_maestro.storage import MaestroError, atomic_json

DEFAULT_CACHE_DIR = Locations.discover().cache


class Registry:
    """Index catalog identities and aliases, keeping disk state separate."""

    def __init__(
        self, models: list[ModelSpec], settings: list[SettingSpec], endpoints: EndpointSpec
    ) -> None:
        self._by_slug: dict[str, ModelSpec] = {}
        self._aliases: dict[str, str] = {}
        for row in models:
            if row.slug in self._by_slug:
                raise ValueError(f"duplicate catalog identity: {row.slug}")
            self._by_slug[row.slug] = row
            for alias in [row.slug, row.display_name, *row.aliases]:
                self._aliases.setdefault(alias.casefold(), row.slug)
        self._settings = {s.key: s for s in settings}
        if len(self._settings) != len(settings):
            raise ValueError("duplicate setting")
        self.endpoints = endpoints
        self.model_metadata: dict[str, dict[str, Any]] = {}

    @classmethod
    def load_bundled(cls, catalog_dir: Path | None = None) -> Registry:
        """Load the packaged snapshot and schema without network access."""
        folder = catalog_dir or CATALOG_DIR
        models: list[ModelSpec] = []
        settings: list[SettingSpec] = []
        endpoints = None
        for path in sorted(folder.glob("*.yaml")):
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "endpoints" in data:
                endpoints = build_endpoint(data)
            elif isinstance(data, dict) and "keys" in data:
                settings.extend(build_setting(raw) for raw in data["keys"])
            elif isinstance(data, list):
                models.extend(build_row(raw) for raw in data)
            else:
                raise ValueError(f"invalid catalog document: {path.name}")
        if endpoints is None or not models:
            raise ValueError("bundled catalog or endpoints missing")
        registry = cls(models, settings, endpoints)
        for kind in ("vr", "mdx"):
            path = folder / f"{kind}_metadata.json"
            if path.is_file():
                metadata = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(metadata, dict):
                    raise TypeError("model metadata must be an object")
                registry.model_metadata[kind] = metadata
        return registry

    @classmethod
    def load_cache(cls, cache_path: Path) -> Registry | None:
        """Fall back when a cache is absent, corrupted or incompatible."""
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if data.get("schema_version") != 1:
                return None
            registry = cls(
                [build_row(raw) for raw in data["models"]],
                [build_setting(raw) for raw in data["settings"]],
                build_endpoint({"endpoints": data["endpoints"]}),
            )
            metadata = data.get("model_metadata", {})
            if not isinstance(metadata, dict) or any(
                not isinstance(value, dict) for value in metadata.values()
            ):
                return None
            registry.model_metadata = metadata
            return registry
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return None

    def all(self) -> list[ModelSpec]:
        """Return models in stable snapshot order."""
        return list(self._by_slug.values())

    def installed(self) -> list[ModelSpec]:
        """Return models whose complete artifact set is present."""
        return [row for row in self.all() if row.installed]

    def get(self, slug: str) -> ModelSpec:
        """Resolve a catalog identity, filename alias or display name."""
        return self._by_slug[self._aliases[slug.casefold()]]

    def by_arch(self, arch: ModelArch) -> list[ModelSpec]:
        """Select an architecture."""
        return [row for row in self.all() if row.arch == arch]

    def by_stem(self, stem: Stem) -> list[ModelSpec]:
        """Select a source or restoration target."""
        return [row for row in self.all() if stem in (row.primary_stem, row.secondary_stem)]

    def filter(
        self,
        *,
        arch: ModelArch | str | None = None,
        primary_stem: Stem | str | None = None,
        only_installed: bool = False,
        only_available: bool = False,
        query: str | None = None,
    ) -> list[ModelSpec]:
        """Filter by domain values, installation state and text."""
        if only_installed and only_available:
            raise ValueError("installed and available filters are mutually exclusive")
        needle = (query or "").casefold()
        return [
            row
            for row in self.all()
            if (arch is None or row.arch.value == arch)
            and (primary_stem is None or row.primary_stem.value == primary_stem)
            and (not only_installed or row.installed)
            and (not only_available or not row.installed)
            and (
                not needle
                or needle
                in " ".join([row.slug, row.display_name, *row.aliases, *row.purposes]).casefold()
            )
        ]

    def settings(self) -> list[SettingSpec]:
        """Return the settings schema."""
        return list(self._settings.values())

    def setting(self, key: str) -> SettingSpec:
        """Resolve a declared setting."""
        return self._settings[key]

    def synchronize_with_disk(
        self, models_dir: Path, extra_roots: list[Path] | None = None
    ) -> None:
        """Read artifact state without downloading or importing the engine."""
        from uvr_maestro.engine.disk import reconcile

        roots = [models_dir, *(extra_roots or [])]
        for row in self.all():
            reconcile(row, roots, owned_root=models_dir)
            if (
                row.installed
                and row.path
                and row.arch
                in (
                    ModelArch.VR_5,
                    ModelArch.VR_4,
                    ModelArch.MDX,
                )
            ):
                from uvr_maestro.engine.uvr_bridge import complementary_stem, metadata_for

                kind = "vr" if row.arch.value.startswith("vr_") else "mdx"
                try:
                    parameters = metadata_for(
                        row.path, row.arch.value, self.model_metadata.get(kind)
                    )
                    primary = parameters["primary_stem"]
                    row.stems = [primary, complementary_stem(primary)]
                except (MaestroError, KeyError, ValueError, OSError):
                    row.stems = []

    def payload(self) -> dict[str, Any]:
        """Serialize only durable catalog data, excluding local paths."""
        rows = []
        for row in self.all():
            value: dict[str, Any] = asdict(row)
            for key in ("path", "status", "artifact_paths", "owned"):
                value.pop(key)
            value["architecture"] = value.pop("arch").value
            value["primary_stem"] = row.primary_stem.value
            value["secondary_stem"] = row.secondary_stem.value if row.secondary_stem else None
            rows.append(value)
        return {
            "schema_version": 1,
            "models": rows,
            "settings": [asdict(s) for s in self.settings()],
            "endpoints": asdict(self.endpoints),
            "model_metadata": self.model_metadata,
        }

    def write_cache(self, cache_path: Path) -> None:
        """Commit a complete validated snapshot atomically."""
        atomic_json(cache_path, self.payload())


_default: Registry | None = None
_default_key: tuple[str, str, int] | None = None


def default_registry(*, models_dir: Path | None = None, cache_dir: Path | None = None) -> Registry:
    """Load a registry for the configured storage and refresh its disk snapshot."""
    global _default, _default_key
    locations = Locations.discover()
    models_dir = models_dir or locations.models
    cache_dir = cache_dir or locations.cache
    custom = cache_dir / "local-models.json"
    key = (str(models_dir), str(cache_dir), custom.stat().st_mtime_ns if custom.exists() else 0)
    if _default is None or _default_key != key:
        bundled = Registry.load_bundled()
        _default = Registry.load_cache(cache_dir / "catalog.json") or bundled
        _default._settings = bundled._settings
        for row in _default.all():
            known = bundled._by_slug.get(row.slug)
            if known is not None and known.metadata.get("catalog_source_override"):
                row.artifacts = copy.deepcopy(known.artifacts)
                row.download_url = known.download_url
                row.primary_stem = known.primary_stem
                row.secondary_stem = known.secondary_stem
                row.stems = list(known.stems)
                row.purposes = list(known.purposes)
                row.verified = known.verified
                row.notes = known.notes
                row.metadata.update(known.metadata)
            if known is not None and known.arch == row.arch and known.artifacts == row.artifacts:
                row.metadata = {**known.metadata, **row.metadata}
                row.verified = row.verified or known.verified
        _default_key = key
    extra = [locations.desktop / "models"] if locations.desktop else []
    _default.synchronize_with_disk(models_dir, extra)
    from uvr_maestro.custom_models import records, row

    for name, record in records(cache_dir).items():
        local = row(record)
        if (
            name in _default._by_slug
            and "local_fingerprints" not in _default._by_slug[name].metadata
        ):
            raise ValueError(f"local identity conflicts with the catalog: {name}")
        if name.casefold() in _default._aliases and _default._aliases[name.casefold()] != name:
            raise ValueError(f"local identity conflicts with an alias: {name}")
        _default._by_slug[name] = local
        _default._aliases[name.casefold()] = name
    return _default


def reset_default_registry() -> None:
    """Invalidate the process cache after a committed refresh."""
    global _default, _default_key
    _default = None
    _default_key = None


def summarize(registry: Registry) -> dict[str, dict[str, int]]:
    """Count installation states by architecture."""
    result: dict[str, dict[str, int]] = {}
    for row in registry.all():
        counts = result.setdefault(row.arch.value, {})
        counts[row.status.value] = counts.get(row.status.value, 0) + 1
    return result
