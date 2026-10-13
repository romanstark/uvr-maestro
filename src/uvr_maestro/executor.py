"""Application operations and preflight before file writes or job creation."""

from __future__ import annotations

import json
import shutil
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from uvr_maestro import runtime
from uvr_maestro.discovery import Locations
from uvr_maestro.engine import disk, fetch
from uvr_maestro.engine import settings as profiles
from uvr_maestro.engine.uvr_bridge import metadata_for
from uvr_maestro.jobs import JobManager
from uvr_maestro.models import ModelArch, ModelSpec
from uvr_maestro.registry import default_registry, reset_default_registry, summarize
from uvr_maestro.storage import MaestroError, atomic_json, contained
from uvr_maestro.toolargs import (
    AudioContract,
    ChainStep,
    EnsembleEntry,
    LocalModel,
    SeparationOptions,
)


class Workbench:
    """Own path boundaries, catalog mutations and fully validated job requests."""

    def __init__(self, locations: Locations | None = None) -> None:
        self.locations = locations or Locations.discover()
        self.manager = JobManager(self.locations)
        self._local_lock = threading.Lock()

    def register(self, spec: LocalModel, confirm: bool) -> dict[str, Any]:
        """Register explicit artifacts, preserving the original files and upstream catalog."""
        from uvr_maestro import custom_models
        from uvr_maestro.storage import safe_name

        safe_name(spec.name)
        with self._local_lock:
            saved = custom_models.records(self.locations.cache)
            registry = self.registry()
            if spec.name not in saved:
                try:
                    registry.get(spec.name)
                except KeyError:
                    pass
                else:
                    raise ValueError("name conflicts with an existing model or alias")
            plan = {
                "record": spec.name,
                "artifacts": spec.model_dump(mode="json"),
                "before": saved.get(spec.name),
            }
            if not confirm:
                return {"success": False, "blocked": True, "plan": plan}
            record = custom_models.prepare(spec)
            with self.mutation("register_model", plan):
                custom_models.save(self.locations.cache, {**saved, spec.name: record})
            reset_default_registry()
            return {"success": True, "model": self.row_payload(self.registry().get(spec.name))}

    def unregister(self, name: str, confirm: bool) -> dict[str, Any]:
        """Remove only a registration record, never its original model artifacts."""
        from uvr_maestro import custom_models

        with self._local_lock:
            saved = custom_models.records(self.locations.cache)
            if name not in saved:
                raise MaestroError("model_unknown", "no local registration with that name")
            if not confirm:
                return {"success": False, "blocked": True, "plan": saved[name]}
            with self.mutation("unregister_model", {"before": saved[name]}):
                custom_models.save(
                    self.locations.cache,
                    {key: value for key, value in saved.items() if key != name},
                )
            reset_default_registry()
            return {"success": True, "unregistered": name, "artifacts_preserved": True}

    def registry(self) -> Any:
        """Read the offline catalog and actual disk state."""
        return default_registry(models_dir=self.locations.models, cache_dir=self.locations.cache)

    def status(self) -> dict[str, Any]:
        """Report light-server readiness separately from engine availability."""
        from uvr_maestro import __version__

        info = runtime.status(self.locations.runtime)
        from uvr_maestro.roformer_runtime import status as roformer_status

        info["roformer"] = roformer_status(self.locations.runtime)
        preflight = self.locations.runtime / "engine.json"
        if preflight.is_file():
            info["last_preflight"] = json.loads(preflight.read_text(encoding="utf-8")).get(
                "preflight"
            )
        return {
            "success": True,
            "version": __version__,
            "engine": info,
            "catalog": summarize(self.registry()),
            "models_dir": str(self.locations.models),
            "reused_installation": str(self.locations.desktop) if self.locations.desktop else None,
            "profiles_dir": str(self.locations.profiles),
            "executables": self.executables(),
        }

    @staticmethod
    def row_payload(row: ModelSpec) -> dict[str, Any]:
        """Serialize model capabilities and live installation state."""
        result: dict[str, Any] = asdict(row)
        result["path"] = str(row.path) if row.path else None
        result["arch"] = row.arch.value
        result["primary_stem"] = row.primary_stem.value
        result["secondary_stem"] = row.secondary_stem.value if row.secondary_stem else None
        result["status"] = row.status.value
        result["installed"] = row.installed
        result["supported"] = row.supported
        return result

    def executables(self) -> dict[str, str]:
        """Locate optional codecs without running or installing them."""
        result = {}
        for name in ("ffmpeg", "rubberband"):
            located = shutil.which(name)
            if not located and self.locations.desktop:
                suffix = ".exe" if sys.platform == "win32" else ""
                candidate = self.locations.desktop / (name + suffix)
                located = str(candidate) if candidate.is_file() else None
            if located:
                result[name] = located
        return result

    @contextmanager
    def mutation(self, operation: str, plan: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Record intent before a confirmed change and preserve its final outcome."""
        identifier = uuid.uuid4().hex
        path = self.locations.data / "journal" / (identifier + ".json")
        entry: dict[str, Any] = {
            "id": identifier,
            "operation": operation,
            "plan": plan,
            "state": "started",
            "started_at": time.time(),
        }
        atomic_json(path, entry)
        try:
            yield entry
        except Exception as error:
            entry.update(state="failed", error=str(error), finished_at=time.time())
            atomic_json(path, entry)
            raise
        else:
            entry.update(state="completed", finished_at=time.time())
            atomic_json(path, entry)

    def journal(self) -> list[dict[str, Any]]:
        """Read the most recent fifty mutation records."""
        paths = sorted(
            (self.locations.data / "journal").glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:50]
        return [json.loads(path.read_text(encoding="utf-8")) for path in paths]

    def model(self, name: str) -> dict[str, Any]:
        """Resolve a complete model and its actual engine metadata."""
        try:
            row = self.registry().get(name)
        except KeyError as error:
            raise MaestroError("model_unknown", f"unknown model: {name}") from error
        if not row.supported:
            raise MaestroError(
                "architecture_unsupported", f"runtime does not support {row.arch.value}"
            )
        if not row.installed or row.path is None:
            raise MaestroError("model_not_installed", f"required artifacts missing: {row.slug}")
        result = self.row_payload(row)
        if "local_fingerprints" in row.metadata:
            from uvr_maestro.storage import sha256

            for artifact_name, digest in row.metadata["local_fingerprints"].items():
                if sha256(Path(row.artifact_paths[artifact_name])) != digest:
                    raise MaestroError(
                        "model_modified", "registered artifact changed; register it again"
                    )
        if not row.metadata and row.arch in (ModelArch.VR_5, ModelArch.VR_4, ModelArch.MDX):
            kind = "vr" if row.arch.value.startswith("vr_") else "mdx"
            result["metadata"] = metadata_for(
                row.path, row.arch.value, self.registry().model_metadata.get(kind)
            )
        if row.arch is ModelArch.MDXC and "config_yaml" not in result["metadata"]:
            result["metadata"] = metadata_for(row.path, row.arch.value)
        return result

    def settings(
        self,
        profile: str | None,
        options: SeparationOptions | None = None,
    ) -> dict[str, Any]:
        """Resolve defaults, selected profile and explicitly supplied overrides."""
        values = profiles.load_named(
            self.registry(),
            settings_root=self.locations.profiles,
            name=profile,
        )
        overrides = options.model_dump(mode="json", exclude_none=True) if options else {}
        values.update(overrides)
        profiles.validate_map(self.registry(), values)
        if values["primary_stem_only"] and values["secondary_stem_only"]:
            raise ValueError("primary-only and secondary-only cannot both be selected")
        if not values.get("gpu_conversion", True) and "device" not in overrides:
            values["device"] = "cpu"
        return values

    @staticmethod
    def input_path(path: str) -> str:
        """Resolve an existing audio file before creating any output directory."""
        target = Path(path).expanduser().resolve()
        if not target.is_file():
            raise MaestroError("input_missing", f"audio file not found: {target}")
        if target.stat().st_size == 0:
            raise MaestroError("invalid_audio", "input file is empty")
        return str(target)

    def base_request(
        self,
        output_dir: str,
        *,
        audio_path: str | None = None,
        profile: str | None = None,
        options: SeparationOptions | None = None,
        audio: AudioContract | None = None,
    ) -> dict[str, Any]:
        """Validate runtime, inputs and options without creating output paths."""
        if not runtime.status(self.locations.runtime)["ready"]:
            raise MaestroError("engine_not_loaded", "run uvr-maestro-setup --confirm first")
        output = Path(output_dir).expanduser().resolve()
        if output.exists() and not output.is_dir():
            raise ValueError("output root is not a directory")
        if self.locations.desktop and output.is_relative_to(self.locations.desktop.resolve()):
            raise ValueError("outputs cannot be written inside the reused UVR installation")
        result: dict[str, Any] = {
            "output_root": str(output),
            "settings": self.settings(profile, options),
            "executables": self.executables(),
            "audio_contract": (audio or AudioContract()).model_dump(mode="json"),
        }
        if result["settings"]["save_format"] == "mp3" and result["audio_contract"]["exact_length"]:
            raise MaestroError("lossy_exactness", "MP3 requires audio.exact_length=False")
        if audio_path:
            result["audio_path"] = self.input_path(audio_path)
        return result

    @staticmethod
    def validate_options(model: dict[str, Any], options: SeparationOptions | None) -> None:
        """Reject explicit controls that the selected architecture cannot use."""
        if options is None:
            return
        arch = model["arch"]
        for key in options.model_dump(exclude_none=True):
            if key in ("roformer_segment_seconds", "roformer_overlap") and arch != "roformer":
                raise ValueError(f"{key} is a Roformer control")
            if key == "stem" and arch not in ("demucs_v3", "demucs_v4", "mdxc"):
                raise ValueError("stem is a Demucs/MDXC source selector")
            if key in (
                "window_size",
                "aggression",
                "tta",
                "post_process",
                "post_process_threshold",
                "high_end",
            ) and not arch.startswith("vr_"):
                raise ValueError(f"{key} is a VR control")
            if key in ("overlap", "shifts", "demucs_segment") and not arch.startswith("demucs"):
                raise ValueError(f"{key} is a Demucs control")
            if (
                key in ("segment_size", "overlap_mdx", "denoise")
                and arch != "mdx"
                and (key != "segment_size" or arch != "mdxc")
            ):
                raise ValueError(f"{key} is an MDX control")
            if key in ("overlap_mdx23", "mdxc_model_segment") and arch != "mdxc":
                raise ValueError(f"{key} is an MDXC control")

    def start(
        self,
        audio_path: str,
        model: str,
        output_dir: str,
        profile: str | None = None,
        options: SeparationOptions | None = None,
        audio: AudioContract | None = None,
    ) -> dict[str, Any]:
        """Queue one real separation after complete preflight."""
        resolved = self.model(model)
        self.validate_options(resolved, options)
        request = self.base_request(
            output_dir, audio_path=audio_path, profile=profile, options=options, audio=audio
        )
        request.update(operation="separation", model=resolved)
        return self.manager.submit("separation", request)

    def download(self, name: str, confirm: bool) -> dict[str, Any]:
        """Download only into owned storage, with a concrete refusal plan."""
        row = self.registry().get(name)
        if "local_fingerprints" in row.metadata:
            raise ValueError("registered local models are supplied by the owner, not downloaded")
        plan = [
            {
                "url": artifact.url,
                "path": str(
                    contained(
                        self.locations.models,
                        self.locations.models / artifact.directory / artifact.filename,
                    )
                ),
                "expected_bytes": artifact.size_bytes or None,
            }
            for artifact in row.artifacts
        ]
        if not confirm:
            return {"success": False, "blocked": True, "model": row.slug, "plan": plan}
        with self.mutation("download_model", {"model": row.slug, "artifacts": plan}) as entry:
            result = fetch.download_one(row, dest_dir=self.locations.models)
            entry["result"] = result
        self.registry()
        return {"success": result["ok"], **result}

    def remove(self, name: str, confirm: bool) -> dict[str, Any]:
        """Remove exclusively owned, unshared artifacts after confirmation."""
        registry = self.registry()
        row = registry.get(name)
        if not row.owned:
            raise MaestroError("model_not_owned", "reused or missing models cannot be removed")
        other_files = {
            artifact.filename
            for other in registry.all()
            if other.slug != row.slug
            for artifact in other.artifacts
        }
        removed = [
            path for filename, path in row.artifact_paths.items() if filename not in other_files
        ]
        if not confirm:
            return {"success": False, "blocked": True, "model": row.slug, "paths": removed}
        with self.mutation("remove_model", {"model": row.slug, "paths": removed}):
            for path in removed:
                contained(self.locations.models, Path(path)).unlink()
        registry.synchronize_with_disk(self.locations.models)
        return {
            "success": True,
            "removed": removed,
            "shared_artifacts_preserved": len(row.artifacts) - len(removed),
        }

    def utility(
        self,
        operation: str,
        output_dir: str,
        audio_path: str,
        **values: Any,
    ) -> dict[str, Any]:
        """Queue a real utility after validating every input."""
        request = self.base_request(output_dir, audio_path=audio_path)
        if "reference" in values:
            values["reference"] = self.input_path(values["reference"])
        if (
            operation in ("time_stretch", "pitch_shift")
            and "rubberband" not in request["executables"]
        ):
            raise MaestroError("rubberband_missing", "Rubber Band executable not found")
        request.update(operation=operation, **values)
        return self.manager.submit(operation, request)

    def ensemble(
        self,
        audio_path: str,
        entries: list[EnsembleEntry],
        output_dir: str,
        algorithm: str,
        profile: str | None,
        options: SeparationOptions | None,
        audio: AudioContract | None = None,
        keep_intermediates: bool = True,
    ) -> dict[str, Any]:
        """Queue an ensemble of explicit compatible source outputs."""
        if not 2 <= len(entries) <= 12:
            raise ValueError("an ensemble needs 2 through 12 entries")
        request = self.base_request(
            output_dir, audio_path=audio_path, profile=profile, options=options, audio=audio
        )
        resolved = []
        for entry in entries:
            model = self.model(entry.model)
            common = options.model_dump(exclude_none=True) if options else {}
            member = entry.options.model_dump(exclude_none=True) if entry.options else {}
            effective = SeparationOptions.model_validate({**common, **member})
            self.validate_options(model, effective)
            settings = self.settings(profile, effective)
            settings.update(save_format="wav", wav_subtype="FLOAT")
            resolved.append(
                {"model": model, "stem": entry.stem, "weight": entry.weight, "settings": settings}
            )
        if algorithm != "average" and any(entry.weight != 1 for entry in entries):
            raise ValueError("spectral ensembles use unit weights")
        request.update(
            operation="ensemble",
            entries=resolved,
            algorithm=algorithm,
            keep_intermediates=keep_intermediates,
        )
        return self.manager.submit("ensemble", request)

    def chain(
        self,
        audio_path: str,
        steps: list[ChainStep],
        output_dir: str,
        profile: str | None,
        audio: AudioContract | None = None,
        keep_intermediates: bool = True,
    ) -> dict[str, Any]:
        """Queue a source-routed separation chain."""
        if not 1 <= len(steps) <= 8:
            raise ValueError("a chain needs 1 through 8 steps")
        request = self.base_request(output_dir, audio_path=audio_path, profile=profile, audio=audio)
        resolved = []
        for step in steps:
            model = self.model(step.model)
            self.validate_options(model, step.options)
            resolved.append(
                {
                    "model": model,
                    "output_stem": step.output_stem,
                    "settings": self.settings(profile, step.options),
                }
            )
        request.update(operation="chain", steps=resolved, keep_intermediates=keep_intermediates)
        return self.manager.submit("chain", request)

    def verify(self) -> dict[str, Any]:
        """Refresh disk state and check any available trusted checksums."""
        registry = self.registry()
        failures = []
        for row in registry.installed():
            failures.extend(
                {"model": row.slug, **failure} for failure in disk.verify_artifacts(row)
            )
        return {
            "success": True,
            "catalog": summarize(registry),
            "integrity_failures": failures,
            "orphans": [
                str(path) for path in disk.list_orphan_files(registry.all(), self.locations.models)
            ],
        }
