"""MCP access to local UVR separation, model management and audio jobs."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from uvr_maestro.engine import fetch
from uvr_maestro.engine import settings as profiles
from uvr_maestro.executor import Workbench
from uvr_maestro.models import ModelArch, Stem
from uvr_maestro.registry import reset_default_registry, summarize
from uvr_maestro.toolargs import (
    ADDS,
    DESTROYS,
    NETWORK_READS,
    NETWORK_WRITES,
    READS,
    AudioContract,
    AudioPath,
    ChainStep,
    Confirm,
    EnsembleEntry,
    JobId,
    LocalModel,
    ModelName,
    OutputDir,
    Profile,
    SeparationOptions,
)

_workbench: Workbench | None = None
_workbench_lock = threading.Lock()


def workbench() -> Workbench:
    """Resolve the application lazily, without writing during discovery."""
    global _workbench
    with _workbench_lock:
        if _workbench is None:
            _workbench = Workbench()
    return _workbench


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Cancel managed workers when the MCP session shuts down."""
    del server
    try:
        yield {}
    finally:
        if _workbench is not None:
            _workbench.manager.close()


mcp = FastMCP(
    "uvr-maestro",
    lifespan=lifespan,
    log_level="WARNING",
    instructions=(
        "Local UVR5 audio partner. Begin with uvr_status and find_model or list_models. "
        "The light server works before an engine is installed. Install the engine with "
        "uvr-maestro-setup --confirm. Model downloads, deletion, profile replacement and "
        "batch runs require confirm=True. Existing UVR desktop models are reused read-only. "
        "start_separation and the audio workflow tools return queued jobs, not finished "
        "audio. Poll get_job until succeeded, failed or cancelled. Only a succeeded job "
        "has validated outputs and a saved manifest. Use cancel_job to stop its worker. "
        "Read workflow guidance with uvr_skills or uvr://skill/{name}."
    ),
)


def response(action, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Translate application failures at the public protocol boundary."""
    try:
        result = action(*args, **kwargs)
        return result if "success" in result else {"success": True, **result}
    except Exception as error:  # noqa: BLE001
        return {
            "success": False,
            "code": getattr(error, "code", "invalid_request"),
            "error": str(error),
        }


@mcp.tool(annotations=READS)
def uvr_status() -> dict[str, Any]:
    """Report engine readiness, model roots and last measured backend support."""
    return response(workbench().status)


@mcp.tool(annotations=READS)
def list_models(
    arch: Annotated[
        ModelArch | None, Field(description="Architecture enum. None selects all.")
    ] = None,
    primary_stem: Annotated[
        Stem | None, Field(description="Source/restoration target enum.")
    ] = None,
    only_installed: Annotated[
        bool, Field(description="True includes only complete local artifact sets.")
    ] = False,
    only_available: Annotated[
        bool, Field(description="True includes models that are not installed.")
    ] = False,
    query: Annotated[
        str | None, Field(description="Substring in IDs, aliases, names or purposes.")
    ] = None,
) -> dict[str, Any]:
    """Read the offline catalog and current disk state without downloading."""

    def action():
        registry = workbench().registry()
        rows = registry.filter(
            arch=arch,
            primary_stem=primary_stem,
            only_installed=only_installed,
            only_available=only_available,
            query=query,
        )
        return {
            "success": True,
            "rows": [workbench().row_payload(row) for row in rows],
            "summary": summarize(registry),
        }

    return response(action)


@mcp.tool(annotations=READS)
def find_model(
    intent: Annotated[
        str, Field(min_length=1, description="Task in your own words, English or German.")
    ],
) -> dict[str, Any]:
    """Rank compatible supported models and expose the task-based selection."""
    from uvr_maestro.engine.intent import rank_intent

    def action():
        rows = rank_intent(intent, registry=workbench().registry())
        return {
            "success": True,
            "intent": intent,
            "candidates": [
                {
                    **workbench().row_payload(row),
                    "score": score,
                    "reason": "compatible task/architecture, with preference for installed models",
                }
                for row, score in rows
            ],
        }

    return response(action)


@mcp.tool(annotations=NETWORK_WRITES)
async def refresh_catalog(confirm: Confirm = False) -> dict[str, Any]:
    """Fetch upstream metadata and atomically replace the validated offline snapshot."""
    wb = workbench()
    if not confirm:
        return {
            "success": False,
            "blocked": True,
            "plan": {
                "cache": str(wb.locations.cache / "catalog.json"),
                "source": wb.registry().endpoints.download_checks,
            },
        }

    def action():
        with wb.mutation("refresh_catalog", {"cache": str(wb.locations.cache / "catalog.json")}):
            result = fetch.refresh_catalog(
                cache_path=wb.locations.cache / "catalog.json", endpoints=wb.registry().endpoints
            )
        reset_default_registry()
        return {"success": result["ok"], **result}

    return await asyncio.to_thread(response, action)


@mcp.tool(annotations=NETWORK_WRITES)
async def download_model(slug: ModelName, confirm: Confirm = False) -> dict[str, Any]:
    """Download a complete model artifact set into Maestro-owned storage."""
    return await asyncio.to_thread(response, workbench().download, slug, confirm)


@mcp.tool(annotations=DESTROYS)
async def register_model(
    spec: Annotated[
        LocalModel, Field(description="Explicit local weights/config and output labels.")
    ],
    confirm: Confirm = False,
) -> dict[str, Any]:
    """Register or replace a local model without moving its artifacts or refreshing upstream."""
    return await asyncio.to_thread(response, workbench().register, spec, confirm)


@mcp.tool(annotations=DESTROYS)
async def unregister_model(slug: ModelName, confirm: Confirm = False) -> dict[str, Any]:
    """Remove a local registration while preserving all original weights and configuration."""
    return await asyncio.to_thread(response, workbench().unregister, slug, confirm)


@mcp.tool(annotations=NETWORK_WRITES)
async def download_missing_models(
    arch: Annotated[
        ModelArch, Field(description="Architecture whose supported missing models to install.")
    ],
    confirm: Confirm = False,
) -> dict[str, Any]:
    """Plan or download missing models for one supported architecture."""

    def action():
        wb = workbench()
        rows = [row for row in wb.registry().by_arch(arch) if row.supported and not row.installed]
        plans = [wb.download(row.slug, False) for row in rows]
        if not confirm:
            return {"success": False, "blocked": True, "models": plans}
        results = [wb.download(row.slug, True) for row in rows]
        return {"success": all(item["success"] for item in results), "models": results}

    return await asyncio.to_thread(response, action)


@mcp.tool(annotations=DESTROYS)
def remove_model(slug: ModelName, confirm: Confirm = False) -> dict[str, Any]:
    """Remove owned artifacts while preserving reused files and shared checkpoints."""
    return response(workbench().remove, slug, confirm)


@mcp.tool(annotations=READS)
def verify_catalog() -> dict[str, Any]:
    """Refresh installation state, check trusted hashes and report orphan weights."""
    return response(workbench().verify)


@mcp.tool(annotations=READS)
def get_settings(name: Profile = None) -> dict[str, Any]:
    """Read defaults overlaid by the selected persisted profile."""
    return response(
        lambda: {
            "success": True,
            "values": workbench().settings(name),
            "saved": profiles.list_saved(workbench().locations.profiles),
            "schema": [vars(setting) for setting in workbench().registry().settings()],
        }
    )


@mcp.tool(annotations=DESTROYS)
def set_settings(
    values: Annotated[
        dict[str, Any], Field(description="Declared setting keys and values from get_settings.")
    ],
    name: Profile = None,
    confirm: Confirm = False,
) -> dict[str, Any]:
    """Validate and merge profile changes, refusing to write without confirmation."""

    def action():
        wb = workbench()
        profiles.validate_map(wb.registry(), values)
        target = profiles.profile_path(wb.locations.profiles, name)
        if not confirm:
            return {
                "success": False,
                "blocked": True,
                "path": str(target),
                "before": wb.settings(name),
                "changes": values,
            }
        with wb.mutation(
            "set_settings", {"path": str(target), "before": wb.settings(name), "changes": values}
        ):
            written = profiles.write_named(
                wb.registry(), settings_root=wb.locations.profiles, name=name, values=values
            )
        return {"success": True, "path": str(written), "read_back": wb.settings(name)}

    return response(action)


@mcp.tool(annotations=DESTROYS)
def reset_settings(name: Profile = None, confirm: Confirm = False) -> dict[str, Any]:
    """Plan or delete a Maestro profile so future runs use bundled defaults."""

    def action():
        wb = workbench()
        target = profiles.profile_path(wb.locations.profiles, name)
        if not confirm:
            return {
                "success": False,
                "blocked": True,
                "path": str(target),
                "exists": target.exists(),
            }
        with wb.mutation("reset_settings", {"path": str(target), "before": wb.settings(name)}):
            removed = profiles.reset_named(settings_root=wb.locations.profiles, name=name)
        return {"success": True, "removed": str(removed) if removed else None}

    return response(action)


@mcp.tool(annotations=ADDS)
def start_separation(
    audio_path: AudioPath,
    model: ModelName,
    output_dir: OutputDir,
    profile: Profile = None,
    options: Annotated[
        SeparationOptions | None, Field(description="Explicit overrides on the selected profile.")
    ] = None,
    audio: Annotated[
        AudioContract | None,
        Field(description="Source-relative excerpt and lossless export contract."),
    ] = None,
) -> dict[str, Any]:
    """Start one separation and return a queued job. Poll get_job for verified files."""
    return response(workbench().start, audio_path, model, output_dir, profile, options, audio)


@mcp.tool(annotations=ADDS)
async def separate_audio(
    audio_path: AudioPath,
    model: ModelName,
    output_dir: OutputDir,
    profile: Profile = None,
    options: Annotated[
        SeparationOptions | None, Field(description="Explicit architecture-specific overrides.")
    ] = None,
    audio: Annotated[
        AudioContract | None,
        Field(description="Source-relative excerpt and lossless export contract."),
    ] = None,
) -> dict[str, Any]:
    """Run one separation and await verified outputs. Use start_separation for long work."""
    started = start_separation(audio_path, model, output_dir, profile, options, audio)
    if not started["success"]:
        return started
    identifier = started["id"]
    try:
        while True:
            job = workbench().manager.get(identifier)
            if job["state"] in ("succeeded", "failed", "cancelled", "interrupted"):
                return {"success": job["state"] == "succeeded", **job}
            await asyncio.sleep(0.25)
    except asyncio.CancelledError:
        workbench().manager.cancel(identifier)
        raise


@mcp.tool(annotations=READS)
def get_job(job_id: JobId) -> dict[str, Any]:
    """Read a job's state, progress, actual outputs and durable manifest path."""
    return response(lambda: workbench().manager.get(job_id))


@mcp.tool(annotations=READS)
def list_jobs() -> dict[str, Any]:
    """List the latest hundred persisted and active jobs."""
    return response(lambda: {"success": True, "jobs": workbench().manager.list()})


@mcp.tool(annotations=READS)
def change_journal() -> dict[str, Any]:
    """Read the latest fifty confirmed catalog/profile change records."""
    return response(lambda: {"success": True, "entries": workbench().journal()})


@mcp.tool(annotations=DESTROYS)
def cancel_job(job_id: JobId) -> dict[str, Any]:
    """Stop queued work or terminate its native worker, preserving partial-run evidence."""
    return response(workbench().manager.cancel, job_id)


@mcp.tool(annotations=ADDS)
def separate_ensemble(
    audio_path: AudioPath,
    entries: Annotated[
        list[EnsembleEntry], Field(description="2 through 12 model/source/weight entries.")
    ],
    output_dir: OutputDir,
    algorithm: Annotated[
        Literal["average", "min_spec", "max_spec"],
        Field(
            description=(
                "average uses weights. min_spec/max_spec use UVR spectral "
                "algorithms and unit weights."
            )
        ),
    ] = "average",
    profile: Profile = None,
    options: Annotated[
        SeparationOptions | None,
        Field(description="Common overrides valid for every selected architecture."),
    ] = None,
    audio: Annotated[
        AudioContract | None, Field(description="Common source selection and export contract.")
    ] = None,
    keep_intermediates: Annotated[
        bool, Field(description="Retain individual model outputs after verified success.")
    ] = True,
) -> dict[str, Any]:
    """Queue actual model runs and combine matching sources. Poll get_job."""
    return response(
        workbench().ensemble,
        audio_path,
        entries,
        output_dir,
        algorithm,
        profile,
        options,
        audio,
        keep_intermediates,
    )


@mcp.tool(annotations=ADDS)
def separate_chain(
    audio_path: AudioPath,
    steps: Annotated[
        list[ChainStep], Field(description="1 through 8 models with explicit output routing.")
    ],
    output_dir: OutputDir,
    profile: Profile = None,
    audio: Annotated[
        AudioContract | None, Field(description="Selection from the original source; applied once.")
    ] = None,
    keep_intermediates: Annotated[
        bool, Field(description="Retain earlier step files after verified final output.")
    ] = True,
) -> dict[str, Any]:
    """Queue a cleanup/separation chain using each declared source as the next input."""
    return response(
        workbench().chain, audio_path, steps, output_dir, profile, audio, keep_intermediates
    )


@mcp.tool(annotations=ADDS)
def separate_vocal_splitter(
    audio_path: AudioPath,
    output_dir: OutputDir,
    vocal_model: ModelName = "Kim_Vocal_2",
    split_model: ModelName = "UVR-BVE-4B_SN-44100-1",
    profile: Profile = None,
) -> dict[str, Any]:
    """Queue UVR's lead/backing vocal splitter using a vocal model and a BVE model."""

    def action():
        wb = workbench()
        primary = wb.model(vocal_model)
        split = wb.model(split_model)
        if not split["metadata"].get("is_bv_model"):
            raise ValueError("split_model must be identified as a backing-vocal model")
        request = wb.base_request(output_dir, audio_path=audio_path, profile=profile)
        request.update(operation="vocal_splitter", model=primary, split_model=split)
        return wb.manager.submit("vocal_splitter", request)

    return response(action)


@mcp.tool(annotations=DESTROYS)
def separate_batch(
    input_dir: Annotated[
        str, Field(description="Local directory. Direct child audio files are processed.")
    ],
    model: ModelName,
    output_dir: OutputDir,
    profile: Profile = None,
    options: Annotated[
        SeparationOptions | None, Field(description="Explicit overrides for every input file.")
    ] = None,
    confirm: Confirm = False,
) -> dict[str, Any]:
    """Plan or queue per-file separation with aggregate results and cancellation."""

    def action():
        wb = workbench()
        folder = Path(input_dir).expanduser().resolve()
        if not folder.is_dir():
            raise ValueError("input_dir is not a directory")
        files = sorted(
            str(path)
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.casefold() in (".wav", ".flac", ".mp3", ".ogg", ".m4a")
        )
        if not files:
            raise ValueError("input_dir has no supported audio files")
        resolved = wb.model(model)
        wb.validate_options(resolved, options)
        if not confirm:
            return {
                "success": False,
                "blocked": True,
                "files": files,
                "model": resolved["slug"],
                "output_root": str(Path(output_dir).resolve()),
            }
        request = wb.base_request(output_dir, profile=profile, options=options)
        request.update(operation="batch", inputs=files, model=resolved)
        return wb.manager.submit("batch", request)

    return response(action)


@mcp.tool(annotations=READS)
def probe_audio(audio_path: AudioPath) -> dict[str, Any]:
    """Read file metadata without decoding the whole recording."""

    def action():
        import soundfile as sf

        path = workbench().input_path(audio_path)
        info = sf.info(path)
        return {
            "success": True,
            "path": path,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "subtype": info.subtype,
            "duration_seconds": info.duration,
        }

    return response(action)


@mcp.tool(annotations=READS)
def verify_audio(audio_path: AudioPath) -> dict[str, Any]:
    """Decode the complete file in blocks and reject empty or non-finite samples."""

    def action():
        import numpy as np
        import soundfile as sf

        path = workbench().input_path(audio_path)
        frames = 0
        for block in sf.blocks(path, blocksize=65536, always_2d=True):
            if not np.isfinite(block).all():
                raise ValueError("audio has non-finite samples")
            frames += len(block)
        if not frames:
            raise ValueError("audio has no decoded frames")
        return {"success": True, "verified_frames": frames, **probe_audio(path)}

    return response(action)


@mcp.tool(annotations=ADDS)
def create_sample(
    audio_path: AudioPath,
    output_dir: OutputDir,
    duration_seconds: Annotated[
        float, Field(gt=0, le=120, description="Preview duration in seconds, above 0 through 120.")
    ] = 10,
    offset_seconds: Annotated[
        float | None,
        Field(ge=0, description="Start offset in seconds. None starts one third into the input."),
    ] = None,
) -> dict[str, Any]:
    """Queue a real preview clip in a unique output directory."""
    return response(
        workbench().utility,
        "create_sample",
        output_dir,
        audio_path,
        duration_seconds=duration_seconds,
        offset_seconds=offset_seconds,
    )


@mcp.tool(annotations=ADDS)
def pitch_shift(
    audio_path: AudioPath,
    output_dir: OutputDir,
    semitones: Annotated[
        float, Field(ge=-24, le=24, description="Pitch offset in semitones. Positive shifts up.")
    ],
) -> dict[str, Any]:
    """Queue Rubber Band pitch shifting while preserving duration."""
    return response(workbench().utility, "pitch_shift", output_dir, audio_path, semitones=semitones)


@mcp.tool(annotations=ADDS)
def time_stretch(
    audio_path: AudioPath,
    output_dir: OutputDir,
    rate: Annotated[
        float,
        Field(
            gt=0.1,
            le=10,
            description="Playback speed multiplier. 1.05 is 5% faster. Pitch is preserved.",
        ),
    ],
) -> dict[str, Any]:
    """Queue Rubber Band time stretching while preserving pitch."""
    return response(workbench().utility, "time_stretch", output_dir, audio_path, rate=rate)


@mcp.tool(annotations=ADDS)
def align_inputs(
    reference: AudioPath,
    target: AudioPath,
    output_dir: OutputDir,
) -> dict[str, Any]:
    """Queue sample-offset estimation and write the target aligned to the reference."""
    return response(workbench().utility, "align_inputs", output_dir, target, reference=reference)


@mcp.tool(annotations=ADDS)
def matchering_inputs(
    reference: AudioPath,
    target: AudioPath,
    output_dir: OutputDir,
) -> dict[str, Any]:
    """Queue reference mastering with Matchering and write a verified 24-bit WAV."""
    return response(workbench().utility, "matchering", output_dir, target, reference=reference)


@mcp.tool(annotations=ADDS)
def analyze_audio(audio_path: AudioPath, output_dir: OutputDir) -> dict[str, Any]:
    """Queue full-file peak, RMS, clipping and BS.1770 integrated loudness analysis."""
    return response(workbench().utility, "analyze_audio", output_dir, audio_path)


@mcp.tool(annotations=NETWORK_READS)
async def check_update() -> dict[str, Any]:
    """Compare the pinned UVR patch with current platform-specific upstream metadata."""

    def action():
        endpoint = workbench().registry().endpoints
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            metadata = client.get(endpoint.download_checks)
            metadata.raise_for_status()
            bulletin = client.get(endpoint.bulletin)
            bulletin.raise_for_status()
        from uvr_maestro.engine.uvr_version import load

        current = load()
        checks = metadata.json()
        platform_key, running_patch = {
            "darwin": ("current_version_mac", current.patch_mac),
            "linux": ("current_version_linux", current.patch_linux),
        }.get(sys.platform, ("current_version", current.patch))
        return {
            "success": True,
            "running": vars(current),
            "upstream": {
                key: value for key, value in checks.items() if key.startswith("current_version")
            },
            "update_available": checks.get(platform_key) != running_patch,
            "platform": sys.platform,
            "bulletin": bulletin.text,
            "installers": list_updates(),
        }

    return await asyncio.to_thread(response, action)


@mcp.tool(annotations=READS)
def list_updates() -> dict[str, Any]:
    """Read full-installer URLs and patch source without downloading an installer."""
    endpoints = workbench().registry().endpoints
    return {
        "success": True,
        "windows": endpoints.windows_setup,
        "mac_arm": endpoints.mac_arm_installer,
        "mac_x86": endpoints.mac_x86_installer,
        "linux": endpoints.linux_source,
        "patch_base": endpoints.update_patch_repo_base,
    }


@mcp.tool(annotations=READS)
def uvr_skills() -> dict[str, Any]:
    """List client-neutral workflow guidance available through uvr://skill/{name}."""
    folder = Path(__file__).parent / "skills"
    return {
        "success": True,
        "skills": [
            {"name": path.parent.name, "uri": "uvr://skill/" + path.parent.name}
            for path in sorted(folder.glob("*/SKILL.md"))
        ],
    }


@mcp.resource("uvr://catalog")
def catalog_resource() -> str:
    """Read the compact offline model map and lookup routes."""
    registry = workbench().registry()
    return json.dumps(
        {
            "summary": summarize(registry),
            "models": len(registry.all()),
            "tools": ["list_models", "find_model", "verify_catalog"],
        }
    )


@mcp.resource("uvr://settings")
def settings_resource() -> str:
    """Read the current profile and its typed schema."""
    return json.dumps(get_settings(), ensure_ascii=False)


@mcp.resource("uvr://status")
def status_resource() -> str:
    """Read readiness without requiring a loaded engine."""
    return json.dumps(uvr_status(), ensure_ascii=False)


@mcp.resource("uvr://skill/{name}")
def skill_resource(name: str) -> str:
    """Read one shipped workflow skill by its confined folder name."""
    from uvr_maestro.storage import safe_name

    path = Path(__file__).parent / "skills" / safe_name(name) / "SKILL.md"
    return path.read_text(encoding="utf-8")


@mcp.prompt()
def separate_stems(
    audio_path: AudioPath,
    intent: Annotated[str, Field(description="Desired stems or restoration task.")],
) -> str:
    """Guide a local separation from readiness through verified outputs."""
    return (
        f"Read uvr://skill/uvr-separation. Check uvr_status, find_model for {intent!r}, "
        f"and separate {audio_path!r}. Poll get_job until completion and inspect the manifest."
    )


@mcp.prompt()
def restore_recording(
    audio_path: AudioPath,
    intent: Annotated[str, Field(description="Noise, echo or reverb to address.")],
) -> str:
    """Guide a source-routed restoration chain with actual output labels."""
    return (
        f"Read uvr://skill/uvr-restoration. Find suitable models for {intent!r}, "
        f"then prepare a separate_chain for {audio_path!r} with explicit output routing."
    )


def main() -> None:
    """Serve the published tools over MCP stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
