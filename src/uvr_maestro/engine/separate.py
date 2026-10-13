"""Synchronous compatibility dispatcher over the same durable job engine."""

from __future__ import annotations

import time
from pathlib import Path

from uvr_maestro.discovery import Locations
from uvr_maestro.jobs import JobManager
from uvr_maestro.models import PipelineSpec, RunManifest
from uvr_maestro.storage import MaestroError, sha256
from uvr_maestro.workbench import Workbench


def run_pipeline(spec: PipelineSpec, *, audio_path: Path) -> RunManifest:
    """Run a resolved pipeline and return its verified manifest value object."""
    manager = JobManager(Locations.discover())
    model = Workbench.row_payload(spec.model_row)
    job = manager.submit(
        "separation",
        {
            "operation": "separation",
            "model": model,
            "settings": spec.settings,
            "audio_path": str(audio_path.resolve()),
            "output_root": str(spec.output_dir.resolve()),
        },
    )
    try:
        while job["state"] not in ("succeeded", "failed", "cancelled", "interrupted"):
            time.sleep(0.1)
            job = manager.get(job["id"])
        if job["state"] != "succeeded":
            raise MaestroError(job.get("code", "separation_failed"), job.get("error", job["state"]))
        result = job["result"]
        from uvr_maestro import __version__

        return RunManifest(
            tool="separate_audio",
            model_slug=spec.model_row.slug,
            model_arch=spec.model_row.arch.value,
            primary_stem=spec.model_row.primary_stem.value,
            secondary_stem=spec.model_row.secondary_stem.value
            if spec.model_row.secondary_stem
            else None,
            input_sha256=sha256(audio_path),
            inputs=[str(audio_path)],
            outputs=[item["path"] for item in result["outputs"]],
            settings=result["settings"],
            gpu=result["device"],
            runtime_seconds=result["runtime_seconds"],
            fallback_to_cpu=bool(result.get("fallback_reason")),
            version=__version__,
            fallback_reason=result.get("fallback_reason"),
            engine_revision=result["engine_revision"],
            audio=result["outputs"],
        )
    finally:
        manager.close()
