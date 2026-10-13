"""Verify actual audio utilities and advanced workflows on owned fixtures."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from uvr_maestro import server
from uvr_maestro.discovery import Locations
from uvr_maestro.models import SaveFormat
from uvr_maestro.toolargs import AudioContract, ChainStep, EnsembleEntry, SeparationOptions
from uvr_maestro.workbench import Workbench

ROOT = Path(__file__).resolve().parents[1]


def wait(started: dict[str, Any], wb: Workbench) -> dict[str, Any]:
    """Await one queued job and preserve its terminal evidence."""
    if not started.get("success"):
        return {"state": "failed", **started}
    job = started
    while job["state"] not in ("succeeded", "failed", "cancelled", "interrupted"):
        time.sleep(0.5)
        job = wb.manager.get(job["id"])
    return job


def main() -> None:
    """Exercise every utility with real engine dependencies and executables."""
    os.environ.setdefault("UVR_MAESTRO_DATA_DIR", str(ROOT / "scratch/workflow-data"))
    os.environ.setdefault("UVR_MAESTRO_RUNTIME_DIR", str(ROOT / ".runtime"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--vr-model", default="3_HP-Vocal-UVR")
    parser.add_argument("--cleanup-model", default="UVR-De-Echo-Aggressive")
    parser.add_argument("--cleanup-stem", default="No Echo")
    args = parser.parse_args()
    wb = Workbench(Locations.discover())
    server._workbench = wb
    audio = ROOT / "scratch/voice-music.wav"
    output = str(ROOT / "scratch/workflow-runs")
    samples, rate = sf.read(audio, always_2d=True)
    delayed = ROOT / "scratch/delayed.wav"
    sf.write(delayed, np.pad(samples, ((512, 0), (0, 0))), rate, subtype="PCM_16")
    batch = ROOT / "scratch/batch-inputs"
    batch.mkdir(exist_ok=True)
    sf.write(batch / "take-a.wav", samples, rate)
    sf.write(batch / "take-b.wav", samples, rate)
    options = SeparationOptions(device=args.device)
    actions = [
        ("sample", lambda: server.create_sample(str(audio), output, 1.0, 0)),
        ("pitch", lambda: server.pitch_shift(str(audio), output, 2)),
        ("time", lambda: server.time_stretch(str(audio), output, 1.05)),
        ("align", lambda: server.align_inputs(str(audio), str(delayed), output)),
        ("analyze", lambda: server.analyze_audio(str(audio), output)),
        ("matchering", lambda: server.matchering_inputs(str(audio), str(delayed), output)),
        (
            "mp3",
            lambda: server.start_separation(
                str(audio),
                "Kim Vocal 2",
                output,
                options=SeparationOptions(device=args.device, save_format=SaveFormat.MP3),
                audio=AudioContract(exact_length=False),
            ),
        ),
        (
            "flac",
            lambda: server.start_separation(
                str(audio),
                "Kim Vocal 2",
                output,
                options=SeparationOptions(device=args.device, save_format=SaveFormat.FLAC),
            ),
        ),
        (
            "batch",
            lambda: server.separate_batch(
                str(batch), "Kim Vocal 2", output, options=options, confirm=True
            ),
        ),
        (
            "ensemble_average",
            lambda: server.separate_ensemble(
                str(audio),
                [
                    EnsembleEntry(model="Kim Vocal 2", weight=0.6),
                    EnsembleEntry(model=args.vr_model, weight=0.4),
                ],
                output,
                options=options,
            ),
        ),
        (
            "ensemble_min",
            lambda: server.separate_ensemble(
                str(audio),
                [EnsembleEntry(model="Kim Vocal 2"), EnsembleEntry(model=args.vr_model)],
                output,
                algorithm="min_spec",
                options=options,
            ),
        ),
        (
            "ensemble_max",
            lambda: server.separate_ensemble(
                str(audio),
                [EnsembleEntry(model="Kim Vocal 2"), EnsembleEntry(model=args.vr_model)],
                output,
                algorithm="max_spec",
                options=options,
            ),
        ),
        (
            "chain",
            lambda: server.separate_chain(
                str(audio),
                [
                    ChainStep(model="Kim Vocal 2", output_stem="Vocals", options=options),
                    ChainStep(
                        model=args.cleanup_model, output_stem=args.cleanup_stem, options=options
                    ),
                ],
                output,
            ),
        ),
        ("vocal_splitter", lambda: server.separate_vocal_splitter(str(audio), output)),
    ]
    results = []
    try:
        for name, action in actions:
            job = wait(action(), wb)
            if name == "align" and job["state"] == "succeeded":
                assert job["result"]["measurements"]["offset_samples"] == 512
            if name == "sample" and job["state"] == "succeeded":
                assert job["result"]["outputs"][0]["frames"] == rate
            results.append({"case": name, **job})
            (ROOT / "scratch/workflow-results.json").write_text(
                json.dumps(results, indent=2),
                encoding="utf-8",
            )
            print(
                json.dumps({"case": name, "state": job["state"], "error": job.get("error")}),
                flush=True,
            )
        # Cancellation is verified against a real active separation, then another job runs.
        job = server.start_separation(str(audio), "htdemucs_ft", output, options=options)
        identifier = job["id"]
        deadline = time.monotonic() + 30
        while wb.manager.get(identifier)["state"] == "queued" and time.monotonic() < deadline:
            time.sleep(0.1)
        cancelled = server.cancel_job(identifier)
        assert cancelled["state"] == "cancelled"
        following = wait(server.create_sample(str(audio), output, 1, 0), wb)
        assert following["state"] == "succeeded"
        results.append(
            {
                "case": "cancel_and_resume",
                "state": "succeeded",
                "cancelled": cancelled,
                "following": following,
            }
        )
        (ROOT / "scratch/workflow-results.json").write_text(json.dumps(results, indent=2))
        print('{"case":"cancel_and_resume","state":"succeeded"}', flush=True)
    finally:
        wb.manager.close()
        server._workbench = None


if __name__ == "__main__":
    main()
