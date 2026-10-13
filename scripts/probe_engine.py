"""Run actual local architecture probes on an owned speech/music fixture."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf

from uvr_maestro.discovery import Locations
from uvr_maestro.toolargs import SeparationOptions
from uvr_maestro.workbench import Workbench

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("UVR_MAESTRO_DATA_DIR", str(ROOT / "scratch/integration-data"))
os.environ.setdefault("UVR_MAESTRO_RUNTIME_DIR", str(ROOT / ".runtime"))


def fixture() -> Path:
    """Mix optional owned speech or a generated harmonic signal with accompaniment."""
    path = ROOT / "scratch/voice-music.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    speech_path = ROOT / "scratch/speech.wav"
    if speech_path.is_file():
        speech, rate = sf.read(speech_path, always_2d=True)
    else:
        rate = 44100
        clock = np.arange(rate * 4) / rate
        voice = np.sin(2 * np.pi * 220 * clock) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * clock))
        speech = voice[:, None]
    import math

    from uvr_maestro.engine.dependencies import library

    resample_poly = library("scipy.signal").resample_poly

    divisor = math.gcd(rate, 44100)
    speech = resample_poly(speech.mean(axis=1), 44100 // divisor, rate // divisor)
    rate = 44100
    # Four seconds are sufficient for smoke tests and avoid private recordings.
    speech = speech[: rate * 4]
    speech = np.pad(speech, (0, max(0, rate * 4 - len(speech))))
    speech *= 0.25 / max(float(np.abs(speech).max()), 1e-6)
    t = np.arange(len(speech)) / rate
    accompaniment = (
        sum(np.sin(2 * np.pi * frequency * t) for frequency in (130.81, 164.81, 196.0)) * 0.03
    )
    sf.write(
        path,
        np.stack([speech + accompaniment, speech + accompaniment * 0.9], axis=1),
        rate,
        subtype="PCM_16",
    )
    return path


def wait(wb: Workbench, started: dict) -> dict:
    """Wait while the outer tool cell remains available for user updates."""
    identifier = started["id"]
    while True:
        job = wb.manager.get(identifier)
        if job["state"] in ("succeeded", "failed", "cancelled", "interrupted"):
            return job
        time.sleep(0.5)


def main() -> None:
    """Record truthful results for each advertised architecture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", help="Repeat for each local catalog model")
    parser.add_argument("--device", action="append", choices=("auto", "cpu", "cuda", "mps"))
    args = parser.parse_args()
    audio = fixture()
    wb = Workbench(Locations.discover())
    results: list[dict[str, Any]] = []
    models = args.model or [row.slug for row in wb.registry().filter(only_installed=True)]
    devices: list[Literal["auto", "cpu", "cuda", "mps"]] = args.device or ["cpu"]
    cases: list[tuple[str, Literal["auto", "cpu", "cuda", "mps"]]] = [
        (model, device) for model in models for device in devices
    ]
    try:
        for model, device in cases:
            started = time.monotonic()
            try:
                job = wait(
                    wb,
                    wb.start(
                        str(audio),
                        model,
                        str(ROOT / "scratch/engine-runs"),
                        options=SeparationOptions(device=device),
                    ),
                )
                result: dict[str, Any] = {
                    "model": model,
                    "requested_device": device,
                    "elapsed": time.monotonic() - started,
                    **job,
                }
            except Exception as error:  # noqa: BLE001
                result = {
                    "model": model,
                    "requested_device": device,
                    "state": "failed",
                    "error": str(error),
                }
            results.append(result)
            (ROOT / "scratch/engine-results.json").write_text(
                json.dumps(results, indent=2),
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "model": model,
                        "requested_device": device,
                        "state": result["state"],
                        "error": result.get("error"),
                        "outputs": len(result.get("result", {}).get("outputs", [])),
                    }
                ),
                flush=True,
            )
    finally:
        wb.manager.close()


if __name__ == "__main__":
    main()
