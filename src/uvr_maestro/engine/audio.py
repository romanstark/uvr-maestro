"""Real file-based audio transforms and measurements in the isolated runtime."""

from __future__ import annotations

import io
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from uvr_maestro.engine.dependencies import library
from uvr_maestro.storage import MaestroError, sha256


def read_audio(path: str) -> tuple[Any, int]:
    """Read finite mono/stereo audio, refusing empty and multichannel files."""
    sf, np = library("soundfile"), library("numpy")
    try:
        samples, rate = sf.read(path, always_2d=True)
    except (RuntimeError, OSError):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise MaestroError("codec_unavailable", "input codec requires FFmpeg") from None
        process = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-i",
                path,
                "-f",
                "wav",
                "-acodec",
                "pcm_f32le",
                "pipe:1",
            ],
            capture_output=True,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        samples, rate = sf.read(io.BytesIO(process.stdout), always_2d=True)
    if not samples.size or samples.shape[1] > 2 or not np.isfinite(samples).all():
        raise MaestroError("invalid_audio", "audio must be finite, nonempty mono or stereo")
    return samples, rate


def describe(path: Path) -> dict[str, Any]:
    """Verify produced audio and return measured characteristics."""
    sf, np = library("soundfile"), library("numpy")
    samples, rate = read_audio(str(path))
    info = sf.info(path)
    peak = float(np.abs(samples).max())
    loudness = None
    if len(samples) >= int(rate * 0.4) and peak > 0:
        value = float(library("pyloudnorm").Meter(rate).integrated_loudness(samples))
        if math.isfinite(value):
            loudness = value
    return {
        "path": str(path),
        "stem": path.stem.rsplit("_(", 1)[-1].rstrip(")"),
        "sample_rate": rate,
        "channels": samples.shape[1],
        "frames": len(samples),
        "duration_seconds": len(samples) / rate,
        "peak": peak,
        "rms": float(np.sqrt(np.mean(samples**2))),
        "integrated_lufs": loudness,
        "clipped_samples": int(np.count_nonzero(np.abs(samples) >= 1)),
        "subtype": info.subtype,
        "sha256": sha256(path),
    }


def convert_outputs(
    paths: list[Path],
    settings: dict[str, Any],
    executables: dict[str, str],
) -> list[Path]:
    """Convert staged WAV outputs only after a successful source write."""
    sf = library("soundfile")
    output_format = settings.get("save_format", "wav")
    if output_format == "wav":
        return paths
    converted = []
    for path in paths:
        target = path.with_suffix("." + output_format)
        if target.exists():
            raise MaestroError("output_collision", str(target))
        if output_format == "flac":
            samples, rate = read_audio(str(path))
            sf.write(target, samples, rate, subtype="PCM_24")
        elif output_format == "mp3":
            ffmpeg = executables.get("ffmpeg")
            if not ffmpeg:
                raise MaestroError("ffmpeg_missing", "MP3 output requires ffmpeg")
            subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-v",
                    "error",
                    "-n",
                    "-i",
                    str(path),
                    "-b:a",
                    settings.get("mp3_bitrate", "320k"),
                    str(target),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        else:
            raise ValueError("unsupported output format")
        describe(target)
        path.unlink()
        converted.append(target)
    return converted


def execute_audio(request: dict[str, Any]) -> dict[str, Any]:
    """Execute a utility, persisting actual audio or measured analysis."""
    started = time.monotonic()
    import os

    for filename in request.get("executables", {}).values():
        os.environ["PATH"] = str(Path(filename).parent) + os.pathsep + os.environ.get("PATH", "")
    sf, np = library("soundfile"), library("numpy")
    operation = request["operation"]
    output = Path(request["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    measurements: dict[str, Any] = {}
    settings = request.get("settings", {})
    subtype = settings.get("wav_subtype", "PCM_16")
    if operation == "ensemble":
        for group in request["groups"]:
            files = group["files"]
            target = output / (group["stem"].replace(" ", "_") + ".wav")
            algorithm = request["algorithm"]
            if algorithm == "average":
                arrays = [read_audio(item["path"]) for item in files]
                rate = arrays[0][1]
                channels = arrays[0][0].shape[1]
                if any(item[1] != rate or item[0].shape[1] != channels for item in arrays):
                    raise MaestroError("incompatible_outputs", "ensemble rates or channels differ")
                frames = max(len(item[0]) for item in arrays)
                if frames - min(len(item[0]) for item in arrays) > rate * 0.05:
                    raise MaestroError("incompatible_outputs", "ensemble lengths differ over 50 ms")
                measurements["input_padding_frames"] = [frames - len(item[0]) for item in arrays]
                arrays = [
                    (np.pad(item[0], ((0, frames - len(item[0])), (0, 0))), item[1])
                    for item in arrays
                ]
                weight = sum(item["weight"] for item in files)
                mixed = sum(item[0] * file["weight"] for item, file in zip(arrays, files)) / weight
                sf.write(target, mixed, rate, subtype=subtype)
            else:
                if any(item["weight"] != 1 for item in files):
                    raise ValueError("spectral min/max ensembles require unit weights")
                from uvr_maestro.engine.uvr_bridge import bootstrap

                bootstrap(Path(request["source"]), Path(request["support"]))
                utils = library("lib_v5.spec_utils")
                algorithm_name = "Min Spec" if algorithm == "min_spec" else "Max Spec"
                utils.ensemble_inputs(
                    [item["path"] for item in files],
                    algorithm_name,
                    settings.get("normalize_output", False),
                    subtype,
                    str(target),
                )
            paths.append(target)
    elif operation == "matchering":
        matchering = library("matchering")
        target = output / "mastered.wav"
        matchering.process(
            target=request["audio_path"],
            reference=request["reference"],
            results=[matchering.pcm24(str(target))],
        )
        paths.append(target)
    else:
        samples, rate = read_audio(request["audio_path"])
        target = output / (Path(request["audio_path"]).stem + "_" + operation + ".wav")
        if operation in ("pitch_shift", "time_stretch"):
            rubberband = request.get("executables", {}).get("rubberband")
            if not rubberband:
                raise MaestroError("rubberband_missing", "Rubber Band executable not found")
            import os

            os.environ["PATH"] = (
                str(Path(rubberband).parent) + os.pathsep + os.environ.get("PATH", "")
            )
            pyrb = library("pyrubberband")
            if operation == "pitch_shift":
                transformed = pyrb.pitch_shift(samples, rate, request["semitones"])
            else:
                transformed = pyrb.time_stretch(samples, rate, request["rate"])
            sf.write(target, transformed, rate, subtype=subtype)
            paths.append(target)
        elif operation == "create_sample":
            offset = request.get("offset_seconds")
            start = int((len(samples) / rate / 3 if offset is None else offset) * rate)
            frames = int(request["duration_seconds"] * rate)
            clipped = samples[start : start + frames]
            if not len(clipped):
                raise ValueError("sample offset is beyond the end of the audio")
            sf.write(target, clipped, rate, subtype=subtype)
            paths.append(target)
        elif operation == "align_inputs":
            reference, reference_rate = read_audio(request["reference"])
            if rate != reference_rate:
                signal = library("scipy.signal")
                divisor = math.gcd(rate, reference_rate)
                samples = signal.resample_poly(
                    samples,
                    reference_rate // divisor,
                    rate // divisor,
                    axis=0,
                )
                rate = reference_rate
            signal = library("scipy.signal")
            left, right = reference.mean(axis=1), samples.mean(axis=1)
            if not np.any(left) or not np.any(right):
                raise MaestroError("alignment_unresolved", "cannot align silent inputs")
            correlation = signal.correlate(right, left, mode="full", method="fft")
            lag = int(np.argmax(np.abs(correlation)) - (len(left) - 1))
            aligned = samples[max(0, lag) :]
            if lag < 0:
                aligned = np.pad(aligned, ((-lag, 0), (0, 0)))
            aligned = aligned[: len(reference)]
            if len(aligned) < len(reference):
                aligned = np.pad(aligned, ((0, len(reference) - len(aligned)), (0, 0)))
            sf.write(target, aligned, rate, subtype=subtype)
            paths.append(target)
            measurements = {
                "offset_samples": lag,
                "offset_seconds": lag / rate,
                "meaning": "positive means target lagged behind reference",
            }
        elif operation == "analyze_audio":
            peak = float(np.abs(samples).max())
            rms = float(np.sqrt(np.mean(samples**2)))
            loudness = None
            if len(samples) >= int(rate * 0.4) and peak > 0:
                meter = library("pyloudnorm").Meter(rate)
                measured = float(meter.integrated_loudness(samples))
                if math.isfinite(measured):
                    loudness = measured
            measurements = {
                "sample_rate": rate,
                "channels": samples.shape[1],
                "duration_seconds": len(samples) / rate,
                "peak": peak,
                "rms": rms,
                "integrated_lufs": loudness,
                "clipped_samples": int(np.count_nonzero(np.abs(samples) >= 1)),
            }
        else:
            raise ValueError(f"unknown audio operation: {operation}")
    paths = convert_outputs(paths, settings, request.get("executables", {}))
    return {
        "success": True,
        "outputs": [describe(path) for path in paths],
        "measurements": measurements,
        "device": "cpu",
        "runtime_seconds": time.monotonic() - started,
        "settings": settings,
    }
