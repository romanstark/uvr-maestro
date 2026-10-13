"""Source-relative excerpts, measured export invariants and descriptive tag removal."""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any

from uvr_maestro.engine.dependencies import library
from uvr_maestro.storage import MaestroError


def select_source(request: dict[str, Any]) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Round boundaries to nearest source sample and stage only the selected input."""
    from uvr_maestro.engine.audio import read_audio

    samples, rate = read_audio(request["audio_path"])
    contract = request.get("audio_contract", {})
    start = math.floor(contract.get("start_seconds", 0) * rate + 0.5)
    end_s = contract.get("end_seconds")
    end = len(samples) if end_s is None else math.floor(end_s * rate + 0.5)
    if not 0 <= start < end <= len(samples):
        raise MaestroError("invalid_selection", "selection lies outside the decoded source")
    selected = samples[start:end]
    target_rate = contract.get("output_sample_rate") or rate
    expected = (len(selected) * target_rate * 2 + rate) // (2 * rate)
    timeline = {
        "schema_version": 1,
        "source_path": request["audio_path"],
        "source_sample_rate": rate,
        "source_frames": len(samples),
        "start_sample": start,
        "end_sample": end,
        "source_offset_seconds": start / rate,
        "output_sample_rate": target_rate,
        "expected_output_frames": expected,
        "boundary_rounding": "nearest sample, ties upward; end exclusive",
        "alignment": "source timeline preserved; model attack latency is not calibrated",
    }
    if start or end != len(samples):
        target = Path(request["output_dir"]) / "_input" / "selection.wav"
        target.parent.mkdir(parents=True, exist_ok=True)
        library("soundfile").write(target, selected, rate, subtype="FLOAT")
        request = {**request, "audio_path": str(target)}
    return request, selected, timeline


def strip_tags(path: Path) -> None:
    """Retain mandatory audio structure and remove descriptive WAV/FLAC metadata."""
    raw = path.read_bytes()
    if path.suffix.casefold() == ".wav":
        if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
            raise MaestroError("unsupported_container", "exact export requires RIFF WAV")
        chunks = []
        offset = 12
        while offset + 8 <= len(raw):
            tag, length = struct.unpack_from("<4sI", raw, offset)
            end = offset + 8 + length
            if end > len(raw):
                raise MaestroError("invalid_output", "truncated WAV chunk")
            if tag in (b"fmt ", b"fact", b"data"):
                chunks.append(raw[offset:end] + (b"\0" if length % 2 else b""))
            offset = end + length % 2
        body = b"WAVE" + b"".join(chunks)
        path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    elif path.suffix.casefold() == ".flac":
        if raw[:4] != b"fLaC":
            raise MaestroError("invalid_output", "invalid FLAC container")
        offset, stream_info = 4, None
        while offset + 4 <= len(raw):
            flags = raw[offset]
            length = int.from_bytes(raw[offset + 1 : offset + 4], "big")
            end = offset + 4 + length
            if end > len(raw):
                raise MaestroError("invalid_output", "truncated FLAC metadata")
            if flags & 127 == 0:
                stream_info = raw[offset + 4 : end]
            offset = end
            if flags & 128:
                break
        if stream_info is None:
            raise MaestroError("invalid_output", "FLAC STREAMINFO is absent")
        path.write_bytes(
            b"fLaC\x80" + len(stream_info).to_bytes(3, "big") + stream_info + raw[offset:]
        )


def finalize(
    paths: list[Path], request: dict[str, Any], source: Any, timeline: dict[str, Any]
) -> tuple[list[Path], dict[str, Any]]:
    """Resample and reconcile frames before encoding, reporting every adjustment."""
    from uvr_maestro.engine.audio import convert_outputs, read_audio

    sf, np = library("soundfile"), library("numpy")
    settings = request["settings"]
    exact = request.get("audio_contract", {}).get("exact_length", True)
    if settings.get("save_format", "wav") == "mp3" and exact:
        raise MaestroError("lossy_exactness", "MP3 requires audio.exact_length=False")
    rate, frames = timeline["output_sample_rate"], timeline["expected_output_frames"]
    adjustments = []
    arrays = []
    for path in paths:
        samples, native_rate = read_audio(str(path))
        native_frames = len(samples)
        if native_rate != rate:
            divisor = math.gcd(native_rate, rate)
            samples = library("scipy.signal").resample_poly(
                samples, rate // divisor, native_rate // divisor, axis=0
            )
        if source.shape[1] == 1 and samples.shape[1] == 2:
            samples = samples.mean(axis=1, keepdims=True)
        elif source.shape[1] == 2 and samples.shape[1] == 1:
            samples = np.repeat(samples, 2, axis=1)
        delta = frames - len(samples)
        if abs(delta) > max(round(rate * 0.05), round(frames * 0.005)):
            raise MaestroError("duration_mismatch", "native duration differs beyond frame rounding")
        samples = samples[:frames]
        if delta > 0:
            samples = np.pad(samples, ((0, delta), (0, 0)))
        sf.write(path, samples, rate, subtype=settings.get("wav_subtype", "PCM_16"))
        arrays.append(samples)
        adjustments.append(
            {
                "stem": path.stem,
                "native_rate": native_rate,
                "native_frames": native_frames,
                "padded_frames": max(0, delta),
                "trimmed_frames": max(0, -delta),
            }
        )
    measurements: dict[str, Any] = {"frame_adjustments": adjustments}
    expected_stems = request.get("model", {}).get("stems", [])
    actual_labels = {path.stem.rsplit("_(", 1)[-1].rstrip(")").casefold() for path in paths}
    complete = actual_labels == {stem.casefold() for stem in expected_stems} and len(paths) >= 2
    if complete and not settings.get("normalize_output", False):
        source_rate = timeline["source_sample_rate"]
        if source_rate != rate:
            divisor = math.gcd(source_rate, rate)
            source = library("scipy.signal").resample_poly(
                source, rate // divisor, source_rate // divisor, axis=0
            )
        source = source[:frames]
        if len(source) < frames:
            source = np.pad(source, ((0, frames - len(source)), (0, 0)))
        if source.shape == arrays[0].shape:
            residual = source - sum(arrays)
            measurements["reconstruction"] = {
                "available": True,
                "rms": float(np.sqrt(np.mean(residual**2))),
                "peak": float(np.abs(residual).max()),
                "meaning": "aligned source minus sum of complementary outputs; not a quality score",
            }
    measurements.setdefault(
        "reconstruction",
        {
            "available": False,
            "reason": "requires all complementary sources, same channels, no normalization",
        },
    )
    converted = convert_outputs(paths, settings, request.get("executables", {}))
    for path in converted:
        strip_tags(path)
        info = sf.info(path)
        if exact and (info.frames != frames or info.samplerate != rate):
            raise MaestroError("export_mismatch", "encoded output violates rate/frame contract")
    timeline = {**timeline, "exact_length": exact, "descriptive_tags": "removed"}
    return converted, {"timeline": timeline, "measurements": measurements}
