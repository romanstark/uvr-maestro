"""Classic Roformer adapter with explicit local weights and bounded overlap-add inference."""

from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Any

from uvr_maestro.engine.audio import describe
from uvr_maestro.engine.config import classic_roformer_variant, load_config
from uvr_maestro.engine.dependencies import library
from uvr_maestro.engine.export import finalize, select_source
from uvr_maestro.storage import MaestroError, safe_name, sha256


def inference(request: dict[str, Any], emit: Any) -> dict[str, Any]:
    """Load one classic architecture strictly; no Hub, downloader or training code."""
    torch, np, sf = library("torch"), library("numpy"), library("soundfile")
    started = time.monotonic()
    package = library("bs_roformer")
    torch.set_num_threads(4)
    request, source, timeline = select_source(request)
    model = request["model"]
    config_path = model["artifact_paths"][model["metadata"]["config_yaml"]]
    config = load_config(Path(config_path).read_text(encoding="utf-8"))
    classic_roformer_variant(config)
    settings = dict(request["settings"])
    requested = settings.get("device", "auto")
    device, reason = "cpu", None
    if requested in ("cuda", "auto") and torch.cuda.is_available():
        device = "cuda"
    elif requested in ("mps", "auto") and torch.backends.mps.is_available():
        device = "mps"
    elif requested not in ("cpu", "auto"):
        reason = f"{requested} backend is unavailable"
    variant = model["metadata"]["roformer_variant"]
    cls = package.MelBandRoformer if variant == "mel_band" else package.BSRoformer
    options = dict(config["model"])
    if options.pop("linear_transformer_depth", 0) != 0:
        raise MaestroError(
            "architecture_unsupported", "linear-transformer extensions need another adapter"
        )
    for key in ("freqs_per_bands", "multi_stft_resolutions_window_sizes"):
        if key in options:
            options[key] = tuple(options[key])
    options["flash_attn"] = device == "cuda"
    checkpoint = torch.load(model["path"], map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    state = {
        key.removeprefix("module.").removeprefix("_orig_mod."): value
        for key, value in state.items()
    }
    # Upstream configs count hidden MLP layers; the classic package counts all
    # affine layers. Resolve that convention from the actual checkpoint, then
    # require a strict full state-dict match. Never discard unmatched weights.
    affine = [
        key
        for key in state
        if re.fullmatch(r"mask_estimators\.0\.to_freqs\.0\.0\.\d+\.weight", key)
    ]
    configured_depth = options.get("mask_estimator_depth", 2)
    if affine:
        options["mask_estimator_depth"] = len(affine)
    network = cls(**options)
    network.load_state_dict(state, strict=True)
    network = network.to(device).eval()
    native_rate = int(config["audio"].get("sample_rate", 44100))
    source_rate = timeline["source_sample_rate"]
    samples = source
    if source_rate != native_rate:
        divisor = math.gcd(source_rate, native_rate)
        samples = library("scipy.signal").resample_poly(
            samples, native_rate // divisor, source_rate // divisor, axis=0
        )
    channels = 2 if options.get("stereo", False) else 1
    if channels == 2 and samples.shape[1] == 1:
        samples = np.repeat(samples, 2, axis=1)
    if channels == 1:
        samples = samples.mean(axis=1, keepdims=True)
    instruments = list(config["training"]["instruments"])
    target = config["training"].get("target_instrument")
    labels = [target] if target else instruments
    for label in labels:
        safe_name(label)
    nfft = int(options.get("stft_n_fft", 2048))
    length = len(samples)
    configured = int(
        config.get("inference", {}).get(
            "chunk_size", config["audio"].get("chunk_size", native_rate * 8)
        )
    )
    seconds = settings.get("roformer_segment_seconds")
    chunk = max(
        nfft * 2, min(configured if seconds is None else round(seconds * native_rate), length)
    )
    hop = int(options.get("stft_hop_length", 512))
    chunk = math.ceil(chunk / hop) * hop
    overlap = settings.get("roformer_overlap", 0.25)
    stride = max(1, round(chunk * (1 - overlap)))
    result = np.zeros((len(labels), length, channels), dtype=np.float32)
    weights = np.zeros(length, dtype=np.float32)
    starts = list(range(0, length, stride))
    with torch.inference_mode():
        for index, start in enumerate(starts):
            fragment = samples[start : start + chunk]
            actual = len(fragment)
            if actual < chunk:
                fragment = np.pad(fragment, ((0, chunk - actual), (0, 0)))
            tensor = torch.from_numpy(fragment.T.astype(np.float32)).unsqueeze(0).to(device)
            predicted = network(tensor).float().cpu().numpy()
            if predicted.ndim == 3:
                predicted = predicted[:, None]
            predicted = predicted[0, :, :, :actual].transpose(0, 2, 1)
            if (
                predicted.shape != (len(labels), actual, channels)
                or not np.isfinite(predicted).all()
            ):
                raise MaestroError(
                    "invalid_output", "Roformer output dimensions or values are invalid"
                )
            window = np.maximum(0.05, np.hanning(chunk))[:actual]
            result[:, start : start + actual] += predicted * window[None, :, None]
            weights[start : start + actual] += window
            emit({"event": "progress", "progress": (index + 1) / len(starts) * 0.98})
    if not np.all(weights > 0):
        raise MaestroError("invalid_output", "overlap-add left uncovered samples")
    result /= weights[None, :, None]
    paths = []
    output = Path(request["output_dir"])
    for index, label in enumerate(labels):
        path = output / f"source_({label}).wav"
        sf.write(path, result[index], native_rate, subtype="FLOAT")
        paths.append(path)
    if target and len(instruments) == 2:
        complement = next(label for label in instruments if label != target)
        safe_name(complement)
        path = output / f"source_({complement}).wav"
        sf.write(path, samples - result[0], native_rate, subtype="FLOAT")
        paths.append(path)
    paths, export = finalize(paths, request, source, timeline)
    return {
        "success": True,
        "outputs": [describe(path) for path in paths],
        "device": device,
        "requested_device": requested,
        "fallback_reason": reason,
        "settings": settings,
        "engine_revision": "bs-roformer-0.3.0",
        "torch_version": torch.__version__,
        "model_fingerprints": {
            name: sha256(Path(path)) for name, path in model["artifact_paths"].items()
        },
        "model_parameter_adjustments": {
            "mask_estimator_depth": {
                "configured": configured_depth,
                "effective": options.get("mask_estimator_depth", 2),
                "basis": "checkpoint affine-layer count; strict state-dict load",
            }
        },
        "runtime_seconds": time.monotonic() - started,
        **export,
    }
