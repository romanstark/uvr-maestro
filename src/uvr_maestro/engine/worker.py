"""Local inference worker with a JSON-lines progress/result channel."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Load only Maestro's package, never the control interpreter's site-packages.
# The native worker may use Python 3.11 while the server uses Python 3.12+.
if "uvr_maestro" not in sys.modules:
    package_root = Path(__file__).resolve().parent.parent
    package_spec = importlib.util.spec_from_file_location(
        "uvr_maestro",
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    if package_spec is None or package_spec.loader is None:
        raise RuntimeError("Maestro worker package is missing")
    package_module = importlib.util.module_from_spec(package_spec)
    sys.modules["uvr_maestro"] = package_module
    package_spec.loader.exec_module(package_module)

from uvr_maestro.engine.audio import describe, execute_audio
from uvr_maestro.engine.dependencies import library
from uvr_maestro.engine.uvr_bridge import (
    REVISION,
    bootstrap,
    build_model_data,
    separate_class_for_arch,
)
from uvr_maestro.models import ModelArch

CHANNEL = sys.stdout


def emit(payload: dict[str, Any]) -> None:
    """Write one worker protocol record outside redirected engine stdout."""
    print(json.dumps(payload, allow_nan=False), file=CHANNEL, flush=True)


def select_device(
    requested: str, arch: ModelArch, cuda_available: bool, mps_available: bool
) -> tuple[str, str | None]:
    """Select an available backend, including measured pinned-engine limitations."""
    device, reason = "cpu", None
    if requested in ("cuda", "auto") and cuda_available:
        device = "cuda"
    elif requested in ("mps", "auto") and mps_available:
        device = "mps"
    elif requested not in ("cpu", "auto"):
        reason = f"{requested} backend is unavailable"
    if device == "mps" and arch is ModelArch.DEMUCS_V4:
        device = "cpu"
        reason = (
            "Demucs v4 uses CPU: the pinned PyTorch MPS backend exceeded its "
            "65536-channel convolution limit in native htdemucs and htdemucs_6s tests"
        )
    return device, reason


def inference(request: dict[str, Any]) -> dict[str, Any]:
    """Run one resolved model and validate its actual audio outputs."""
    started = time.monotonic()
    for filename in request.get("executables", {}).values():
        os.environ["PATH"] = str(Path(filename).parent) + os.pathsep + os.environ.get("PATH", "")
    if request["model"]["arch"] == "roformer":
        from uvr_maestro.engine.roformer import inference as roformer_inference

        return roformer_inference(request, emit)
    source = Path(request["source"])
    executable_paths = request.get("executables", {})
    for filename in executable_paths.values():
        os.environ["PATH"] = str(Path(filename).parent) + os.pathsep + os.environ.get("PATH", "")
    bootstrap(source, Path(request["support"]))
    from uvr_maestro.engine.export import finalize, select_source

    request, selected_source, timeline = select_source(request)
    torch = library("torch")

    torch.set_num_threads(int(request.get("threads", 4)))
    settings = dict(request["settings"])
    requested = settings.get("device", "cpu")
    arch = ModelArch(request["model"]["arch"])
    device, reason = select_device(
        requested, arch, torch.cuda.is_available(), torch.backends.mps.is_available()
    )
    if arch is ModelArch.MDX and device == "cuda":
        ort = library("onnxruntime")
        # ONNX models can run through UVR's Torch adapter when ORT CUDA is absent.
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            settings["segment_size"] = 512
    settings["device"] = device
    model = dict(request["model"])
    model["repo"] = str(Path(request["output_dir"]) / ".demucs-repo")
    data = build_model_data(model, settings, source)
    if request.get("operation") == "vocal_splitter":
        split = build_model_data(request["split_model"], settings, source)
        split.is_vocal_split_model = True
        split.is_secondary_model = True
        split.primary_model_primary_stem = "Vocals"
        backing_primary = split.is_bv_model == (split.primary_stem_native == "Vocals")
        split.primary_stem = "backing_only" if backing_primary else "lead_only"
        split.secondary_stem = "lead_only" if backing_primary else "backing_only"
        data.vocal_split_model = split
        data.is_save_inst_vocal_splitter = False
        data.is_sec_bv_rebalance = bool(split.bv_model_rebalance)
    if arch is ModelArch.MDX and device == "cuda":
        ort = library("onnxruntime")
        if (
            "CUDAExecutionProvider" not in ort.get_available_providers()
            and data.mdx_segment_size == 2**data.mdx_dim_t_set
        ):
            data.mdx_segment_size *= 2
            settings["segment_size"] = data.mdx_segment_size
    output = Path(request["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    executables = request.get("executables", {})
    if executables.get("ffmpeg"):
        pydub = library("pydub")
        pydub.AudioSegment.converter = executables["ffmpeg"]
    if executables.get("rubberband"):
        executable = Path(executables["rubberband"])
        os.environ["PATH"] = str(executable.parent) + os.pathsep + os.environ.get("PATH", "")
    cached: dict[str, Any] = {}

    def progress(value: float, *args: Any, **kwargs: Any) -> None:
        del kwargs
        total = float(value)
        if args and isinstance(args[0], (int, float)):
            total += float(args[0])
        emit({"event": "progress", "progress": min(0.98, max(0.0, total))})

    def console(*args: Any, **kwargs: Any) -> None:
        del kwargs
        emit({"event": "log", "message": " ".join(str(arg) for arg in args)})

    def get_cache(architecture: str, model_name: str) -> tuple[str | None, Any]:
        return model_name, cached.get(architecture + model_name)

    def put_cache(architecture: str, sources: Any, model_name: str) -> None:
        cached[architecture + model_name] = sources

    process_data = {
        "audio_file": request["audio_path"],
        "audio_file_base": Path(request["audio_path"]).stem,
        "export_path": str(output),
        "set_progress_bar": progress,
        "write_to_console": console,
        "cached_source_callback": get_cache,
        "cached_model_source_holder": put_cache,
        "is_4_stem_ensemble": False,
        "is_ensemble_master": False,
        "list_all_models": [data.model_basename],
        "process_iteration": lambda: None,
    }
    klass = separate_class_for_arch(Path(data.model_path), arch)
    if klass is None:
        raise ValueError(f"unsupported architecture: {arch.value}")
    instance = klass(data, process_data)
    instance.seperate()
    paths, export = finalize(
        sorted(output.glob("*.wav")), {**request, "settings": settings}, selected_source, timeline
    )
    results = [describe(path) for path in paths]
    if not results:
        raise ValueError("engine produced no audio files")
    from uvr_maestro.storage import sha256

    fingerprints = {
        filename: sha256(Path(location))
        for filename, location in model.get("artifact_paths", {}).items()
    }
    return {
        "success": True,
        "outputs": results,
        "device": str(instance.device),
        "requested_device": requested,
        "fallback_reason": reason,
        "settings": settings,
        "runtime_seconds": time.monotonic() - started,
        "engine_revision": REVISION,
        "torch_version": torch.__version__,
        "model_fingerprints": fingerprints,
        **export,
    }


def main() -> None:
    """Read a request file and keep engine output off the protocol stream."""
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        with contextlib.redirect_stdout(sys.stderr):
            if request.get("operation") == "roformer_preflight":
                library("bs_roformer")
                torch = library("torch")
                result = {
                    "success": True,
                    "torch_version": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "mps_available": torch.backends.mps.is_available(),
                    "architectures": ["mel_band", "band_split"],
                }
            elif request.get("operation") == "preflight":
                bootstrap(Path(request["source"]), Path(request["support"]))
                torch, ort = library("torch"), library("onnxruntime")
                result = {
                    "success": True,
                    "torch_version": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "mps_available": torch.backends.mps.is_available(),
                    "onnx_providers": ort.get_available_providers(),
                    "engine_revision": REVISION,
                }
            elif request.get("operation") in (None, "separation", "vocal_splitter"):
                result = inference(request)
            else:
                result = execute_audio(request)
        emit({"event": "result", **result})
    except Exception as error:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        emit(
            {
                "event": "result",
                "success": False,
                "code": getattr(error, "code", "engine_error"),
                "error": str(error),
            }
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
