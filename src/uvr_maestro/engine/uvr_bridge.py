"""Headless configuration adapter for the pinned UVR separation interface."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from uvr_maestro.catalog import CATALOG_DIR
from uvr_maestro.engine.dependencies import library
from uvr_maestro.models import ModelArch
from uvr_maestro.storage import MaestroError, contained, safe_name

REVISION = "a5f88453bfb2b38b05a965bcf67727243e0cbf19"


@lru_cache(maxsize=256)
def _identification_hash(path: str, size: int, modified: int) -> str:
    """Compute UVR's model-identification MD5 of the final ten MiB."""
    del modified
    with Path(path).open("rb") as stream:
        stream.seek(max(0, size - 10000 * 1024))
        return hashlib.md5(stream.read(), usedforsecurity=False).hexdigest()


def identification_hash(path: Path) -> str:
    """Cache UVR identification by resolved path, size and modification time."""
    details = path.stat()
    return _identification_hash(str(path.resolve()), details.st_size, details.st_mtime_ns)


def metadata_for(path: Path, arch: str, refreshed: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve model parameters by UVR identification hash, without inference."""
    kind = "vr" if arch.startswith("vr_") else "mdx"
    filename = CATALOG_DIR / f"{kind}_metadata.json"
    candidates = [path.parent / "model_data/model_data.json", filename]
    digest = identification_hash(path)
    explicit = path.parent / "model_data" / (digest + ".json")
    if explicit.is_file():
        result = json.loads(explicit.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("custom model parameters must be a JSON object")
        return result
    if refreshed:
        for key, metadata in refreshed.items():
            if digest in key:
                return dict(metadata)
    for candidate in candidates:
        if not candidate.is_file():
            continue
        mapping = json.loads(candidate.read_text(encoding="utf-8"))
        for key, metadata in mapping.items():
            if digest in key:
                return dict(metadata)
    raise MaestroError("metadata_missing", f"no UVR parameters for {path.name} ({digest})")


def complementary_stem(primary: str) -> str:
    """Map source names to UVR's actual complementary label without GUI imports."""
    pairs = {
        "Vocals": "Instrumental",
        "Instrumental": "Vocals",
        "lead_only": "backing_only",
        "backing_only": "lead_only",
    }
    return pairs.get(primary, primary[3:] if primary.startswith("No ") else "No " + primary)


def bootstrap(source: Path, support: Path) -> Any:
    """Load pinned source and the portable DiffQ decoder inside a worker."""
    if not (source / "separate.py").is_file():
        raise MaestroError("engine_not_loaded", f"missing UVR source: {source}")
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(support))
    from uvr_maestro.engine.compat import librosa_legacy_arguments

    librosa_legacy_arguments()
    sys.modules["diffq.bitpack"] = importlib.import_module("uvr_maestro.engine.diffq_bitpack")
    return importlib.import_module("separate")


def separate_class_for_arch(model_path: Path, arch: ModelArch) -> type | None:
    """Look up a class only after worker bootstrap has loaded the engine."""
    del model_path
    module = sys.modules.get("separate")
    names = {
        ModelArch.VR_5: "SeperateVR",
        ModelArch.VR_4: "SeperateVR",
        ModelArch.MDX: "SeperateMDX",
        ModelArch.MDXC: "SeperateMDXC",
        ModelArch.DEMUCS_V3: "SeperateDemucs",
        ModelArch.DEMUCS_V4: "SeperateDemucs",
    }
    return getattr(module, names.get(arch, ""), None) if module else None


def build_model_data(model: dict[str, Any], settings: dict[str, Any], source: Path) -> Any:
    """Populate the real ModelData attribute contract without importing UVR.py."""
    constants = importlib.import_module("gui_data.constants")
    arch = model["arch"]
    path = Path(model["path"])
    metadata = dict(model.get("metadata", {}))
    if not metadata and not arch.startswith("demucs"):
        metadata = metadata_for(path, arch)
    primary = metadata.get("primary_stem", model.get("primary_stem", "Vocals").title())
    safe_name(primary)
    safe_name(settings.get("stem", "All Stems"))
    data: dict[str, Any] = {
        key: False
        for key in (
            "is_pre_proc_model",
            "is_secondary_model",
            "is_secondary_model_activated",
            "is_primary_model_primary_stem_only",
            "is_primary_model_secondary_stem_only",
            "is_ensemble_mode",
            "is_multi_stem_ensemble",
            "is_mixer_mode",
            "is_demucs_pre_proc_model_inst_mix",
            "is_deverb_vocals",
            "is_denoise_model",
            "is_vocal_split_model",
            "is_inst_only_voc_splitter",
            "is_save_inst_vocal_splitter",
            "is_save_vocal_only",
            "is_sec_bv_rebalance",
            "is_use_opencl",
            "is_chunk_demucs",
        )
    }
    data.update(
        {
            "model_path": str(path),
            "model_basename": path.stem,
            "model_name": path.stem,
            "process_method": constants.VR_ARCH_TYPE
            if arch.startswith("vr_")
            else constants.DEMUCS_ARCH_TYPE
            if arch.startswith("demucs")
            else constants.MDX_ARCH_TYPE,
            "model_samplerate": 44100,
            "model_capacity": (32, 128),
            "primary_stem": primary,
            "primary_stem_native": primary,
            "secondary_stem": constants.secondary_stem(primary),
            "primary_model_primary_stem": primary,
            "ensemble_primary_stem": primary,
            "ensemble_secondary_stem": constants.secondary_stem(primary),
            "secondary_model": None,
            "secondary_model_scale": None,
            "secondary_model_4_stem": [],
            "secondary_model_4_stem_scale": [],
            "pre_proc_model": None,
            "vocal_split_model": None,
            "DENOISER_MODEL": "",
            "DEVERBER_MODEL": "",
            "deverb_vocal_opt": "Vocals",
            "is_karaoke": metadata.get("is_karaoke", False),
            "is_bv_model": metadata.get("is_bv_model", False),
            "bv_model_rebalance": metadata.get("is_bv_model_rebalanced", 0),
            "semitone_shift": settings.get("semitone_shift", 0.0),
            "is_pitch_change": settings.get("semitone_shift", 0.0) != 0,
            "is_match_frequency_pitch": settings.get("is_match_frequency_pitch", True),
            "is_gpu_conversion": 0 if settings.get("device", "cpu") != "cpu" else -1,
            "device_set": "Default",
            "is_normalization": settings.get("normalize_output", False),
            "is_primary_stem_only": settings.get("primary_stem_only", False),
            "is_secondary_stem_only": settings.get("secondary_stem_only", False),
            "is_invert_spec": settings.get("is_invert_spec", False),
            "wav_type_set": settings.get("wav_subtype", "PCM_16"),
            "mp3_bit_set": settings.get("mp3_bitrate", "320k"),
            "save_format": "WAV",
            "overlap": settings.get("overlap", 0.25),
            "overlap_mdx": settings.get("overlap_mdx", 0.25),
            "overlap_mdx23": settings.get("overlap_mdx23", 8),
            "is_mdx_combine_stems": True,
            "is_demucs_combine_stems": True,
            "mdxnet_stem_select": settings.get("stem", "All Stems"),
            "is_mdx_c": arch == "mdxc",
            "mdx_c_configs": None,
            "mixer_path": "",
            "is_vr_51_model": False,
            "is_mdx_ckpt": path.suffix == ".ckpt" and arch == "mdx",
            "is_denoise": settings.get("denoise", False),
            "is_mdx_c_seg_def": settings.get("mdxc_model_segment", True),
            "mdx_batch_size": settings.get("batch_size", 1),
            "mdx_segment_size": settings.get("segment_size", 256),
            "mdx_dim_f_set": metadata.get("mdx_dim_f_set", 0),
            "mdx_dim_t_set": metadata.get("mdx_dim_t_set", 8),
            "mdx_n_fft_scale_set": metadata.get("mdx_n_fft_scale_set", 0),
            "compensate": metadata.get("compensate", 1.0),
            "chunks": 0,
            "margin": 44100,
            "is_high_end_process": "mirroring" if settings.get("high_end", False) else "None",
            "is_tta": settings.get("tta", False),
            "is_post_process": settings.get("post_process", False),
            "batch_size": settings.get("batch_size", 1),
            "window_size": settings.get("window_size", 512),
            "post_process_threshold": settings.get("post_process_threshold", 0.2),
            "aggression_setting": settings.get("aggression", 5) / 100,
            "demucs_version": "v3" if arch == "demucs_v3" else "v4",
            "demucs_stems": settings.get("stem", "All Stems"),
            "demucs_source_list": ["drums", "bass", "other", "vocals"],
            "demucs_source_map": {"Bass": 0, "Drums": 1, "Other": 2, "Vocals": 3},
            "demucs_stem_count": 4,
            "segment": settings.get("demucs_segment", "Default"),
            "shifts": settings.get("shifts", 1),
            "is_split_mode": True,
            "vr_model_param": None,
        }
    )
    if arch.startswith("vr_"):
        folder = source / "lib_v5/vr_network/modelparams"
        param = contained(folder, folder / f"{safe_name(metadata['vr_model_param'])}.json")
        parameters = importlib.import_module("lib_v5.vr_network.model_param_init")
        data["vr_model_param"] = parameters.ModelParameters(str(param))
        data["model_samplerate"] = data["vr_model_param"].param["sr"]
        if "nout" in metadata and "nout_lstm" in metadata:
            data["is_vr_51_model"] = True
            data["model_capacity"] = (metadata["nout"], metadata["nout_lstm"])
    if arch == "mdx":
        if not all(data[key] for key in ("mdx_dim_f_set", "mdx_n_fft_scale_set")):
            raise MaestroError("metadata_missing", "MDX dimensions are missing")
        if data["overlap_mdx"] == "Default":
            data["overlap_mdx"] = 0.25
    if arch == "mdxc":
        from uvr_maestro.engine.config import load_config

        ConfigDict = library("ml_collections").ConfigDict
        config_name = metadata.get("config_yaml")
        config_path = model.get("artifact_paths", {}).get(config_name)
        if not config_path:
            raise MaestroError("config_missing", f"MDXC config missing: {config_name}")
        data["mdx_c_configs"] = ConfigDict(
            load_config(Path(config_path).read_text(encoding="utf-8"))
        )
        training = data["mdx_c_configs"].training
        data["primary_stem"] = training.target_instrument or training.instruments[0]
        for label in training.instruments:
            safe_name(label)
        requested_stem = settings.get("stem", "All Stems")
        if requested_stem != "All Stems":
            if requested_stem not in training.instruments:
                raise MaestroError("stem_unsupported", f"MDXC does not produce {requested_stem}")
            if training.target_instrument and requested_stem != training.target_instrument:
                raise MaestroError("stem_unsupported", "this MDXC model has one target source")
            data["primary_stem"] = requested_stem
        data["secondary_stem"] = constants.secondary_stem(data["primary_stem"])
        if len(training.instruments) == 2 and not training.target_instrument:
            data["mdxnet_stem_select"] = data["primary_stem"]
    if arch.startswith("demucs"):
        paths = model.get("artifact_paths", {})
        repo = Path(model["repo"])
        repo.mkdir(parents=True, exist_ok=True)
        for name, location in paths.items():
            target = repo / name
            if not target.exists():
                try:
                    os.link(location, target)
                except OSError:
                    shutil.copyfile(location, target)
        data["model_path"] = str(repo / path.name)
        if "6s" in path.stem:
            data["demucs_stem_count"] = 6
            data["demucs_source_map"].update({"Guitar": 4, "Piano": 5})
        elif "UVR_Demucs" in path.stem:
            data["demucs_stem_count"] = 2
            data["demucs_source_list"] = ["instrumental", "vocals"]
            data["demucs_source_map"] = {"Instrumental": 0, "Vocals": 1}
        if data["demucs_stems"] == "All Stems":
            data["primary_stem"] = "Primary Stem"
            data["secondary_stem"] = "Secondary Stem"
        else:
            if data["demucs_stems"] not in data["demucs_source_map"]:
                raise MaestroError(
                    "stem_unsupported", f"Demucs does not produce {data['demucs_stems']}"
                )
            data["primary_stem"] = data["demucs_stems"]
            data["secondary_stem"] = constants.secondary_stem(data["primary_stem"])
    return SimpleNamespace(**data)
