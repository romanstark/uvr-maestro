"""Translate UVR's download lists into explicit model artifact sets."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

from uvr_maestro.models import Artifact, ModelArch, ModelSpec, Stem
from uvr_maestro.spec import EndpointSpec
from uvr_maestro.storage import safe_name


def purposes_for(name: str) -> list[str]:
    """Describe a model's task without renaming its audio outputs."""
    normalized = name.casefold()
    result = []
    for words, purpose in (
        (("de-echo", "deecho"), "de_echo"),
        (("dereverb", "deverb", "reverb"), "de_reverb"),
        (("denoise",), "de_noise"),
        (("karaoke",), "karaoke"),
        (("bve", "backing"), "backing_vocals"),
        (("wind",), "wind"),
    ):
        if any(word in normalized for word in words):
            result.append(purpose)
    return result


def infer_stem(name: str, purposes: list[str]) -> Stem:
    """Supply discovery hints until metadata identifies the actual model."""
    if purposes:
        return Stem(purposes[0])
    lowered = name.casefold()
    for word, stem in (
        ("bass", Stem.BASS),
        ("drum", Stem.DRUMS),
        ("other", Stem.OTHER),
        ("vocal", Stem.VOCALS),
        ("inst", Stem.INSTRUMENTAL),
    ):
        if word in lowered:
            return stem
    return Stem.INSTRUMENTAL if "HP" in name and "Vocal" not in name else Stem.VOCALS


def parse_downloads(
    payload: dict[str, Any], endpoints: EndpointSpec, names: dict[str, Any] | None = None
) -> list[ModelSpec]:
    """Parse the supported public lists and disclose newer unsupported networks."""
    if not isinstance(payload.get("vr_download_list"), dict):
        raise TypeError("upstream download list lacks VR catalog")
    rows: dict[str, ModelSpec] = {}

    def add(label: str, arch: ModelArch, artifacts: list[Artifact]) -> None:
        main = next((item for item in artifacts if item.role == "bag"), artifacts[0])
        slug = safe_name(Path(main.filename).stem)
        purposes = purposes_for(label)
        primary = infer_stem(label, purposes)
        aliases = [label.split(": ", 1)[-1], main.filename]
        if names:
            mapped = names.get(Path(main.filename).stem) or names.get(main.filename)
            if isinstance(mapped, str):
                aliases.append(mapped)
        metadata = {}
        configs = [item.filename for item in artifacts if item.role == "config"]
        if arch is ModelArch.MDXC and configs:
            metadata["config_yaml"] = configs[0]
        stems = ["Vocals", "Instrumental"]
        if arch in (ModelArch.DEMUCS_V3, ModelArch.DEMUCS_V4):
            stems = ["Drums", "Bass", "Other", "Vocals"]
            if "6s" in slug:
                stems += ["Guitar", "Piano"]
        rows[slug] = ModelSpec(
            slug=slug,
            display_name=label,
            arch=arch,
            primary_stem=primary,
            secondary_stem=Stem.INSTRUMENTAL if primary is Stem.VOCALS else Stem.VOCALS,
            sample_rate=44100,
            download_url=main.url,
            artifacts=artifacts,
            metadata=metadata,
            aliases=aliases,
            purposes=purposes,
            stems=stems,
            notes="Output names and model parameters are resolved from UVR hash metadata.",
        )

    for key, directory, default_arch in (
        ("vr_download_list", "VR_Models", ModelArch.VR_5),
        ("mdx_download_list", "MDX_Net_Models", ModelArch.MDX),
    ):
        for label, filename in payload.get(key, {}).items():
            safe_name(filename)
            arch = (
                ModelArch.VR_4
                if default_arch is ModelArch.VR_5 and "v4:" in label
                else default_arch
            )
            add(
                label,
                arch,
                [
                    Artifact(
                        filename, directory, endpoints.model_repo_release_base + quote(filename)
                    )
                ],
            )
    for key in ("mdx23_download_list", "mdx23c_download_list"):
        for label, config_map in payload.get(key, {}).items():
            for filename, config in config_map.items():
                safe_name(filename)
                safe_name(config)
                add(
                    label,
                    ModelArch.MDXC,
                    [
                        Artifact(
                            filename,
                            "MDX_Net_Models",
                            endpoints.model_repo_release_base + quote(filename),
                        ),
                        Artifact(
                            config,
                            "MDX_Net_Models/model_data/mdx_c_configs",
                            endpoints.mdx23_configs_base + quote(config),
                            role="config",
                        ),
                    ],
                )
    for label, downloads in payload.get("demucs_download_list", {}).items():
        arch = (
            ModelArch.DEMUCS_V4
            if "v4:" in label
            else (ModelArch.DEMUCS_V3 if "v3:" in label else ModelArch.LEGACY_DEMUCS)
        )
        directory = (
            "Demucs_Models/v3_v4_repo" if arch is not ModelArch.LEGACY_DEMUCS else ("Demucs_Models")
        )
        artifacts = [
            Artifact(
                safe_name(filename),
                directory,
                url,
                role="bag" if filename.endswith(".yaml") else "weights",
            )
            for filename, url in downloads.items()
        ]
        add(label, arch, artifacts)
    for key in ("other_network_list", "other_network_list_new"):
        for label, downloads in payload.get(key, {}).items():
            artifacts = [
                Artifact(
                    safe_name(filename),
                    "MDX_Net_Models",
                    url,
                    role="config" if filename.endswith(".yaml") else "weights",
                )
                for filename, url in downloads.items()
            ]
            if artifacts:
                architecture = (
                    ModelArch.ROFORMER if "roformer" in label.casefold() else ModelArch.UNKNOWN
                )
                add(label, architecture, artifacts)
    return list(rows.values())
