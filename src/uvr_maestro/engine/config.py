"""Model configurations are data, including upstream's harmless tuple notation."""

from typing import Any

import yaml

from uvr_maestro.storage import MaestroError


class ConfigLoader(yaml.SafeLoader):
    """Support sequences tagged as tuples without enabling arbitrary Python objects."""


def _tuple(loader: ConfigLoader, node: yaml.Node) -> tuple[Any, ...]:
    if not isinstance(node, yaml.SequenceNode):
        raise TypeError("tuple configuration tag requires a sequence")
    return tuple(loader.construct_sequence(node))


ConfigLoader.add_constructor("tag:yaml.org,2002:python/tuple", _tuple)


def load_config(text: str) -> dict[str, Any]:
    """Accept a mapping and reject all other unsafe Python tags."""
    data = yaml.load(text, Loader=ConfigLoader)
    if not isinstance(data, dict):
        raise TypeError("model configuration must be a mapping")
    return data


def classic_roformer_variant(config: dict[str, Any]) -> str:
    """Validate the classic constructor contract before registering a model."""
    model = config["model"]
    common = {
        "dim", "depth", "stereo", "num_stems", "time_transformer_depth",
        "freq_transformer_depth", "dim_head", "heads", "attn_dropout", "ff_dropout",
        "flash_attn", "dim_freqs_in", "stft_n_fft", "stft_hop_length", "stft_win_length",
        "stft_normalized", "mask_estimator_depth", "multi_stft_resolution_loss_weight",
        "multi_stft_resolutions_window_sizes", "multi_stft_hop_size", "multi_stft_normalized",
        "linear_transformer_depth",
    }
    if "num_bands" in model and "freqs_per_bands" not in model:
        variant, allowed = "mel_band", common | {"num_bands", "sample_rate"}
    elif "freqs_per_bands" in model and "num_bands" not in model:
        variant, allowed = "band_split", common | {"freqs_per_bands"}
    else:
        raise MaestroError("architecture_unsupported", "cannot identify a classic Roformer config")
    extras = set(model) - allowed
    if extras or model.get("linear_transformer_depth", 0) != 0:
        raise MaestroError(
            "architecture_unsupported", f"modified Roformer configuration: {sorted(extras)}"
        )
    return variant
