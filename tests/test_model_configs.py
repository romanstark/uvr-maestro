"""Upstream tuple data is accepted without enabling Python object construction."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from uvr_maestro.engine import uvr_bridge
from uvr_maestro.engine.config import classic_roformer_variant, load_config
from uvr_maestro.storage import MaestroError


def test_harmless_tuple_and_unsafe_object_tag():
    assert load_config("model:\n  bands: !!python/tuple [2, 4, 8]\n")["model"]["bands"] == (2, 4, 8)
    with pytest.raises(yaml.constructor.ConstructorError):
        load_config("model: !!python/object/apply:builtins.str ['do not execute']")


def test_modified_roformer_does_not_silently_use_the_classic_adapter():
    config = {"model": {"dim": 384, "depth": 6, "num_bands": 60}}
    assert classic_roformer_variant(config) == "mel_band"
    with pytest.raises(MaestroError, match="modified"):
        classic_roformer_variant({"model": {"num_bands": 60, "add_value_residual": True}})


def test_mdxc_inference_uses_the_same_restricted_config_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DrumSep's augmentation tuple must survive registration AND inference preparation."""
    config = tmp_path / "drumsep.yaml"
    config.write_text(
        "training:\n  target_instrument: null\n  instruments: [kick, snare]\n"
        "  mixup_probs: !!python/tuple [0.5, 0.25]\n",
        encoding="utf-8",
    )
    original = uvr_bridge.importlib.import_module
    constants = SimpleNamespace(
        VR_ARCH_TYPE="VR",
        DEMUCS_ARCH_TYPE="Demucs",
        MDX_ARCH_TYPE="MDX",
        secondary_stem=lambda stem: "No " + stem,
    )
    monkeypatch.setattr(
        uvr_bridge.importlib,
        "import_module",
        lambda name: constants if name == "gui_data.constants" else original(name),
    )

    def nested(data):
        return SimpleNamespace(
            **{k: nested(v) if isinstance(v, dict) else v for k, v in data.items()}
        )

    monkeypatch.setattr(uvr_bridge, "library", lambda name: SimpleNamespace(ConfigDict=nested))
    model = {
        "arch": "mdxc",
        "path": str(tmp_path / "weights.ckpt"),
        "metadata": {"config_yaml": config.name},
        "artifact_paths": {config.name: str(config)},
    }
    prepared = uvr_bridge.build_model_data(model, {"stem": "kick"}, tmp_path)
    assert prepared.mdx_c_configs.training.mixup_probs == (0.5, 0.25)
    assert prepared.primary_stem == "kick"
    config.write_text("training: !!python/object/apply:builtins.str ['unsafe']", encoding="utf-8")
    with pytest.raises(yaml.constructor.ConstructorError):
        uvr_bridge.build_model_data(model, {}, tmp_path)
