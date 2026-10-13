"""Narrow argument compatibility for UVR's librosa 0.9 call conventions."""

from __future__ import annotations

from functools import wraps
from typing import Any

from uvr_maestro.engine.dependencies import library


def librosa_legacy_arguments() -> None:
    """Translate legacy positional arguments into modern keyword-only arguments."""
    librosa = library("librosa")
    if getattr(librosa, "_uvr_maestro_compat", False):
        return
    parameters = {
        "load": ("path", "sr", "mono", "offset", "duration", "dtype", "res_type"),
        "resample": ("y", "orig_sr", "target_sr", "res_type", "fix", "scale"),
        "stft": ("y", "n_fft", "hop_length", "win_length", "window", "center", "dtype", "pad_mode"),
    }

    def adapter(function: Any, names: tuple[str, ...]) -> Any:
        @wraps(function)
        def call(*args: Any, **kwargs: Any) -> Any:
            if len(args) > len(names):
                raise TypeError("too many positional arguments")
            values = dict(zip(names, args))
            if values.keys() & kwargs.keys():
                raise TypeError("argument supplied both positionally and by keyword")
            values.update(kwargs)
            return function(**values)

        return call

    for name, names in parameters.items():
        setattr(librosa, name, adapter(getattr(librosa, name), names))
    librosa._uvr_maestro_compat = True
