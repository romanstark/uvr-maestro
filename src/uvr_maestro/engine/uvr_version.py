"""Constants for the exact source revision installed by runtime setup."""

from dataclasses import dataclass


@dataclass(frozen=True)
class UVRVersion:
    """The engine source version, distinct from Maestro's package version."""

    version: str = "v5.6.0"
    patch: str = "UVR_Patch_9_29_23_1_39"
    patch_mac: str = "UVR_Patch_9_29_23_1_39"
    patch_linux: str = "UVR_Patch_9_29_23_1_39"


def load() -> UVRVersion:
    """Return the pinned source identity without importing its GUI."""
    return UVRVersion()
