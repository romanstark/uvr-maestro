"""Pinned UVR source identity. Maestro's own version is in __init__.py."""

from uvr_maestro.engine.uvr_version import load

_version = load()
VERSION = _version.version
PATCH = _version.patch
PATCH_MAC = _version.patch_mac
PATCH_LINUX = _version.patch_linux
