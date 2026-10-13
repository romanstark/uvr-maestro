"""Read-only desktop discovery and Maestro-owned storage locations."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def data_root() -> Path:
    """Return Maestro's independent per-user data directory."""
    override = os.environ.get("UVR_MAESTRO_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local")))
    elif sys.platform == "darwin":
        base = Path.home() / "Library/Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share")))
    return base / "uvr-maestro"


def desktop_install() -> Path | None:
    """Find a desktop install without importing or starting its application."""
    explicit = os.environ.get("UVR_MAESTRO_UVR_DIR")
    if explicit:
        target = Path(explicit).expanduser().resolve()
        return target if (target / "models").is_dir() else None
    candidates = [
        Path.home() / "AppData/Local/Programs/Ultimate Vocal Remover",
        Path("/Applications/Ultimate Vocal Remover.app/Contents/Resources"),
        Path.home() / "ultimatevocalremovergui",
    ]
    return next((path for path in candidates if (path / "models").is_dir()), None)


@dataclass(frozen=True)
class Locations:
    """Resolved ownership boundaries, with no filesystem side effects."""

    data: Path
    models: Path
    profiles: Path
    cache: Path
    runtime: Path
    desktop: Path | None

    @classmethod
    def discover(cls) -> Locations:
        """Resolve explicit overrides and desktop discovery."""
        data = data_root()
        return cls(
            data=data,
            models=Path(os.environ.get("UVR_MAESTRO_MODELS_DIR", str(data / "models"))),
            profiles=Path(os.environ.get("UVR_MAESTRO_SETTINGS_ROOT", str(data / "profiles"))),
            cache=data / "cache",
            runtime=Path(os.environ.get("UVR_MAESTRO_RUNTIME_DIR", str(data / "runtime"))),
            desktop=desktop_install(),
        )

    def model_roots(self) -> list[Path]:
        """Search owned models first, then the reused read-only installation."""
        roots = [self.models]
        if self.desktop:
            roots.append(self.desktop / "models")
        return roots
