"""Packaged offline model snapshot and UVR parameter metadata."""

from pathlib import Path

CATALOG_DIR = Path(__file__).parent
DEFAULT_CATALOG = CATALOG_DIR / "10-vr5.yaml"
__all__ = ["CATALOG_DIR", "DEFAULT_CATALOG"]
