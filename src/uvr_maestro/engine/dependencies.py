"""Resolve optional runtime libraries only inside the isolated worker."""

from __future__ import annotations

import importlib
from typing import Any


def library(name: str) -> Any:
    """Import a runtime dependency at the explicit heavy boundary."""
    return importlib.import_module(name)
