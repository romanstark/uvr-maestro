"""Atomic persistence and confinement helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


class MaestroError(Exception):
    """A domain failure with a stable machine-readable code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def safe_name(value: str) -> str:
    """Refuse traversal, absolute paths and Windows device filenames."""
    if not isinstance(value, str) or not value or value in (".", ".."):
        raise ValueError("name must be a nonempty filename")
    if any(ord(c) < 32 or c in '/\\:*?"<>|' for c in value):
        raise ValueError("name contains path separators or reserved characters")
    if value.endswith((" ", ".")):
        raise ValueError("name ends with a space or dot")
    if re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])", value.split(".")[0]):
        raise ValueError("reserved device name")
    return value


def contained(root: Path, candidate: Path) -> Path:
    """Resolve a path and require it to remain inside its owning root."""
    target = candidate.resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("path escapes its owning directory")
    return target


def atomic_json(path: Path, data: Any) -> None:
    """Write JSON through a temporary file and atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".maestro-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def sha256(path: Path) -> str:
    """Fingerprint a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
