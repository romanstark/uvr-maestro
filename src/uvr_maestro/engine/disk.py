"""Resolve complete artifact sets across owned and reused model roots."""

from __future__ import annotations

from pathlib import Path

from uvr_maestro.models import ModelSpec, ModelStatus
from uvr_maestro.storage import contained, sha256


def extension_for_row(row: ModelSpec) -> str:
    """Read a weight extension for legacy catalog rows."""
    if row.download_url.startswith("pytorch_hub:"):
        return ""
    return Path(row.download_url.split("?", 1)[0]).suffix.lower() or ".pth"


def reconcile(row: ModelSpec, roots: list[Path], *, owned_root: Path) -> None:
    """Mark a row installed only when every required artifact exists."""
    row.path = None
    row.artifact_paths = {}
    row.owned = False
    if not row.supported:
        row.status = ModelStatus.UNSUPPORTED
        return
    if not row.artifacts:
        status, path = row_disk_status(row, roots[0])
        row.status, row.path = status, path
        row.owned = path is not None and path.resolve().is_relative_to(owned_root.resolve())
        return
    broken = False
    all_owned = True
    for artifact in row.artifacts:
        candidates = [
            contained(root, root / artifact.directory / artifact.filename) for root in roots
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            row.status = ModelStatus.AVAILABLE
            return
        actual = path.stat().st_size
        broken |= actual == 0 or bool(artifact.size_bytes and actual != artifact.size_bytes)
        row.artifact_paths[artifact.filename] = str(path)
        all_owned &= path.is_relative_to(owned_root.resolve())
        if artifact.role in ("weights", "bag") and row.path is None:
            row.path = path
        if artifact.role == "bag":
            row.path = path
    row.owned = all_owned
    row.status = ModelStatus.BROKEN if broken else ModelStatus.INSTALLED


def row_disk_status(row: ModelSpec, models_dir: Path) -> tuple[ModelStatus, Path | None]:
    """Support simple rows and delegate real artifact sets."""
    if row.artifacts:
        reconcile(row, [models_dir], owned_root=models_dir)
        return row.status, row.path
    ext = extension_for_row(row)
    if not ext:
        return ModelStatus.AVAILABLE, None
    target = contained(models_dir, models_dir / f"{row.slug}{ext}")
    if not target.is_file():
        return ModelStatus.AVAILABLE, None
    size = target.stat().st_size
    broken = size == 0 or bool(row.size_bytes and size != row.size_bytes)
    return (ModelStatus.BROKEN if broken else ModelStatus.INSTALLED), target


def verify_artifacts(row: ModelSpec) -> list[dict[str, str]]:
    """Hash resolved artifacts only for an explicit integrity verification."""
    failures = []
    for artifact in row.artifacts:
        path = row.artifact_paths.get(artifact.filename)
        if path and artifact.sha256 and sha256(Path(path)) != artifact.sha256:
            failures.append({"filename": artifact.filename, "code": "checksum_mismatch"})
    return failures


def list_orphan_files(registry_rows: list[ModelSpec], models_dir: Path) -> list[Path]:
    """Report model weights absent from the catalog without deleting them."""
    declared = {f"{a.directory}/{a.filename}" for row in registry_rows for a in row.artifacts}
    declared.update(
        f"{row.slug}{extension_for_row(row)}" for row in registry_rows if not row.artifacts
    )
    extensions = {".pth", ".onnx", ".ckpt", ".th", ".yaml"}
    return sorted(
        path
        for path in models_dir.rglob("*")
        if path.is_file()
        and path.suffix in extensions
        and path.relative_to(models_dir).as_posix() not in declared
    )
