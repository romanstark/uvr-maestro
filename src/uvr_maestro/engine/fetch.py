"""Explicit catalog refresh and staged model artifact downloads."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

import httpx

from uvr_maestro.catalog_sync import parse_downloads
from uvr_maestro.engine.config import load_config
from uvr_maestro.registry import Registry, default_registry
from uvr_maestro.spec import EndpointSpec, ModelSpec
from uvr_maestro.storage import atomic_json, contained, sha256


async def fetch_json(url: str, *, timeout: float = 30.0) -> dict[str, Any]:
    """Read an upstream JSON object, refusing invalid top-level values."""
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError("upstream metadata must be a JSON object")
        return payload


def refresh_catalog(
    *,
    cache_path: Path | None = None,
    endpoints: EndpointSpec | None = None,
) -> dict[str, Any]:
    """Parse real upstream lists and commit a validated catalog diff."""
    previous = default_registry()
    endpoints = endpoints or previous.endpoints

    async def retrieve() -> list[dict[str, Any]]:
        return list(
            await asyncio.gather(
                fetch_json(endpoints.download_checks),
                fetch_json(endpoints.mdx_model_names),
                fetch_json(endpoints.vr_catalog),
                fetch_json(endpoints.mdx_catalog),
            )
        )

    checks, names, vr, mdx = asyncio.run(retrieve())
    rows = parse_downloads(checks, endpoints, names)
    previous_rows = {row.slug: row for row in previous.all()}
    for row in rows:
        old = previous_rows.get(row.slug)
        if old is not None and old.artifacts == row.artifacts and old.arch == row.arch:
            row.verified = old.verified
            row.notes = old.notes
            row.metadata = {**old.metadata, **row.metadata}
    replacement = Registry(rows, previous.settings(), endpoints)
    target = cache_path or Path.home() / ".uvr_maestro/cache/catalog.json"
    # The catalog embeds the metadata snapshots so a refresh is one atomic commit.
    payload = replacement.payload()
    payload["model_metadata"] = {"vr": vr, "mdx": mdx}
    old = {row.slug: row for row in previous.all()}
    new = {row.slug: row for row in rows}
    changed = [
        slug
        for slug in old.keys() & new.keys()
        if old[slug].artifacts != new[slug].artifacts
        or old[slug].display_name != new[slug].display_name
        or old[slug].arch != new[slug].arch
    ]
    atomic_json(target, payload)
    return {
        "ok": True,
        "added": sorted(new.keys() - old.keys()),
        "removed": sorted(old.keys() - new.keys()),
        "changed": sorted(changed),
        "metadata_changed": {
            "vr": previous.model_metadata.get("vr") != vr,
            "mdx": previous.model_metadata.get("mdx") != mdx,
        },
        "models": len(rows),
        "cache_path": str(target),
        "patch": checks.get("current_version"),
    }


def download_one(
    row: ModelSpec,
    *,
    dest_dir: Path,
    chunk_bytes: int | None = None,
    timeout_seconds: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Stage every artifact and verify its declared integrity before installation."""
    if not row.supported or not row.artifacts:
        raise ValueError("model has no supported artifact download plan")
    results: list[dict[str, Any]] = []
    with httpx.Client(timeout=timeout_seconds or 300, follow_redirects=True) as client:
        for artifact in row.artifacts:
            target = contained(dest_dir, dest_dir / artifact.directory / artifact.filename)
            if target.is_file() and not force:
                size = target.stat().st_size
                valid = size > 0 and (not artifact.size_bytes or size == artifact.size_bytes)
                valid &= not artifact.sha256 or sha256(target) == artifact.sha256
                if valid:
                    results.append(
                        {"filename": artifact.filename, "skipped": True, "path": str(target)}
                    )
                    continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=target.parent, prefix=".download-", delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    with client.stream("GET", artifact.url) as response:
                        response.raise_for_status()
                        for block in response.iter_bytes(chunk_size=chunk_bytes or 65536):
                            stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                size = temporary.stat().st_size
                if size == 0 or (artifact.size_bytes and size != artifact.size_bytes):
                    raise ValueError(f"size mismatch for {artifact.filename}")
                fingerprint = sha256(temporary)
                if artifact.sha256 and fingerprint != artifact.sha256:
                    raise ValueError(f"checksum mismatch for {artifact.filename}")
                if artifact.role in ("config", "bag"):
                    payload = load_config(temporary.read_text(encoding="utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError(f"invalid YAML config: {artifact.filename}")
                    if artifact.role == "bag" and not payload.get("models"):
                        raise ValueError("Demucs bag lacks model signatures")
                os.replace(temporary, target)
                results.append(
                    {
                        "filename": artifact.filename,
                        "path": str(target),
                        "bytes": size,
                        "sha256": fingerprint,
                        "upstream_checksum_verified": bool(artifact.sha256),
                    }
                )
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
    return {"ok": True, "slug": row.slug, "artifacts": results}
