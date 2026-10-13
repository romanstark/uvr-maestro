"""Explicit additional runtime for classic Roformer; never alter the UVR interpreter."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from uvr_maestro.runtime import check_setup_interpreter, checked_download
from uvr_maestro.storage import MaestroError, atomic_json

PACKAGE_URL = (
    "https://files.pythonhosted.org/packages/40/7d/"
    "1f76749ca55319857aac0b34bcc62a77040e6978d71da2d5a0b01e10c250/"
    "BS_RoFormer-0.3.0-py3-none-any.whl"
)
PACKAGE_SHA256 = "ddfe5eab6690b3ca02aec2971f6d90624270d955be83ae5c19dcc3aa8d5eff17"


def status(root: Path) -> dict[str, Any]:
    """Read readiness only after a successful isolated preflight."""
    directory = root / "roformer"
    python = Path(
        os.environ.get("UVR_MAESTRO_ROFORMER_PYTHON")
        or directory / "venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    )
    return {
        "ready": python.is_file() and (directory / "engine.json").is_file(),
        "python": str(python),
        "source": "",
        "support": "",
        "revision": "bs-roformer-0.3.0",
        "root": str(directory),
    }


def setup(root: Path, *, acceleration: str = "cpu", confirm: bool = False) -> dict[str, Any]:
    """Install the verified package and locked dependencies in its own interpreter."""
    plan = {
        "runtime_root": str(root / "roformer"),
        "package_url": PACKAGE_URL,
        "package_sha256": PACKAGE_SHA256,
        "acceleration": acceleration,
    }
    if not confirm:
        return {"success": False, "blocked": True, "plan": plan}
    if acceleration not in ("cpu", "cuda"):
        raise ValueError("acceleration must be cpu or cuda")
    check_setup_interpreter()
    uv = shutil.which("uv")
    if uv is None:
        raise MaestroError("uv_missing", "runtime setup requires uv on PATH")
    directory = root / "roformer"
    directory.mkdir(parents=True, exist_ok=True)
    python = Path(status(root)["python"])
    if not python.exists():
        subprocess.run(
            [uv, "venv", "--managed-python", "--python", "3.11", str(directory / "venv")],
            check=True,
        )
    package = directory / "BS_RoFormer-0.3.0-py3-none-any.whl"
    package.write_bytes(checked_download(PACKAGE_URL, PACKAGE_SHA256))
    index = "https://download.pytorch.org/whl/" + ("cu124" if acceleration == "cuda" else "cpu")
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--index-url",
            index,
            "torch==2.5.1",
            "torchaudio==2.5.1",
            "torchvision==0.20.1",
        ],
        check=True,
    )
    lock = directory / "requirements.txt"
    baseline = (Path(__file__).parent / "engine/requirements.txt").read_text(encoding="utf-8")
    lock.write_text(
        baseline.replace("einops==0.6.1", "einops==0.8.0")
        + "\nbeartype==0.19.0\nrotary-embedding-torch==0.6.4\n",
        encoding="utf-8",
    )
    subprocess.run([uv, "pip", "install", "--python", str(python), "-r", str(lock)], check=True)
    subprocess.run(
        [uv, "pip", "install", "--python", str(python), "--no-deps", str(package)], check=True
    )
    probe = directory / "preflight.json"
    atomic_json(probe, {"operation": "roformer_preflight"})
    result = subprocess.run(
        [str(python), "-I", str(Path(__file__).parent / "engine/worker.py"), str(probe)],
        capture_output=True,
        text=True,
        check=False,
    )
    events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    receipt = next((event for event in reversed(events) if event.get("event") == "result"), {})
    if result.returncode or not receipt.get("success"):
        raise MaestroError("runtime_preflight_failed", receipt.get("error", result.stderr[-2000:]))
    atomic_json(directory / "engine.json", {**plan, "preflight": receipt})
    return {"success": True, **status(root), "preflight": receipt}
