"""Reproducible setup of UVR source and an isolated inference interpreter."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path, PureWindowsPath
from typing import Any

import httpx

from uvr_maestro.discovery import Locations
from uvr_maestro.engine.uvr_bridge import REVISION
from uvr_maestro.storage import MaestroError, atomic_json, contained, sha256

SOURCE_URL = f"https://codeload.github.com/Anjok07/ultimatevocalremovergui/zip/{REVISION}"
SOURCE_SHA256 = "7a4484dbb128c7ca2f05d2fd6560058f39948b031f750ab426360c91eb7f8972"
DIFFQ_URL = (
    "https://files.pythonhosted.org/packages/5a/fd/"
    "4c58807bf855c5929ffa6da55f26dd6b9ae462a4193f5e09cc49fbbfd451/diffq-0.2.4.tar.gz"
)
DIFFQ_SHA256 = "049064861e974ebf00d0badab8b324c775037371419eda3150985b9d477b5bd2"
DEPENDENCIES = [
    "torch==2.5.1",
    "torchaudio==2.5.1",
    "torchvision==0.20.1",
    "torchmetrics==1.5.2",
    "numpy==1.26.4",
    "librosa==0.10.2.post1",
    "soundfile==0.12.1",
    "scipy==1.13.1",
    "onnx==1.17.0",
    "onnxruntime==1.20.1",
    "onnx2pytorch==0.4.1",
    "pytorch-lightning==2.4.0",
    "ml-collections==0.1.1",
    "omegaconf==2.3.0",
    "einops==0.6.1",
    "julius==0.2.7",
    "pydub==0.25.1",
    "pyrubberband==0.4.0",
    "resampy==0.4.3",
    "samplerate==0.2.1",
    "pyyaml==6.0.3",
    "matchering==2.0.6",
    "requests>=2.31,<3",
    "tqdm>=4.66,<5",
    "psutil>=5.9,<8",
    "pyloudnorm==0.1.1",
]


def checked_download(url: str, expected: str) -> bytes:
    """Fetch a pinned source artifact and verify its full checksum."""
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
    if hashlib.sha256(response.content).hexdigest() != expected:
        raise MaestroError("checksum_mismatch", f"source checksum mismatch: {url}")
    return response.content


def source_setup(root: Path) -> tuple[Path, Path]:
    """Install pinned Python source without evaluating upstream setup scripts."""
    source = root / "source" / REVISION
    support = root / "support"
    marker = root / "source.json"
    if marker.is_file():
        data = json.loads(marker.read_text(encoding="utf-8"))
        if data.get("revision") == REVISION and sha256(source / "separate.py") == data["separate"]:
            return source, support
        raise MaestroError(
            "runtime_modified", "managed source was modified; use a new runtime root"
        )
    if source.exists() and any(source.iterdir()):
        raise MaestroError("runtime_partial", "source setup is incomplete; use a new runtime root")
    payload = checked_download(SOURCE_URL, SOURCE_SHA256)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for item in archive.infolist():
            parts = Path(item.filename).parts[1:]
            if not parts or item.is_dir():
                continue
            if parts[0] not in (
                "separate.py",
                "__version__.py",
                "LICENSE",
                "demucs",
                "lib_v5",
                "gui_data",
            ):
                continue
            if (
                parts[0] == "gui_data"
                and len(parts) > 1
                and parts[1]
                not in (
                    "__init__.py",
                    "constants.py",
                    "error_handling.py",
                )
            ):
                continue
            target = contained(source, source.joinpath(*parts))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(item))
    payload = checked_download(DIFFQ_URL, DIFFQ_SHA256)
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        for item in archive.getmembers():
            parts = Path(item.name).parts[1:]
            if not item.isfile() or not parts:
                continue
            if parts[0] != "LICENSE" and not (parts[0] == "diffq" and parts[-1].endswith(".py")):
                continue
            target = contained(support, support.joinpath(*parts))
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(item)
            if stream is None:
                raise ValueError("missing archive member")
            target.write_bytes(stream.read())
    atomic_json(
        marker,
        {
            "revision": REVISION,
            "archive_sha256": SOURCE_SHA256,
            "diffq_sha256": DIFFQ_SHA256,
            "separate": sha256(source / "separate.py"),
        },
    )
    return source, support


def interpreter(root: Path) -> Path:
    """Resolve the managed interpreter or an explicit development override."""
    explicit = os.environ.get("UVR_MAESTRO_ENGINE_PYTHON")
    if explicit:
        return Path(explicit).expanduser().resolve()
    return root / "venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def status(root: Path | None = None) -> dict[str, Any]:
    """Report installed runtime paths without importing Torch or downloading."""
    root = (root or Locations.discover().runtime).resolve()
    python = interpreter(root)
    source = root / "source" / REVISION
    support = root / "support"
    ready = python.is_file() and (source / "separate.py").is_file() and (support / "diffq").is_dir()
    return {
        "ready": ready,
        "python": str(python),
        "source": str(source),
        "support": str(support),
        "revision": REVISION,
        "root": str(root),
    }


def check_setup_interpreter() -> None:
    """Refuse Store Python's private AppData view before external setup writes."""
    parts = [part.casefold() for part in PureWindowsPath(sys.base_prefix).parts]
    if (
        sys.platform == "win32"
        and "windowsapps" in parts
        and any(part.startswith("pythonsoftwarefoundation.python.") for part in parts)
    ):
        raise MaestroError(
            "store_python_unsupported",
            "Microsoft Store Python redirects AppData files that the external engine cannot "
            "read. Reinstall with: uv tool install --force --reinstall --managed-python "
            "--python 3.12 git+https://github.com/romanstark/uvr-maestro.git",
        )


def setup_runtime(
    root: Path,
    *,
    acceleration: str = "cpu",
    confirm: bool = False,
) -> dict[str, Any]:
    """Install source and a validated dependency set through explicit setup."""
    plan = {
        "runtime_root": str(root.resolve()),
        "acceleration": acceleration,
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "dependencies": DEPENDENCIES,
        "python_version": "3.11",
    }
    if not confirm:
        return {"success": False, "blocked": True, "plan": plan}
    if acceleration not in ("cpu", "cuda"):
        raise ValueError("acceleration must be cpu or cuda")
    check_setup_interpreter()
    source, support = source_setup(root)
    uv = shutil.which("uv")
    python = interpreter(root)
    explicit = bool(os.environ.get("UVR_MAESTRO_ENGINE_PYTHON"))
    if not explicit:
        if uv is None:
            raise MaestroError("uv_missing", "install uv before running uvr-maestro-setup")
        if not python.exists():
            subprocess.run(
                [uv, "venv", "--managed-python", "--python", "3.11", str(root / "venv")],
                check=True,
            )
        index = "https://download.pytorch.org/whl/" + ("cu124" if acceleration == "cuda" else "cpu")
        subprocess.run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                *DEPENDENCIES[:3],
                "--index-url",
                index,
            ],
            check=True,
        )
        requirements = Path(__file__).parent / "engine/requirements.txt"
        subprocess.run(
            [uv, "pip", "install", "--python", str(python), "-r", str(requirements)],
            check=True,
        )
    if not python.is_file():
        raise MaestroError("python_missing", f"engine interpreter is missing: {python}")
    probe = root / "preflight.json"
    atomic_json(probe, {"operation": "preflight", "source": str(source), "support": str(support)})
    worker = Path(__file__).parent / "engine/worker.py"
    process = subprocess.run(
        [str(python), "-I", str(worker), str(probe)],
        capture_output=True,
        text=True,
        check=False,
    )
    results = [json.loads(line) for line in process.stdout.splitlines() if line.startswith("{")]
    result = next((item for item in reversed(results) if item.get("event") == "result"), None)
    if process.returncode or result is None or not result.get("success"):
        raise MaestroError(
            "runtime_preflight_failed", (result or {}).get("error", process.stderr[-2000:])
        )
    atomic_json(root / "engine.json", {**plan, "preflight": result})
    return {"success": True, **status(root), "preflight": result}


def main() -> None:
    """Install the engine from an installed wheel or source checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=Locations.discover().runtime)
    parser.add_argument("--acceleration", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--confirm", action="store_true", help="Download and install the runtime")
    parser.add_argument(
        "--roformer", action="store_true", help="Install the separate classic Roformer runtime"
    )
    args = parser.parse_args()
    if args.roformer:
        from uvr_maestro.roformer_runtime import setup

        print(
            json.dumps(
                setup(args.runtime_root, acceleration=args.acceleration, confirm=args.confirm),
                indent=2,
            )
        )
        return
    print(
        json.dumps(
            setup_runtime(args.runtime_root, acceleration=args.acceleration, confirm=args.confirm),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
