"""Validate wheel and sdist from a clean environment on the current platform."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path


def run(command: list[str], cwd: Path, environment: dict[str, str]) -> None:
    """Run one release gate with its explicitly selected interpreter."""
    print("CHECK", Path(command[0]).name, " ".join(command[1:3]), flush=True)
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def main() -> None:
    """Install only built artifacts and test away from the development checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    args = parser.parse_args()
    wheel, sdist = args.wheel.resolve(), args.sdist.resolve()
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert len([name for name in names if name.endswith("10-vr5.yaml")]) == 1
        assert any(name.endswith("engine/requirements.txt") for name in names)
        assert all(name.startswith("uvr_maestro/") or ".dist-info/" in name for name in names)
    with tarfile.open(sdist) as archive:
        names = ["/".join(member.name.split("/")[1:]) for member in archive.getmembers()]
        roots = {name.split("/")[0] for name in names if name}
        allowed = {
            "src", "tests", "scripts", "docs", ".github", ".claude-plugin", ".agents",
            ".claude", ".codex", ".cursor", ".vscode", ".mcp.json", "README.md", "LICENSE",
            "THIRD-PARTY.md", "CONTRIBUTING.md", "glama.json", ".gitattributes", ".gitignore",
            "pyproject.toml", "PKG-INFO",
        }
        assert roots <= allowed, f"unlisted sdist roots: {roots - allowed}"
    with tempfile.TemporaryDirectory(prefix="uvr-maestro-package-") as directory:
        root = Path(directory)
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment.pop("UVR_MAESTRO_ENGINE_PYTHON", None)
        environment["UVR_MAESTRO_DATA_DIR"] = str(root / "data")
        environment["UVR_MAESTRO_UVR_DIR"] = str(root / "absent-desktop")
        venv.create(root / "venv", with_pip=True)
        binary = root / "venv" / ("Scripts" if sys.platform == "win32" else "bin")
        python = binary / ("python.exe" if sys.platform == "win32" else "python")
        environment["VIRTUAL_ENV"] = str(root / "venv")
        environment["PATH"] = str(binary) + os.pathsep + environment.get("PATH", "")
        run([str(python), "-m", "pip", "install", "-q", str(wheel) + "[dev]"], root, environment)
        run(
            [
                str(python),
                "-c",
                (
                    "import asyncio; from uvr_maestro.server import mcp,uvr_status; "
                    "assert len(asyncio.run(mcp.list_tools()))==34; assert uvr_status()['success']"
                ),
            ],
            root,
            environment,
        )
        export = binary / (
            "uvr-maestro-skills.exe" if sys.platform == "win32" else "uvr-maestro-skills"
        )
        run([str(export), "export", "--output", str(root / "skills")], root, environment)
        assert len(list((root / "skills").glob("*/SKILL.md"))) == 5
        run(
            [
                str(python),
                str(Path(__file__).parent / "probe_mcp.py"),
                "--python",
                str(python),
                "--data",
                str(root / "wheel-protocol"),
            ],
            root,
            environment,
        )
        source = root / "source"
        source.mkdir()
        # This is our own built archive. Validate member confinement before extraction.
        with tarfile.open(sdist) as archive:
            for member in archive.getmembers():
                if member.issym() or member.islnk():
                    raise ValueError("sdist must not contain links")
                target = (source / member.name).resolve()
                if not target.is_relative_to(source.resolve()):
                    raise ValueError("sdist member escapes its root")
            archive.extractall(source, filter="data" if sys.version_info >= (3, 12) else None)
        project = next(source.iterdir())
        run(
            [str(python), "-m", "pip", "install", "-q", "-e", str(project) + "[dev]"],
            root,
            environment,
        )
        run([str(python), "-m", "pytest", "-q"], project, environment)
        suffix = ".exe" if sys.platform == "win32" else ""
        run(
            [str(binary / ("ruff" + suffix)), "check", "src", "scripts", "tests"],
            project,
            environment,
        )
        run([str(binary / ("pyright" + suffix)), "--pythonpath", str(python)], project, environment)
        run(
            [str(binary / ("pyrefly" + suffix)), "check", "--python-interpreter-path", str(python)],
            project,
            environment,
        )
        run(
            [
                str(python),
                str(project / "scripts/probe_mcp.py"),
                "--python",
                str(python),
                "--data",
                str(root / "protocol-data"),
            ],
            root,
            environment,
        )
        print(f"PACKAGE PASSED: {sys.platform}, Python {sys.version.split()[0]}", flush=True)


if __name__ == "__main__":
    main()
