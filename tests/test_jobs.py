"""Real subprocess lifecycle tests without Torch or a UVR installation."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from uvr_maestro import jobs
from uvr_maestro.discovery import Locations
from uvr_maestro.jobs import JobManager

WORKER = """
import hashlib,json,pathlib,sys,time,wave
request=json.loads(pathlib.Path(sys.argv[1]).read_text())
if request.get('delay'):
    print(json.dumps({'event':'progress','progress':.2}),flush=True)
    time.sleep(request['delay'])
output=pathlib.Path(request['output_dir'])/'actual.wav'
with wave.open(str(output),'wb') as audio:
    audio.setnchannels(2);audio.setsampwidth(2);audio.setframerate(8000)
    audio.writeframes(b'\\x01\\x00'*1600)
files=[] if request.get('missing_output') else [
    {'path':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}]
print(json.dumps({'event':'result','success':True,'outputs':files}),flush=True)
"""


@pytest.fixture
def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Replace only the worker program, retaining actual Popen and lifecycle behavior."""
    root = tmp_path / "data"
    locations = Locations(
        root, root / "models", root / "profiles", root / "cache", root / "runtime", None
    )
    worker = tmp_path / "worker.py"
    worker.write_text(WORKER)
    original = subprocess.Popen

    def start(command, **kwargs):
        if command[0] == "taskkill":
            return original(command, **kwargs)
        return original([sys.executable, str(worker), command[-1]], **kwargs)

    monkeypatch.setattr(jobs.subprocess, "Popen", start)
    monkeypatch.setattr(
        jobs.runtime,
        "status",
        lambda root: {
            "ready": True,
            "python": sys.executable,
            "source": str(tmp_path),
            "support": str(tmp_path),
        },
    )
    result = JobManager(locations)
    yield result
    result.close()


def terminal(manager: JobManager, identifier: str) -> dict:
    """Fail promptly if a test worker cannot reach its terminal state."""
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        result = manager.get(identifier)
        if result["state"] in jobs.TERMINAL:
            return result
        time.sleep(0.05)
    raise AssertionError("worker did not finish")


def test_success_requires_actual_output_and_manifest(manager: JobManager, tmp_path: Path) -> None:
    """A real written file is validated and recorded durably."""
    job = manager.submit(
        "separation", {"operation": "separation", "output_root": str(tmp_path / "outputs")}
    )
    result = terminal(manager, job["id"])
    assert result["state"] == "succeeded"
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert Path(manifest["result"]["outputs"][0]["path"]).is_file()
    assert manifest["schema_version"] == 1


def test_false_worker_receipt_never_becomes_success(manager: JobManager, tmp_path: Path) -> None:
    """Even a worker's successful receipt cannot substitute for declared output files."""
    job = manager.submit(
        "separation",
        {
            "operation": "separation",
            "missing_output": True,
            "output_root": str(tmp_path / "outputs"),
        },
    )
    result = terminal(manager, job["id"])
    assert result["state"] == "failed" and result["code"] == "output_missing"
    assert not (Path(result["output_dir"]) / "manifest.json").exists()


def test_cancel_active_and_queued_then_resume(manager: JobManager, tmp_path: Path) -> None:
    """Cancellation stops the actual worker and does not poison subsequent jobs."""
    root = str(tmp_path / "outputs")
    first = manager.submit(
        "separation", {"operation": "separation", "delay": 5, "output_root": root}
    )
    second = manager.submit("separation", {"operation": "separation", "output_root": root})
    deadline = time.monotonic() + 5
    while first["id"] not in manager._processes and time.monotonic() < deadline:
        time.sleep(0.05)
    assert manager.cancel(second["id"])["state"] == "cancelled"
    assert manager.cancel(first["id"])["state"] == "cancelled"
    following = manager.submit("separation", {"operation": "separation", "output_root": root})
    assert terminal(manager, following["id"])["state"] == "succeeded"
    assert terminal(manager, first["id"])["state"] == "cancelled"
    assert terminal(manager, second["id"])["state"] == "cancelled"


def test_persisted_unfinished_job_is_reported_interrupted(
    manager: JobManager,
    tmp_path: Path,
) -> None:
    """Recovery does not mistake an old running record for an active worker."""
    from uvr_maestro.storage import atomic_json

    atomic_json(manager.locations.data / "jobs/old.json", {"id": "old", "state": "running"})
    assert manager.get("old")["state"] == "interrupted"
