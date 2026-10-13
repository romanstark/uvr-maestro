"""Serialized local jobs, cancellation and durable verified run records."""

from __future__ import annotations

import copy
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from uvr_maestro import __version__, runtime
from uvr_maestro.discovery import Locations
from uvr_maestro.storage import MaestroError, atomic_json, safe_name, sha256

TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


class JobManager:
    """Queue at most one native worker at a time, keeping MCP reads responsive."""

    def __init__(self, locations: Locations | None = None) -> None:
        self.locations = locations or Locations.discover()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="uvr-maestro")
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._processes: dict[str, subprocess.Popen[str]] = {}

    def _save(self, job: dict[str, Any]) -> None:
        atomic_json(self.locations.data / "jobs" / (job["id"] + ".json"), job)

    def submit(self, operation: str, request: dict[str, Any]) -> dict[str, Any]:
        """Enqueue a validated request and return before inference starts."""
        identifier = uuid.uuid4().hex
        job = {
            "id": identifier,
            "operation": operation,
            "state": "queued",
            "progress": 0.0,
            "created_at": time.time(),
            "logs": [],
            "output_dir": str(Path(request["output_root"]).resolve() / f"uvr-{identifier}"),
        }
        with self._lock:
            self._jobs[identifier] = job
            self._save(job)
        self._pool.submit(self._execute, identifier, copy.deepcopy(request))
        return self.get(identifier)

    def get(self, identifier: str) -> dict[str, Any]:
        """Read a live or persisted job, reporting unfinished old jobs as interrupted."""
        safe_name(identifier)
        with self._lock:
            if identifier in self._jobs:
                return copy.deepcopy(self._jobs[identifier])
            path = self.locations.data / "jobs" / (identifier + ".json")
            if not path.is_file():
                raise MaestroError("job_unknown", f"unknown job: {identifier}")
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["state"] not in TERMINAL:
                result["state"] = "interrupted"
                result["error"] = "server stopped before this job reached a terminal state"
            return result

    def list(self) -> list[dict[str, Any]]:
        """List the latest hundred jobs without reading their audio."""
        paths = sorted(
            (self.locations.data / "jobs").glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:100]
        return [self.get(path.stem) for path in paths]

    def cancel(self, identifier: str) -> dict[str, Any]:
        """Cancel queued work or terminate only its active native worker."""
        with self._lock:
            job = self._jobs.get(identifier)
            if job is None:
                return self.get(identifier)
            if job["state"] in TERMINAL:
                return copy.deepcopy(job)
            job["state"] = "cancelled"
            job["finished_at"] = time.time()
            self._save(job)
            process = self._processes.get(identifier)
            if process is not None and process.poll() is None:
                self._terminate(process)
            return copy.deepcopy(job)

    @staticmethod
    def _terminate(process: subprocess.Popen[str]) -> None:
        """Terminate this worker's process tree, including codecs and Rubber Band."""
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check=False,
            )
            if process.poll() is None:
                process.terminate()
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return

    def _check_cancelled(self, identifier: str) -> None:
        with self._lock:
            if self._jobs[identifier]["state"] == "cancelled":
                raise MaestroError("cancelled", "job cancelled")

    def _worker(
        self,
        identifier: str,
        request: dict[str, Any],
        output: Path,
        portion: tuple[float, float] = (0.0, 1.0),
    ) -> dict[str, Any]:
        self._check_cancelled(identifier)
        info = runtime.status(self.locations.runtime)
        if request.get("model", {}).get("arch") == "roformer":
            from uvr_maestro.roformer_runtime import status as roformer_status

            info = roformer_status(self.locations.runtime)
        if not info["ready"]:
            raise MaestroError("engine_not_loaded", "run uvr-maestro-setup --confirm first")
        output.mkdir(parents=True, exist_ok=True)
        request = {
            **request,
            "source": info["source"],
            "support": info["support"],
            "output_dir": str(output),
        }
        atomic_json(output / "request.json", request)
        command = [
            info["python"],
            "-I",
            str(Path(__file__).parent / "engine/worker.py"),
            str(output / "request.json"),
        ]
        result = None
        with (output / "engine.log").open("w", encoding="utf-8") as errors:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=errors,
                text=True,
                encoding="utf-8",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                start_new_session=sys.platform != "win32",
            )
            with self._lock:
                self._processes[identifier] = process
                if self._jobs[identifier]["state"] == "cancelled":
                    self._terminate(process)
            try:
                if process.stdout is None:
                    raise RuntimeError("worker stdout is missing")
                for line in process.stdout:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    with self._lock:
                        job = self._jobs[identifier]
                        if event.get("event") == "result":
                            result = event
                        elif event.get("event") == "progress":
                            job["progress"] = max(
                                job["progress"],
                                min(0.99, portion[0] + portion[1] * event["progress"]),
                            )
                        elif event.get("event") == "log":
                            job["logs"] = [*job["logs"][-11:], event["message"][:1000]]
                        self._save(job)
                process.wait()
            finally:
                if process.stdout:
                    process.stdout.close()
                with self._lock:
                    self._processes.pop(identifier, None)
                if process.poll() is None:
                    self._terminate(process)
                    process.wait(timeout=10)
        self._check_cancelled(identifier)
        if result is None:
            raise MaestroError(
                "worker_crashed", f"worker exited {process.returncode}; see engine.log"
            )
        if process.returncode or not result.get("success"):
            message = result.get("error", "worker failed")
            if len(message) > 2000:
                message = message[:2000] + "; full diagnostics in engine.log"
            raise MaestroError(result.get("code", "engine_error"), message)
        return result

    def _workflow(self, identifier: str, request: dict[str, Any], output: Path) -> dict[str, Any]:
        operation = request["operation"]
        if operation == "batch":
            results = []
            inputs = request["inputs"]
            for index, path in enumerate(inputs):
                child = {**request, "operation": "separation", "audio_path": path}
                try:
                    result = self._worker(
                        identifier,
                        child,
                        output / f"file-{index + 1:04d}",
                        (index / len(inputs), 1 / len(inputs)),
                    )
                    results.append({"input": path, **result})
                except MaestroError as error:
                    if error.code == "cancelled":
                        raise
                    results.append(
                        {"input": path, "success": False, "code": error.code, "error": str(error)}
                    )
            outputs = [item for result in results for item in result.get("outputs", [])]
            return {
                "success": all(item["success"] for item in results),
                "outputs": outputs,
                "files": results,
                "device": "see per-file results",
            }
        if operation == "ensemble":
            groups: dict[str, list[dict[str, Any]]] = {}
            steps = []
            entries = request["entries"]
            for index, entry in enumerate(entries):
                child: dict[str, Any] = {
                    **request,
                    "operation": "separation",
                    "model": entry["model"],
                    "settings": entry["settings"],
                }
                result = self._worker(
                    identifier,
                    child,
                    output / ".intermediate" / str(index),
                    (index / (len(entries) + 1), 1 / (len(entries) + 1)),
                )
                steps.append(result)
                match = next(
                    (
                        item
                        for item in result["outputs"]
                        if item["stem"].casefold() == entry["stem"].casefold()
                    ),
                    None,
                )
                if match is None:
                    raise MaestroError("stem_missing", f"model did not output {entry['stem']}")
                groups.setdefault(entry["stem"], []).append(
                    {"path": match["path"], "weight": entry["weight"]}
                )
            if any(len(items) < 2 for items in groups.values()):
                raise ValueError("each ensemble source needs at least two model outputs")
            combined = self._worker(
                identifier,
                {
                    **request,
                    "groups": [{"stem": stem, "files": files} for stem, files in groups.items()],
                },
                output,
                (len(entries) / (len(entries) + 1), 1 / (len(entries) + 1)),
            )
            return {**combined, "steps": steps, "timeline": steps[0].get("timeline")}
        if operation == "chain":
            current = request["audio_path"]
            results = []
            steps = request["steps"]
            for index, step in enumerate(steps):
                child: dict[str, Any] = {
                    **request,
                    "operation": "separation",
                    "audio_path": current,
                    "model": step["model"],
                    "settings": step["settings"],
                }
                if index > 0:
                    child["audio_contract"] = {
                        **request.get("audio_contract", {}),
                        "start_seconds": 0,
                        "end_seconds": None,
                    }
                if index < len(steps) - 1:
                    child["settings"] = {
                        **child["settings"],
                        "save_format": "wav",
                        "wav_subtype": "FLOAT",
                    }
                result = self._worker(
                    identifier,
                    child,
                    output / f"step-{index + 1:02d}",
                    (index / len(steps), 1 / len(steps)),
                )
                chosen = next(
                    (
                        item
                        for item in result["outputs"]
                        if item["stem"].casefold() == step["output_stem"].casefold()
                    ),
                    None,
                )
                if chosen is None:
                    raise MaestroError("stem_missing", f"step did not output {step['output_stem']}")
                current = chosen["path"]
                results.append(result)
            return {
                **results[-1],
                "steps": results,
                "routed_output": current,
                "timeline": results[0].get("timeline"),
            }
        return self._worker(identifier, request, output)

    def _execute(self, identifier: str, request: dict[str, Any]) -> None:
        started = time.monotonic()
        job = self._jobs[identifier]
        output = Path(job["output_dir"])
        try:
            with self._lock:
                self._check_cancelled(identifier)
                job["state"] = "running"
                job["started_at"] = time.time()
                self._save(job)
            output.mkdir(parents=True, exist_ok=False)
            result = self._workflow(identifier, request, output)
            with self._lock:
                job["result"] = result
            self._check_cancelled(identifier)
            if not result.get("success"):
                raise MaestroError("batch_partial_failure", "one or more batch files failed")
            outputs = result.get("outputs", [])
            if not outputs and request["operation"] != "analyze_audio":
                raise MaestroError("output_missing", "completed worker returned no audio files")
            import numpy as np
            import soundfile as sf

            for item in outputs:
                path = Path(item["path"]).resolve()
                if not path.is_relative_to(output.resolve()) or not path.is_file():
                    raise MaestroError("invalid_output", "output file is absent or outside the run")
                for block in sf.blocks(path, blocksize=65536, always_2d=True):
                    if not np.isfinite(block).all():
                        raise MaestroError("invalid_output", "output contains non-finite samples")
                if sf.info(path).frames == 0 or sha256(path) != item["sha256"]:
                    raise MaestroError("invalid_output", "output is empty or its hash changed")
            input_paths = request.get(
                "inputs", [request["audio_path"]] if "audio_path" in request else []
            )
            if "reference" in request:
                input_paths = [*input_paths, request["reference"]]
            manifest = {
                "schema_version": 1,
                "id": identifier,
                "operation": job["operation"],
                "version": __version__,
                "request": request,
                "result": result,
                "runtime_seconds": time.monotonic() - started,
                "input_sha256": {path: sha256(Path(path)) for path in input_paths},
                "created_at": job["created_at"],
                "finished_at": time.time(),
            }
            with self._lock:
                self._check_cancelled(identifier)
                if not request.get("keep_intermediates", True):
                    removed = []
                    candidates = [output / ".intermediate"]
                    if request["operation"] == "chain":
                        candidates.extend(sorted(output.glob("step-*"))[:-1])
                    for directory in candidates:
                        resolved = directory.resolve()
                        if not resolved.is_relative_to(output.resolve()):
                            raise MaestroError(
                                "invalid_output", "intermediate escaped run directory"
                            )
                        if directory.exists():
                            shutil.rmtree(directory)
                            removed.append(str(directory))
                    result["removed_intermediates"] = removed
                atomic_json(output / "manifest.json", manifest)
                job.update(
                    state="succeeded",
                    progress=1.0,
                    result=result,
                    manifest=str(output / "manifest.json"),
                    finished_at=time.time(),
                )
                self._save(job)
        except Exception as error:  # noqa: BLE001
            with self._lock:
                if job["state"] != "cancelled":
                    job.update(
                        state="failed",
                        code=getattr(error, "code", "job_failed"),
                        error=str(error),
                        finished_at=time.time(),
                    )
                job["partial_output_dir"] = str(output) if output.exists() else None
                self._save(job)

    def close(self) -> None:
        """Stop this manager's workers and prevent background threads on shutdown."""
        for identifier in list(self._jobs):
            if self._jobs[identifier]["state"] not in TERMINAL:
                self.cancel(identifier)
        self._pool.shutdown(wait=True, cancel_futures=True)


_manager: JobManager | None = None


def default_manager() -> JobManager:
    """Create a manager lazily on the first job operation."""
    global _manager
    if _manager is None:
        _manager = JobManager()
    return _manager
