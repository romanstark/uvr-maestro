"""Setup must share real files with its external interpreter, before any downloads."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from uvr_maestro import roformer_runtime, runtime
from uvr_maestro.storage import MaestroError

Setup = Callable[..., dict[str, Any]]


@pytest.mark.parametrize("setup", [runtime.setup_runtime, roformer_runtime.setup])
def test_store_python_setup_fails_before_writes(
    setup: Setup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime.sys, "platform", "win32")
    monkeypatch.setattr(
        runtime.sys,
        "base_prefix",
        r"C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_store",
    )
    target = tmp_path / "runtime"
    with pytest.raises(MaestroError) as error:
        setup(target, confirm=True)
    assert error.value.code == "store_python_unsupported"
    assert "--managed-python" in str(error.value)
    assert not target.exists()


@pytest.mark.parametrize("setup", [runtime.setup_runtime, roformer_runtime.setup])
def test_store_python_can_preview_without_writes(
    setup: Setup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime.sys, "platform", "win32")
    monkeypatch.setattr(
        runtime.sys,
        "base_prefix",
        r"C:\Users\example\AppData\Local\Microsoft\WindowsApps"
        r"\PythonSoftwareFoundation.Python.3.12_store",
    )
    target = tmp_path / "runtime"
    assert setup(target, confirm=False)["blocked"]
    assert not target.exists()


@pytest.mark.parametrize(
    ("platform", "prefix"),
    [
        ("win32", r"C:\Users\example\AppData\Roaming\uv\python\cpython-3.12"),
        ("win32", r"C:\Users\example\AppData\Local\Programs\Python\Python312"),
        ("darwin", "/Library/Frameworks/Python.framework/Versions/3.12"),
        ("linux", "/usr"),
    ],
)
def test_regular_python_is_accepted(
    platform: str, prefix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime.sys, "platform", platform)
    monkeypatch.setattr(runtime.sys, "base_prefix", prefix)
    runtime.check_setup_interpreter()
