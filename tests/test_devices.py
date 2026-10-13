"""Backend selection respects measured native incompatibilities and explicit choices."""

import pytest

from uvr_maestro.engine.worker import select_device
from uvr_maestro.models import ModelArch


@pytest.mark.parametrize("requested", ["auto", "mps"])
def test_demucs_v4_mps_request_falls_back_with_a_reason(requested: str) -> None:
    device, reason = select_device(requested, ModelArch.DEMUCS_V4, False, True)
    assert device == "cpu"
    assert reason and "65536" in reason


@pytest.mark.parametrize("arch", [ModelArch.VR_5, ModelArch.MDX, ModelArch.MDXC])
def test_apple_gpu_remains_available_for_verified_families(arch: ModelArch) -> None:
    assert select_device("auto", arch, False, True) == ("mps", None)


def test_cuda_and_explicit_cpu_choices_are_preserved() -> None:
    assert select_device("auto", ModelArch.DEMUCS_V4, True, True) == ("cuda", None)
    assert select_device("cpu", ModelArch.DEMUCS_V4, True, True) == ("cpu", None)
    assert select_device("cuda", ModelArch.MDX, False, True) == (
        "cpu", "cuda backend is unavailable"
    )
