"""Spec validation: bad stem / bad arch / bad sample rate are refused."""

from __future__ import annotations

import pytest

from uvr_maestro.catalog import CATALOG_DIR
from uvr_maestro.registry import Registry
from uvr_maestro.spec import (
    SettingSpec,
    build_endpoint,
    build_row,
    validate_setting_value,
)


@pytest.fixture
def registry() -> Registry:
    return Registry.load_bundled(CATALOG_DIR)


def test_build_row_happy_path() -> None:
    row = build_row(
        {
            "slug": "Test Model",
            "display_name": "Test Model",
            "architecture": "vr_5",
            "primary_stem": "vocals",
            "secondary_stem": "instrumental",
            "sample_rate": 44100,
            "download_url": "https://example.com/test.pth",
            "size_bytes": 1024,
        }
    )
    assert row.arch.value == "vr_5"
    assert row.primary_stem.value == "vocals"
    assert row.sample_rate == 44100


def test_build_row_rejects_unknown_arch() -> None:
    """Unknown architectures raise :class:`ValueError`."""
    with pytest.raises(ValueError):
        build_row(
            {
                "slug": "Test",
                "display_name": "Test",
                "architecture": "wax_tablet",
                "primary_stem": "vocals",
                "sample_rate": 44100,
                "download_url": "",
                "size_bytes": 0,
            }
        )


def test_build_row_rejects_unknown_stem() -> None:
    with pytest.raises(ValueError):
        build_row(
            {
                "slug": "Test",
                "display_name": "Test",
                "architecture": "vr_5",
                "primary_stem": "maracas",
                "sample_rate": 44100,
                "download_url": "",
                "size_bytes": 0,
            }
        )


def test_build_row_rejects_bad_sample_rate() -> None:
    with pytest.raises(ValueError):
        build_row(
            {
                "slug": "Test",
                "display_name": "Test",
                "architecture": "vr_5",
                "primary_stem": "vocals",
                "sample_rate": 48000,
                "download_url": "",
                "size_bytes": 0,
            }
        )


def test_build_endpoint_loads_url() -> None:
    payload = {
        "endpoints": {
            "vr_catalog": "u1",
            "mdx_catalog": "u2",
            "mdx_model_names": "u3",
            "demucs_model_names": "u4",
            "mdx23_configs_base": "u5",
            "model_repo_release_base": "u6",
            "update_patch_repo_base": "u7",
            "bulletin": "u8",
            "download_checks": "u9",
            "mac_arm_installer": "u10",
            "mac_x86_installer": "u11",
            "windows_setup": "u12",
            "linux_source": "u13",
        },
        "defaults": {"bulletin_refresh_seconds": 7200},
    }
    endpoint = build_endpoint(payload)
    assert endpoint.vr_catalog == "u1"
    assert endpoint.bulletin_refresh_seconds == 7200


def test_setting_validation_enum() -> None:
    spec = SettingSpec(key="save_format", type="enum", default="wav", enum=["wav", "mp3", "flac"])
    validate_setting_value(spec, "mp3")
    with pytest.raises(ValueError):
        validate_setting_value(spec, "aac")


def test_setting_validation_int_type() -> None:
    spec = SettingSpec(key="segment_size", type="int", default=256, enum=None)
    validate_setting_value(spec, 1024)
    with pytest.raises(ValueError):
        validate_setting_value(spec, "1024")
