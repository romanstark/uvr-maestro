"""Impulse timing, source selection, exact frames and honest residual reporting."""

import struct
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from uvr_maestro.engine.export import finalize, select_source, strip_tags
from uvr_maestro.storage import MaestroError


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("rate", [32000, 44100, 48000])
def test_excerpt_resample_impulse_and_complementary_sum(tmp_path: Path, channels: int, rate: int):
    source = np.zeros((rate + 7, channels))
    source[rate // 4] = 0.5
    input_path = tmp_path / "source.wav"
    sf.write(input_path, source, rate, subtype="FLOAT")
    request = {
        "audio_path": str(input_path),
        "output_dir": str(tmp_path),
        "audio_contract": {"start_seconds": 0.1, "end_seconds": 0.7, "output_sample_rate": 48000},
        "settings": {"save_format": "wav", "wav_subtype": "FLOAT"},
        "model": {"stems": ["Vocals", "Instrumental"]},
    }
    selected_request, selected, timeline = select_source(request)
    assert sf.info(selected_request["audio_path"]).frames == round(0.6 * rate)
    assert timeline["start_sample"] == round(0.1 * rate)
    paths = [tmp_path / "x_(Vocals).wav", tmp_path / "x_(Instrumental).wav"]
    sf.write(paths[0], selected * 0.4, rate, subtype="FLOAT")
    sf.write(paths[1], selected * 0.6, rate, subtype="FLOAT")
    outputs, result = finalize(paths, request, selected, timeline)
    for path in outputs:
        samples, output_rate = sf.read(path, always_2d=True)
        assert output_rate == 48000 and len(samples) == 28800
        assert samples.shape[1] == channels
        assert abs(int(np.argmax(np.abs(samples[:, 0]))) - 7200) <= 1
        assert b"PEAK" not in path.read_bytes() and b"LIST" not in path.read_bytes()
    assert result["measurements"]["reconstruction"]["rms"] < 1e-7
    assert result["timeline"]["source_offset_seconds"] == 0.1


def test_flac_decodes_identically_after_removing_comments(tmp_path: Path):
    path = tmp_path / "audio.flac"
    sf.write(path, np.linspace(-0.5, 0.5, 1001), 44100, subtype="PCM_24")
    before, rate = sf.read(path)
    strip_tags(path)
    after, final_rate = sf.read(path)
    assert final_rate == rate and np.array_equal(before, after)
    assert path.read_bytes()[4] == 128


def test_wav_descriptive_chunks_removed(tmp_path: Path):
    path = tmp_path / "tagged.wav"
    sf.write(path, np.zeros(100), 44100)
    raw = path.read_bytes() + b"LIST" + struct.pack("<I", 4) + b"INFO"
    path.write_bytes(raw[:4] + struct.pack("<I", len(raw) - 8) + raw[8:])
    strip_tags(path)
    assert b"LIST" not in path.read_bytes() and sf.info(path).frames == 100


def test_large_missing_tail_is_not_padded_into_fake_success(tmp_path: Path):
    path = tmp_path / "broken_(Vocals).wav"
    sf.write(path, np.zeros(100), 44100)
    with pytest.raises(MaestroError, match="native duration"):
        finalize(
            [path],
            {"settings": {}},
            np.zeros((44100, 1)),
            {
                "output_sample_rate": 44100,
                "expected_output_frames": 44100,
                "source_sample_rate": 44100,
            },
        )
