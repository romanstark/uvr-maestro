# Compatibility and validation

The control server runs separately from the native inference runtime. Control-server
and package checks cover Windows, Linux and macOS. Native inference coverage is
specific to the architecture, model and backend below.

## Native engine coverage

| Architecture | Windows CPU | Windows CUDA | Apple Silicon CPU | Apple Silicon MPS |
| --- | --- | --- | --- | --- |
| VR v5 | Verified | Verified | Verified | Verified |
| VR v4 | Verified | Not measured | Not measured | Not measured |
| MDX | Verified | Verified | Verified | Verified |
| MDX23C | Verified | Verified | Verified | Verified |
| Demucs v3 | Verified | Verified | Not measured | Not measured |
| Demucs v4 | Verified | Verified | Verified | CPU fallback |
| Classic MelBand/BandSplit Roformer | Verified | Verified | Not measured | Not measured |

The standard engine uses pinned UVR source and Torch 2.5.1 in a separate Python
3.11 environment. Classic Roformer uses its optional BS-RoFormer 0.3.0 runtime.
Windows CUDA coverage uses an NVIDIA RTX 3070 Ti; Apple Silicon coverage uses
an M2 Pro. These are compatibility checks, not minimum hardware specifications.

Native model coverage includes HP VR models, Kim Vocal 2, MDX23C InstVoc,
htdemucs, htdemucs_6s, quantized Demucs v3, custom MDX23C/DrumSep and the classic
Roformer pairs described in [audio workflows](audio-workflows.md).
Compatibility with one model does not verify every catalog entry or custom checkpoint.

Demucs v4 selects CPU when `auto` or `mps` would otherwise select MPS. The pinned
Torch MPS backend exceeds its convolution channel limit for the tested htdemucs
models. Results preserve `requested_device` and report the actual backend and
`fallback_reason`. Other verified MPS families continue to use the Apple GPU.

## Workflow coverage

Native checks cover separation, batches, weighted and spectral ensembles, routed
cleanup chains, vocal splitting, cancellation and subsequent jobs. Audio utilities
cover preview creation, pitch/time changes, constant-offset alignment, loudness,
reference mastering and WAV/FLAC/MP3 export. FFmpeg and Rubber Band are discovered
from PATH or an existing desktop installation.

The export contract is checked with mono/stereo sources, multiple source rates,
impulses, excerpts, resampling, frame reconciliation and container-tag inspection.
Full-length and excerpt inference also exercise real recordings. WAV/FLAC checks
establish decoded file length, rate, channels and finite samples; they do not
calibrate model attack latency or measure perceptual separation quality.

## Package and protocol coverage

Clean wheel and extracted-sdist checks cover packaged assets, installation away
from the checkout, catalog loading, skill export and real MCP stdio calls.
The light suite covers schemas, confirmation guards, path confinement, cache and
download failures, subprocess job lifecycle and artifact verification. Ruff,
Pyright and Pyrefly cover the control server and development scripts.

CI covers the light server on Windows, Linux and macOS with Python 3.11 through
3.14. Engine compatibility is a separate, pinned dependency contract.
Generated audio, model weights, runtimes and detailed run records are excluded
from distributions.

## Boundaries

Linux native inference and the unmeasured matrix cells are not covered by these
local engine checks. Modified modern Roformer/Bandit and Demucs v1/v2 are unsupported.
Long-song resource usage, listening quality, automatic identification of voices
and reference-free crosstalk are not guaranteed. See [operational limits](limits.md)
and [the source export contract](audio-workflows.md) for caller-visible behavior.
