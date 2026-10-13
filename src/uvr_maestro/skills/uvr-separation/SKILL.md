---
name: uvr-separation
description: Separate audio into verified stems with UVR Maestro. Use for single-file or batch source separation and job progress or cancellation.
---

# uvr separation

Check `uvr_status`, resolve a supported installed model, and use `probe_audio`
or `verify_audio` where the codec is supported. For a pilot, pass the audio object's
source-relative start_seconds/end_seconds directly. Choose an output root outside
the UVR installation. Lossless WAV/FLAC preserves selected source duration and rate
unless output_sample_rate is supplied; MP3 requires exact_length false.

`start_separation` returns a queued job. Poll `get_job` until succeeded, failed
or cancelled. Report progress only when it changes meaningfully. Use
`cancel_job` if requested. The server serializes native workers; multiple queued
requests do not imply parallel GPU use.

Options override only explicitly provided values. The selected profile overrides
defaults. VR window/aggression controls, MDX frame count and Demucs segment
seconds are different controls. Use source labels advertised by the selected
model. Demucs defaults to All Stems.

Only a succeeded job has checked output files and a persisted manifest. Inspect
its device, fallback reason, output duration/channels and manifest before reporting
completion. Native CPU and CUDA support is measured per architecture; do not
infer that ONNX CUDA was used from Torch CUDA availability.

The succeeded timeline reports actual source boundaries and source_offset_seconds.
Pass that offset and the original source_sample_rate to an independent transcription
consumer. Equal frame counts do not prove calibrated model attack latency. Per-stem
levels and complete-source reconstruction residual are not measured crosstalk.

`separate_audio` waits for completion; prefer start_separation for long work.
`separate_batch(confirm=False)` previews direct-child audio files. With authorized
confirmation, it runs real per-file jobs and preserves individual failure results.

