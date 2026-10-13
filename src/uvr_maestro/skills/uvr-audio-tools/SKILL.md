---
name: uvr-audio-tools
description: Transform, align, inspect or reference-master audio with UVR Maestro. Use for pitch, speed, previews, loudness, alignment and Matchering.
---

# uvr audio tools

Utilities return queued jobs. Use get_job for measured results and verified files.

pitch_shift changes pitch in semitones while preserving duration. time_stretch
takes a playback-speed multiplier: 1.05 is five percent faster with preserved
pitch. Both require a Rubber Band executable, discovered from PATH or the local
UVR installation. Preserve the source and use the new output.

align_inputs compares a reference and target and writes the target aligned to
the reference. A positive offset means the target lagged behind the reference.
It corrects a constant offset, not tempo drift. Silent inputs cannot resolve
alignment. Inspect the result before treating unrelated performances as aligned.

analyze_audio measures peak, RMS, clipping and integrated BS.1770 loudness.
Loudness can be null for silence or clips too short for the gating window.

matchering_inputs creates a 24-bit WAV from a target and reference. Select the
reference according to the user's production intent; do not describe the result
as a subjective quality improvement without listening. create_sample makes a
real preview and defaults to one third into the recording.

