# Source timeline and model workflows

The single-model, ensemble and chain tools accept an `audio` object:

```json
{
  "start_seconds": 30,
  "end_seconds": 40,
  "output_sample_rate": 48000,
  "exact_length": true
}
```

Omit the output rate to retain the decoded source rate. Start/end are relative to
the original file; end is exclusive. Boundaries round to the nearest source
sample, with ties rounded up. Requests outside the decoded file are refused.
Succeeded results report the actual boundaries, `source_offset_seconds`, source
rate/frames and expected output frames in `timeline`.

WAV/FLAC outputs preserve channel count and the selected source duration. Rate
conversion uses a polyphase resampler. Small engine frame differences are
explicitly reconciled at the end; a discrepancy exceeding the larger of 50 ms
and 0.5% of duration fails. Files are decoded and checked again after export.
Descriptive tags are stripped; essential container/rate headers remain.
MP3 cannot promise this contract and requires `exact_length: false`.

This establishes a shared source time axis, not calibrated model attack latency.
Separation can change the shape or timing of an attack. Use measured detector
calibration when transcribing, and audition against the source. Do not infer
physical attack accuracy from matching file lengths.

Each output reports peak, RMS, integrated LUFS where defined and clipping.
Reconstruction residual is source minus the sum of all complementary outputs,
available only with a complete source set, matching channels and normalization
disabled. It is not a crosstalk or subjective quality score. Chains and incomplete
stem sets do not receive a misleading reconstruction score.

## Custom models and ensembles

`register_model` previews or stores a read-only local registration. Supply a unique
name, architecture, weights, output labels and required parameters/configuration.
MDX23C and Roformer need their matching YAML. Registration fingerprints artifacts
and refuses execution if they change. `unregister_model` removes the record only.
Registrations survive catalog refresh; downloaded models and desktop-owned models
keep their existing ownership rules.

An ensemble member has its own `options`, applied after shared options. Segment,
overlap and shifts retain architecture-specific units and validation. Chain steps
route an explicitly named output to the next model. Source selection applies once,
before the first step. `keep_intermediates` defaults to true; false removes only
the job's intermediate files after final output verification succeeds.

Compatible custom MDX23C pairs include MDX23C_D1581 and the original
aufr33/jarredou DrumSep checkpoint with its matching YAML. DrumSep exposes kick,
snare, toms, hh, ride and crash. Other checkpoints require their own configuration
and inference check. Model weights are not distributed with Maestro; obtain them
from their publisher and review their separate license. The DrumSep curator's
[mirror](https://github.com/openmirlab/mdxnet-infer/releases/tag/weights-drumsep-v1)
lists the upstream checkpoint license as unknown.

## Optional classic Roformer runtime

```powershell
uvr-maestro-setup --roformer --acceleration cuda --confirm
```

Use `--acceleration cpu` for CPU. This provisions a separate interpreter with
BS-RoFormer 0.3.0 and pinned dependencies, preserving the UVR engine runtime.
`UVR_MAESTRO_ROFORMER_PYTHON` selects an explicitly provisioned expert runtime.
Setup verifies its capability before use. Inference never installs dependencies
or fetches weights.

Tested pairs are Kim's MelBandRoformer and anvuew's archived classic
`deverb_bs_roformer_8_256dim_8depth`. The latter deliberately uses the author's
archive weights/config pair; the newer main-branch config describes a modified
architecture. The catalog preserves this curated pair across refreshes.

Strict state-dict loading and config validation reject incompatible variants.
Harmless Python tuple tags are supported by a restricted YAML loader; arbitrary
Python objects are refused. Modified modern Roformer/Bandit architectures remain
unsupported. Chunk duration and overlap use `roformer_segment_seconds` and
`roformer_overlap`; actual CPU/CUDA/MPS selection is reported.

## Independent consumers

UVR Maestro does not require Live Maestro. Any file consumer can use the audio
and run manifest. For Live's optional drum analysis, pass each selected output
file with `source_offset_seconds` and the original `source_sample_rate` from
`timeline`. Beat maps, MIDI, instrument assignment and DAW placement belong to
that consumer. No cross-server imports, calls or shared installation are needed.
