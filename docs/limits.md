# Operational limits

- File-based processing only. There is no real-time stream or live GUI control.
- Runtime setup requires a regular or uv-managed Python interpreter. Microsoft
  Store Python's private AppData redirection is incompatible with the separate
  engine process; confirmed setup refuses it before creating files or downloading.
- UVR's engine operates on mono/stereo input and commonly emits stereo at 44.1 kHz.
  VR parameter metadata controls its internal sample rates. Multichannel processing
  is not advertised. Final separation output restores decoded source channels/rate
  or the explicitly requested rate; WAV/FLAC frames match the selected duration.
- WAV/FLAC and codecs supported by libsndfile can be inspected in the light server.
  UVR's librosa/audioread path and an installed FFmpeg provide additional input
  codecs. Probe/decode tools return explicit codec errors when unavailable.
- WAV supports PCM_16, PCM_24 and FLOAT. FLAC conversion uses PCM_24. MP3 export
  needs FFmpeg and `audio.exact_length=false`. Lossy formats may introduce
  duration/padding differences. See [the export contract](audio-workflows.md).
- One native worker runs at a time. CPU work can be slow, and memory/VRAM depends
  on model, input length and options. There are no invented memory guarantees.
- GPU requests report the actual backend and CPU fallback reasons. Native Apple
  Silicon checks verified MPS for VR, MDX and MDXC. Demucs v4 uses CPU on Apple
  Silicon because the pinned PyTorch MPS backend exceeds its convolution channel
  limit. Demucs v3 and optional Roformer MPS behavior remain unmeasured.
  ONNX provider availability is distinct from Torch CUDA availability.
- Jobs preserve failure/cancellation evidence in their unique run folder. They
  are not resumed after a server restart. Persisted unfinished jobs read as
  interrupted. Worker cancellation loses unflushed inference progress.
- Batch scans direct children of the selected directory and reports each file.
- Average ensembles require matching channels/sample rates and permit at most
  50 ms of frame-rounding difference, padded explicitly and reported in the result.
  Spectral min/max use UVR's algorithms and unit weights. No tempo/phase correction
  is implied by an ensemble.
- Alignment corrects a constant sample offset. It does not correct tempo drift
  or guarantee alignment of unrelated performances.
- Rubber Band pitch shifting preserves duration; time stretching preserves pitch.
  The time rate is playback speed, so 1.05 is five percent faster.
- BS.1770 integrated loudness is null for silence or clips below the gating window.
  Matchering is reference mastering, not a subjective quality guarantee.
- Vocal splitting uses a metadata-identified BVE model on separated vocals. It is
  distinct from denoise/dereverb chains. Cleanup output labels must be routed
  explicitly; a requested missing source fails the chain.
- Classic MelBand/BandSplit Roformer run in an optional separate pinned runtime.
  Modified modern Roformer/Bandit and Demucs v1/v2 remain discovery-only.
- Source timeline preservation does not calibrate a model's attack latency.
  Reconstruction residual requires all complementary sources; measured crosstalk
  requires references and is not inferred from one mixture.
- Runtime setup and explicit catalog/download/update tools use the network.
  Audio itself is never uploaded and inference never downloads missing weights.
