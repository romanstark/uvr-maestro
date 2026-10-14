# UVR Maestro

> An AI stem separation and audio restoration partner for **Ultimate Vocal Remover (UVR5)**. Speak in natural language to extract vocals, drums, bass, guitar and piano, combine separation models and clean up recordings, then bring the resulting stems back to your DAW. Your audio stays on your computer.

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE) [![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/) [![UVR5](https://img.shields.io/badge/UVR-5-orange.svg)](https://github.com/Anjok07/ultimatevocalremovergui) [![No telemetry](https://img.shields.io/badge/telemetry-none-brightgreen.svg)](#your-music-stays-yours) [![CI](https://github.com/romanstark/uvr-maestro/actions/workflows/ci.yml/badge.svg)](https://github.com/romanstark/uvr-maestro/actions/workflows/ci.yml) [![Glama](https://glama.ai/mcp/servers/romanstark/uvr-maestro/badges/score.svg)](https://glama.ai/mcp/servers/romanstark/uvr-maestro)

---

## Stay in the creative flow

Finding the right separation often means trying several models, repeating settings
and moving files between processing steps. That work can interrupt the listening
that tells you whether a result is useful.

**UVR Maestro** acts like an assistant engineer beside you in the studio:

- **Start with your musical goal:** Ask for a vocal, an instrumental, drum stems or a cleanup pass in ordinary studio language.
- **Try a small section first:** Compare models on an excerpt before processing the whole recording.
- **Build repeatable workflows:** Choose model settings, combine compatible models and route stems through cleanup chains in one request.
- **Keep the source timeline:** Lossless exports preserve the selected duration and source sample rate, or use a rate you choose. Excerpts report their position in the original recording.

Maestro uses UVR's engine in a separate local worker and can reuse your installed
models. You listen to the results and decide what belongs in the production.

---

## How it feels in the studio

### 1. Find the right vocal model

> **You:** *"Try Kim Vocal 2 and a compatible VR model on the first 20 seconds of the chorus. Keep both results so I can compare them."*

Maestro checks which models are installed, processes the same excerpt and returns
the vocal and complementary files with their model settings and source offset.
Audition the results, choose a model, then use it for the full song.

### 2. Work with your own drum models

> **You:** *"Extract drums with htdemucs, then run my registered DrumSep model on that stem. Keep the intermediate drums too."*

Maestro routes the drum output into a compatible custom MDX23C model and keeps
the requested intermediate. The resulting audio files can go into any DAW.
Independent Audio Maestro can estimate MIDI from the files;
[Live Maestro](https://github.com/romanstark/live-maestro) handles pad loading and DAW placement.

### 3. Clean up a recording

> **You:** *"Reduce reverb in this vocal, then denoise it. Keep each step so I can hear whether the cleanup goes too far."*

Maestro checks the installed cleanup models, routes each selected output into the
next step and returns the intermediate and final files. Inspect their levels,
then listen for lost detail before choosing the result.

---

## What you can do

| Workflow | Example prompt | What Maestro does |
| --- | --- | --- |
| **🎙️ Vocals & Instrumentals** | *"Separate the lead vocal and instrumental from this recording"* | Runs a selected compatible model and returns checked audio files |
| **🥁 Drums & Custom Models** | *"Use my DrumSep model on this drum stem"* | Registers local model/config pairs read-only and runs supported custom architectures |
| **🎛️ Ensembles & Chains** | *"Combine these two vocal models, then remove reverb"* | Combines compatible outputs and routes explicit sources through processing steps |
| **✂️ Pilot Excerpts** | *"Only process 45 to 65 seconds for now"* | Separates the chosen section and reports its source-time offset |
| **🎧 Audio Preparation** | *"Align these recordings and make a short preview"* | Creates local previews, corrects constant offsets and provides pitch/time utilities |
| **📊 Levels & Inspection** | *"Check the stem levels before I import them"* | Reports measured audio properties and marks unavailable comparisons explicitly |
| **⏳ Long Jobs & Batches** | *"Process this folder and show progress"* | Runs serialized background jobs with per-file results and cancellation |

### Which stems can you separate?

Each model has a trained set of outputs. Choose a model for the sources you need;
an ensemble combines model estimates, and a chain can split a resulting stem further.

| Sources | Supported route |
| --- | --- |
| **Vocals + instrumental** | Compatible VR, MDX, MDX23C or classic Roformer models |
| **Vocals, drums, bass + other** | Four-source Demucs models such as `htdemucs` |
| **Vocals, drums, bass, guitar, piano + other** | `htdemucs_6s`; piano separation is experimental and can contain substantial leakage and artifacts |
| **Lead + backing vocals** | `separate_vocal_splitter` with a compatible BVE model, operating on separated vocals |
| **Kick, snare, toms, hi-hat, ride + crash** | The tested original DrumSep checkpoint with its matching MDX23C configuration, registered locally |
| **Synths, strings, brass, general keyboards or choir** | No dedicated verified route in the bundled model set. These can remain in `other` or vocals; distinct extraction requires a suitable trained model with a compatible adapter |

Piano does not cover every keyboard instrument, and backing vocals do not guarantee
a separate choir or an individual singer. Declaring a new output label does not
teach a model to extract that source. See [model workflows](docs/audio-workflows.md)
and the [Demucs model documentation](https://github.com/facebookresearch/demucs#available-models)
for model-specific outputs and limitations.

---

## Why Maestro is reliable

- **Checked output files:** Completed separation results verify decoded frames, sample rate, channels and finite samples. WAV/FLAC exports strip container tags.
- **Traceable processing:** Saved manifests fingerprint the input, model artifacts and output files, with the actual settings and backend.
- **Clear job state:** A queued job is still processing. Only `state: succeeded` includes checked outputs; failures and cancellations remain visible.
- **Concrete change previews:** Batch processing and destructive model/profile operations require `confirm=True`. Without it, Maestro reports the proposed action.
- **Respect for existing files:** Desktop model files are reused read-only. Outputs receive unique run directories.
- **Honest backend reporting:** GPU availability, the selected device and CPU fallback reasons are reported rather than inferred from the machine's hardware.

---

## Quick start

You need [uv](https://docs.astral.sh/uv/). An existing UVR5 installation lets
Maestro reuse its models and audio utilities; models can also be downloaded
explicitly through Maestro.

### 1. Install it

```powershell
uv tool install --managed-python --python 3.12 git+https://github.com/romanstark/uvr-maestro.git
```

### 2. Set up the local engine

```powershell
uvr-maestro-setup --confirm
```

Setup downloads pinned UVR source and provisions a separate Python 3.11 inference
environment. Your desktop UVR installation alone does not replace this step.
Use uv-managed Python: Microsoft Store Python redirects AppData writes into a
private folder that the separate engine cannot share.

For an NVIDIA system using the CUDA 12.4 build:

```powershell
uvr-maestro-setup --acceleration cuda --confirm
```

### 3. Connect your AI assistant

Add the stdio server to your MCP client's configuration:

```json
{
  "mcpServers": {
    "uvr-maestro": {
      "command": "uvr-maestro"
    }
  }
}
```

Use the command's absolute path if your client does not inherit your shell's PATH.
Examples for agents, Claude, Codex, Cursor and VS Code are included; see
[MCP client examples](docs/mcp-clients.md) for configuration and skill export.

Start with: *"Check my UVR setup and show me the installed vocal models."*
Maestro uses `uvr_status` and `find_model` to establish readiness before separation.

---

## Your music stays yours

- **Local audio processing:** Your recordings, stems and settings remain on your disk. Inference runs on your computer.
- **No telemetry:** UVR Maestro collects no analytics and uploads no audio, prompts or settings.
- **Explicit network access:** Runtime setup, catalog refresh, model downloads and update checks use the network when requested. Separation does not fetch missing weights.

Model downloads can use publishers' Hugging Face repositories or Meta's public-file
host. Maestro does not bundle model weights.

---

## What stays in your hands

| Decision | Your role |
| --- | --- |
| **Model and ensemble choice** | Audition pilot results. Selection scores describe task suitability, not measured separation quality |
| **Cleanup strength** | Listen for removed detail and artifacts; intermediate files help compare processing steps |
| **Quality and crosstalk** | Levels and residuals describe measured signals. Reference-free leakage and perceptual quality are not guaranteed |
| **Musical interpretation** | Choose the beat grid, MIDI notes and placement in your DAW. UVR Maestro supplies audio and source timing |

---

## Compatibility and limits

The packaged catalog snapshot contains **104 models**. Classic MelBand/BandSplit
Roformer use an optional separate runtime; modified modern Roformer/Bandit and
legacy Demucs v1/v2 remain unsupported. Catalog presence does not establish
algorithm support or model quality.

Native coverage includes VR, MDX, MDX23C, Demucs v3/v4 and classic Roformer on
Windows, with CPU/CUDA availability depending on the architecture. Apple Silicon
coverage includes VR v5, MDX and MDXC on CPU/MPS, and Demucs v4 on CPU. Demucs v4
uses an explicit CPU fallback for the pinned Torch MPS convolution limit.
See [the compatibility matrix](docs/verification.md) for backend coverage and limits.

Single-model, ensemble and chain jobs support direct excerpts, source-rate or
selected-rate lossless output, per-member options and intermediate retention.
See [audio workflows](docs/audio-workflows.md) for frame guarantees, read-only
custom models, Roformer setup and independent consumers.

Models have separate installation and verification states. An unknown upstream
size/hash is reported as unknown integrity, not a successful trusted-checksum
comparison. Manifests fingerprint the actual inputs, model artifacts and outputs.
A model-selection score is a task heuristic, not a quality benchmark.

---

## MCP tools and resources

<details>
<summary>Browse the tool reference</summary>

| Workflow | Tools | Behavior |
| --- | --- | --- |
| Readiness and model choice | `uvr_status`, `list_models`, `find_model`, `verify_catalog` | Browse the offline snapshot, actual artifact sets and supported architectures. |
| Manage models | `refresh_catalog`, `download_model`, `download_missing_models`, `remove_model`, `register_model`, `unregister_model` | Explicit downloads and read-only local model registration, with confirmation. |
| Profiles | `get_settings`, `set_settings`, `reset_settings`, `change_journal` | Validated persisted defaults, concrete previews and durable mutation records. |
| Separate audio | `start_separation`, `separate_audio`, `separate_batch` | Real model inference and verified output files, including four/six-source Demucs. |
| Combine or restore | `separate_ensemble`, `separate_chain`, `separate_vocal_splitter` | Weighted or spectral ensembles, explicitly routed cleanup and lead/backing vocals. |
| Inspect and transform | `probe_audio`, `verify_audio`, `analyze_audio`, `create_sample`, `pitch_shift`, `time_stretch`, `align_inputs`, `matchering_inputs` | Measured audio characteristics and actual file-producing utilities. |
| Follow jobs | `get_job`, `list_jobs`, `cancel_job` | Serialized workers, progress, cancellation and terminal results. |
| Upstream updates | `check_update`, `list_updates` | Text bulletin and platform patch/installer metadata. |
| Workflow guidance | `uvr_skills` | Five client-neutral skills served through MCP and exportable from the package. |

The server exposes **34 tools**, three static resources and one resource template:
`uvr://catalog`, `uvr://settings`, `uvr://status`, `uvr://skill/{name}`.
Two prompts, `separate_stems` and `restore_recording`, guide the complete workflows.

</details>

Five client-neutral skills cover setup, model choice, separation, restoration and
audio utilities. Read them through `uvr_skills` or export them with
`uvr-maestro-skills`.

## Paths and audio utilities

<details>
<summary>Data directories, environment variables and external utilities</summary>

Maestro's own data directory contains models, profiles, cache, jobs, journal and
runtime. The default is `%LOCALAPPDATA%/uvr-maestro` on Windows,
`~/Library/Application Support/uvr-maestro` on macOS and
`$XDG_DATA_HOME/uvr-maestro` or `~/.local/share/uvr-maestro` on Linux.

| Environment variable | Purpose |
| --- | --- |
| `UVR_MAESTRO_DATA_DIR` | Override Maestro's owned data root. |
| `UVR_MAESTRO_UVR_DIR` | Explicit desktop/source installation root containing `models/`. |
| `UVR_MAESTRO_MODELS_DIR` | Override the managed model root. |
| `UVR_MAESTRO_SETTINGS_ROOT` | Override the independent profile directory. |
| `UVR_MAESTRO_RUNTIME_DIR` | Override the managed engine runtime root. |
| `UVR_MAESTRO_ENGINE_PYTHON` | Development/expert override for an already provisioned interpreter. Setup preflight still verifies it. |

FFmpeg is required for MP3 export and additional input codecs. Rubber Band is
required for pitch/time utilities. Both are discovered on PATH or in the existing
Windows/macOS UVR installation. Maestro does not bundle those executables.

An existing UVR5 desktop installation supplies reusable models and executables.
Maestro additionally provisions its own pinned headless engine; the desktop
installation alone does not replace that setup step.

</details>

---

## Related Maestro projects

- [Dorico Maestro](https://github.com/romanstark/dorico-maestro) controls notation workflows in Steinberg Dorico.
- [Live Maestro](https://github.com/romanstark/live-maestro) controls Ableton Live, musical timing, MIDI and sample placement.

Each server works independently. Use ordinary MIDI, audio and JSON files when
combining their workflows.

---

## Development

From a local checkout:

```powershell
uv venv --managed-python --python 3.12
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
.venv/Scripts/python -m pytest
.venv/Scripts/ruff check src scripts tests
.venv/Scripts/pyright
.venv/Scripts/pyrefly check
uv build
```

On macOS and Linux, use the corresponding commands under `.venv/bin/`.
The unit suite requires neither Torch nor UVR. CI checks the light server on
Windows, Linux and macOS with Python 3.11 through 3.14, then installs wheels and
extracted sdists away from the checkout.

- [Architecture](docs/architecture.md)
- [Catalog and ownership](docs/catalog.md)
- [Operational limits](docs/limits.md)
- [Compatibility and validation](docs/verification.md)
- [Audio workflows](docs/audio-workflows.md)
- [Contributing](CONTRIBUTING.md)
- [Third-party notices](THIRD-PARTY.md)
- [AGPL-3.0-or-later license](LICENSE)

UVR Maestro is independent of the upstream projects and their trademark owners.
