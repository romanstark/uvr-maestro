# Architecture

The MCP server is a light control plane. It discovers models and validates
requests without importing Torch. `executor.Workbench` owns preflight, guarded
mutations, profile precedence and job submission. `workbench.py` is a compatibility
import of that same executor, not a second implementation.

The registry loads the packaged YAML snapshot or the last validated cache.
Model IDs and aliases are separate from exact artifact filenames. Disk scans
resolve architecture subfolders across owned storage and a read-only desktop
installation. New architectures require an adapter, not only new catalog data.

`JobManager` serializes native inference and audio utilities through one worker.
It creates a unique output directory and stores queued/running/terminal records.
The worker's stdout is a JSON-lines progress channel. Engine prints and native
warnings go into its stderr log, preserving MCP stdio. Cancellation terminates
only the job's worker. An old unfinished record is reported as interrupted.

The worker runs from a packaged Python file in a separately provisioned Python
3.11 interpreter. Runtime setup downloads the exact UVR source revision
`a5f88453bfb2b38b05a965bcf67727243e0cbf19` and verifies archive checksums.
It installs a locked dependency set and works from an installed wheel.

The adapter creates UVR's real model_data attributes and process callbacks.
It calls `SeperateVR`, `SeperateMDX`, `SeperateMDXC` or `SeperateDemucs` through
`seperate()`. GUI-bound `UVR.py` is never imported. Compatibility wrappers
translate old librosa positional arguments; a portable integer decoder handles
DiffQ's legacy bit layout. No signal-separation algorithm is replaced.

Torch GPU capability and ONNX providers are separate. If ONNX CUDA is unavailable,
UVR's Torch ONNX-conversion path can run MDX on CUDA. The completed result records
actual device use and effective frame settings. No inference-time model fetches
are permitted: Demucs resolves a complete local YAML/checkpoint repository.

The worker verifies audio files. The manager independently checks confinement,
file existence, finite decoded samples and hashes before marking success and
atomically writing the manifest. Individual batch failures remain in per-file
results. Cancellation/failure preserves its isolated partial-run directory.

Profiles apply in this order: bundled defaults, selected persisted profile,
explicit per-call overrides. Confirmed catalog/profile mutations record an intent
before writing and a completed/failed journal outcome afterward. Profiles are
independent of UVR's GUI settings.

Skills and prompts describe operations using the published MCP surface. The
package includes the same client-neutral skill folders served by resource URIs.

The export layer stages a source-relative selection once, restores the requested
rate/channel/frame contract and strips descriptive tags before independent file
verification. Ensembles merge per-member options; chains retain the original
timeline. Retention cleanup is confined to the job's own intermediate folders.

Read-only custom registrations live separately from refreshed catalog data.
Their artifact fingerprints must still match before inference. Classic Roformer
uses an additional pinned runtime, strict checkpoint/config loading and the same
export/job verification layer. The adapter does not import or start another MCP
server. File consumers receive audio and explicit source-time metadata.
