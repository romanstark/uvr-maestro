---
name: uvr-session
description: Start or diagnose a UVR Maestro session. Use when setting up local separation, checking engine readiness or resolving model/runtime failures.
---

# uvr session

Begin with `uvr_status`. The MCP server can browse the catalog before the heavy
runtime is installed. Inspect engine.ready, reused_installation and executables.
The pinned engine version and the desktop GUI version are distinct.

If the engine is missing, use the owner's authorized scope to run
`uvr-maestro-setup --confirm` locally. CPU is the default setup; CUDA uses
`--acceleration cuda`. This downloads source and dependencies, not model weights.
An installed wheel contains the setup command; no development checkout is needed.

Read `uvr_skills` for task guidance. Newer catalog architectures may be marked
unsupported even when their files exist. Do not turn catalog presence into a
claim of engine compatibility. CPU/GPU capability in status is the last measured
preflight; the completed job reports the device actually used.

Existing UVR desktop files are reused read-only. Profiles and new downloads
belong to Maestro. Never propose editing UVR's frozen Python installation to
satisfy a missing dependency.

