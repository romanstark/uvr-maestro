# Contributing

Use Python 3.11 or newer for the control server. Install `.[dev]` and run pytest,
Ruff, Pyright and Pyrefly before submitting a change. The ordinary suite does not
need a UVR installation, Torch or a GPU. Native engine probes run separately on
owned fixtures and produce evidence outside release artifacts.

Model data belongs in the catalog. Preserve upstream filenames and explicit
artifact roles. A new architecture needs an adapter and real integration results.
A successful file download is not evidence that a model separates correctly.

Public MCP parameters need Field descriptions, units, fixed choices and honest
side-effect annotations. Unsupported operations must fail clearly. Destructive
operations must provide a concrete no-write preview and respect confirm=True.

Keep the existing desktop installation read-only. Do not add audio uploads,
telemetry or hidden inference-time model downloads. Never commit user recordings,
model weights, local runtimes or generated native results.

Build wheel and sdist, install them outside the checkout and run their own tests.
Catalog metadata, workflow skills, the worker, compatibility adapters and runtime
requirements must be included in the wheel. CI also checks the extracted sdist.

Keep the version in pyproject.toml, the package and plugin manifest consistent.
Update the compatibility matrix when an architecture or backend gains native
coverage. Public documentation describes current behavior, setup and limits;
detailed development logs belong outside the published documentation.

Include source and license notices for third-party contributions. UVR Maestro's
own code is AGPL-3.0-or-later. Separately downloaded runtimes and weights retain
their own upstream licenses.
