---
name: uvr-models
description: Choose and manage UVR models by task and installed capabilities. Use when finding, downloading, verifying or removing a separation model.
---

# uvr models

Use `find_model` with the desired source or restoration task, then inspect
`list_models` for supported, installed, purposes and artifacts. The score is a
task/installation heuristic, not a measured quality benchmark. Model names,
filenames and actual output source labels are different concepts.

A Demucs model can require a YAML bag plus several shared checkpoints. An MDXC
model also needs a YAML config. `verify_catalog` reports complete installation
state and available trusted checksums; an unknown upstream size/hash does not
constitute verified integrity.

For a missing model, `download_model(confirm=False)` shows the URLs and owned
targets. Execute with confirm=True when the user has authorized that download.
Do not request the same authorization repeatedly. Bulk downloads can be large;
prefer the model needed for the current task.

`refresh_catalog` changes the cached snapshot without replacing existing model
files. `remove_model` refuses reused installation files and preserves shared
checkpoints. Unknown local files are reported as orphans, never silently deleted.

`register_model` previews/stores a read-only local identity with explicit architecture,
weights, output labels and matching configuration/parameters. Fingerprints must still
match before execution. `unregister_model` removes only the record. A successful
registration does not prove inference quality; probe the owner's actual model files.
Classic MelBand/BandSplit Roformer requires its separate setup runtime and matching
tested constructor/config. Modified modern variants remain unsupported.

