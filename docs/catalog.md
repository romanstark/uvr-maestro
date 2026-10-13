# Catalog and ownership

The bundled snapshot is ordered YAML under `uvr_maestro/catalog/`, supplemented
by VR/MDX model-identification metadata and snapshot provenance. Refresh reads
UVR's `download_checks.json`, name map and hash metadata and atomically commits
one versioned cache. Failed parsing leaves the previous cache intact.

A `ModelSpec` contains an ID, aliases, architecture, purpose hints, possible
source labels, optional verified status and explicit `Artifact` records. Each
artifact has a safe filename, relative directory, source URL, role and optional
trusted size/SHA-256. The runtime resolves actual model parameters by UVR's MD5
of the final ten MiB. That MD5 identifies parameters; it is not a full-file
integrity proof. Run manifests use full SHA-256 fingerprints.

| Family | Required local layout |
| --- | --- |
| VR | `VR_Models/<filename>.pth` plus packaged VR parameter definitions |
| MDX | `MDX_Net_Models/<filename>.onnx` or checkpoint, with identified dimensions |
| MDX23C | `MDX_Net_Models/<filename>.ckpt` and `model_data/mdx_c_configs/<config>.yaml` |
| Classic Roformer | Matching weights/config pair in the MDX folder, optional separate runtime |
| Demucs v3/v4 | `Demucs_Models/v3_v4_repo/<bag>.yaml` and all referenced hash-named `.th` files |

Owned storage is searched first, then a discovered/configured desktop install.
`installed` means every required file exists and passes declared-size checks.
`broken` means resolved files fail those checks. `available` means required
artifacts are missing. `unsupported` denotes an unimplemented architecture.
A runtime call can separately fail with `metadata_missing` or `config_missing`.

Downloads use temporary files and validated replacement. An existing zero/partial
file is repaired rather than blindly skipped. Known size/hash checks are applied;
unknown values remain unknown. Configs must decode into YAML objects, and Demucs
bags must name model signatures. Partial artifact sets never count as installed.

Removal applies only to complete owned artifact sets and preserves checkpoints
referenced by another catalog row. Read-only reused files and orphan weights are
never automatically deleted. Named profiles and artifact paths are confined by
resolved paths, including symlink resolution.

The snapshot contains newer networks to make discovery truthful. Classic
MelBand/BandSplit Roformer use a separate adapter/runtime; modified modern
Roformer/Bandit and legacy Demucs v1/v2 cannot execute.
A catalog refresh does not update the engine, silently activate a new architecture
or install/remove model files.
