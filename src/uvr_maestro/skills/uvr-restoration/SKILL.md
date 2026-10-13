---
name: uvr-restoration
description: Route UVR cleanup models or split lead/backing vocals. Use for de-echo, denoise, dereverb and explicit separation chains.
---

# uvr restoration

Find models by the restoration purpose, then inspect their real source labels.
A purpose such as de_echo is not necessarily the emitted stem name. Use a short
sample before a full recording when the cleanup choice is uncertain.

`separate_chain` requires each step's model and output_stem. The selected output
becomes the next input. Choose the actual clean source, not the removed-noise or
removed-reverb residue. Poll get_job and inspect each step's outputs. A missing
source label fails the chain rather than choosing another output silently.

The original source selection applies once, before step one. keep_intermediates
defaults to true; false prunes only this job's earlier steps after verified final
output. Do not sum consecutive chain outputs as complementary sources.

`separate_vocal_splitter` implements UVR's lead/backing-vocal workflow using a
vocal extraction model and a BVE model identified by metadata. It is a different
workflow from denoise/dereverb cleanup. The splitter can retain the initial
Vocals/Instrumental files as well as split vocal outputs.

For an ensemble, every named source needs at least two compatible model outputs.
Average accepts positive weights. min_spec and max_spec use UVR's spectral
algorithms and require unit weights. Verify lengths and source meaning before
comparing output quality; no algorithm guarantees the best result for every song.
Each member's options override shared ensemble options; validate segment/overlap
units for that member's architecture rather than assuming one universal setting.

