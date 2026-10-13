# Third-party notices

UVR Maestro's own source is AGPL-3.0-or-later; see LICENSE. This package includes
catalog metadata and original adapter code. Model weights, the UVR engine source,
DiffQ source, FFmpeg and Rubber Band binaries are not included in the wheel.
Runtime setup downloads the pinned sources separately and preserves their notices.

## Ultimate Vocal Remover GUI

Source: https://github.com/Anjok07/ultimatevocalremovergui
Pinned revision: a5f88453bfb2b38b05a965bcf67727243e0cbf19.
UVR's About text in gui_data/constants.py identifies the application as MIT
licensed and contains its permission notice. The installed source preserves that
file unchanged. The archive at this revision has no standalone root LICENSE file.
The headless adapter invokes separation classes without starting/importing UVR.py.

## Model and application metadata

Sources: https://github.com/TRvlvr/application_data and
https://github.com/TRvlvr/model_repo . Bundled VR/MDX parameter metadata and catalog
filenames/URLs are attributed to these repositories. Snapshot provenance is stored
with the catalog. Model weights and their terms belong to their respective
publishers. No blanket weight-license or commercial-use claim is made by Maestro.

## Demucs and DiffQ

Demucs source is obtained through the pinned UVR archive. Its source headers
attribute Facebook/Meta and the original contributors.
Original project: https://github.com/facebookresearch/demucs .

DiffQ 0.2.4 source is downloaded from its published source artifact. Its LICENSE
is Attribution-NonCommercial 4.0 International and is preserved under the runtime
support directory. Original project: https://github.com/facebookresearch/diffq .
The portable decoder is Maestro's own implementation of the documented legacy
integer storage layout. It avoids compiling DiffQ's optional native bitpack
extension. Quantized Demucs inference was checked against actual local models.
Maestro's license does not replace these separately downloaded source terms.

## Audio/runtime dependencies

- Optional BS-RoFormer 0.3.0: https://github.com/lucidrains/BS-RoFormer , MIT.
  Its distribution and notices are installed in a separate runtime. Maestro's
  adapter invokes the classic MelBandRoformer/BSRoformer classes; the library is
  not embedded in Maestro's wheel. Checkpoint terms remain publisher-specific.

- PyTorch/TorchAudio/TorchVision: https://pytorch.org/ . Their notices are installed
  with the corresponding Python distributions.
- ONNX/ONNX Runtime/onnx2pytorch: https://onnx.ai/ and
  https://github.com/microsoft/onnxruntime . Runtime distributions retain notices.
- librosa: https://github.com/librosa/librosa , ISC.
- SoundFile/libsndfile: https://github.com/bastibe/python-soundfile . Installed
  distributions include their library notices.
- Matchering: https://github.com/sergree/matchering , GPL-3.0.
- Rubber Band: https://breakfastquay.com/rubberband/ . Executable licenses depend
  on the selected upstream distribution. pyrubberband is a separate Python binding.
- FFmpeg: https://ffmpeg.org/ . The user's selected executable carries its own
  build-specific licensing. Maestro does not distribute it.

Ultimate Vocal Remover, PyTorch, ONNX and related names belong to their owners.
This independent project is not endorsed by the upstream projects.
