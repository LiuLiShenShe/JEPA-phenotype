# M1 verified implementation sources

Accessed 2026-10-11. Data pixels were never uploaded to an external API.

- Official DINOv2 source: https://github.com/facebookresearch/dinov2
  commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
- Official backbone definitions:
  https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/hub/backbones.py
- Official pretrained weight:
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
  SHA-256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
- Official ImageNet normalization and bicubic transform implementation:
  https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/data/transforms.py
  M1 deliberately replaces cropping with aspect-preserving letterboxing.
- PyTorch / torchvision / CUDA 12.8 version compatibility:
  https://pytorch.org/get-started/previous-versions/
- Context7 resolved PyTorch to `/pytorch/pytorch`, queried `/pytorch/pytorch/v2_8_0`
  for CUDA setup, inference, deterministic seeds and checkpoint APIs. Its output
  included the official v2.8.0 hub, modules and randomness documentation:
  https://github.com/pytorch/pytorch/blob/v2.8.0/docs/source/hub.md
  https://github.com/pytorch/pytorch/blob/v2.8.0/docs/source/notes/modules.rst
  https://github.com/pytorch/pytorch/blob/v2.8.0/docs/source/notes/randomness.rst
- Context7 DINOv2 lookup returned only RADIO, not an exact official match;
  therefore DINO APIs and preprocessing were checked in official source above.
- User-requested Bocha web search returned HTTP 403. The authorization credential
  is not stored in repository files. Built-in retrieval returned no usable text;
  authoritative documentation was fetched directly with HTTPS instead.
- Data version and attribution: M0 `reports/m0/SOURCE_AND_LICENSE.md`, official
  https://huggingface.co/datasets/rick77a/PhenoLeaf_TS and
  https://github.com/pisyntor/PhenoLeaf-TS . No new dataset download took place.
