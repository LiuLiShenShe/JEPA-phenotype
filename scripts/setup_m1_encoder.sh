#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
commit=7764ea0f912e53c92e82eb78a2a1631e92725fc8
weights_sha256=b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9
mkdir -p outputs/m1/pretrained reports/m1/docs
if [ ! -e third_party/dinov2/hubconf.py ]; then
  mkdir -p third_party/dinov2
  archive=$(mktemp)
  trap 'rm -f "$archive"' EXIT
  curl --fail -L --retry 2 "https://codeload.github.com/facebookresearch/dinov2/tar.gz/$commit" -o "$archive"
  tar -xzf "$archive" --strip-components=1 -C third_party/dinov2
fi
printf '%s\n' "$commit" > reports/m1/docs/dinov2_commit.txt
if [ ! -e outputs/m1/pretrained/dinov2_vits14_pretrain.pth ]; then
  curl --fail -L --retry 2 https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth -o outputs/m1/pretrained/dinov2_vits14_pretrain.pth
fi
printf '%s  %s\n' "$weights_sha256" outputs/m1/pretrained/dinov2_vits14_pretrain.pth | sha256sum --check
