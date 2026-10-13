from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/m6_b"
LOCAL = ROOT / "data/raw/m6_b/official"


def main():
    commit = "02fc608bea4c0549b0a7b00ca1bf15dee4a0b228"
    url = f"https://codeload.github.com/NVIDIA/MinkowskiEngine/zip/{commit}"
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    archive = LOCAL / "minkowski.zip"
    archive.write_bytes(response.content)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if not (LOCAL / member.filename).resolve().is_relative_to(LOCAL.resolve()):
                raise ValueError("Unsafe archive")
        bundle.extractall(LOCAL)
    interpreter = str(ROOT / ".venv-m6-b-organ3d/bin/python")
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "PYTHONDONTWRITEBYTECODE": "1", "MAX_JOBS": "2"}
    command = [interpreter, "setup.py", "build_ext", "--inplace", "--force_cuda", "--blas=openblas"]
    with (REPORT / "organ3d_minkowski_build.log").open("w") as stream:
        try:
            result = subprocess.run(command, cwd=LOCAL / ("MinkowskiEngine-" + commit), env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=180)
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            exit_code = "timeout after 180 seconds; bounded dependency build"
    metadata = json.loads((REPORT / "sources/organ3dnet_checkpoint_zenodo.json").read_text())
    file = metadata["files"][0]
    checkpoint = LOCAL / "organ3dnet_last-epoch.ckpt"
    if not checkpoint.exists():
        with requests.get(file["links"]["self"], timeout=(30, 120), stream=True) as response:
            response.raise_for_status()
            with checkpoint.open("wb") as stream:
                for block in response.iter_content(8 * 1024 * 1024):
                    stream.write(block)
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            md5.update(block)
            sha.update(block)
    if checkpoint.stat().st_size != file["size"] or "md5:" + md5.hexdigest() != file["checksum"]:
        raise ValueError("Official checkpoint download checksum mismatch")
    with (REPORT / "organ3d_official_model_import.log").open("w") as stream:
        imported = subprocess.run([interpreter, "-c", "from models.Organ3DNet import Organ3DNet; print('OFFICIAL_IMPORT_SUCCESS')"],
                                  cwd=LOCAL / "Organ3DNet-06f711fc3d9884933c61766c009b2bcd86837139", env=environment,
                                  stdout=stream, stderr=subprocess.STDOUT, timeout=60)
    result = {"method": "S4 Organ3DNet", "status": "BLOCKED", "official_commit": "06f711fc3d9884933c61766c009b2bcd86837139",
              "minkowski_commit": commit, "python": "3.10", "torch": "1.12.1+cu113", "installed_torch_cuda": "11.3",
              "available_nvcc": "12.6", "pointnet2_extension_build_exit": 1, "minkowski_build_exit": exit_code,
              "official_model_import_exit": imported.returncode, "checkpoint": {"source": file["links"]["self"],
              "path": str(checkpoint.relative_to(ROOT)), "bytes": checkpoint.stat().st_size, "md5": md5.hexdigest(), "sha256": sha.hexdigest(),
              "zenodo_record_license": metadata["metadata"].get("license"), "strict_load": False},
              "official_example_smoke": "not completed: native dependency build prevents official model import",
              "target_supervised_steps": 0, "simplified_replacement": False,
              "policy": "No alternate architecture. Keep sources and weights local; checkpoint record CC-BY-4.0 does not establish repository code license. Stop bounded port attempt."}
    (REPORT / "organ3d_feasibility.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
