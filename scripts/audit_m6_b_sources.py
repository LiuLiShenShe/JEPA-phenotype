from __future__ import annotations

import base64
import datetime
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from m6a.common import digest, write_json


def main():
    destination = ROOT / "reports/m6_b/sources"
    local = ROOT / "data/raw/m6_b/official"
    destination.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    evidence = []
    resources = {
        "sugar4d_page": "https://www.ipb.uni-bonn.de/data/sugar4d/index.html",
        "sugar4d_metadata": "https://bonndata.uni-bonn.de/api/datasets/:persistentId/?persistentId=doi:10.60507/FK2/IS8YBZ",
        "organ3dnet_checkpoint_reference": "https://raw.githubusercontent.com/Huang2002200/Organ3DNet/06f711fc3d9884933c61766c009b2bcd86837139/checkpoint/checkpoint_file.txt",
        "organ3dnet_paper": "https://doi.org/10.1016/j.aiia.2025.10.011",
    }
    for name, url in resources.items():
        try:
            response = session.get(url, timeout=60)
            path = destination / (name + (".json" if name.endswith("metadata") else ".txt"))
            path.write_text(response.text)
            evidence.append({"name": name, "url": url, "http": response.status_code,
                             "final_url": response.url, "bytes": len(response.content), "sha256": digest(path)})
            print(name, response.status_code, len(response.content), flush=True)
        except requests.RequestException as error:
            evidence.append({"name": name, "url": url, "error": str(error)})
    commit = "06f711fc3d9884933c61766c009b2bcd86837139"
    url = f"https://codeload.github.com/Huang2002200/Organ3DNet/zip/{commit}"
    response = session.get(url, timeout=120)
    response.raise_for_status()
    archive = local / "organ3dnet.zip"
    archive.write_bytes(response.content)
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            target = (local / entry.filename).resolve()
            if not target.is_relative_to(local.resolve()):
                raise ValueError("Archive traversal")
        bundle.extractall(local)
    evidence.append({"name": "organ3dnet_archive", "url": url, "http": response.status_code,
                     "bytes": len(response.content), "sha256": digest(archive), "commit": commit})
    files = []
    for path in (local / f"Organ3DNet-{commit}").rglob("*"):
        if path.is_file():
            files.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path)})
    for path in (ROOT / "data/raw/m6_a/official/plantnet").rglob("*"):
        if path.is_file():
            files.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path)})
    write_json(destination / "retrieval_audit.json", {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "http": evidence, "files": files, "licenses": "no declared repository code/weight license; do not redistribute"})
    print("Downloaded official source; hashes recorded. No upstream code redistributed.")


if __name__ == "__main__":
    main()
