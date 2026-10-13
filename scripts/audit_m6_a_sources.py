from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from m6a.common import digest, write_json


REPOSITORIES = {
    "pepper4d": "foysalahmed10/Pepper-4D",
    "plantnet": "Huang2002200/PlantNet-and-PSegNet",
    "trackplant3d": "entarot/TrackPlant3D-3D-organ-growth-tracking-framework-for-organ-level-dynamic-phenotyping",
}
PINNED_REVISIONS = {"plantnet": "5efe9ced49473f0d59c618b7e140ffeeb692d9b1",
                    "trackplant3d": "3c04919dc86d75acd1a3159a1e1f74893e0bc5e4"}


def collect(download=False, search=False):
    destination = ROOT / "reports/m6_a/sources"
    destination.mkdir(parents=True, exist_ok=True)
    responses = []
    for name, repository in REPOSITORIES.items():
        for endpoint, suffix in [("", "metadata.json"), ("/readme", "README.md"), ("/git/trees/HEAD?recursive=1", "tree.json"), ("/license", "license_response.json")]:
            url = "https://api.github.com/repos/" + repository + endpoint
            try:
                response = requests.get(url, headers={"Accept": "application/vnd.github.raw+json"} if suffix.endswith(".md") else {}, timeout=40)
                (destination / f"{name}_{suffix}").write_bytes(response.content)
                responses.append({"project": name, "url": url, "http": response.status_code, "bytes": len(response.content)})
            except Exception as error:
                responses.append({"project": name, "url": url, "error": repr(error)})
        if download and name in PINNED_REVISIONS:
            commit = PINNED_REVISIONS[name]
            tree_response = requests.get("https://api.github.com/repos/" + repository + "/git/trees/" + commit + "?recursive=1", timeout=40)
            tree_response.raise_for_status()
            tree = tree_response.json()
            write_json(destination / f"{name}_pinned_tree.json", tree)
            for item in tree.get("tree", []):
                path = item["path"]
                selected = item["type"] == "blob" and ((name == "plantnet" and "_pytorch/" in path) or
                                                       (name == "trackplant3d" and (path.endswith(".py") or path == "requirements.txt")))
                if not selected or "__pycache__" in path:
                    continue
                target = ROOT / "data/raw/m6_a/official" / name / path
                url = f"https://raw.githubusercontent.com/{repository}/{commit}/{path}"
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and digest(target) != hashlib.sha256(response.content).hexdigest():
                    raise ValueError("Pinned source differs from existing file; refusing overwrite: " + path)
                target.write_bytes(response.content)
                responses.append({"project": name, "path": str(target.relative_to(ROOT)), "url": url,
                                  "http": response.status_code, "bytes": len(response.content), "sha256": digest(target)})
    attempts = []
    for subset, resource in [(1, "1BYQbJAQ2jogPlnH5Y-nNrnuZz1ZdOumB"), (2, "16Pg3GRM-4XEM3jLrwn1CgnuZHAaJng9o"), (3, "1LViEisq8OkjMJyOSXFBs5qji5-q_iJSL")]:
        url = "https://drive.google.com/uc?export=download&id=" + resource
        try:
            response = requests.get(url, stream=True, timeout=25)
            prefix = next(response.iter_content(4096), b"")
            attempts.append({"subset": subset, "url": url, "http": response.status_code, "content_type": response.headers.get("Content-Type"),
                             "prefix": prefix[:300].decode(errors="replace"), "dataset_downloaded": False})
            response.close()
        except Exception as error:
            attempts.append({"subset": subset, "url": url, "error": repr(error), "dataset_downloaded": False})
    write_json(destination / "pepper_download_attempts.json", attempts)
    if search:
        key = os.environ.get("BOCHA_API_KEY")
        searches = []
        for repository in REPOSITORIES.values():
            query = repository + " official plant phenotyping dataset license pretrained model"
            if not key:
                searches.append({"query": query, "error": "BOCHA_API_KEY unset; no secret is stored in the repository"})
                continue
            response = requests.post("https://api.bocha.cn/v1/web-search", headers={"Authorization": "Bearer " + key},
                                     json={"query": query, "summary": True, "count": 50}, timeout=40)
            searches.append({"query": query, "http": response.status_code, "response": response.json()})
        write_json(destination / "bocha_search.json", searches)
    write_json(destination / "http_audit.json", {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "requests": responses})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--search", action="store_true")
    arguments = parser.parse_args()
    collect(arguments.download, arguments.search)
