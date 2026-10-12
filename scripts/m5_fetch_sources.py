#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from m4.common import write_json


SOURCES = {
    "crops_repository": "https://api.github.com/repos/clawCa/Crops3D",
    "crops_revision": "https://api.github.com/repos/clawCa/Crops3D/commits/main",
    "crops_readme": "https://raw.githubusercontent.com/clawCa/Crops3D/main/README.md",
    "crops_metadata": "https://api.figshare.com/v2/articles/27313272",
    "crops_article": "https://www.nature.com/articles/s41597-024-04290-0",
    "beet_metadata": "https://bonndata.uni-bonn.de/api/datasets/:persistentId/?persistentId=doi:10.60507/FK2/34W30T",
    "beet_readme": "https://bonndata.uni-bonn.de/api/access/datafile/4278",
    "beet_article": "https://arxiv.org/html/2312.14706",
}


def fetch(root):
    folder = Path(root) / "reports/m5/sources"
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    session = requests.Session()
    for name, url in SOURCES.items():
        destination = folder / (name + (".json" if name.endswith(("metadata", "repository", "revision")) else ".txt"))
        if destination.exists():
            continue
        for attempt in range(1, 4):
            try:
                response = session.get(url, timeout=90)
                record = {"name": name, "url": url, "status": response.status_code,
                          "final_url": response.url, "attempt": attempt, "bytes": len(response.content)}
                if response.ok:
                    if destination.suffix == ".json":
                        write_json(destination, response.json())
                    else:
                        destination.write_bytes(response.content)
                    records.append(record)
                    print(json.dumps(record), flush=True)
                    break
                record["error"] = response.text[:200]
            except Exception as error:
                record = {"name": name, "url": url, "attempt": attempt, "error": str(error)}
            records.append(record)
            print(json.dumps(record), flush=True)
            time.sleep(2)
    write_json(folder / "fetch_status.json", records)
    write_json(folder / "bocha_status.json", {"status": "UNAVAILABLE", "http_status": 403,
               "reason": "You do not have enough money or package quota", "date": "2026-10-12",
               "fallback": "direct retrieval of user-provided official source URLs; not training-memory source verification"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    arguments = parser.parse_args()
    fetch(arguments.root)
