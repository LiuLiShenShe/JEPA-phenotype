import datetime
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import requests
import torch

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "reports/m7/sources"

if __name__ == "__main__":
    DESTINATION.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, url in [("sugar4d_official", "https://www.ipb.uni-bonn.de/data/sugar4d/index.html"),
                      ("sugar4d_dataverse", "https://bonndata.uni-bonn.de/api/datasets/:persistentId/?persistentId=doi:10.60507/FK2/IS8YBZ")]:
        try:
            response = requests.get(url, timeout=60)
            (DESTINATION / (name + (".json" if "dataverse" in name else ".html"))).write_bytes(response.content)
            rows.append({"name": name, "url": url, "HTTP": response.status_code, "bytes": len(response.content),
                         "sha256": hashlib.sha256(response.content).hexdigest()})
        except requests.RequestException as error:
            rows.append({"name": name, "url": url, "error": str(error)})
    (DESTINATION / "retrieval_audit.json").write_text(json.dumps({"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "Bocha": {"HTTP": 403, "message": "You do not have enough money or package quota", "general_search_available": False}, "HTTP": rows}, indent=2))
    environment = {"python": subprocess.check_output([str(ROOT / ".venv-m6-b/bin/python"), "--version"], text=True).strip(),
                   "historical_environment_modified": False, "reuse": ".venv-m6-b read-only; PYTHONDONTWRITEBYTECODE=1", "torch_cuda": torch.version.cuda,
                   "packages": {name: importlib.metadata.version(name) for name in ["torch", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "pytest"]},
                   "GPU": subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader"], text=True)}
    (ROOT / "reports/m7/environment.json").write_text(json.dumps(environment, indent=2))
    print(json.dumps(rows, indent=2))
