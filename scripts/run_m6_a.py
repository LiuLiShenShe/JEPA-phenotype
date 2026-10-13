from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m6a.common import freeze, verify_protection
from m6a.experiment import evaluate, prepare, smoke


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "smoke", "prepare", "evaluate", "report", "verify", "all"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["val", "test"])
    parser.add_argument("--wait-for-cache", action="store_true")
    args = parser.parse_args()
    if args.stage in ["freeze", "all"]:
        freeze(ROOT)
    if args.stage in ["smoke", "all"]:
        smoke(ROOT, args.device)
    if args.stage in ["prepare", "all"]:
        prepare(ROOT, args.device, tuple(args.splits))
    if args.stage in ["evaluate", "all"]:
        evaluate(ROOT, tuple(args.splits), args.wait_for_cache)
    if args.stage in ["report", "all"]:
        from m6a.reporting import deliver

        deliver(ROOT)
    if args.stage in ["verify", "all"]:
        print(verify_protection(ROOT))


if __name__ == "__main__":
    main()
