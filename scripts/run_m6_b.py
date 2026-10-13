from __future__ import annotations

import argparse
import fcntl
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


def execute(args):
    if args.stage in {"snapshot", "freeze", "verify"}:
        from m6b.common import snapshot, freeze, verify
        print({"snapshot": snapshot, "freeze": freeze, "verify": verify}[args.stage](ROOT))
    elif args.stage == "smoke":
        from m6b.models import smoke
        smoke(ROOT, args.device)
    elif args.stage == "data":
        from m6b.data import prepare_data
        prepare_data(ROOT)
    elif args.stage == "training-cache":
        from m6b.data import prepare_training_cache
        prepare_training_cache(ROOT)
    elif args.stage == "train":
        from m6b.training import train
        train(ROOT, args.method, args.seed, args.device)
    elif args.stage == "infer":
        from m6b.evaluation import infer
        infer(ROOT, args.method, args.seed, args.device, args.splits)
    elif args.stage == "track":
        from m6b.tracking import track
        track(ROOT, args.method, args.seed, args.splits)
    elif args.stage == "report":
        from m6b.reporting import deliver
        deliver(ROOT)
    elif args.stage == "manifest":
        from m6b.reporting import delivery_manifest
        delivery_manifest(ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["snapshot", "smoke", "freeze", "data", "training-cache", "train", "infer", "track", "report", "verify", "manifest"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--method", choices=["S1", "S2", "S3"], default="S2")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val"])
    args = parser.parse_args()
    if args.stage in {"train", "infer", "track"}:
        directory = ROOT / f"outputs/m6_b/{args.method}/seed_{args.seed}"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "execution.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            execute(args)
    else:
        execute(args)


if __name__ == "__main__":
    main()
