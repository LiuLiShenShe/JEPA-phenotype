#!/usr/bin/env python3
"""Rerun M0 tests, archive/label audit, local figures and final acceptance."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    report_dir = PROJECT_ROOT / "reports/m0"
    report_dir.mkdir(parents=True, exist_ok=True)
    execution = {"status": "RUNNING", "started_at_utc": datetime.now(timezone.utc).isoformat(), "stages": []}
    (report_dir / "acceptance_run.json").write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
    commands = [
        [sys.executable, "-m", "pytest", "-q", "--junitxml=reports/m0/test_results.xml"],
        [sys.executable, "scripts/check_reproducibility.py"],
        [sys.executable, "scripts/audit_dataset.py"],
        [sys.executable, "scripts/visualize_m0.py"],
        [sys.executable, "scripts/summarize_m0.py"],
        [sys.executable, "scripts/validate_m0.py"],
    ]
    for command in commands:
        print("+ " + " ".join(command), flush=True)
        started = time.monotonic()
        result = subprocess.run(command, cwd=PROJECT_ROOT)
        execution["stages"].append({"command": command, "returncode": result.returncode, "elapsed_seconds": round(time.monotonic() - started, 3)})
        (report_dir / "acceptance_run.json").write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
        if result.returncode:
            execution["status"] = "FAIL"
            execution["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            (report_dir / "acceptance_run.json").write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
            (report_dir / "validation_results.json").write_text(json.dumps({"status": "FAIL", "failed_stage": execution["stages"][-1]}, indent=2) + "\n", encoding="utf-8")
            if (report_dir / "M0_REPORT.md").exists():
                report = (report_dir / "M0_REPORT.md").read_text(encoding="utf-8")
                report = report.replace("**M0 状态：PASS**", "**M0 状态：FAIL**（最新验收失败，先前结果仅作历史参考）")
                (report_dir / "M0_REPORT.md").write_text(report, encoding="utf-8")
            return result.returncode
    execution["status"] = "PASS"
    execution["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (report_dir / "acceptance_run.json").write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
