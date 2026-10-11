#!/usr/bin/env python3
"""Download PhenoLeaf-TS through the official Hugging Face helper.

The script never edits an existing extracted dataset.  It records every
download/extraction failure in ``reports/m0/download_report.json`` so a
network-blocked environment cannot be mistaken for a completed M0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO_COMMIT = "0d9a24ff146d12cc8fca6bdfa2538d196754f456"
HF_MIRROR_URL = "https://hf-mirror.net/datasets/rick77a/PhenoLeaf_TS/resolve/main/PhenoLeaf-TS_DS.zip"
EXPECTED_ARCHIVE_BYTES = 5_486_793_580
EXPECTED_ARCHIVE_SHA256 = "608c7b07125281fb92517182c0873d1a9c87d696e2adedcb11c002173887de2f"
HF_DATASET_COMMIT = "ee4c8dac44593a69c8a72e1ca09d686133161d92"
sys.path.insert(0, str(PROJECT_ROOT / "third_party/PhenoLeaf-TS"))


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def find_dataset_root(path: Path) -> Path | None:
    if (path / "01_raw_dataset").is_dir() and (path / "02_leaf_labels").is_dir():
        return path
    for name in ("leaf_dataset_colour", "PhenoLeaf-TS_DS"):
        candidate = path / name
        if (candidate / "01_raw_dataset").is_dir() and (candidate / "02_leaf_labels").is_dir():
            return candidate
    return None


def safe_extract(archive_path: Path, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted_files = 0
    with zipfile.ZipFile(archive_path) as archive:
        destination_resolved = destination.resolve()
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise ValueError(f"Unsafe archive member path: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size != member.file_size:
                    raise ValueError(f"Existing raw file differs from archive; not overwriting: {target}")
                checksum = 0
                with target.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        checksum = zlib.crc32(chunk, checksum)
                if checksum & 0xFFFFFFFF != member.CRC:
                    raise ValueError(f"Existing raw file differs from archive; not overwriting: {target}")
                continue
            temporary = target.with_name(target.name + ".extracting")
            try:
                with archive.open(member) as source, temporary.open("wb") as sink:
                    while chunk := source.read(1024 * 1024):
                        sink.write(chunk)
                os.link(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            extracted_files += 1
    return {"extracted_files": extracted_files}


def download_from_mirror(archive_path: Path, url: str) -> dict[str, Any]:
    """Download the public archive from an explicitly selected mirror.

    This is opt-in.  Range resume is used so an interrupted multi-gigabyte
    transfer can continue without restarting from zero.
    """

    import requests

    existing_bytes = archive_path.stat().st_size if archive_path.exists() else 0
    headers = {"Range": f"bytes={existing_bytes}-"} if existing_bytes else {}
    response = requests.get(url, headers=headers, stream=True, timeout=(30, 120), allow_redirects=True)
    response.raise_for_status()
    append = existing_bytes > 0 and response.status_code == 206
    if append and not response.headers.get("Content-Range", "").startswith(f"bytes {existing_bytes}-"):
        response.close()
        raise ValueError("Mirror returned an unexpected byte range")
    if existing_bytes > 0 and not append:
        existing_bytes = 0
    mode = "ab" if append else "wb"
    written = existing_bytes
    with archive_path.open(mode) as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
                written += len(chunk)
    response.close()
    return {
        "download_url": url,
        "http_status": response.status_code,
        "resumed": append,
        "bytes_written": written,
        "etag": response.headers.get("ETag", ""),
        "content_length": response.headers.get("Content-Length", ""),
        "content_range": response.headers.get("Content-Range", ""),
    }


def mirror_headers(url: str) -> dict[str, str]:
    try:
        import requests

        response = requests.head(url, timeout=(15, 30), allow_redirects=True)
        return {
            "status_code": str(response.status_code),
            "etag": response.headers.get("ETag", ""),
            "content_length": response.headers.get("Content-Length", ""),
            "content_type": response.headers.get("Content-Type", ""),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def verify_hub_metadata() -> dict[str, Any]:
    import requests

    url = f"https://hf-mirror.com/api/datasets/rick77a/PhenoLeaf_TS/tree/{HF_DATASET_COMMIT}"
    response = requests.get(url, timeout=(15, 30))
    response.raise_for_status()
    archive = next(item for item in response.json() if item["path"] == "PhenoLeaf-TS_DS.zip")
    if archive["lfs"]["oid"] != EXPECTED_ARCHIVE_SHA256 or archive["size"] != EXPECTED_ARCHIVE_BYTES:
        raise ValueError("Pinned Hugging Face LFS metadata changed unexpectedly")
    return {"metadata_url": url, "dataset_commit": HF_DATASET_COMMIT, "archive": archive}


def download_data(
    data_root: str | Path,
    report_path: str | Path,
    *,
    no_extract: bool = False,
    source: str = "auto",
    mirror_url: str = HF_MIRROR_URL,
) -> dict[str, Any]:
    data_root = Path(data_root)
    report_path = Path(report_path)
    data_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "archive_name": "PhenoLeaf-TS_DS.zip",
        "official_repository": "https://github.com/pisyntor/PhenoLeaf-TS",
        "official_repository_commit": OFFICIAL_REPO_COMMIT,
        "dataset_source": "https://huggingface.co/datasets/rick77a/PhenoLeaf_TS",
        "download_method": "phenoleaf_ts.data.download_phenoleaf",
        "source_mode": source,
        "dataset_commit": HF_DATASET_COMMIT,
        "expected_archive_sha256": EXPECTED_ARCHIVE_SHA256,
    }
    if source == "hf-mirror":
        report["download_method"] = "explicit mirror fallback with resumable HTTP download"
        report["download_url"] = mirror_url
        report["mirror_headers_before_download"] = mirror_headers(mirror_url)
    archive_path = data_root / "PhenoLeaf-TS_DS.zip"
    extracted_root = data_root / "leaf_dataset_colour"
    try:
        if archive_path.exists() and archive_path.stat().st_size == EXPECTED_ARCHIVE_BYTES:
            report["archive_reused"] = True
        else:
            if source == "hf-mirror":
                report.update(download_from_mirror(archive_path, mirror_url))
            else:
                from phenoleaf_ts.data import download_phenoleaf

                try:
                    downloaded = Path(download_phenoleaf(str(data_root), extract=False))
                    archive_path = downloaded
                except Exception as exc:
                    if source == "official":
                        raise
                    report["official_attempt_error"] = f"{type(exc).__name__}: {exc}"
                    report["download_method"] = "official helper failed; SHA-256-verified mirror fallback"
                    report["source_mode"] = "hf-mirror"
                    report.update(download_from_mirror(archive_path, mirror_url))
            report["archive_reused"] = False
        if not archive_path.exists():
            raise FileNotFoundError(f"Official helper returned missing archive: {archive_path}")
        report["archive_path"] = str(archive_path)
        report["archive_bytes"] = archive_path.stat().st_size
        if source == "hf-mirror":
            report["mirror_expected_archive_bytes"] = EXPECTED_ARCHIVE_BYTES
            if archive_path.stat().st_size != EXPECTED_ARCHIVE_BYTES:
                raise ValueError(
                    f"Mirror archive size mismatch: {archive_path.stat().st_size} != {EXPECTED_ARCHIVE_BYTES}"
                )
        report["archive_sha256"] = sha256(archive_path)
        report["sha256_matches_hf_lfs"] = report["archive_sha256"] == EXPECTED_ARCHIVE_SHA256
        if not report["sha256_matches_hf_lfs"]:
            raise ValueError("Archive SHA-256 does not match the pinned Hugging Face LFS object")
        try:
            report["hub_metadata_verification"] = verify_hub_metadata()
        except Exception as exc:
            report["hub_metadata_verification"] = {"status": "unavailable", "error": str(exc)}
        with zipfile.ZipFile(archive_path) as archive:
            bad_member = archive.testzip()
            report["archive_testzip"] = "passed" if bad_member is None else f"failed:{bad_member}"
            if bad_member is not None:
                raise ValueError(f"Archive CRC check failed at {bad_member}")
        if not no_extract:
            report.update(safe_extract(archive_path, data_root))
        dataset_root = find_dataset_root(data_root)
        report["extracted_root"] = str(dataset_root or extracted_root)
        report["layout_verified"] = dataset_root is not None
        if not no_extract and not report["layout_verified"]:
            raise ValueError("Archive extracted but documented leaf_dataset_colour layout was not found")
        report["status"] = "ok"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["next_action"] = "Retry from a network with official Hugging Face access; do not use an unverified mirror."
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--report", default="reports/m0/download_report.json")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument(
        "--source",
        choices=("auto", "official", "hf-mirror"),
        default="auto",
        help="Use the official helper by default; hf-mirror is an explicit fallback recorded in the report.",
    )
    parser.add_argument("--mirror-url", default=HF_MIRROR_URL)
    args = parser.parse_args()
    report = download_data(
        args.data_root,
        args.report,
        no_extract=args.no_extract,
        source=args.source,
        mirror_url=args.mirror_url,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
