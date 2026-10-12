#!/usr/bin/env python
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import time
from pathlib import Path

import requests


def download(url, destination, total, workers, record_path):
    destination = Path(destination)
    prefix = destination.stat().st_size if destination.exists() else 0
    if prefix >= total:
        raise ValueError("Range downloader requires a genuine partial prefix")
    with destination.open("ab") as stream:
        stream.truncate(total)
    segment = math.ceil((total - prefix) / workers)
    intervals = [(index, prefix + index * segment, min(total - 1, prefix + (index + 1) * segment - 1))
                 for index in range(workers) if prefix + index * segment < total]

    def transfer(interval):
        index, start, end = interval
        offset = start
        for attempt in range(1, 4):
            try:
                with requests.get(url, headers={"Range": f"bytes={offset}-{end}"}, stream=True, timeout=(30, 180)) as response:
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {offset}-{end}/{total}":
                        raise ValueError("Server did not honour exact byte range")
                    with destination.open("r+b") as output:
                        output.seek(offset)
                        for block in response.iter_content(1024 * 1024):
                            output.write(block)
                            offset += len(block)
                if offset != end + 1:
                    raise ValueError("Truncated byte range")
                print(f"RANGE {index} completed {end - start + 1} bytes", flush=True)
                return {"index": index, "start": start, "end": end, "status": "COMPLETED", "attempts": attempt}
            except Exception as error:
                print(f"RANGE {index} attempt={attempt} offset={offset} error={error}", flush=True)
                if attempt == 3:
                    raise
                time.sleep(5)

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(transfer, intervals))
    Path(record_path).write_text(json.dumps({"url": url, "bytes": total, "preserved_prefix": prefix,
                                           "seconds": time.monotonic() - started, "ranges": records}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--bytes", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--record", required=True)
    arguments = parser.parse_args()
    download(arguments.url, arguments.destination, arguments.bytes, arguments.workers, arguments.record)
