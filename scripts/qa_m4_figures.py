#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import requests

from m4.common import write_json


def inspect(root, api_key, files=None):
    root = Path(root)
    images = sorted((root / "reports/m4/figures").glob("*.png"))
    records = []
    if files:
        images = [path for path in images if path.name in files]
        previous = root / "reports/m4/figure_visual_qa.json"
        if previous.exists():
            records = [record for record in json.loads(previous.read_text())["images"] if record["file"] not in files]
    session = requests.Session()
    for path in images:
        image = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
        prompt = ("客观检查这张科研结果图。请提取标题、轴名、图例、关键数值/趋势，并判断有无标题/图例/轴标签重叠、"
                  "文字截断、坐标/误差条误导。注意它是实际实验结果，禁止猜测未显示的数据。"
                  "语义分割三维点图请描述GT和Prediction的可见不同，不声称实例分割质量。最后给出视觉质量问题清单。")
        try:
            response = session.post("https://apihub.agnes-ai.cn/v1/chat/completions",
                                    headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
                                    json={"model": "agnes-2.5-flash", "messages": [{"role": "user", "content": [
                                        {"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image}}]}]}, timeout=120)
            response.raise_for_status()
            description = response.json()["choices"][0]["message"]["content"]
            records.append({"file": path.name, "status": "API_RECOGNIZED", "description": description})
            print(path.name, "API_RECOGNIZED", flush=True)
        except (requests.RequestException, KeyError, IndexError, ValueError) as error:
            records.append({"file": path.name, "status": "FAILED", "error": type(error).__name__ + ": " + str(error)})
            print(path.name, "识图 API 调用失败，无法获取图片内容", flush=True)
        write_json(root / "reports/m4/figure_visual_qa.json", {"image_content_source": "图片内容由 Agnes API 识别",
                                                              "images": records, "credentials_saved": False})
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--files", default=None)
    arguments = parser.parse_args()
    key = os.environ.get("AGNES_API_KEY")
    if not key:
        raise SystemExit("Set AGNES_API_KEY locally; never store API credentials in repository")
    inspect(Path(arguments.root), key, arguments.files.split(",") if arguments.files else None)
