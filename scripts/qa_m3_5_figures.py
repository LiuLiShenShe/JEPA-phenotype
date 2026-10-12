#!/usr/bin/env python
"""User-authorized external vision QA of the six local quantitative figures."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
from pathlib import Path

import requests


def inspect_image(path, output):
    encoded = base64.b64encode(path.read_bytes()).decode()
    prompt = ("这是科研定量图，请只进行可视化质量审查，不推测不存在的数据或重新判断科研结论。"
              "详细客观描述标题、坐标、图例和布局；判断是否有文字/曲线被裁剪、标签重叠、图例遮挡重要数据、单位缺失、"
              "字号不可读等问题。清楚区分确定问题与不确定问题。最终给出PASS或REVIEW以及具体理由。")
    payload = {"model": "agnes-2.5-flash", "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}]}]}
    try:
        response = requests.post("https://apihub.agnes-ai.cn/v1/chat/completions",
                                 headers={"Authorization": f"Bearer {os.environ['AGNES_API_KEY']}", "Content-Type": "application/json"},
                                 json=payload, timeout=180)
        data = response.json()
        if response.status_code != 200:
            result = {"image": path.name, "status": "API_FAILED", "http_status": response.status_code, "body": data}
        else:
            result = {"image": path.name, "status": "API_RECOGNIZED", "attribution": "图片内容由 Agnes API 识别", "response": data}
    except Exception as error:
        result = {"image": path.name, "status": "API_FAILED", "error": f"{type(error).__name__}: {error}"}
    (output / f"{path.stem}_agnes_qa.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"{path.name}: {result['status']}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.root)
    output = root / "reports/m3_5/visual_qa"
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted((root / "reports/m3_5/figures").glob("*.png"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda path: inspect_image(path, output), paths))
    (root / "reports/m3_5/figure_visual_qa.json").write_text(json.dumps({"attribution": "图片内容由 Agnes API 识别", "images": results}, ensure_ascii=False, indent=2) + "\n")
