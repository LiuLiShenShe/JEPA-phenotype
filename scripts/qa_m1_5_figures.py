#!/usr/bin/env python3
"""Rendered-figure inspection through the user-requested vision API."""

import base64
import json
import os
from pathlib import Path
import sys

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.common import file_hash, write_json


def main():
    report = ROOT / "reports/m1_5"
    key = os.environ.get("AGNES_API_KEY")
    if not key:
        raise RuntimeError("Set AGNES_API_KEY to run the explicitly requested vision API")
    records = []
    prompt = (
        "仅对这张科研统计图做客观视觉QA，必须返回JSON，不要Markdown围栏："
        '{"description":"简述各面板、坐标、图例、文字", "issues":["仅列确实看见的重叠、截断、缺失点、不可读标注或歧义"], '
        '"has_blocking_visual_issue":false}。'
        "不要推测图外的数据或统计有效性；图中n是独立植物，点和区间来自真实CSV。"
    )
    for name in ("capacity_control", "time_conditions", "calendar_feasibility", "difficulty_day5"):
        path = report / "figures" / f"{name}.png"
        encoded = base64.b64encode(path.read_bytes()).decode()
        payload = {"model": "agnes-2.5-flash", "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}}
        ]}]}
        record = {"figure": name, "png_sha256": file_hash(path)}
        try:
            response = requests.post("https://apihub.agnes-ai.cn/v1/chat/completions",
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                                     json=payload, timeout=180)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            text = content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            record.update({"status": "inspected", "provider": "Agnes API", "response": json.loads(text)})
        except Exception as error:
            record.update({"status": "unavailable", "error": f"{type(error).__name__}: {str(error)}"})
        records.append(record)
        print(json.dumps({key: value for key, value in record.items() if key != "response"}, ensure_ascii=False), flush=True)
    write_json(report / "figure_visual_qa.json", {"provider": "Agnes API", "records": records})
    sections = ["# M1.5 渲染图视觉 QA", "", "图片内容由 Agnes API 识别；不使用模型猜测替代未成功识别。实际数值有效性由独立CSV/自动测试核验。", ""]
    for record in records:
        sections += ["## " + record["figure"], "", "状态：" + record["status"], ""]
        if record["status"] == "inspected":
            sections += [record["response"]["description"], "", "问题：" + json.dumps(record["response"]["issues"], ensure_ascii=False),
                         "", "阻塞视觉问题：" + str(record["response"]["has_blocking_visual_issue"]), ""]
        else:
            sections += ["识图 API 调用失败，无法获取图片内容；没有声称视觉QA通过。", record["error"], ""]
    (report / "figure_visual_qa.md").write_text("\n".join(sections))


if __name__ == "__main__":
    main()
