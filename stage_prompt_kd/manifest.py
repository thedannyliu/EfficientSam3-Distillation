from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def point_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    idx = len(xs) // 2
    return [int(xs[idx]), int(ys[idx])]


def build_prompt_record(
    *,
    image: str,
    source_id: str,
    text: str | None = None,
    point: list[int] | None = None,
    box_xyxy: list[int] | None = None,
    mask: str | None = None,
) -> dict:
    return {
        "image": image,
        "source_id": source_id,
        "text": text,
        "point": point,
        "box_xyxy": box_xyxy,
        "mask": mask,
    }


def write_jsonl(records: Iterable[dict], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    return output_path


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
