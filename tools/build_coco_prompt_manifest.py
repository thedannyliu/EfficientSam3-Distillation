#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build a COCO prompt manifest for EfficientSAM3 eval")
    parser.add_argument("--coco-root", required=True, help="COCO root with images/ and annotations/.")
    parser.add_argument("--split", default="val2017", choices=["val2017", "test2017"])
    parser.add_argument("--output", required=True, help="Output JSONL manifest.")
    parser.add_argument("--max-images", type=int, default=-1, help="Limit images for smoke manifests.")
    parser.add_argument("--min-area", type=float, default=1.0)
    return parser.parse_args()


def image_path(coco_root: Path, split: str, file_name: str) -> str:
    candidates = [
        coco_root / "images" / split / file_name,
        coco_root / split / file_name,
        coco_root / "images" / file_name,
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return str(candidates[0])


def mask_centroid(mask: np.ndarray, bbox_xywh: list[float]) -> list[float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        x, y, w, h = bbox_xywh
        return [float(x + 0.5 * w), float(y + 0.5 * h)]
    return [float(xs.mean()), float(ys.mean())]


def build_val_rows(coco_root: Path, split: str, max_images: int, min_area: float) -> list[dict[str, Any]]:
    from pycocotools.coco import COCO

    ann_file = coco_root / "annotations" / f"instances_{split}.json"
    if not ann_file.is_file():
        raise FileNotFoundError(f"Missing COCO annotation file: {ann_file}")
    coco = COCO(str(ann_file))
    cats = coco.loadCats(coco.getCatIds())
    cat_names = {cat["id"]: cat["name"] for cat in cats}

    rows: list[dict[str, Any]] = []
    for image_id in coco.getImgIds():
        image = coco.loadImgs(image_id)[0]
        ann_ids = coco.getAnnIds(imgIds=image_id, iscrowd=False)
        anns = [
            ann for ann in coco.loadAnns(ann_ids)
            if float(ann.get("area", 0.0)) >= min_area and ann.get("bbox")
        ]
        if not anns:
            continue
        ann = max(anns, key=lambda item: float(item.get("area", 0.0)))
        x, y, w, h = [float(v) for v in ann["bbox"]]
        mask = coco.annToMask(ann)
        category = cat_names.get(ann["category_id"], str(ann["category_id"]))
        rows.append({
            "sample_id": f"coco_{split}_{image_id}_{ann['id']}",
            "split": split,
            "image_id": int(image_id),
            "annotation_id": int(ann["id"]),
            "category_id": int(ann["category_id"]),
            "category_name": category,
            "text_prompt": category,
            "file_name": image["file_name"],
            "image_path": image_path(coco_root, split, image["file_name"]),
            "height": int(image["height"]),
            "width": int(image["width"]),
            "bbox_xywh": [x, y, w, h],
            "box_xyxy": [x, y, x + w, y + h],
            "point": mask_centroid(mask, [x, y, w, h]),
            "point_label": 1,
        })
        if max_images > 0 and len(rows) >= max_images:
            break
    return rows


def build_test_rows(coco_root: Path, split: str, max_images: int) -> list[dict[str, Any]]:
    image_dir = coco_root / "images" / split
    if not image_dir.is_dir():
        image_dir = coco_root / split
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing COCO test image directory under: {coco_root}")
    rows = []
    for path in sorted(image_dir.glob("*.jpg")):
        rows.append({
            "sample_id": f"coco_{split}_{path.stem}",
            "split": split,
            "image_id": path.stem,
            "file_name": path.name,
            "image_path": str(path),
            "text_prompt": "",
            "box_xyxy": None,
            "point": None,
            "point_label": 1,
        })
        if max_images > 0 and len(rows) >= max_images:
            break
    return rows


def main() -> int:
    args = parse_args()
    coco_root = Path(args.coco_root).expanduser().resolve()
    rows = (
        build_val_rows(coco_root, args.split, args.max_images, args.min_area)
        if args.split == "val2017"
        else build_test_rows(coco_root, args.split, args.max_images)
    )
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    meta = output.with_suffix(output.suffix + ".meta.json")
    meta.write_text(json.dumps({
        "coco_root": str(coco_root),
        "split": args.split,
        "num_rows": len(rows),
        "selection": "largest_non_crowd_object_per_image" if args.split == "val2017" else "image_scaffold",
        "text_prompt": "COCO category name" if args.split == "val2017" else "requires external prompts",
    }, indent=2) + "\n")
    print(f"Wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
