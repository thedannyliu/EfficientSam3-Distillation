#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SAM3_SRC = REPO_ROOT / "sam3"
for path in (REPO_ROOT, SAM3_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


MODEL_SPECS = {
    "rv_s": ("ES-RV-S", "repvit", "m0.9", "efficient_sam3_repvit_s.pt"),
    "rv_m": ("ES-RV-M", "repvit", "m1.1", "efficient_sam3_repvit_m.pt"),
    "rv_l": ("ES-RV-L", "repvit", "m2.3", "efficient_sam3_repvit_l.pt"),
    "tv_s": ("ES-TV-S", "tinyvit", "5m", "efficient_sam3_tinyvit_s.pt"),
    "tv_m": ("ES-TV-M", "tinyvit", "11m", "efficient_sam3_tinyvit_m.pt"),
    "tv_l": ("ES-TV-L", "tinyvit", "21m", "efficient_sam3_tinyvit_l.pt"),
    "ev_s": ("ES-EV-S", "efficientvit", "b0", "efficient_sam3_efficientvit_s.pt"),
    "ev_m": ("ES-EV-M", "efficientvit", "b1", "efficient_sam3_efficientvit_m.pt"),
    "ev_l": ("ES-EV-L", "efficientvit", "b2", "efficient_sam3_efficientvit_l.pt"),
    "vit_s": ("ES-VIT-S", "vit", "tiny", "efficient_sam3_vit_s.pt"),
    "vit_m": ("ES-VIT-M", "vit", "small", "efficient_sam3_vit_m.pt"),
    "vit_l": ("ES-VIT-L", "vit", "base", "efficient_sam3_vit_l.pt"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate EfficientSAM3 checkpoints on COCO point/box/text prompts")
    parser.add_argument("--checkpoint-dir", default="../efficientsam3_distill_runs/output")
    parser.add_argument("--checkpoint-suffix", default="", help="Inserted before .pt, e.g. _e2e_ft.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--coco-root", default=None, help="Needed for val IoU via COCO annotations.")
    parser.add_argument("--split", default="val2017", choices=["val2017", "test2017"])
    parser.add_argument("--output-dir", default="../efficientsam3_distill_runs/eval/coco_prompts")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--models", nargs="+", default=sorted(MODEL_SPECS), choices=sorted(MODEL_SPECS))
    parser.add_argument("--prompt-modes", nargs="+", default=["point", "box", "text"], choices=["point", "box", "text"])
    parser.add_argument("--max-rows", type=int, default=-1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--text-threshold", type=float, default=0.5)
    parser.add_argument("--multimask-output", action="store_true")
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device: torch.device, fn):
    sync(device)
    start = time.perf_counter()
    result = fn()
    sync(device)
    return result, time.perf_counter() - start


def autocast_context(device: torch.device):
    if device.type in ("cuda", "mps"):
        from sam3.device import get_autocast_device_type, get_autocast_dtype
        return torch.autocast(get_autocast_device_type(device), dtype=get_autocast_dtype(device))
    return torch.inference_mode()


def load_rows(path: Path, max_rows: int) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if max_rows > 0 and len(rows) >= max_rows:
                    break
    return rows


def checkpoint_name(default_name: str, suffix: str) -> str:
    if not suffix:
        return default_name
    path = Path(default_name)
    return path.with_name(path.stem + suffix + path.suffix).name


def load_model(spec_key: str, ckpt_dir: Path, suffix: str, device: torch.device):
    from sam3.model_builder import build_efficientsam3_image_model
    label, backbone, model_name, default_ckpt = MODEL_SPECS[spec_key]
    checkpoint = ckpt_dir / checkpoint_name(default_ckpt, suffix)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing {label} checkpoint: {checkpoint}")
    model = build_efficientsam3_image_model(
        enable_inst_interactivity=True,
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        backbone_type=backbone,
        model_name=model_name,
        device=device,
    )
    model.eval()
    return model, checkpoint


def select_mask(masks: Any, scores: Any, threshold: float, image_shape: tuple[int, int]) -> tuple[np.ndarray, float]:
    masks_np = masks.detach().cpu().numpy() if isinstance(masks, torch.Tensor) else np.asarray(masks)
    scores_np = scores.detach().float().cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
    if masks_np.size == 0 or masks_np.shape[0] == 0:
        return np.zeros(image_shape, dtype=bool), float("nan")
    best_idx = int(np.argmax(scores_np)) if scores_np.size else 0
    mask = np.squeeze(masks_np[best_idx])
    return mask > threshold, float(scores_np[best_idx]) if scores_np.size else float("nan")


def predict(model, processor, image: Image.Image, row: dict[str, Any], mode: str, device: torch.device, args):
    h, w = image.height, image.width
    with torch.inference_mode(), autocast_context(device):
        state, set_image_sec = timed(device, lambda: processor.set_image(image))
        if mode == "box":
            box = row.get("box_xyxy")
            if not box:
                return None
            pred, prompt_sec = timed(
                device,
                lambda: model.predict_inst(
                    state,
                    point_coords=None,
                    point_labels=None,
                    box=np.asarray(box, dtype=np.float32)[None, :],
                    multimask_output=args.multimask_output,
                ),
            )
            masks, scores, _ = pred
            mask, score = select_mask(masks, scores, args.threshold, (h, w))
        elif mode == "point":
            point = row.get("point")
            if not point:
                return None
            pred, prompt_sec = timed(
                device,
                lambda: model.predict_inst(
                    state,
                    point_coords=np.asarray(point, dtype=np.float32)[None, :],
                    point_labels=np.array([int(row.get("point_label", 1))], dtype=np.int32),
                    multimask_output=args.multimask_output,
                ),
            )
            masks, scores, _ = pred
            mask, score = select_mask(masks, scores, args.threshold, (h, w))
        elif mode == "text":
            prompt = str(row.get("text_prompt") or "").strip()
            if not prompt:
                return None
            state, prompt_sec = timed(device, lambda: processor.set_text_prompt(prompt=prompt, state=state))
            if "masks" not in state or len(state["masks"]) == 0:
                mask, score = np.zeros((h, w), dtype=bool), float("nan")
            else:
                mask, score = select_mask(state["masks"], state.get("scores", []), args.text_threshold, (h, w))
        else:
            raise ValueError(mode)
    return mask, score, {"set_image_sec": set_image_sec, "prompt_sec": prompt_sec, "total_sec": set_image_sec + prompt_sec}


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in rows:
        key = f"{row['model_key']}:{row['prompt_mode']}"
        item = out.setdefault(key, {"num": 0, "ious": [], "times": []})
        item["num"] += 1
        if row.get("iou") != "":
            item["ious"].append(float(row["iou"]))
        item["times"].append(float(row["total_sec"]))
    for item in out.values():
        ious = item.pop("ious")
        times = item.pop("times")
        item["mean_iou"] = float(np.mean(ious)) if ious else None
        item["median_iou"] = float(np.median(ious)) if ious else None
        item["mean_total_sec"] = float(np.mean(times)) if times else None
    return out


def main() -> int:
    args = parse_args()
    from sam3.device import get_device
    from sam3.model.sam3_image_processor import Sam3Processor

    device = torch.device(args.device) if args.device else get_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    manifest = Path(args.manifest).expanduser().resolve()
    rows = load_rows(manifest, args.max_rows)
    coco = None
    if args.split == "val2017" and args.coco_root:
        from pycocotools.coco import COCO

        ann_file = Path(args.coco_root).expanduser().resolve() / "annotations" / f"instances_{args.split}.json"
        coco = COCO(str(ann_file)) if ann_file.is_file() else None

    out_dir = Path(args.output_dir).expanduser() / (args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve()
    metric_rows: list[dict[str, Any]] = []

    for model_key in args.models:
        label, _, _, _ = MODEL_SPECS[model_key]
        print(f"Loading {label}")
        model, checkpoint = load_model(model_key, ckpt_dir, args.checkpoint_suffix, device)
        processor = Sam3Processor(model, device=device)
        for row in rows:
            image = Image.open(row["image_path"]).convert("RGB")
            gt_mask = None
            if coco is not None and row.get("annotation_id") is not None:
                gt_mask = coco.annToMask(coco.loadAnns([int(row["annotation_id"])])[0]).astype(bool)
            for mode in args.prompt_modes:
                result = predict(model, processor, image, row, mode, device, args)
                if result is None:
                    continue
                pred_mask, score, timing = result
                metric_rows.append({
                    "model_key": model_key,
                    "model_label": label,
                    "checkpoint": str(checkpoint),
                    "prompt_mode": mode,
                    "sample_id": row.get("sample_id", ""),
                    "image_id": row.get("image_id", ""),
                    "annotation_id": row.get("annotation_id", ""),
                    "category": row.get("category_name", ""),
                    "text_prompt": row.get("text_prompt", "") if mode == "text" else "",
                    "iou": iou(pred_mask, gt_mask) if gt_mask is not None else "",
                    "score": score,
                    **timing,
                })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "manifest": str(manifest),
        "checkpoint_dir": str(ckpt_dir),
        "checkpoint_suffix": args.checkpoint_suffix,
        "split": args.split,
        "num_manifest_rows": len(rows),
        "by_model_prompt": summarize(metric_rows),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if metric_rows:
        fieldnames = sorted({key for row in metric_rows for key in row})
        with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metric_rows)
    print(f"Wrote eval output to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
