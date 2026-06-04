#!/usr/bin/env python3
"""Time and sanity-check distilled EfficientSAM3 image encoders.

This script evaluates the three RepViT Stage 1 image encoder distillation
checkpoints with text, point, and box prompts. It uses the same fixed COCO-10
manifest as the EfficientSAM3 benchmark repo and writes all metrics/overlays
under RUN_ROOT/eval by default, never under the repository root.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
SAM3_SRC = REPO_ROOT / "sam3"
for path in (REPO_ROOT, SAM3_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


DEFAULT_RUN_ROOT = "/storage/scratch1/9/eliu354/efficientsam3_distill_smoke"
DEFAULT_BENCHMARK_ROOT = "/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/efficientsam3-benchmark"
COCO_VAL_IMAGE_URL = "http://images.cocodataset.org/val2017/{file_name}"

FIXED10_MANIFEST = REPO_ROOT / "data" / "manifests" / "coco_val2017_fixed10.jsonl"
FIXED10_PROMPTS = REPO_ROOT / "configs" / "datasets" / "coco_val2017_fixed10_prompts.json"

MODEL_SPECS = {
    "s": {
        "label": "ES-RV-S",
        "checkpoint": "efficient_sam3_repvit_s_smoke.pt",
        "backbone_type": "repvit",
        "model_name": "m0.9",
    },
    "m": {
        "label": "ES-RV-M",
        "checkpoint": "efficient_sam3_repvit_m_smoke.pt",
        "backbone_type": "repvit",
        "model_name": "m1.1",
    },
    "l": {
        "label": "ES-RV-L",
        "checkpoint": "efficient_sam3_repvit_l_smoke.pt",
        "backbone_type": "repvit",
        "model_name": "m2.3",
    },
}


@dataclass(frozen=True)
class PromptExample:
    sample_id: str
    image_path: Path
    image_id: int | str
    annotation_id: int | None
    box_xyxy: np.ndarray
    point: np.ndarray
    point_label: int
    text_prompt: str
    gt_mask: np.ndarray | None = None
    category: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ES-RV-S/M/L distilled image encoders with timing, IoU, and overlays."
    )
    parser.add_argument("--run-root", default=os.environ.get("RUN_ROOT", DEFAULT_RUN_ROOT))
    parser.add_argument("--checkpoint-dir", default=None, help="Defaults to RUN_ROOT/output.")
    parser.add_argument("--checkpoint-s", default=None, help="Override ES-RV-S checkpoint path.")
    parser.add_argument("--checkpoint-m", default=None, help="Override ES-RV-M checkpoint path.")
    parser.add_argument("--checkpoint-l", default=None, help="Override ES-RV-L checkpoint path.")
    parser.add_argument("--sizes", nargs="+", default=["s", "m", "l"], choices=sorted(MODEL_SPECS))
    parser.add_argument(
        "--prompt-modes",
        nargs="+",
        default=["text", "point", "box"],
        choices=["text", "point", "box"],
        help="Prompt modes to run for single image and COCO-10.",
    )
    parser.add_argument("--output-root", default=None, help="Defaults to RUN_ROOT/eval/image_encoder_distill.")
    parser.add_argument("--run-name", default=None, help="Output subdirectory name. Defaults to a timestamp.")
    parser.add_argument("--single-image", default=None, help="Single image path for timing and overlay.")
    parser.add_argument("--single-text", default="truck", help="Single-image text prompt.")
    parser.add_argument("--single-point", nargs=2, type=float, metavar=("X", "Y"), default=None)
    parser.add_argument(
        "--single-box",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=None,
        help="Single-image box prompt in pixel xyxy coordinates.",
    )
    parser.add_argument("--manifest", default=str(FIXED10_MANIFEST), help="Fixed COCO-10 JSONL manifest.")
    parser.add_argument("--prompt-config", default=str(FIXED10_PROMPTS), help="Fixed COCO-10 prompt metadata JSON.")
    parser.add_argument("--coco-root", default=None, help="Root containing only fixed10 images by default.")
    parser.add_argument("--benchmark-root", default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument(
        "--prepare-coco10",
        action="store_true",
        help="Copy fixed10 images from the benchmark repo, falling back to COCO URLs if missing.",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Prepare fixed COCO-10 images/manifests and exit.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--warmup", type=int, default=1, help="Warmup prompt predictions for single-image timing.")
    parser.add_argument("--multimask-output", action="store_true", help="Keep SAM-style multi-mask output.")
    parser.add_argument("--threshold", type=float, default=0.0, help="Mask threshold for interactive prompt masks.")
    parser.add_argument("--skip-coco", action="store_true", help="Only run the single-image pass.")
    parser.add_argument("--skip-single", action="store_true", help="Only run COCO-10.")
    return parser.parse_args()


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device: torch.device, fn):
    sync_if_needed(device)
    start = time.perf_counter()
    result = fn()
    sync_if_needed(device)
    return result, time.perf_counter() - start


def autocast_context(device: torch.device):
    if device.type in ("cuda", "mps"):
        from sam3.device import get_autocast_device_type, get_autocast_dtype

        return torch.autocast(get_autocast_device_type(device), dtype=get_autocast_dtype(device))
    return torch.inference_mode()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, dict[str, Path], Path, Path]:
    run_root = Path(args.run_root).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve() if args.checkpoint_dir else run_root / "output"
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else run_root / "eval" / "image_encoder_distill"
    )
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_root / run_name
    overrides = {"s": args.checkpoint_s, "m": args.checkpoint_m, "l": args.checkpoint_l}
    checkpoints = {
        size: Path(overrides[size]).expanduser().resolve()
        if overrides[size]
        else checkpoint_dir / spec["checkpoint"]
        for size, spec in MODEL_SPECS.items()
    }
    return run_root, checkpoints, output_root, out_dir


def load_model(size: str, checkpoint_path: Path, device: torch.device):
    from sam3.model_builder import build_efficientsam3_image_model

    spec = MODEL_SPECS[size]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"{spec['label']} checkpoint not found: {checkpoint_path}")
    model = build_efficientsam3_image_model(
        enable_inst_interactivity=True,
        checkpoint_path=str(checkpoint_path),
        load_from_HF=False,
        backbone_type=spec["backbone_type"],
        model_name=spec["model_name"],
        device=device,
    )
    model.eval()
    return model


def default_single_image() -> Path:
    return REPO_ROOT / "sam3" / "assets" / "images" / "truck.jpg"


def centered_box(image: Image.Image) -> np.ndarray:
    width, height = image.size
    return np.array([0.2 * width, 0.2 * height, 0.8 * width, 0.8 * height], dtype=np.float32)


def centered_point(image: Image.Image) -> np.ndarray:
    width, height = image.size
    return np.array([0.5 * width, 0.5 * height], dtype=np.float32)


def normalize_mask(mask: Any, threshold: float = 0.0) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask after squeeze, got shape {mask.shape}")
    if mask.dtype == np.bool_:
        return mask
    return mask > threshold


def image_mask_shape(image: Image.Image) -> tuple[int, int]:
    width, height = image.size
    return height, width


def empty_mask(image: Image.Image) -> np.ndarray:
    return np.zeros(image_mask_shape(image), dtype=bool)


def select_best_mask(
    masks: Any,
    scores: Any,
    threshold: float,
    empty_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, float]:
    if isinstance(scores, torch.Tensor):
        scores_np = scores.detach().float().cpu().numpy()
    else:
        scores_np = np.asarray(scores, dtype=np.float32)
    masks_np = masks.detach().cpu().numpy() if isinstance(masks, torch.Tensor) else np.asarray(masks)
    if masks_np.size == 0 or masks_np.shape[0] == 0:
        if empty_shape is not None:
            return np.zeros(empty_shape, dtype=bool), float("nan")
        raise RuntimeError("model returned no masks")
    best_idx = int(np.argmax(scores_np)) if scores_np.size else 0
    score = float(scores_np[best_idx]) if scores_np.size else float("nan")
    return normalize_mask(masks_np[best_idx], threshold), score


def predict_prompt(
    model,
    processor,
    image: Image.Image,
    prompt_mode: str,
    device: torch.device,
    threshold: float,
    multimask_output: bool,
    box_xyxy: np.ndarray | None = None,
    point: np.ndarray | None = None,
    point_label: int = 1,
    text_prompt: str | None = None,
) -> tuple[np.ndarray, float, dict[str, float]]:
    with torch.inference_mode(), autocast_context(device):
        state, set_image_sec = timed(device, lambda: processor.set_image(image))

        if prompt_mode == "box":
            if box_xyxy is None:
                box_xyxy = centered_box(image)

            def _predict_box():
                return model.predict_inst(
                    state,
                    point_coords=None,
                    point_labels=None,
                    box=box_xyxy.astype(np.float32)[None, :],
                    multimask_output=multimask_output,
                )

            pred, prompt_sec = timed(device, _predict_box)
            masks, scores, _ = pred
            mask, score = select_best_mask(masks, scores, threshold, image_mask_shape(image))
        elif prompt_mode == "point":
            if point is None:
                point = centered_point(image)

            def _predict_point():
                return model.predict_inst(
                    state,
                    point_coords=point.astype(np.float32)[None, :],
                    point_labels=np.array([point_label], dtype=np.int32),
                    multimask_output=multimask_output,
                )

            pred, prompt_sec = timed(device, _predict_point)
            masks, scores, _ = pred
            mask, score = select_best_mask(masks, scores, threshold, image_mask_shape(image))
        elif prompt_mode == "text":
            if not text_prompt:
                raise ValueError("text prompt mode requires text_prompt")

            def _predict_text():
                return processor.set_text_prompt(prompt=text_prompt, state=state)

            state, prompt_sec = timed(device, _predict_text)
            if "masks" not in state or len(state["masks"]) == 0:
                mask, score = empty_mask(image), float("nan")
            else:
                mask, score = select_best_mask(
                    state["masks"],
                    state.get("scores", []),
                    0.5,
                    image_mask_shape(image),
                )
        else:
            raise ValueError(f"unsupported prompt mode: {prompt_mode}")

    timing = {
        "set_image_sec": set_image_sec,
        "prompt_sec": prompt_sec,
        "total_sec": set_image_sec + prompt_sec,
    }
    return mask, score, timing


def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(intersection / union) if union else 0.0


def mask_to_rgba(mask: np.ndarray, color: tuple[int, int, int], alpha: int) -> Image.Image:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[..., :3] = color
    rgba[..., 3] = mask.astype(np.uint8) * alpha
    return Image.fromarray(rgba, mode="RGBA")


def save_overlay(
    image: Image.Image,
    pred_mask: np.ndarray,
    output_path: Path,
    box_xyxy: np.ndarray | None = None,
    point: np.ndarray | None = None,
    gt_mask: np.ndarray | None = None,
    title: str | None = None,
) -> None:
    base = image.convert("RGBA")
    if gt_mask is not None:
        base.alpha_composite(mask_to_rgba(gt_mask, (255, 80, 80), 95))
    base.alpha_composite(mask_to_rgba(pred_mask, (40, 160, 255), 120))

    draw = ImageDraw.Draw(base)
    if box_xyxy is not None:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        draw.rectangle([x1, y1, x2, y2], outline=(40, 255, 40, 255), width=3)
    if point is not None:
        x, y = [float(v) for v in point]
        r = 6
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 220, 0, 255), outline=(0, 0, 0, 255), width=2)
        draw.line([x - 12, y, x + 12, y], fill=(255, 220, 0, 255), width=2)
        draw.line([x, y - 12, x, y + 12], fill=(255, 220, 0, 255), width=2)
    if title:
        draw.rectangle([0, 0, min(base.width, 980), 30], fill=(0, 0, 0, 170))
        draw.text((8, 8), title, fill=(255, 255, 255, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(output_path)


def decode_manifest_mask(row: dict[str, Any]) -> np.ndarray:
    height = int(row["height"])
    width = int(row["width"])
    segmentation = row.get("segmentation")
    if isinstance(segmentation, list):
        mask_img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_img)
        for polygon in segmentation:
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            if len(points) >= 3:
                draw.polygon([tuple(point) for point in points], fill=1)
        return np.asarray(mask_img, dtype=bool)
    elif isinstance(segmentation, dict):
        from pycocotools import mask as mask_utils

        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        decoded = mask_utils.decode(rle)
    else:
        raise ValueError(f"unsupported segmentation for sample {row.get('sample_id')}")
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    return np.asarray(decoded, dtype=bool)


def read_manifest(manifest: Path) -> list[dict[str, Any]]:
    rows = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if len(rows) != 10:
        raise ValueError(f"fixed COCO manifest must contain exactly 10 rows, found {len(rows)}: {manifest}")
    return rows


def fixed_image_path(coco_root: Path, file_name: str) -> Path:
    return coco_root / "images" / "val2017" / file_name


def copy_or_download_fixed10(coco_root: Path, benchmark_root: Path, manifest: Path, prompt_config: Path) -> None:
    rows = read_manifest(manifest)
    image_dir = coco_root / "images" / "val2017"
    image_dir.mkdir(parents=True, exist_ok=True)
    (coco_root / "manifests").mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest, coco_root / "manifests" / manifest.name)
    if prompt_config.is_file():
        shutil.copy2(prompt_config, coco_root / "manifests" / prompt_config.name)

    benchmark_image_dir = benchmark_root / "data" / "coco" / "images" / "val2017"
    for row in rows:
        file_name = row["file_name"]
        dst = fixed_image_path(coco_root, file_name)
        if dst.is_file():
            continue
        src = benchmark_image_dir / file_name
        if src.is_file():
            shutil.copy2(src, dst)
        else:
            url = COCO_VAL_IMAGE_URL.format(file_name=file_name)
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, dst)


def load_fixed10_examples(coco_root: Path, manifest: Path) -> list[PromptExample]:
    examples = []
    for row in read_manifest(manifest):
        file_name = row["file_name"]
        image_path = fixed_image_path(coco_root, file_name)
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Missing fixed COCO-10 image: {image_path}. "
                "Run with --prepare-coco10 to copy/download the 10 images."
            )
        x, y, w, h = [float(v) for v in row["bbox_xywh"]]
        examples.append(
            PromptExample(
                sample_id=row["sample_id"],
                image_path=image_path,
                image_id=int(row["image_id"]),
                annotation_id=int(row["annotation_id"]),
                box_xyxy=np.array([x, y, x + w, y + h], dtype=np.float32),
                point=np.asarray(row["point"], dtype=np.float32),
                point_label=int(row.get("point_label", 1)),
                text_prompt=str(row.get("text_prompt") or row.get("category_name")),
                gt_mask=decode_manifest_mask(row),
                category=str(row.get("category_name")),
            )
        )
    return examples


def write_outputs(out_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def prompt_for_overlay(prompt_mode: str, example: PromptExample) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if prompt_mode == "box":
        return example.box_xyxy, None, "box"
    if prompt_mode == "point":
        return None, example.point, f"point={example.point.tolist()}"
    if prompt_mode == "text":
        return None, None, f"text={example.text_prompt}"
    raise ValueError(prompt_mode)


def summarize_prompt_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in rows:
        prompt_mode = row["prompt_mode"]
        item = out.setdefault(prompt_mode, {"ious": [], "times": []})
        if row.get("iou") != "":
            item["ious"].append(float(row["iou"]))
        item["times"].append(float(row["total_sec"]))
    for prompt_mode, item in out.items():
        ious = item.pop("ious")
        times = item.pop("times")
        item["num"] = len(times)
        item["mean_total_sec"] = float(np.mean(times)) if times else None
        item["mean_iou"] = float(np.mean(ious)) if ious else None
        item["median_iou"] = float(np.median(ious)) if ious else None
    return out


def run_single(
    model,
    processor,
    spec: dict[str, Any],
    size: str,
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    single_image_path = Path(args.single_image).expanduser().resolve() if args.single_image else default_single_image()
    image = Image.open(single_image_path).convert("RGB")
    box = np.asarray(args.single_box, dtype=np.float32) if args.single_box else centered_box(image)
    point = np.asarray(args.single_point, dtype=np.float32) if args.single_point else centered_point(image)
    rows = []
    for prompt_mode in args.prompt_modes:
        for _ in range(max(args.warmup, 0)):
            predict_prompt(
                model,
                processor,
                image,
                prompt_mode,
                device,
                args.threshold,
                args.multimask_output,
                box_xyxy=box,
                point=point,
                text_prompt=args.single_text,
            )
        pred_mask, score, timing = predict_prompt(
            model,
            processor,
            image,
            prompt_mode,
            device,
            args.threshold,
            args.multimask_output,
            box_xyxy=box,
            point=point,
            text_prompt=args.single_text,
        )
        overlay_path = out_dir / "single_image" / spec["label"].lower() / f"{prompt_mode}_overlay.jpg"
        save_overlay(
            image,
            pred_mask,
            overlay_path,
            box_xyxy=box if prompt_mode == "box" else None,
            point=point if prompt_mode == "point" else None,
            title=f"{spec['label']} {prompt_mode} score={score:.3f} total={timing['total_sec']:.3f}s",
        )
        rows.append(
            {
                "mode": "single",
                "prompt_mode": prompt_mode,
                "model_size": size,
                "model_label": spec["label"],
                "image": str(single_image_path),
                "text_prompt": args.single_text if prompt_mode == "text" else "",
                "point": point.tolist() if prompt_mode == "point" else "",
                "box_xyxy": box.tolist() if prompt_mode == "box" else "",
                "iou": "",
                "score": score,
                "overlay": str(overlay_path),
                **timing,
            }
        )
        print(f"single {prompt_mode}: total={timing['total_sec']:.3f}s")
    return rows


def run_coco10(
    model,
    processor,
    spec: dict[str, Any],
    size: str,
    examples: list[PromptExample],
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        image = Image.open(example.image_path).convert("RGB")
        for prompt_mode in args.prompt_modes:
            pred_mask, score, timing = predict_prompt(
                model,
                processor,
                image,
                prompt_mode,
                device,
                args.threshold,
                args.multimask_output,
                box_xyxy=example.box_xyxy,
                point=example.point,
                point_label=example.point_label,
                text_prompt=example.text_prompt,
            )
            iou = calculate_iou(pred_mask, example.gt_mask)
            overlay_box, overlay_point, prompt_desc = prompt_for_overlay(prompt_mode, example)
            overlay_path = (
                out_dir
                / "coco10"
                / spec["label"].lower()
                / prompt_mode
                / f"{example.sample_id}_overlay.jpg"
            )
            save_overlay(
                image,
                pred_mask,
                overlay_path,
                box_xyxy=overlay_box,
                point=overlay_point,
                gt_mask=example.gt_mask,
                title=f"{spec['label']} {prompt_desc} {example.category} IoU={iou:.3f}",
            )
            rows.append(
                {
                    "mode": "coco10",
                    "prompt_mode": prompt_mode,
                    "model_size": size,
                    "model_label": spec["label"],
                    "sample_id": example.sample_id,
                    "image_id": example.image_id,
                    "annotation_id": example.annotation_id,
                    "category": example.category,
                    "image": str(example.image_path),
                    "text_prompt": example.text_prompt if prompt_mode == "text" else "",
                    "point": example.point.tolist() if prompt_mode == "point" else "",
                    "box_xyxy": example.box_xyxy.tolist() if prompt_mode == "box" else "",
                    "iou": iou,
                    "score": score,
                    "overlay": str(overlay_path),
                    **timing,
                }
            )
        print(f"coco10 {spec['label']} {example.sample_id}: done")
    return rows


def main() -> int:
    args = parse_args()
    run_root, checkpoints, _, out_dir = resolve_paths(args)
    manifest = Path(args.manifest).expanduser().resolve()
    prompt_config = Path(args.prompt_config).expanduser().resolve()
    coco_root = Path(args.coco_root).expanduser().resolve() if args.coco_root else run_root / "data" / "coco_fixed10"
    benchmark_root = Path(args.benchmark_root).expanduser().resolve()

    if args.prepare_coco10 and not args.skip_coco:
        copy_or_download_fixed10(coco_root, benchmark_root, manifest, prompt_config)
        if args.prepare_only:
            print(f"Prepared fixed COCO-10 under: {coco_root}")
            return 0

    from sam3.device import get_device
    from sam3.model.sam3_image_processor import Sam3Processor

    device = torch.device(args.device) if args.device else get_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    examples = [] if args.skip_coco else load_fixed10_examples(coco_root, manifest)
    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "run_root": str(run_root),
        "output_dir": str(out_dir),
        "manifest": str(manifest),
        "prompt_config": str(prompt_config),
        "coco_root": str(coco_root),
        "prompt_modes": args.prompt_modes,
        "models": {},
    }

    for size in args.sizes:
        spec = MODEL_SPECS[size]
        checkpoint_path = checkpoints[size]
        print(f"\n=== {spec['label']} ({checkpoint_path}) ===")
        model, load_sec = timed(device, lambda: load_model(size, checkpoint_path, device))
        processor = Sam3Processor(model, device=device)
        model_rows: list[dict[str, Any]] = []

        if not args.skip_single:
            model_rows.extend(run_single(model, processor, spec, size, args, out_dir, device))
        if examples:
            model_rows.extend(run_coco10(model, processor, spec, size, examples, args, out_dir, device))

        all_rows.extend(model_rows)
        summary["models"][size] = {
            "label": spec["label"],
            "checkpoint": str(checkpoint_path),
            "load_sec": load_sec,
            "by_prompt": summarize_prompt_rows(model_rows),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_outputs(out_dir, all_rows, summary)
    print(f"\nWrote metrics and overlays to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
