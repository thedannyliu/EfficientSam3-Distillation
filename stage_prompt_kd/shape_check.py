from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


TINYVIT_21M_CHANNELS = 576


def tinyvit_final_resolution(img_size: int, patch_size: int = 4, stages: int = 4) -> int:
    size = img_size // patch_size
    for _ in range(stages - 1):
        size = (size - 1) // 2 + 1
    return size


def expected_tinyvit21_shapes(
    *,
    batch_size: int,
    img_size: int,
    embed_dim: int,
    embed_size: int,
) -> dict[str, list[int]]:
    raw_size = tinyvit_final_resolution(img_size)
    return {
        "tinyvit_raw": [batch_size, TINYVIT_21M_CHANNELS, raw_size, raw_size],
        "student_projected": [batch_size, embed_dim, embed_size, embed_size],
        "sam3_teacher_trunk": [batch_size, embed_dim, embed_size, embed_size],
    }


def run_student_forward(args: argparse.Namespace) -> dict[str, list[int]]:
    from stage1_geometry_finetune.model import StudentTrunk

    device = torch.device(args.device)
    model = StudentTrunk(
        backbone_name=args.backbone,
        embed_dim=args.embed_dim,
        embed_size=args.embed_size,
        img_size=args.img_size,
    ).to(device)
    model.eval()
    x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
    with torch.no_grad():
        raw = model.backbone(x)
        projected = model(x)
    return {
        "tinyvit_raw_forward": list(raw.shape),
        "student_projected_forward": list(projected.shape),
    }


def run_teacher_forward(args: argparse.Namespace) -> dict[str, list[int]]:
    from stage1.model import SAM3ImageTeacherEncoder

    device = torch.device(args.device)
    teacher = SAM3ImageTeacherEncoder(
        checkpoint_path=args.sam3_checkpoint,
        embed_size=args.embed_size,
    ).to(device)
    teacher.eval()
    x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
    with torch.no_grad():
        teacher_features = teacher(x)
    return {"sam3_teacher_trunk_forward": list(teacher_features.shape)}


def main() -> None:
    parser = argparse.ArgumentParser("TinyViT-21M to SAM3 shape audit")
    parser.add_argument("--backbone", default="tiny_vit_21m")
    parser.add_argument("--img-size", type=int, default=1008)
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--embed-size", type=int, default=72)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sam3-checkpoint", default="")
    parser.add_argument("--run-forward", action="store_true")
    parser.add_argument("--run-teacher", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    report: dict = {
        "backbone": args.backbone,
        "img_size": args.img_size,
        "embed_dim": args.embed_dim,
        "embed_size": args.embed_size,
        "expected_shapes": expected_tinyvit21_shapes(
            batch_size=args.batch_size,
            img_size=args.img_size,
            embed_dim=args.embed_dim,
            embed_size=args.embed_size,
        ),
        "notes": [
            "TinyViT-21M raw output is 576 channels at 32x32 for 1008 input.",
            "The current student head projects 576->1024 and interpolates 32x32->72x72.",
            "SAM3 teacher trunk target is 1024 channels at 72x72 for 1008 input.",
        ],
    }

    if args.run_forward:
        report["forward_shapes"] = run_student_forward(args)
    if args.run_teacher:
        if not args.sam3_checkpoint:
            raise ValueError("--run-teacher requires --sam3-checkpoint")
        report.setdefault("forward_shapes", {}).update(run_teacher_forward(args))

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
