from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from stage_prompt_kd.checkpointing import (
    load_training_checkpoint,
    resolve_latest_checkpoint,
    resolve_wandb_run_id,
    save_training_checkpoint,
)
from stage_prompt_kd.losses import feature_mse_loss


def init_wandb(args: argparse.Namespace, output_dir: Path):
    if not args.use_wandb:
        return None, None
    import wandb

    run_id = resolve_wandb_run_id(
        output_dir,
        args.wandb_run_id,
        generate_id=wandb.util.generate_id,
    )
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        id=run_id,
        resume=args.wandb_resume,
        dir=str(output_dir / "wandb"),
        config=vars(args),
    )
    return run, run_id


def main() -> None:
    parser = argparse.ArgumentParser("Prompt-KD resumable TinyViT-21M smoke trainer")
    parser.add_argument("--output", default="/storage/scratch1/9/eliu354/efficientsam3_prompt_kd/tinyvit21_sam3")
    parser.add_argument("--backbone", default="tiny_vit_21m")
    parser.add_argument("--img-size", type=int, default=1008)
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--embed-size", type=int, default=72)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default="")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="efficientsam3-prompt-kd")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-run-id", default=os.environ.get("WANDB_RUN_ID", ""))
    parser.add_argument("--wandb-resume", default="allow", choices=["allow", "must", "never", "auto"])
    args = parser.parse_args()

    from stage1_geometry_finetune.model import StudentTrunk

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    device = torch.device(args.device)
    model = StudentTrunk(
        backbone_name=args.backbone,
        embed_dim=args.embed_dim,
        embed_size=args.embed_size,
        img_size=args.img_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs * args.steps_per_epoch),
    )

    resume_path = Path(args.resume) if args.resume else None
    if args.auto_resume and resume_path is None:
        resume_path = resolve_latest_checkpoint(output_dir)

    start_epoch = 0
    global_step = 0
    wandb_run_id = None
    if resume_path:
        checkpoint = load_training_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        wandb_run_id = checkpoint.get("wandb_run_id")

    if wandb_run_id and not args.wandb_run_id:
        args.wandb_run_id = wandb_run_id
    wandb_run, wandb_run_id = init_wandb(args, output_dir)

    model.train()
    start = time.time()
    for epoch in range(start_epoch, args.epochs):
        for _ in range(args.steps_per_epoch):
            images = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
            with torch.no_grad():
                teacher = torch.randn(
                    args.batch_size,
                    args.embed_dim,
                    args.embed_size,
                    args.embed_size,
                    device=device,
                )

            pred = model(images)
            loss = feature_mse_loss(pred, teacher)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            metrics = {
                "loss/feature_mse": float(loss.detach().cpu()),
                "train/global_step": global_step,
                "train/lr": optimizer.param_groups[0]["lr"],
            }
            if wandb_run is not None:
                wandb_run.log(metrics, step=global_step)

        latest = save_training_checkpoint(
            output_dir,
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            wandb_run_id=wandb_run_id,
            extra={"elapsed_sec": time.time() - start},
        )
        print(f"saved {latest}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
