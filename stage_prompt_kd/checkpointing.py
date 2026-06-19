from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def resolve_latest_checkpoint(output_dir: str | Path) -> Path | None:
    latest = Path(output_dir) / "checkpoints" / "latest.pt"
    return latest if latest.exists() else None


def save_training_checkpoint(
    output_dir: str | Path,
    *,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    wandb_run_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "wandb_run_id": wandb_run_id,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if extra:
        payload.update(extra)

    epoch_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(payload, epoch_path)
    torch.save(payload, latest_path)
    return latest_path


def load_training_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def resolve_wandb_run_id(
    output_dir: str | Path,
    requested_run_id: str | None,
    *,
    generate_id,
) -> str:
    run_id_path = Path(output_dir) / "wandb_run_id.txt"
    if requested_run_id:
        run_id_path.parent.mkdir(parents=True, exist_ok=True)
        run_id_path.write_text(requested_run_id + "\n", encoding="utf-8")
        return requested_run_id
    if run_id_path.exists():
        return run_id_path.read_text(encoding="utf-8").strip()

    run_id = generate_id()
    run_id_path.parent.mkdir(parents=True, exist_ok=True)
    run_id_path.write_text(run_id + "\n", encoding="utf-8")
    return run_id
