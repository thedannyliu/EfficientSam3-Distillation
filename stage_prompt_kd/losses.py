from __future__ import annotations

import torch
import torch.nn.functional as F


def feature_mse_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = F.mse_loss(student_features, teacher_features, reduction="none")
    if valid_mask is None:
        return loss.mean()
    return (loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)


def dice_loss(
    pred_logits: torch.Tensor,
    target_masks: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred = pred_logits.sigmoid().flatten(1)
    target = target_masks.float().flatten(1)
    intersection = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    return (1.0 - (2.0 * intersection + eps) / (denom + eps)).mean()


def mask_bce_dice_loss(
    pred_logits: torch.Tensor,
    target_masks: torch.Tensor,
    *,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(pred_logits, target_masks.float())
    return bce_weight * bce + dice_weight * dice_loss(pred_logits, target_masks)


def box_l1_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = F.l1_loss(pred_boxes, target_boxes, reduction="none")
    if valid_mask is None:
        return loss.mean()
    while valid_mask.ndim < loss.ndim:
        valid_mask = valid_mask.unsqueeze(-1)
    return (loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
