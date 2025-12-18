from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    class_weights: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError(f"Expected logits (B,C,H,W), got {tuple(logits.shape)}")
    if target.ndim != 3:
        raise ValueError(f"Expected target (B,H,W), got {tuple(target.shape)}")

    probs = F.softmax(logits, dim=1)

    if ignore_index is not None:
        valid = target != ignore_index
        target = target.clone()
        target[~valid] = 0
    else:
        valid = torch.ones_like(target, dtype=torch.bool)

    target_oh = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid = valid.unsqueeze(1).float()

    probs = probs * valid
    target_oh = target_oh * valid

    dims = (0, 2, 3)
    intersection = (probs * target_oh).sum(dims)
    denom = (probs + target_oh).sum(dims)
    dice = (2.0 * intersection + eps) / (denom + eps)

    if class_weights is not None:
        dice = dice * class_weights.to(dice.device)
        loss = 1.0 - dice.sum() / (class_weights.sum() + eps)
    else:
        loss = 1.0 - dice.mean()
    return loss


@dataclass
class SegLossConfig:
    num_classes: int = 3
    ce_weight: float = 1.0
    dice_weight: float = 1.0
    ignore_index: Optional[int] = None
    class_weights: Optional[torch.Tensor] = None


class CombinedSegLoss(nn.Module):
    def __init__(self, cfg: SegLossConfig):
        super().__init__()
        self.cfg = cfg
        ignore_idx = cfg.ignore_index if cfg.ignore_index is not None else -100
        self.ce = nn.CrossEntropyLoss(
            weight=cfg.class_weights, ignore_index=ignore_idx
        )
        self.ignore_index = cfg.ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.cfg.ce_weight:
            loss = loss + self.cfg.ce_weight * self.ce(logits, target)
        if self.cfg.dice_weight:
            loss = loss + self.cfg.dice_weight * soft_dice_loss(
                logits,
                target,
                num_classes=self.cfg.num_classes,
                ignore_index=self.ignore_index,
                class_weights=self.cfg.class_weights,
            )
        return loss


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1.0 - prob) * (1.0 - targets)
        loss = bce * ((1.0 - pt) ** self.gamma)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
