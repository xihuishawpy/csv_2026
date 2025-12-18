from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice_coef(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = float(np.logical_and(pred, gt).sum())
    denom = float(pred.sum() + gt.sum())
    if denom == 0.0:
        return 1.0
    return (2.0 * inter + eps) / (denom + eps)


def _surface(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    er = binary_erosion(mask, iterations=1, border_value=0)
    return np.logical_xor(mask, er)


def nsd(pred: np.ndarray, gt: np.ndarray, tolerance: float = 2.0, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0

    s_pred = _surface(pred)
    s_gt = _surface(gt)
    if s_pred.sum() == 0 and s_gt.sum() == 0:
        return 1.0

    dt_gt = distance_transform_edt(~s_gt)
    dt_pred = distance_transform_edt(~s_pred)

    close_pred = float(np.logical_and(s_pred, dt_gt <= tolerance).sum())
    close_gt = float(np.logical_and(s_gt, dt_pred <= tolerance).sum())
    denom = float(s_pred.sum() + s_gt.sum())
    return (close_pred + close_gt + eps) / (denom + eps)


@dataclass
class SegMetrics:
    dice_vessel: float
    dice_plaque: float
    nsd_vessel: float
    nsd_plaque: float


def compute_seg_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    vessel_id: int = 1,
    plaque_id: int = 2,
    tolerance: float = 2.0,
) -> SegMetrics:
    dice_vessel = dice_coef(pred == vessel_id, gt == vessel_id)
    dice_plaque = dice_coef(pred == plaque_id, gt == plaque_id)
    nsd_vessel = nsd(pred == vessel_id, gt == vessel_id, tolerance=tolerance)
    nsd_plaque = nsd(pred == plaque_id, gt == plaque_id, tolerance=tolerance)
    return SegMetrics(dice_vessel, dice_plaque, nsd_vessel, nsd_plaque)


def aggregate_seg_scores(long_m: SegMetrics, trans_m: SegMetrics) -> Dict[str, float]:
    def view_score(m: SegMetrics) -> float:
        s_vessel = 0.5 * (m.dice_vessel + m.nsd_vessel)
        s_plaque = 0.5 * (m.dice_plaque + m.nsd_plaque)
        return 0.4 * s_vessel + 0.6 * s_plaque

    s_long = view_score(long_m)
    s_trans = view_score(trans_m)
    s_seg = 0.5 * (s_long + s_trans)
    return {"S_long": s_long, "S_trans": s_trans, "S_seg": s_seg}

