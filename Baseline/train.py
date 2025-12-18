from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Baseline.unet import UNet, UNetConfig
from Baseline.utils.eval_utils import aggregate_seg_scores, compute_seg_metrics
from Baseline.utils.losses import BinaryFocalLoss, CombinedSegLoss, SegLossConfig
from Baseline.utils.ramps import sigmoid_rampup
from Baseline.utils.util import (
    CSV2026LabeledDataset,
    CSV2026UnlabeledDataset,
    EMA,
    get_device,
    list_case_ids,
    make_teacher,
    maybe_set_float32_matmul_precision,
    save_checkpoint,
    set_seed,
    split_labeled_ids,
)


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def _make_pseudo_labels(
    logits: torch.Tensor, thresh: float, ignore_index: int = 255
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=1)
    conf, pseudo = probs.max(dim=1)
    pseudo = pseudo.long()
    valid = conf >= thresh
    pseudo = pseudo.clone()
    pseudo[~valid] = ignore_index
    return pseudo, valid


def train_one_epoch(
    *,
    student: UNet,
    teacher: UNet,
    ema: EMA,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    seg_loss_sup: CombinedSegLoss,
    cls_loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
    unsup_weight_max: float,
    unsup_rampup: int,
    pseudo_thresh: float,
    amp: bool,
) -> Dict[str, float]:
    student.train()
    teacher.eval()

    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    unlabeled_iter = itertools.cycle(unlabeled_loader)

    loss_meter = {"loss": 0.0, "sup": 0.0, "unsup": 0.0}
    n = 0

    pbar = tqdm(labeled_loader, desc=f"Train {epoch+1}/{epochs}", leave=False)
    for labeled_batch in pbar:
        unlabeled_batch = next(unlabeled_iter)
        labeled_batch = _to_device(labeled_batch, device)
        unlabeled_batch = _to_device(unlabeled_batch, device)

        long_img = labeled_batch["long_img"]
        trans_img = labeled_batch["trans_img"]
        long_mask = labeled_batch["long_mask"]
        trans_mask = labeled_batch["trans_mask"]
        cls_target = labeled_batch["cls"]

        long_w = unlabeled_batch["long_w"]
        trans_w = unlabeled_batch["trans_w"]
        long_s = unlabeled_batch["long_s"]
        trans_s = unlabeled_batch["trans_s"]

        unsup_w = unsup_weight_max * sigmoid_rampup(epoch, unsup_rampup)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            sup_long_logits, sup_trans_logits, sup_cls_logits = student.forward_pair(
                long_img, trans_img
            )
            sup_seg = 0.5 * (
                seg_loss_sup(sup_long_logits, long_mask)
                + seg_loss_sup(sup_trans_logits, trans_mask)
            )
            sup_cls = cls_loss_fn(sup_cls_logits, cls_target)
            sup_loss = sup_seg + sup_cls

            with torch.no_grad():
                t_long_logits, t_trans_logits, t_cls_logits = teacher.forward_pair(long_w, trans_w)
                pseudo_long, _ = _make_pseudo_labels(t_long_logits, pseudo_thresh)
                pseudo_trans, _ = _make_pseudo_labels(t_trans_logits, pseudo_thresh)
                t_prob = torch.sigmoid(t_cls_logits)
                cls_pseudo = (t_prob >= 0.5).float()
                cls_conf = (t_prob >= 0.7) | (t_prob <= 0.3)

            u_long_logits, u_trans_logits, u_cls_logits = student.forward_pair(long_s, trans_s)
            unsup_seg = 0.5 * (
                F.cross_entropy(u_long_logits, pseudo_long, ignore_index=255)
                + F.cross_entropy(u_trans_logits, pseudo_trans, ignore_index=255)
            )
            if cls_conf.any():
                unsup_cls = cls_loss_fn(u_cls_logits[cls_conf], cls_pseudo[cls_conf])
            else:
                unsup_cls = torch.zeros((), device=device)
            unsup_loss = unsup_seg + unsup_cls

            loss = sup_loss + unsup_w * unsup_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        ema.update(student, teacher)

        loss_meter["loss"] += float(loss.detach().cpu())
        loss_meter["sup"] += float(sup_loss.detach().cpu())
        loss_meter["unsup"] += float(unsup_loss.detach().cpu())
        n += 1
        pbar.set_postfix(loss=loss_meter["loss"] / max(1, n), unsup_w=unsup_w)

    return {k: v / max(1, n) for k, v in loss_meter.items()}


@torch.no_grad()
def evaluate(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
    nsd_tolerance: float,
) -> Dict[str, float]:
    model.eval()
    y_true = []
    y_pred = []
    seg_scores = []

    for batch in tqdm(loader, desc="Val", leave=False):
        batch = _to_device(batch, device)
        long_img = batch["long_img"]
        trans_img = batch["trans_img"]
        long_mask = batch["long_mask"]
        trans_mask = batch["trans_mask"]
        cls_target = batch["cls"].long().cpu().numpy()

        long_logits, trans_logits, cls_logits = model.forward_pair(long_img, trans_img)
        cls_prob = torch.sigmoid(cls_logits).detach().cpu().numpy()
        cls_pred = (cls_prob >= 0.5).astype(np.int64)

        y_true.extend(cls_target.tolist())
        y_pred.extend(cls_pred.tolist())

        long_pred = long_logits.argmax(dim=1).detach().cpu().numpy()
        trans_pred = trans_logits.argmax(dim=1).detach().cpu().numpy()
        long_gt = long_mask.detach().cpu().numpy()
        trans_gt = trans_mask.detach().cpu().numpy()

        for i in range(long_pred.shape[0]):
            long_m = compute_seg_metrics(long_pred[i], long_gt[i], tolerance=nsd_tolerance)
            trans_m = compute_seg_metrics(trans_pred[i], trans_gt[i], tolerance=nsd_tolerance)
            seg_scores.append(aggregate_seg_scores(long_m, trans_m))

    macro_f1 = float(f1_score(y_true, y_pred, average="macro")) if y_true else 0.0
    if seg_scores:
        s_seg = float(np.mean([d["S_seg"] for d in seg_scores]))
        s_long = float(np.mean([d["S_long"] for d in seg_scores]))
        s_trans = float(np.mean([d["S_trans"] for d in seg_scores]))
    else:
        s_seg = s_long = s_trans = 0.0

    return {"S_seg": s_seg, "S_long": s_long, "S_trans": s_trans, "F1_macro": macro_f1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--save_dir", type=str, default="Baseline/runs/baseline")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--unsup_weight", type=float, default=1.0)
    parser.add_argument("--unsup_rampup", type=int, default=10)
    parser.add_argument("--pseudo_thresh", type=float, default=0.6)
    parser.add_argument("--nsd_tolerance", type=float, default=2.0)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    maybe_set_float32_matmul_precision()
    device = get_device()

    all_ids = list_case_ids(Path(args.data_root) / "images")
    labeled_ids = [i for i in all_ids if i < 200]
    unlabeled_ids = [i for i in all_ids if i >= 200]
    train_ids, val_ids = split_labeled_ids(labeled_ids, args.val_ratio, seed=args.seed)

    train_ds = CSV2026LabeledDataset(args.data_root, train_ids, augment=True)
    val_ds = CSV2026LabeledDataset(args.data_root, val_ids, augment=False)
    unlabeled_ds = CSV2026UnlabeledDataset(args.data_root, unlabeled_ids)

    labeled_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    unlabeled_loader = DataLoader(
        unlabeled_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=(device.type == "cuda"),
    )

    model = UNet(UNetConfig())
    model.to(device)
    teacher = make_teacher(model).to(device)
    ema = EMA(decay=args.ema_decay)

    class_weights = torch.tensor([0.05, 0.30, 0.65], dtype=torch.float32, device=device)
    seg_loss_sup = CombinedSegLoss(
        SegLossConfig(
            num_classes=3,
            ce_weight=1.0,
            dice_weight=1.0,
            ignore_index=None,
            class_weights=class_weights,
        )
    )
    cls_loss_fn = BinaryFocalLoss(alpha=0.5, gamma=2.0)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0

    for epoch in range(args.epochs):
        train_logs = train_one_epoch(
            student=model,
            teacher=teacher,
            ema=ema,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            seg_loss_sup=seg_loss_sup,
            cls_loss_fn=cls_loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            epochs=args.epochs,
            unsup_weight_max=args.unsup_weight,
            unsup_rampup=args.unsup_rampup,
            pseudo_thresh=args.pseudo_thresh,
            amp=args.amp,
        )
        val_logs = evaluate(teacher, val_loader, device, nsd_tolerance=args.nsd_tolerance)
        score = 0.4 * val_logs["S_seg"] + 0.4 * val_logs["F1_macro"]

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train": train_logs,
            "val": val_logs,
            "args": vars(args),
        }
        save_checkpoint(ckpt, save_dir / "last.pt")
        if score > best:
            best = score
            save_checkpoint(ckpt, save_dir / "best.pt")

        print(
            f"Epoch {epoch+1:03d} "
            f"loss={train_logs['loss']:.4f} "
            f"S_seg={val_logs['S_seg']:.4f} "
            f"F1={val_logs['F1_macro']:.4f} "
            f"score(0.8)={score:.4f} "
            f"best={best:.4f}"
        )


if __name__ == "__main__":
    main()
