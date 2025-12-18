from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

CLASS_TO_MASK_VALUE = {0: 0, 1: 255, 2: 128}


# -----------------------------
#  Model definition (ResUNet + risk head)
# -----------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return out


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ResBlock(in_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block = ResBlock(in_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@dataclass
class UNetConfig:
    in_channels: int = 1
    base_channels: int = 32
    num_classes: int = 3
    dropout: float = 0.1


class UNet(nn.Module):
    def __init__(self, cfg: UNetConfig):
        super().__init__()
        c = cfg.base_channels
        self.cfg = cfg

        self.inc = ResBlock(cfg.in_channels, c, dropout=cfg.dropout * 0.5)
        self.down1 = Down(c, c * 2, dropout=cfg.dropout * 0.5)
        self.down2 = Down(c * 2, c * 4, dropout=cfg.dropout * 0.5)
        self.down3 = Down(c * 4, c * 8, dropout=cfg.dropout)
        self.down4 = Down(c * 8, c * 8, dropout=cfg.dropout)

        self.up1 = Up(c * 16, c * 4, dropout=cfg.dropout)
        self.up2 = Up(c * 8, c * 2, dropout=cfg.dropout * 0.5)
        self.up3 = Up(c * 4, c, dropout=cfg.dropout * 0.5)
        self.up4 = Up(c * 2, c, dropout=cfg.dropout * 0.5)
        self.drop = nn.Dropout2d(cfg.dropout)
        self.outc = OutConv(c, cfg.num_classes)

        feat_dim = c * 8
        risk_dim = 6  # (plaque_area, vessel_area, ratio) x {long, trans}
        self.cls_head = nn.Sequential(
            nn.Linear(feat_dim * 4 + risk_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(feat_dim, 1),
        )

    def forward_single(self, x: torch.Tensor):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.drop(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits, x5

    @staticmethod
    def _feature_from_bottleneck(bottleneck: torch.Tensor, seg_logits: torch.Tensor | None = None):
        global_pool = bottleneck.mean(dim=(2, 3))
        if seg_logits is None:
            return torch.cat([global_pool, global_pool], dim=1)

        with torch.no_grad():
            prob_plaque = torch.softmax(seg_logits, dim=1)[:, 2:3]
        prob_plaque = F.interpolate(
            prob_plaque, size=bottleneck.shape[-2:], mode="bilinear", align_corners=False
        )
        denom = prob_plaque.sum(dim=(2, 3)).clamp_min(1e-6)
        att_pool = (bottleneck * prob_plaque).sum(dim=(2, 3)) / denom
        return torch.cat([global_pool, att_pool], dim=1)

    @staticmethod
    def _risk_features(seg_logits: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probs = torch.softmax(seg_logits, dim=1)
            plaque = probs[:, 2]
            vessel = probs[:, 1]
            plaque_area = plaque.mean(dim=(1, 2))
            vessel_area = vessel.mean(dim=(1, 2))
            ratio = plaque_area / vessel_area.clamp_min(1e-6)
            return torch.stack([plaque_area, vessel_area, ratio], dim=1)

    def forward_cls(self, feat_long: torch.Tensor, feat_trans: torch.Tensor, risk_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([feat_long, feat_trans, risk_feat], dim=1)
        return self.cls_head(x).squeeze(1)

    def forward_pair(self, long_img: torch.Tensor, trans_img: torch.Tensor):
        long_logits, long_bn = self.forward_single(long_img)
        trans_logits, trans_bn = self.forward_single(trans_img)
        feat_long = self._feature_from_bottleneck(long_bn, long_logits)
        feat_trans = self._feature_from_bottleneck(trans_bn, trans_logits)
        risk_feat = torch.cat([self._risk_features(long_logits), self._risk_features(trans_logits)], dim=1)
        cls_logits = self.forward_cls(feat_long, feat_trans, risk_feat=risk_feat.to(feat_long.dtype))
        return long_logits, trans_logits, cls_logits


# -----------------------------
#  Utils
# -----------------------------
def list_case_ids(images_dir: str | Path) -> List[int]:
    images_dir = Path(images_dir)
    ids: List[int] = []
    for p in images_dir.glob("*.h5"):
        try:
            ids.append(int(p.stem))
        except ValueError:
            continue
    return sorted(ids)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device, use_teacher: bool = True) -> None:
    ckpt = torch.load(str(path), map_location=device)
    if isinstance(ckpt, dict) and any(k in ckpt for k in ("model", "teacher", "state_dict")):
        if use_teacher and "teacher" in ckpt:
            state = ckpt["teacher"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt["state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)


class ImagesDataset(Dataset):
    """Inference-only dataset that reads long/trans ultrasound images."""

    def __init__(self, root: str | Path, ids: List[int]) -> None:
        self.root = Path(root)
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case_id = self.ids[idx]
        img_path = self.root / "images" / f"{case_id:04d}.h5"
        with h5py.File(img_path, "r") as f:
            long_img = f["long_img"][...]
            trans_img = f["trans_img"][...]
        long_img = torch.from_numpy(long_img).float().unsqueeze(0) / 255.0
        trans_img = torch.from_numpy(trans_img).float().unsqueeze(0) / 255.0
        return {
            "id": torch.tensor(case_id, dtype=torch.long),
            "long_img": long_img,
            "trans_img": trans_img,
        }


def _to_uint8_mask(pred: np.ndarray) -> np.ndarray:
    out = np.zeros_like(pred, dtype=np.uint8)
    for k, v in CLASS_TO_MASK_VALUE.items():
        out[pred == k] = np.uint8(v)
    return out


@torch.no_grad()
def run_inference(
    model: UNet,
    data_root: str | Path,
    out_dir: str | Path,
    batch_size: int = 4,
    num_workers: int = 0,
) -> None:
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = list_case_ids(data_root / "images")
    if not ids:
        raise FileNotFoundError(f"No .h5 files found under {data_root}/images")
    ds = ImagesDataset(data_root, ids)
    device = next(model.parameters()).device
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    cls_rows = [("case_id", "prob_high_risk")]

    for batch in loader:
        case_ids = batch["id"].tolist()
        long_img = batch["long_img"].to(device, non_blocking=True)
        trans_img = batch["trans_img"].to(device, non_blocking=True)

        long_logits, trans_logits, cls_logits = model.forward_pair(long_img, trans_img)
        cls_prob = torch.sigmoid(cls_logits).cpu().numpy()
        long_pred = long_logits.argmax(dim=1).cpu().numpy()
        trans_pred = trans_logits.argmax(dim=1).cpu().numpy()

        for i, case_id in enumerate(case_ids):
            cls_rows.append((f"{case_id:04d}", f"{float(cls_prob[i]):.6f}"))
            long_u8 = _to_uint8_mask(long_pred[i])
            trans_u8 = _to_uint8_mask(trans_pred[i])
            with h5py.File(out_dir / f"{case_id:04d}_pred.h5", "w") as f:
                f.create_dataset("long_mask", data=long_u8, compression="gzip")
                f.create_dataset("trans_mask", data=trans_u8, compression="gzip")
                f.create_dataset("cls_prob", data=np.float32(cls_prob[i]))

    # Save classification CSV
    import csv  # local import to keep global deps minimal

    with open(out_dir / "pred_cls.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(cls_rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/input", help="Input data root (contains images/).")
    parser.add_argument("--output", type=str, default="/output", help="Output directory.")
    parser.add_argument(
        "--weights", type=str, default="weights/best_model.pth", help="Path to model weights (.pth)."
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--use_teacher", action="store_true", default=True, help="Load EMA teacher weights if available."
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}. Place your trained checkpoint at this path inside the container."
        )

    device = get_device()
    model = UNet(UNetConfig()).to(device)
    load_checkpoint(model, weights_path, device, use_teacher=args.use_teacher)
    model.eval()
    run_inference(model, args.input, args.output, batch_size=args.batch_size, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
