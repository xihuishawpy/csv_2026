from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Baseline.unet import UNet, UNetConfig
from Baseline.utils.util import CLASS_TO_MASK_VALUE, get_device, list_case_ids, load_checkpoint


class CSV2026ImagesOnly(Dataset):
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
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--weights", type=str, default="Baseline/runs/baseline/best.pt")
    parser.add_argument(
        "--use_teacher",
        dest="use_teacher",
        action="store_true",
        help="Load EMA teacher weights if present (default: on).",
    )
    parser.add_argument(
        "--no-teacher", dest="use_teacher", action="store_false", help="Load student weights."
    )
    parser.set_defaults(use_teacher=True)
    parser.add_argument("--out_dir", type=str, default="Baseline/outputs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_png", action="store_true")
    parser.add_argument("--save_h5", action="store_true")
    args = parser.parse_args()

    device = get_device()
    model = UNet(UNetConfig()).to(device)
    load_checkpoint(model, args.weights, device, use_teacher=args.use_teacher)
    model.eval()

    ids = list_case_ids(Path(args.data_root) / "images")
    ds = CSV2026ImagesOnly(args.data_root, ids)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cls_rows = [("case_id", "prob_high_risk")]

    for batch in tqdm(loader, desc="Infer"):
        case_ids = batch["id"].numpy().tolist()
        long_img = batch["long_img"].to(device)
        trans_img = batch["trans_img"].to(device)

        long_logits, trans_logits, cls_logits = model.forward_pair(long_img, trans_img)
        cls_prob = torch.sigmoid(cls_logits).cpu().numpy().tolist()
        long_pred = long_logits.argmax(dim=1).cpu().numpy()
        trans_pred = trans_logits.argmax(dim=1).cpu().numpy()

        for i, case_id in enumerate(case_ids):
            cls_rows.append((f"{case_id:04d}", f"{cls_prob[i]:.6f}"))
            if args.save_png:
                long_u8 = _to_uint8_mask(long_pred[i])
                trans_u8 = _to_uint8_mask(trans_pred[i])
                Image.fromarray(long_u8).save(out_dir / f"{case_id:04d}_long.png")
                Image.fromarray(trans_u8).save(out_dir / f"{case_id:04d}_trans.png")
            if args.save_h5:
                long_u8 = _to_uint8_mask(long_pred[i])
                trans_u8 = _to_uint8_mask(trans_pred[i])
                with h5py.File(out_dir / f"{case_id:04d}_pred.h5", "w") as f:
                    f.create_dataset("long_mask", data=long_u8, compression="gzip")
                    f.create_dataset("trans_mask", data=trans_u8, compression="gzip")
                    f.create_dataset("cls_prob", data=np.float32(cls_prob[i]))

    with open(out_dir / "pred_cls.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(cls_rows)


if __name__ == "__main__":
    main()
