from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from Baseline.augmentations.ctaugment import StrongAugment, WeakAugment


MASK_VALUE_TO_CLASS: Dict[int, int] = {0: 0, 255: 1, 128: 2}
CLASS_TO_MASK_VALUE: Dict[int, int] = {0: 0, 1: 255, 2: 128}


def mask_value_to_class(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.int64)
    for k, v in MASK_VALUE_TO_CLASS.items():
        out[mask == k] = v
    return out


def class_to_mask_value(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.uint8)
    for k, v in CLASS_TO_MASK_VALUE.items():
        out[mask == k] = v
    return out


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(state: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))


def load_checkpoint(
    model: torch.nn.Module, path: str | Path, device: torch.device, use_teacher: bool = False
) -> Dict:
    ckpt = torch.load(str(path), map_location=device)
    if use_teacher and "teacher" in ckpt:
        key = "teacher"
    else:
        key = "model" if "model" in ckpt else "state_dict"
    model.load_state_dict(ckpt[key], strict=True)
    return ckpt


@dataclass
class EMA:
    decay: float = 0.99

    def __post_init__(self) -> None:
        if not (0.0 < self.decay < 1.0):
            raise ValueError("EMA decay must be in (0,1)")

    @torch.no_grad()
    def update(self, student: torch.nn.Module, teacher: torch.nn.Module) -> None:
        s_state = student.state_dict()
        t_state = teacher.state_dict()
        for k, v in t_state.items():
            if k in s_state and v.dtype.is_floating_point:
                t_state[k].mul_(self.decay).add_(s_state[k].detach(), alpha=1.0 - self.decay)
            elif k in s_state:
                t_state[k].copy_(s_state[k])
        teacher.load_state_dict(t_state, strict=True)


def list_case_ids(images_dir: str | Path) -> List[int]:
    images_dir = Path(images_dir)
    ids: List[int] = []
    for p in images_dir.glob("*.h5"):
        try:
            ids.append(int(p.stem))
        except ValueError:
            continue
    return sorted(ids)


def split_labeled_ids(
    labeled_ids: List[int], val_ratio: float, seed: int = 42
) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    ids = labeled_ids[:]
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    val_ids = sorted(ids[:n_val])
    train_ids = sorted(ids[n_val:])
    return train_ids, val_ids


class CSV2026LabeledDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        ids: List[int],
        augment: bool = True,
    ) -> None:
        self.root = Path(root)
        self.ids = ids
        self.augment = augment
        self.weak_aug = WeakAugment()

    def __len__(self) -> int:
        return len(self.ids)

    def _read(self, case_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        img_path = self.root / "images" / f"{case_id:04d}.h5"
        lbl_path = self.root / "labels" / f"{case_id:04d}_label.h5"
        with h5py.File(img_path, "r") as f:
            long_img = f["long_img"][...]
            trans_img = f["trans_img"][...]
        with h5py.File(lbl_path, "r") as f:
            long_mask = f["long_mask"][...]
            trans_mask = f["trans_mask"][...]
            cls = int(f["cls"][()])
        return long_img, trans_img, long_mask, trans_mask, cls

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        case_id = self.ids[index]
        long_img, trans_img, long_mask, trans_mask, cls = self._read(case_id)

        long_img = torch.from_numpy(long_img).float().unsqueeze(0) / 255.0
        trans_img = torch.from_numpy(trans_img).float().unsqueeze(0) / 255.0
        long_mask = torch.from_numpy(mask_value_to_class(long_mask)).long()
        trans_mask = torch.from_numpy(mask_value_to_class(trans_mask)).long()

        if self.augment:
            long_img, long_mask = self.weak_aug(long_img, long_mask)
            trans_img, trans_mask = self.weak_aug(trans_img, trans_mask)

        return {
            "id": torch.tensor(case_id, dtype=torch.long),
            "long_img": long_img,
            "trans_img": trans_img,
            "long_mask": long_mask,
            "trans_mask": trans_mask,
            "cls": torch.tensor(cls, dtype=torch.float32),
        }


class CSV2026UnlabeledDataset(Dataset):
    def __init__(self, root: str | Path, ids: List[int]) -> None:
        self.root = Path(root)
        self.ids = ids
        self.weak_aug = WeakAugment()
        self.strong_aug = StrongAugment()

    def __len__(self) -> int:
        return len(self.ids)

    def _read(self, case_id: int) -> Tuple[np.ndarray, np.ndarray]:
        img_path = self.root / "images" / f"{case_id:04d}.h5"
        with h5py.File(img_path, "r") as f:
            long_img = f["long_img"][...]
            trans_img = f["trans_img"][...]
        return long_img, trans_img

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        case_id = self.ids[index]
        long_img, trans_img = self._read(case_id)

        long_img = torch.from_numpy(long_img).float().unsqueeze(0) / 255.0
        trans_img = torch.from_numpy(trans_img).float().unsqueeze(0) / 255.0

        long_w, _ = self.weak_aug(long_img.clone(), None)
        trans_w, _ = self.weak_aug(trans_img.clone(), None)
        long_s, _ = self.strong_aug(long_img.clone(), None)
        trans_s, _ = self.strong_aug(trans_img.clone(), None)

        return {
            "id": torch.tensor(case_id, dtype=torch.long),
            "long_w": long_w,
            "trans_w": trans_w,
            "long_s": long_s,
            "trans_s": trans_s,
        }


def make_teacher(student: torch.nn.Module) -> torch.nn.Module:
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher


def maybe_set_float32_matmul_precision() -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
