from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def save_overlay(
    image: np.ndarray,
    gt: Optional[np.ndarray],
    pred: Optional[np.ndarray],
    out_path: str | Path,
    title: str = "",
    alpha: float = 0.35,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = image.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0

    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    if gt is not None:
        plt.imshow(gt, cmap="viridis", alpha=alpha)
    if pred is not None:
        plt.imshow(pred, cmap="magma", alpha=alpha)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

