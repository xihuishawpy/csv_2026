from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp_(0.0, 1.0)


def _rand_uniform(a: float, b: float) -> float:
    return a + (b - a) * random.random()


def _random_flip_rot(
    image: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if random.random() < 0.5:
        image = torch.flip(image, dims=(-1,))
        if mask is not None:
            mask = torch.flip(mask, dims=(-1,))
    if random.random() < 0.5:
        image = torch.flip(image, dims=(-2,))
        if mask is not None:
            mask = torch.flip(mask, dims=(-2,))

    k = random.randint(0, 3)
    if k:
        image = torch.rot90(image, k, dims=(-2, -1))
        if mask is not None:
            mask = torch.rot90(mask, k, dims=(-2, -1))
    return image, mask


def _random_intensity(
    image: torch.Tensor, strength: float = 0.25, p: float = 0.8
) -> torch.Tensor:
    if random.random() < p:
        brightness = _rand_uniform(-strength, strength)
        image = image + brightness
    if random.random() < p:
        contrast = _rand_uniform(1.0 - strength, 1.0 + strength)
        image = image * contrast
    if random.random() < p * 0.5:
        gamma = _rand_uniform(1.0 - strength, 1.0 + strength)
        image = _clamp01(image)
        image = image.pow(gamma)
    return _clamp01(image)


def _random_gaussian_noise(
    image: torch.Tensor, sigma: float = 0.08, p: float = 0.5
) -> torch.Tensor:
    if random.random() < p:
        scale = _rand_uniform(0.0, sigma)
        image = image + torch.randn_like(image) * scale
    return _clamp01(image)


def _random_blur(image: torch.Tensor, p: float = 0.3) -> torch.Tensor:
    if random.random() >= p:
        return image
    kernel = torch.ones((1, 1, 3, 3), device=image.device, dtype=image.dtype) / 9.0
    x = image.unsqueeze(0)
    x = F.conv2d(x, kernel, padding=1)
    return x.squeeze(0)


def _random_cutout(
    image: torch.Tensor, max_size: int = 72, p: float = 0.5
) -> torch.Tensor:
    if random.random() >= p:
        return image
    _, h, w = image.shape
    size = random.randint(max(8, max_size // 3), max_size)
    y0 = random.randint(0, max(0, h - size))
    x0 = random.randint(0, max(0, w - size))
    image[:, y0 : y0 + size, x0 : x0 + size] = 0.0
    return image


@dataclass
class WeakAugment:
    intensity_strength: float = 0.15

    def __call__(
        self, image: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        image, mask = _random_flip_rot(image, mask)
        image = _random_intensity(image, strength=self.intensity_strength, p=0.7)
        return image, mask


@dataclass
class StrongAugment:
    intensity_strength: float = 0.3

    def __call__(
        self, image: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        image, mask = _random_flip_rot(image, mask)
        image = _random_intensity(image, strength=self.intensity_strength, p=0.9)
        image = _random_gaussian_noise(image, sigma=0.10, p=0.7)
        image = _random_blur(image, p=0.4)
        image = _random_cutout(image, max_size=80, p=0.6)
        return image, mask

