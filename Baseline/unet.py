from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()

        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.act(out)
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

    def forward_single(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
    def _feature_from_bottleneck(
        bottleneck: torch.Tensor, seg_logits: torch.Tensor | None = None
    ) -> torch.Tensor:
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

    def forward_cls(
        self,
        feat_long: torch.Tensor,
        feat_trans: torch.Tensor,
        risk_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if risk_feat is None:
            risk_feat = torch.zeros(
                (feat_long.shape[0], 6), device=feat_long.device, dtype=feat_long.dtype
            )
        x = torch.cat([feat_long, feat_trans, risk_feat.to(feat_long.dtype)], dim=1)
        return self.cls_head(x).squeeze(1)

    def forward_pair(self, long_img: torch.Tensor, trans_img: torch.Tensor):
        long_logits, long_bn = self.forward_single(long_img)
        trans_logits, trans_bn = self.forward_single(trans_img)
        feat_long = self._feature_from_bottleneck(long_bn, long_logits)
        feat_trans = self._feature_from_bottleneck(trans_bn, trans_logits)
        risk_feat = torch.cat(
            [self._risk_features(long_logits), self._risk_features(trans_logits)], dim=1
        )
        cls_logits = self.forward_cls(feat_long, feat_trans, risk_feat=risk_feat)
        return long_logits, trans_logits, cls_logits
