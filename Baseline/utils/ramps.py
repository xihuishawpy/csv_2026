from __future__ import annotations

import math


def sigmoid_rampup(current: float, rampup_length: float) -> float:
    if rampup_length <= 0:
        return 1.0
    current = float(max(0.0, min(current, rampup_length)))
    phase = 1.0 - current / rampup_length
    return float(math.exp(-5.0 * phase * phase))


def linear_rampup(current: float, rampup_length: float) -> float:
    if rampup_length <= 0:
        return 1.0
    return float(max(0.0, min(1.0, current / rampup_length)))

