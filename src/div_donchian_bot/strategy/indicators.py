from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple
import math

@dataclass
class EmaState:
    length: int
    value: Optional[float] = None

    def update(self, x: float) -> float:
        alpha = 2.0 / (self.length + 1.0)
        if self.value is None:
            self.value = x
        else:
            self.value = alpha * x + (1.0 - alpha) * self.value
        return self.value

class AtrState:
    def __init__(self, length: int):
        self.length = length
        self.ema = EmaState(length)
        self.prev_close: Optional[float] = None

    def update(self, high: float, low: float, close: float) -> float:
        if self.prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
        self.prev_close = close
        return self.ema.update(tr)

def donchian(highs, lows, length: int) -> Tuple[float, float]:
    h = max(list(highs)[-length:])
    l = min(list(lows)[-length:])
    return h, l

def loc_in_range(close: float, lo: float, hi: float) -> float:
    rng = hi - lo
    if rng <= 0:
        return 0.5
    return (close - lo) / rng

def bps(from_price: float, to_price: float) -> float:
    if from_price <= 0:
        return float("nan")
    return ((to_price - from_price) / from_price) * 10000.0
