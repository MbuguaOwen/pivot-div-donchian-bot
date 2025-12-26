from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple, Literal
import math

EmaSeedMode = Literal["first", "sma"]

@dataclass
class EmaState:
    length: int
    seed_mode: EmaSeedMode = "first"
    value: Optional[float] = None
    _seed_buffer: Deque[float] = field(default_factory=deque, init=False)

    def update(self, x: float) -> float:
        """Update EMA with configurable seeding.

        seed_mode:
          - first: first sample initializes EMA (previous behavior)
          - sma: seed with SMA of the first N samples before switching to EMA
        """
        alpha = 2.0 / (self.length + 1.0)
        if self.value is None and self.seed_mode == "sma":
            self._seed_buffer.append(x)
            if len(self._seed_buffer) < self.length:
                # Return running mean until we have full seed window
                return sum(self._seed_buffer) / len(self._seed_buffer)
            self.value = sum(self._seed_buffer) / self.length
            self._seed_buffer.clear()
            return self.value

        if self.value is None:
            self.value = x
            return self.value

        self.value = alpha * x + (1.0 - alpha) * self.value
        return self.value

class AtrState:
    def __init__(self, length: int, ema_seed_mode: EmaSeedMode = "first"):
        self.length = length
        self.ema = EmaState(length, seed_mode=ema_seed_mode)
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
