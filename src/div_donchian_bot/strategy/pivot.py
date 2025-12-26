from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, List

PivotType = Literal["LOW","HIGH"]

@dataclass
class Pivot:
    kind: PivotType
    index: int         # index in the bars deque/list
    price: float
    osc: float
    loc: float
    pivot_time_ms: int # bar open time of pivot bar

def is_pivot_low(lows: List[float], idx: int, left_right: int) -> bool:
    start = idx - left_right
    end = idx + left_right
    if start < 0 or end >= len(lows):
        return False
    window = lows[start:end+1]
    m = min(window)
    return lows[idx] == m

def is_pivot_high(highs: List[float], idx: int, left_right: int) -> bool:
    start = idx - left_right
    end = idx + left_right
    if start < 0 or end >= len(highs):
        return False
    window = highs[start:end+1]
    m = max(window)
    return highs[idx] == m
