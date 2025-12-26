from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, List

PivotType = Literal["LOW","HIGH"]
PivotTieBreak = Literal["strict","tv_like"]

@dataclass
class Pivot:
    kind: PivotType
    index: int         # index in the bars deque/list
    price: float
    osc: float
    loc: float
    pivot_time_ms: int # bar open time of pivot bar

def is_pivot_low(lows: List[float], idx: int, left_right: int, tie_break: PivotTieBreak = "strict") -> bool:
    start = idx - left_right
    end = idx + left_right
    if start < 0 or end >= len(lows):
        return False
    window = lows[start:end+1]
    m = min(window)
    if lows[idx] != m:
        return False
    indices = [j for j in range(start, end + 1) if lows[j] == m]
    if tie_break == "strict":
        return len(indices) == 1
    if tie_break == "tv_like":
        # Deterministic: choose leftmost extremum inside the confirmation window
        return idx == indices[0]
    raise ValueError(f"Unknown tie_break: {tie_break}")

def is_pivot_high(highs: List[float], idx: int, left_right: int, tie_break: PivotTieBreak = "strict") -> bool:
    start = idx - left_right
    end = idx + left_right
    if start < 0 or end >= len(highs):
        return False
    window = highs[start:end+1]
    m = max(window)
    if highs[idx] != m:
        return False
    indices = [j for j in range(start, end + 1) if highs[j] == m]
    if tie_break == "strict":
        return len(indices) == 1
    if tie_break == "tv_like":
        return idx == indices[0]
    raise ValueError(f"Unknown tie_break: {tie_break}")
