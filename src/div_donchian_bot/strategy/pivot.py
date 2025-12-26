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
    # Strict: pivot must be strictly lower than all others in window
    if lows[idx] != m:
        return False
    # If any other bar shares the min, reject
    for j in range(start, end + 1):
        if j != idx and lows[j] == m:
            return False
    return True

def is_pivot_high(highs: List[float], idx: int, left_right: int) -> bool:
    start = idx - left_right
    end = idx + left_right
    if start < 0 or end >= len(highs):
        return False
    window = highs[start:end+1]
    m = max(window)
    if highs[idx] != m:
        return False
    for j in range(start, end + 1):
        if j != idx and highs[j] == m:
            return False
    return True
