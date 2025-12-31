from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal

Side = Literal["LONG", "SHORT"]

@dataclass
class Bar:
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time_ms: int

@dataclass
class Signal:
    symbol: str
    side: Side
    entry_price: float
    pivot_price: float
    pivot_osc_value: float
    pivot_cvd_value: Optional[float]
    slip_bps: float
    loc_at_pivot: float
    oscillator_name: str
    pine_div: bool
    cvd_div: bool
    pivot_time_ms: int
    confirm_time_ms: int
    source: str = "BOT"
    latency_ms: Optional[int] = None

@dataclass
class RiskLevels:
    sl: Optional[float] = None
    tp: Optional[float] = None
