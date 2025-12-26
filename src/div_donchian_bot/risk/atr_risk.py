from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from ..models import RiskLevels

@dataclass
class AtrRiskParams:
    enabled: bool
    length: int
    sl_mult: float
    tp_mult: float

def atr_levels(side: str, entry: float, atr: float, p: AtrRiskParams) -> RiskLevels:
    if not p.enabled or atr is None or atr <= 0:
        return RiskLevels()
    if side == "LONG":
        sl = entry - p.sl_mult * atr
        tp = entry + p.tp_mult * atr
    else:
        sl = entry + p.sl_mult * atr
        tp = entry - p.tp_mult * atr
    return RiskLevels(sl=sl, tp=tp)
