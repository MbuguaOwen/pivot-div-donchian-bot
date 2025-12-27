from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple

Side = Literal["LONG", "SHORT"]


@dataclass
class TrailingParams:
    enabled: bool = False
    trigger_r: float = 2.5
    k_trail: float = 1.6
    lock_r: float = 1.0


@dataclass
class StructureAtrTrailParams:
    enabled: bool
    htf_interval: str = "1h"
    htf_lookback_bars: int = 72
    atr_len: int = 24
    k_init: float = 1.8
    tp_r_mult: float = 2.0
    buffer_bps: float = 10.0
    be_trigger_r: float = 1.0
    be_buffer_bps: float = 10.0
    trailing: TrailingParams = field(default_factory=TrailingParams)
    min_stop_replace_interval_sec: float = 2.0
    min_stop_tick_improvement: int = 2

    @property
    def trail_enabled(self) -> bool:
        return self.trailing.enabled

    @trail_enabled.setter
    def trail_enabled(self, enabled: bool) -> None:
        self.trailing.enabled = enabled


@dataclass
class StopState:
    side: Side
    entry: float
    sl: float
    R: float
    peak: float
    trough: float
    be_done: bool
    trailing_active: bool


def _bps_to_float(bps: float) -> float:
    return bps / 10000.0


def compute_initial_sl(
    entry: float,
    side: Side,
    struct_low: float,
    struct_high: float,
    atr: float,
    params: StructureAtrTrailParams,
) -> Tuple[float, float]:
    """Initial SL using structure with ATR cap."""
    buf = _bps_to_float(params.buffer_bps)

    if side == "LONG":
        sl_struct = struct_low * (1.0 - buf)
        stop_dist = entry - sl_struct
        capped = params.k_init * atr
        sl0 = entry - capped if stop_dist > capped else sl_struct
        R = entry - sl0
    else:
        sl_struct = struct_high * (1.0 + buf)
        stop_dist = sl_struct - entry
        capped = params.k_init * atr
        sl0 = entry + capped if stop_dist > capped else sl_struct
        R = sl0 - entry

    return sl0, R


def compute_tp_from_R(entry: float, side: Side, R: float, tp_r_mult: float) -> float:
    """Compute fixed-R take profit from initial R distance."""
    offset = tp_r_mult * R
    if side == "LONG":
        return entry + offset
    return entry - offset


def update_extremes(state: StopState, price: float | None = None, high: float | None = None, low: float | None = None) -> None:
    """Update best favorable excursion using tick or bar extremes."""
    if state.side == "LONG":
        if high is not None:
            state.peak = max(state.peak, high)
        if price is not None:
            state.peak = max(state.peak, price)
    else:
        if low is not None:
            state.trough = min(state.trough, low)
        if price is not None:
            state.trough = min(state.trough, price)


def best_move_in_R(state: StopState) -> float:
    if state.R == 0:
        return 0.0
    if state.side == "LONG":
        return (state.peak - state.entry) / state.R
    return (state.entry - state.trough) / state.R


def update_stop(state: StopState, atr: float, params: StructureAtrTrailParams) -> float:
    """Monotonic stop update: BE -> lock -> volatility trail."""
    best_r = best_move_in_R(state)

    # 1) Break-even move
    if (not state.be_done) and best_r >= params.be_trigger_r:
        be_buf = _bps_to_float(params.be_buffer_bps)
        if state.side == "LONG":
            be_sl = state.entry * (1.0 + be_buf)
            state.sl = max(state.sl, be_sl)
        else:
            be_sl = state.entry * (1.0 - be_buf)
            state.sl = min(state.sl, be_sl)
        state.be_done = True

    if not params.trailing.enabled:
        return state.sl

    spacing_r = max(0.0, params.trailing.trigger_r - params.trailing.lock_r)

    # 2) Trailing activation + lock
    if (not state.trailing_active) and best_r >= params.trailing.trigger_r:
        if state.side == "LONG":
            lock_sl = state.entry + params.trailing.lock_r * state.R
            state.sl = max(state.sl, lock_sl)
        else:
            lock_sl = state.entry - params.trailing.lock_r * state.R
            state.sl = min(state.sl, lock_sl)
        state.trailing_active = True

    # 3) Volatility trailing after activation
    if state.trailing_active:
        vol_offset = params.trailing.k_trail * (atr or 0.0)
        floor_offset = spacing_r * state.R
        offset = max(vol_offset, floor_offset)

        if state.side == "LONG":
            sl_vol = state.peak - offset
            sl_lock = state.entry + params.trailing.lock_r * state.R
            state.sl = max(state.sl, sl_lock, sl_vol)
        else:
            sl_vol = state.trough + offset
            sl_lock = state.entry - params.trailing.lock_r * state.R
            state.sl = min(state.sl, sl_lock, sl_vol)

    return state.sl
