from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, List, Literal

from ..models import Bar, Signal
from .indicators import EmaState, AtrState, donchian, loc_in_range, bps
from .pivot import is_pivot_low, is_pivot_high

DivergenceMode = Literal["pine","cvd","both","either"]

@dataclass
class StrategyParams:
    don_len: int
    ext_band_pct: float
    pivot_len: int
    osc_ema_len: int
    divergence_mode: DivergenceMode

class SymbolStrategyState:
    def __init__(self, symbol: str, params: StrategyParams, enable_cvd: bool):
        self.symbol = symbol
        self.p = params
        self.enable_cvd = enable_cvd

        self.bars: Deque[Bar] = deque(maxlen=max(2000, params.don_len + 2*params.pivot_len + 50))
        self.osc_ema = EmaState(params.osc_ema_len)
        self.osc_vals: Deque[float] = deque(maxlen=self.bars.maxlen)

        self.don_hi: Deque[float] = deque(maxlen=self.bars.maxlen)
        self.don_lo: Deque[float] = deque(maxlen=self.bars.maxlen)
        self.loc: Deque[float] = deque(maxlen=self.bars.maxlen)

        # CVD per bar close
        self.cvd_vals: Deque[float] = deque(maxlen=self.bars.maxlen)
        self._cvd_running: float = 0.0
        self._delta_this_bar: float = 0.0
        self._current_bar_open_ms: Optional[int] = None

        # Last confirmed pivots (same semantics as Pine)
        self.last_pl_price: Optional[float] = None
        self.last_pl_osc: Optional[float] = None
        self.last_ph_price: Optional[float] = None
        self.last_ph_osc: Optional[float] = None

        self.last_pl_cvd: Optional[float] = None
        self.last_ph_cvd: Optional[float] = None

    def on_agg_trade(self, ts_ms: int, qty: float, is_buyer_maker: bool) -> None:
        """Update delta for current 15m bar.
        Binance aggTrade: isBuyerMaker=true means maker is buyer => aggressor is seller.
        We'll define delta = +qty for buyer-aggressor, -qty for seller-aggressor.
        """
        if not self.enable_cvd:
            return
        if self._current_bar_open_ms is None:
            return
        # Only accumulate deltas that belong to current bar (best-effort)
        # If trades arrive late/early, they'll be slightly off; kline-close will reset.
        if ts_ms < self._current_bar_open_ms:
            return
        if is_buyer_maker:
            self._delta_this_bar -= qty  # seller-aggressor
        else:
            self._delta_this_bar += qty  # buyer-aggressor

    def _finalize_cvd_for_closed_bar(self) -> float:
        self._cvd_running += self._delta_this_bar
        self._delta_this_bar = 0.0
        return self._cvd_running

    def on_bar_close(self, bar: Bar) -> Optional[Signal]:
        self._current_bar_open_ms = bar.open_time_ms

        self.bars.append(bar)

        # Pine oscillator proxy: EMA((close-open)*volume)
        pressure = (bar.close - bar.open) * bar.volume
        osc = self.osc_ema.update(pressure)
        self.osc_vals.append(osc)

        # Donchian + loc
        highs = [b.high for b in self.bars]
        lows = [b.low for b in self.bars]
        if len(self.bars) >= self.p.don_len:
            hi, lo = donchian(highs, lows, self.p.don_len)
        else:
            hi, lo = max(highs), min(lows)
        self.don_hi.append(hi)
        self.don_lo.append(lo)
        self.loc.append(loc_in_range(bar.close, lo, hi))

        # CVD close for this bar (if enabled)
        if self.enable_cvd:
            cvd_close = self._finalize_cvd_for_closed_bar()
        else:
            cvd_close = float("nan")
        self.cvd_vals.append(cvd_close)

        # Need enough bars for pivot confirmation window
        L = len(self.bars)
        pl = self.p.pivot_len
        if L < (2*pl + 1):
            return None

        # Confirm pivots at index mid = L-1-pl (pivot occurs pl bars ago)
        mid = L - 1 - pl
        highs_list = [b.high for b in self.bars]
        lows_list  = [b.low for b in self.bars]

        # Grab pivot bar values (equivalent to Pine: low[pivotLen], osc[pivotLen], loc[pivotLen])
        pivot_bar = list(self.bars)[mid]
        pivot_osc_pine = list(self.osc_vals)[mid]
        pivot_loc = list(self.loc)[mid]
        pivot_cvd = list(self.cvd_vals)[mid] if self.enable_cvd else float("nan")

        # Determine if a pivot is confirmed now
        got_low = is_pivot_low(lows_list, mid, pl)
        got_high = is_pivot_high(highs_list, mid, pl)

        signal: Optional[Signal] = None

        # Evaluate divergence based on mode
        def bull_div_pine() -> bool:
            return (self.last_pl_price is not None and self.last_pl_osc is not None and
                    pivot_bar.low <= self.last_pl_price and pivot_osc_pine > self.last_pl_osc)

        def bear_div_pine() -> bool:
            return (self.last_ph_price is not None and self.last_ph_osc is not None and
                    pivot_bar.high >= self.last_ph_price and pivot_osc_pine < self.last_ph_osc)

        def bull_div_cvd() -> bool:
            return (self.enable_cvd and self.last_pl_price is not None and self.last_pl_cvd is not None and
                    pivot_bar.low <= self.last_pl_price and pivot_cvd > self.last_pl_cvd)

        def bear_div_cvd() -> bool:
            return (self.enable_cvd and self.last_ph_price is not None and self.last_ph_cvd is not None and
                    pivot_bar.high >= self.last_ph_price and pivot_cvd < self.last_ph_cvd)

        def combine(a: bool, b: bool) -> bool:
            if self.p.divergence_mode == "pine":
                return a
            if self.p.divergence_mode == "cvd":
                return b
            if self.p.divergence_mode == "both":
                return a and b
            if self.p.divergence_mode == "either":
                return a or b
            return a

        # LONG side
        if got_low:
            near_lower = pivot_loc <= self.p.ext_band_pct
            pine_ok = bull_div_pine()
            cvd_ok = bull_div_cvd()
            bull = combine(pine_ok, cvd_ok)
            if near_lower and bull:
                entry = bar.close  # tradable entry at confirmation bar close
                slip = bps(pivot_bar.low, entry)
                osc_name = self.p.divergence_mode
                signal = Signal(
                    symbol=self.symbol, side="LONG",
                    entry_price=entry, pivot_price=pivot_bar.low,
                    pivot_osc_value=pivot_osc_pine,
                    pivot_cvd_value=pivot_cvd if self.enable_cvd else None,
                    slip_bps=slip, loc_at_pivot=pivot_loc,
                    oscillator_name=osc_name,
                    pine_div=pine_ok,
                    cvd_div=cvd_ok,
                    pivot_time_ms=pivot_bar.open_time_ms,
                    confirm_time_ms=bar.close_time_ms
                )
            # Update last confirmed pivot low state (as Pine does)
            self.last_pl_price = pivot_bar.low
            self.last_pl_osc = pivot_osc_pine
            if self.enable_cvd:
                self.last_pl_cvd = pivot_cvd

        # SHORT side
        if got_high:
            near_upper = pivot_loc >= (1.0 - self.p.ext_band_pct)
            pine_ok = bear_div_pine()
            cvd_ok = bear_div_cvd()
            bear = combine(pine_ok, cvd_ok)
            if near_upper and bear:
                entry = bar.close
                slip = bps(pivot_bar.high, entry)
                osc_name = self.p.divergence_mode
                signal = Signal(
                    symbol=self.symbol, side="SHORT",
                    entry_price=entry, pivot_price=pivot_bar.high,
                    pivot_osc_value=pivot_osc_pine,
                    pivot_cvd_value=pivot_cvd if self.enable_cvd else None,
                    slip_bps=slip, loc_at_pivot=pivot_loc,
                    oscillator_name=osc_name,
                    pine_div=pine_ok,
                    cvd_div=cvd_ok,
                    pivot_time_ms=pivot_bar.open_time_ms,
                    confirm_time_ms=bar.close_time_ms
                )
            self.last_ph_price = pivot_bar.high
            self.last_ph_osc = pivot_osc_pine
            if self.enable_cvd:
                self.last_ph_cvd = pivot_cvd

        return signal
