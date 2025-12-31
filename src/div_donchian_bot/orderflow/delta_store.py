from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple


MINUTE_MS = 60_000


def _minute_close_ms(ts_ms: int) -> int:
    """Map a trade timestamp (ms) to its 1-minute bar close_time_ms (end-1ms)."""
    start = (ts_ms // MINUTE_MS) * MINUTE_MS
    return start + MINUTE_MS - 1


@dataclass
class DeltaSumResult:
    ok: bool
    reason: str
    delta_sum: float = 0.0
    minutes_found: int = 0
    minutes_needed: int = 0


class Delta1mStore:
    """Rolling 1-minute signed delta store built from aggTrades.

    Delta definition:
      +qty for buyer-aggressor trades
      -qty for seller-aggressor trades

    Binance aggTrade:
      isBuyerMaker=true => maker is buyer => aggressor is seller
    """

    def __init__(self, max_minutes: int = 240):
        self.max_minutes = max(30, int(max_minutes))
        self._deltas: Dict[int, float] = {}
        self._closes: Deque[int] = deque(maxlen=self.max_minutes)
        self._last_close_ms: Optional[int] = None
        self._gap_count: int = 0

    @property
    def last_close_ms(self) -> Optional[int]:
        return self._last_close_ms

    @property
    def gap_count(self) -> int:
        return self._gap_count

    @property
    def had_gap(self) -> bool:
        return self._gap_count > 0

    def minutes_available(self) -> int:
        return len(self._closes)

    def update_trade(self, ts_ms: int, qty: float, is_buyer_maker: bool) -> None:
        close_ms = _minute_close_ms(int(ts_ms))

        # Fill forward gaps so lookups can be strict about continuity.
        if self._last_close_ms is not None and close_ms > self._last_close_ms:
            step = close_ms - self._last_close_ms
            if step > MINUTE_MS:
                # gap of >=2 minutes
                gaps = int(step // MINUTE_MS) - 1
                self._gap_count += max(0, gaps)
                # Fill missing minutes with zero delta
                cur = self._last_close_ms
                for _ in range(gaps):
                    cur += MINUTE_MS
                    self._ensure_minute(cur)

        self._ensure_minute(close_ms)

        # Update delta
        signed = -qty if is_buyer_maker else qty
        self._deltas[close_ms] = self._deltas.get(close_ms, 0.0) + float(signed)

    def _ensure_minute(self, close_ms: int) -> None:
        # Maintain ordered closes; allow out-of-order updates within retention.
        if close_ms not in self._deltas:
            self._deltas[close_ms] = 0.0

        if self._last_close_ms is None:
            self._closes.append(close_ms)
            self._last_close_ms = close_ms
            return

        if close_ms > self._last_close_ms:
            self._closes.append(close_ms)
            self._last_close_ms = close_ms
        else:
            # Out-of-order: ensure it exists in closes if still in window.
            if close_ms not in self._closes:
                self._closes.append(close_ms)
                # Re-sort to preserve ascending order (small deque, ok).
                self._closes = deque(sorted(self._closes), maxlen=self.max_minutes)

        # Trim dict entries beyond deque retention.
        if len(self._closes) == self._closes.maxlen:
            oldest = self._closes[0]
            # remove any keys older than oldest
            for k in list(self._deltas.keys()):
                if k < oldest:
                    self._deltas.pop(k, None)

    def get_delta_sum(self, end_ms: int, window_minutes: int, warmup_minutes: int = 0) -> DeltaSumResult:
        """Sum delta over [end_ms - (window-1)*1m, end_ms] inclusive.

        Requirements:
        - end_ms must align to a minute close_time_ms (i.e., end_ms % 60000 == 59999)
        - all minutes inside the window must exist (including zero-filled gaps)
        - if warmup_minutes>0, require a continuous warmup history ending at end_ms
        """
        end_ms = int(end_ms)
        window_minutes = int(window_minutes)
        warmup_minutes = int(warmup_minutes)
        if window_minutes <= 0:
            return DeltaSumResult(ok=False, reason="bad_window", minutes_needed=window_minutes)

        if end_ms % MINUTE_MS != (MINUTE_MS - 1):
            return DeltaSumResult(ok=False, reason="end_ms_not_minute_close", minutes_needed=window_minutes)

        # Build the full required set of minutes
        needed = window_minutes + max(0, warmup_minutes)
        missing_window = 0
        missing_warmup = 0
        total = 0.0
        for i in range(needed):
            close_ms = end_ms - i * MINUTE_MS
            v = self._deltas.get(close_ms)
            if v is None:
                if i < window_minutes:
                    missing_window += 1
                else:
                    missing_warmup += 1
            else:
                # Only sum the last window_minutes
                if i < window_minutes:
                    total += float(v)

        missing = missing_window + missing_warmup
        if missing > 0:
            return DeltaSumResult(
                ok=False,
                reason=(
                    "insufficient_window" if missing_window > 0 else "insufficient_warmup"
                ),
                delta_sum=0.0,
                minutes_found=needed - missing,
                minutes_needed=needed,
            )

        return DeltaSumResult(
            ok=True,
            reason="ok",
            delta_sum=total,
            minutes_found=needed,
            minutes_needed=needed,
        )

    def snapshot_tail(self, n: int = 20) -> Tuple[Tuple[int, float], ...]:
        """Debug helper: last N (close_ms, delta) pairs."""
        n = max(1, int(n))
        closes = list(self._closes)[-n:]
        return tuple((c, float(self._deltas.get(c, 0.0))) for c in closes)
