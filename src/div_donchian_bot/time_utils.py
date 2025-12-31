from __future__ import annotations

def canonical_close_ms(ms: int) -> int:
    """Canonical close timestamp (already TV-style: end-of-bar minus 1ms)."""
    return int(ms)


def binance_kline_close_to_confirm_ms(close_ms: int) -> int:
    """Convert Binance kline close time (T) to canonical confirm_time_ms."""
    try:
        return int(close_ms) - 1
    except Exception:
        return int(close_ms)
