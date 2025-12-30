from __future__ import annotations

from typing import Iterable, List

from ..models import Bar


def timeframe_to_ms(tf: str) -> int:
    """Convert timeframe strings like '1m' or '4h' to milliseconds."""
    t = tf.strip().lower()
    if t.endswith("ms"):
        return int(t[:-2]) if t[:-2] else 1
    if t.endswith("s"):
        return int(t[:-1]) * 1000
    if t.endswith("m"):
        return int(t[:-1]) * 60_000
    if t.endswith("h"):
        return int(t[:-1]) * 60 * 60_000
    raise ValueError(f"Unsupported timeframe: {tf}")


def infer_interval_ms(bars: Iterable[Bar]) -> int:
    """Infer the base bar interval (milliseconds) from a sequence of bars."""
    arr = list(bars)
    if not arr:
        raise ValueError("Cannot infer interval from empty bar list")
    arr.sort(key=lambda b: b.open_time_ms)
    durations = []
    first_span = arr[0].close_time_ms - arr[0].open_time_ms + 1
    if first_span > 0:
        durations.append(first_span)
    for i in range(1, len(arr)):
        delta = arr[i].open_time_ms - arr[i - 1].open_time_ms
        if delta > 0:
            durations.append(delta)
    if not durations:
        raise ValueError("Unable to infer bar spacing")
    return min(durations)


def resample_bars(bars: List[Bar], target_tf: str) -> List[Bar]:
    """Downsample bars to a coarser timeframe preserving Binance timestamps."""
    if not bars:
        return []
    target_ms = timeframe_to_ms(target_tf)
    src_ms = infer_interval_ms(bars)
    if target_ms < src_ms:
        raise ValueError(f"Cannot upsample from {src_ms}ms to {target_ms}ms")
    if target_ms == src_ms:
        return sorted(bars, key=lambda b: b.open_time_ms)

    out: List[Bar] = []
    bars_sorted = sorted(bars, key=lambda b: b.open_time_ms)
    bucket_start = (bars_sorted[0].open_time_ms // target_ms) * target_ms
    bucket_end = bucket_start + target_ms - 1
    o = bars_sorted[0].open
    h = bars_sorted[0].high
    l = bars_sorted[0].low
    c = bars_sorted[0].close
    v = bars_sorted[0].volume

    for b in bars_sorted[1:]:
        if b.open_time_ms <= bucket_end:
            h = max(h, b.high)
            l = min(l, b.low)
            c = b.close
            v += b.volume
            continue
        out.append(Bar(
            open_time_ms=bucket_start,
            open=o,
            high=h,
            low=l,
            close=c,
            volume=v,
            close_time_ms=bucket_end,
        ))
        bucket_start = (b.open_time_ms // target_ms) * target_ms
        bucket_end = bucket_start + target_ms - 1
        o, h, l, c, v = b.open, b.high, b.low, b.close, b.volume

    out.append(Bar(
        open_time_ms=bucket_start,
        open=o,
        high=h,
        low=l,
        close=c,
        volume=v,
        close_time_ms=bucket_end,
    ))
    return out
