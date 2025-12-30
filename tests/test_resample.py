from div_donchian_bot.data.resample import resample_bars
from div_donchian_bot.models import Bar


def test_resample_close_time_is_end_minus_one():
    bars = [
        Bar(open_time_ms=0, open=1, high=2, low=0.5, close=1.5, volume=10, close_time_ms=59_999),
        Bar(open_time_ms=60_000, open=1.5, high=2.0, low=1.0, close=1.8, volume=9, close_time_ms=119_999),
        Bar(open_time_ms=120_000, open=1.8, high=2.2, low=1.4, close=2.0, volume=8, close_time_ms=179_999),
    ]
    res = resample_bars(bars, "3m")
    assert len(res) == 1
    bar = res[0]
    assert bar.open_time_ms == 0
    assert bar.close_time_ms == 180_000 - 1
    assert bar.high == 2.2
    assert bar.low == 0.5
