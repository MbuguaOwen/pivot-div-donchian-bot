from div_donchian_bot.orderflow.delta_store import Delta1mStore


def test_delta_store_basic_sum_and_signed() -> None:
    ds = Delta1mStore(max_minutes=100)

    # Minute 0: close_ms=59,999
    ds.update_trade(ts_ms=30_000, qty=10.0, is_buyer_maker=False)  # buyer aggressor -> +10
    ds.update_trade(ts_ms=31_000, qty=5.0, is_buyer_maker=True)    # seller aggressor -> -5

    # Minute 1: close_ms=119,999
    ds.update_trade(ts_ms=61_000, qty=4.0, is_buyer_maker=False)   # +4

    res = ds.get_delta_sum(end_ms=119_999, window_minutes=2, warmup_minutes=0)
    assert res.ok
    assert abs(res.delta_sum - (5.0 + 4.0)) < 1e-9

    signed_long = res.delta_sum
    signed_short = -res.delta_sum
    assert signed_long > 0
    assert signed_short < 0


def test_delta_store_gap_fill_and_gap_flag() -> None:
    ds = Delta1mStore(max_minutes=100)
    ds.update_trade(ts_ms=10_000, qty=3.0, is_buyer_maker=False)       # minute 0
    ds.update_trade(ts_ms=130_000, qty=2.0, is_buyer_maker=True)       # minute 2 (gap: minute 1)

    assert ds.had_gap is True

    res = ds.get_delta_sum(end_ms=179_999, window_minutes=3, warmup_minutes=0)
    assert res.ok
    # minute0: +3, minute1: 0, minute2: -2
    assert abs(res.delta_sum - (3.0 + 0.0 - 2.0)) < 1e-9


def test_delta_store_requires_minute_close_end_ms() -> None:
    ds = Delta1mStore(max_minutes=100)
    ds.update_trade(ts_ms=10_000, qty=1.0, is_buyer_maker=False)
    res = ds.get_delta_sum(end_ms=60_000, window_minutes=1, warmup_minutes=0)
    assert res.ok is False
    assert res.reason == "end_ms_not_minute_close"


def test_delta_store_warmup_requirement() -> None:
    ds = Delta1mStore(max_minutes=100)
    ds.update_trade(ts_ms=10_000, qty=1.0, is_buyer_maker=False)
    ds.update_trade(ts_ms=70_000, qty=1.0, is_buyer_maker=False)
    res = ds.get_delta_sum(end_ms=119_999, window_minutes=1, warmup_minutes=3)
    assert res.ok is False
    assert res.reason == "insufficient_warmup"


def test_delta_store_quote_metrics() -> None:
    ds = Delta1mStore(max_minutes=100)
    # minute 0
    ds.update_trade(ts_ms=30_000, qty=1.0, is_buyer_maker=False, price=100.0)  # +100 quote, +100 quote delta
    # minute 1
    ds.update_trade(ts_ms=90_000, qty=2.0, is_buyer_maker=True, price=50.0)   # +100 quote, -100 quote delta

    end_ms = 119_999
    qsum = ds.get_quote_sum(end_ms=end_ms, window_minutes=2, warmup_minutes=0)
    assert qsum.ok
    assert abs(qsum.delta_sum - 200.0) < 1e-9

    qdelta = ds.get_quote_delta_sum(end_ms=end_ms, window_minutes=2, warmup_minutes=0)
    assert qdelta.ok
    assert abs(qdelta.delta_sum - 0.0) < 1e-9

    ratio = ds.get_quote_delta_ratio(end_ms=end_ms, window_minutes=2, warmup_minutes=0)
    assert ratio.ok
    assert abs(ratio.delta_sum - 0.0) < 1e-9
