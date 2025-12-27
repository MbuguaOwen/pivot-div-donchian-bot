import pytest

from div_donchian_bot.risk.structure_atr_trailing import (
    StructureAtrTrailParams,
    StopState,
    compute_initial_sl,
    update_extremes,
    update_stop,
)


def _make_params() -> StructureAtrTrailParams:
    return StructureAtrTrailParams(
        enabled=True,
    )


def test_compute_initial_sl_long_struct_vs_cap():
    params = _make_params()
    sl0, R = compute_initial_sl(
        entry=100.0,
        side="LONG",
        struct_low=90.0,
        struct_high=110.0,
        atr=3.0,
        params=params,
    )
    assert sl0 == pytest.approx(95.2)
    assert R == pytest.approx(4.8)


def test_compute_initial_sl_short_struct_vs_cap():
    params = _make_params()
    sl0, R = compute_initial_sl(
        entry=100.0,
        side="SHORT",
        struct_low=80.0,
        struct_high=102.0,
        atr=3.0,
        params=params,
    )
    expected = 102.0 * 1.001
    assert sl0 == pytest.approx(expected)
    assert R == pytest.approx(expected - 100.0)


def test_update_stop_be_then_lock_then_trail_long():
    params = _make_params()
    state = StopState(
        side="LONG",
        entry=100.0,
        sl=95.0,
        R=5.0,
        peak=100.0,
        trough=100.0,
        be_done=False,
        trailing_active=False,
    )

    update_extremes(state, high=108.0)
    be_sl = update_stop(state, atr=2.0, params=params)
    assert state.be_done is True
    assert state.trailing_active is False
    assert be_sl == pytest.approx(100.0 * 1.001)

    update_extremes(state, high=110.0)
    lock_sl = update_stop(state, atr=2.0, params=params)
    assert state.trailing_active is True
    assert lock_sl == pytest.approx(106.5)

    update_extremes(state, high=115.0)
    trail_sl = update_stop(state, atr=2.0, params=params)
    assert trail_sl == pytest.approx(111.5)


def test_update_stop_monotonic_long():
    params = _make_params()
    state = StopState(
        side="LONG",
        entry=100.0,
        sl=108.0,
        R=5.0,
        peak=112.0,
        trough=100.0,
        be_done=True,
        trailing_active=True,
    )
    new_sl = update_stop(state, atr=10.0, params=params)
    assert new_sl == pytest.approx(108.0)


def test_update_stop_monotonic_short():
    params = _make_params()
    state = StopState(
        side="SHORT",
        entry=100.0,
        sl=95.0,
        R=1.0,
        peak=100.0,
        trough=98.0,
        be_done=True,
        trailing_active=True,
    )
    new_sl = update_stop(state, atr=10.0, params=params)
    assert new_sl == pytest.approx(95.0)


def test_offset_max_vol_vs_spacing():
    params = _make_params()
    state = StopState(
        side="LONG",
        entry=100.0,
        sl=95.0,
        R=5.0,
        peak=110.0,
        trough=100.0,
        be_done=True,
        trailing_active=True,
    )
    new_sl = update_stop(state, atr=2.0, params=params)
    # spacing_r * R = 0.7 * 5 = 3.5 governs because k_trail*ATR=3.2
    assert new_sl == pytest.approx(106.5)
