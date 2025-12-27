import pytest

from div_donchian_bot.risk.structure_atr_trailing import (
    StructureAtrTrailParams,
    TrailingParams,
    StopState,
    compute_initial_sl,
    compute_tp_from_R,
    update_extremes,
    update_stop,
)


def _make_params(trailing_enabled: bool = False) -> StructureAtrTrailParams:
    return StructureAtrTrailParams(
        enabled=True,
        trailing=TrailingParams(enabled=trailing_enabled, trigger_r=2.5, k_trail=1.6, lock_r=1.0),
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
    assert sl0 == pytest.approx(94.6)
    assert R == pytest.approx(5.4)


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


def test_compute_tp_from_R_long_and_short():
    assert compute_tp_from_R(100.0, "LONG", 5.0, 2.0) == pytest.approx(110.0)
    assert compute_tp_from_R(100.0, "SHORT", 5.0, 2.0) == pytest.approx(90.0)


def test_update_stop_be_then_lock_then_trail_long():
    params = _make_params(trailing_enabled=True)
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
    assert state.trailing_active is False
    assert lock_sl == pytest.approx(100.0 * 1.001)

    update_extremes(state, high=115.0)
    trail_sl = update_stop(state, atr=2.0, params=params)
    assert state.trailing_active is True
    assert trail_sl == pytest.approx(107.5)


def test_update_stop_be_only_when_trailing_disabled():
    params = _make_params(trailing_enabled=False)
    state = StopState(
        side="SHORT",
        entry=50.0,
        sl=55.0,
        R=5.0,
        peak=50.0,
        trough=50.0,
        be_done=False,
        trailing_active=False,
    )
    update_extremes(state, low=40.0)
    be_sl = update_stop(state, atr=1.0, params=params)
    assert be_sl == pytest.approx(50.0 * (1 - 0.001))
    update_extremes(state, low=30.0)
    still_sl = update_stop(state, atr=1.0, params=params)
    assert still_sl == pytest.approx(be_sl)
    assert state.trailing_active is False


def test_update_stop_monotonic_long():
    params = _make_params(trailing_enabled=True)
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
    params = _make_params(trailing_enabled=True)
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
    params = _make_params(trailing_enabled=True)
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
    # spacing_r * R = 1.5 * 5 = 7.5 governs because k_trail*ATR=3.2
    assert new_sl == pytest.approx(105.0)
