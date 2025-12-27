from div_donchian_bot.alerts import formatters
from div_donchian_bot.models import Signal
from div_donchian_bot.strategy.pivot_div_donchian import StrategyParams


def test_telegram_html_formatting():
    sig = Signal(
        symbol="TEST<USDT",
        side="LONG",
        entry_price=123.456,
        pivot_price=120.0,
        pivot_osc_value=1.234,
        pivot_cvd_value=None,
        slip_bps=10.5,
        loc_at_pivot=0.1,
        oscillator_name="pine",
        pine_div=True,
        cvd_div=False,
        pivot_time_ms=1700000000000,
        confirm_time_ms=1700000900000,
    )
    strat_params = StrategyParams(don_len=120, ext_band_pct=0.10, pivot_len=5, osc_ema_len=14, divergence_mode="pine")
    stop_info = {
        "enabled": True,
        "ready": True,
        "htf_interval": "1h",
        "htf_lookback_bars": 72,
        "atr_len": 24,
        "atr": 1.2345,
        "buffer_bps": 10,
        "k_init": 1.8,
        "tp_r_mult": 2.0,
        "be_trigger_r": 1.0,
        "be_buffer_bps": 10,
        "sl0": 120.0,
        "tp0": 126.0,
        "R": 3.456,
        "trailing_enabled": False,
        "trail_trigger_r": 2.5,
        "k_trail": 1.6,
        "lock_r": 1.0,
    }
    msg = formatters.format_entry(
        branding="Pivot Bot",
        sig=sig,
        strat_params=strat_params,
        stop_info=stop_info,
        divergence_mode="pine",
        timeframe="15m",
        testnet=True,
        mode="paper",
        exec_mode="paper",
        cooldown_minutes=30,
        max_positions_total=20,
        one_pos_per_symbol=True,
        notional_usdt=25.0,
    )
    assert "<b>" in msg
    assert "<pre>" in msg
    assert "TEST&lt;USDT" in msg  # escaped
    assert "MODE: PAPER" in msg
    assert "Engine: True ready=True" in msg
    assert "BE: 1.0R @ 10" in msg
    assert "SL0:" in msg and "TP0:" in msg
