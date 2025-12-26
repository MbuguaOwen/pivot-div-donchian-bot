from div_donchian_bot.alerts import formatters
from div_donchian_bot.models import Signal
from div_donchian_bot.strategy.pivot_div_donchian import StrategyParams


def test_telegram_html_formatting():
    sig = Signal(
        symbol="TEST<USDT",
        side="LONG",
        entry_price=123.456,
        pivot_price=120.0,
        slip_bps=10.5,
        loc_at_pivot=0.1,
        oscillator_name="pine",
        pivot_time_ms=1700000000000,
        confirm_time_ms=1700000900000,
    )
    strat_params = StrategyParams(don_len=120, ext_band_pct=0.10, pivot_len=5, osc_ema_len=14, divergence_mode="pine")
    msg = formatters.format_entry_signal(
        branding="Pivot Bot",
        sig=sig,
        atr=1.2345,
        strat_params=strat_params,
        divergence_mode="pine",
        timeframe="15m",
        mode="paper",
        cooldown_minutes=30,
        max_positions_total=20,
        one_pos_per_symbol=True,
    )
    assert "<b>" in msg
    assert "<pre>" in msg
    assert "TEST&lt;USDT" in msg  # escaped
    assert "Mode: PAPER" in msg
