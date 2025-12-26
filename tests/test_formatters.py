from div_donchian_bot.alerts import formatters
from div_donchian_bot.models import Signal
from div_donchian_bot.strategy.pivot_div_donchian import StrategyParams
from div_donchian_bot.risk.atr_risk import AtrRiskParams


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
    atr_params = AtrRiskParams(enabled=False, length=14, sl_mult=2.0, tp_mult=3.0)
    msg = formatters.format_entry(
        branding="Pivot Bot",
        sig=sig,
        atr=1.2345,
        strat_params=strat_params,
        atr_params=atr_params,
        divergence_mode="pine",
        timeframe="15m",
        testnet=True,
        mode="paper",
        cooldown_minutes=30,
        max_positions_total=20,
        one_pos_per_symbol=True,
        notional_usdt=25.0,
    )
    assert "<b>" in msg
    assert "<pre>" in msg
    assert "TEST&lt;USDT" in msg  # escaped
    assert "MODE: PAPER" in msg
