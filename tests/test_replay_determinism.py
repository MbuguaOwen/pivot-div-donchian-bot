from div_donchian_bot.cli.replay import _replay_strategy
from div_donchian_bot.models import Bar
from div_donchian_bot.strategy.pivot_div_donchian import StrategyParams


def test_strategy_replay_is_deterministic():
    bars = [
        Bar(open_time_ms=0, open=1, high=2, low=0.5, close=1.5, volume=10, close_time_ms=59_999),
        Bar(open_time_ms=60_000, open=1.5, high=2.0, low=1.0, close=1.8, volume=9, close_time_ms=119_999),
        Bar(open_time_ms=120_000, open=1.8, high=2.2, low=1.4, close=2.0, volume=8, close_time_ms=179_999),
        Bar(open_time_ms=180_000, open=2.0, high=2.4, low=1.6, close=2.2, volume=7, close_time_ms=239_999),
    ]
    params = StrategyParams(
        don_len=2,
        ext_band_pct=0.5,
        pivot_len=1,
        osc_ema_len=1,
        divergence_mode="pine",
        warmup_bars=0,
    )
    sig1, metrics1 = _replay_strategy(bars, [], params, enable_cvd=False, symbol="TEST", expected=set())
    sig2, metrics2 = _replay_strategy(bars, [], params, enable_cvd=False, symbol="TEST", expected=set())
    assert sig1 == sig2
    assert metrics1 == metrics2
