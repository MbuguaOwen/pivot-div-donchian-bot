import pytest

from div_donchian_bot.strategy.indicators import EmaState


def test_ema_first_seed():
    ema = EmaState(3, seed_mode="first")
    inputs = [1.0, 2.0, 3.0, 4.0]
    outputs = [ema.update(x) for x in inputs]
    assert outputs == pytest.approx([1.0, 1.5, 2.25, 3.125])


def test_ema_sma_seed():
    ema = EmaState(3, seed_mode="sma")
    inputs = [1.0, 2.0, 3.0, 4.0]
    outputs = [ema.update(x) for x in inputs]
    assert outputs == pytest.approx([1.0, 1.5, 2.0, 3.0])
