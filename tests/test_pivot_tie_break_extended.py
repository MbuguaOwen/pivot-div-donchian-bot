from div_donchian_bot.strategy.pivot import is_pivot_low, is_pivot_high


def test_pivot_low_tv_like_picks_leftmost_flat():
    lows = [5, 4, 4, 6]
    assert is_pivot_low(lows, 1, 1, tie_break="tv_like") is True
    assert is_pivot_low(lows, 2, 1, tie_break="tv_like") is False
    assert is_pivot_low(lows, 1, 1, tie_break="strict") is False


def test_pivot_high_tv_like_triple_flat():
    highs = [6, 7, 7, 7, 6]
    assert is_pivot_high(highs, 1, 1, tie_break="tv_like") is True
    assert is_pivot_high(highs, 2, 1, tie_break="tv_like") is False
    assert is_pivot_high(highs, 3, 1, tie_break="tv_like") is False


def test_pivot_noise_still_detects_unique_extremes():
    lows = [3, 2, 3, 2, 4]
    assert is_pivot_low(lows, 1, 1, tie_break="tv_like") is True
    assert is_pivot_low(lows, 3, 1, tie_break="tv_like") is True
