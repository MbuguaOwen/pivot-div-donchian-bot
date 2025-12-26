from div_donchian_bot.strategy.pivot import is_pivot_low, is_pivot_high


def test_pivot_low_rejects_ties():
    lows = [5, 4, 4, 6, 7]
    # idx 1 and 2 are equal mins; with left_right=1 neither should be pivot
    assert is_pivot_low(lows, 1, 1) is False
    assert is_pivot_low(lows, 2, 1) is False


def test_pivot_low_unique_min_accepts():
    lows = [5, 4, 3, 4, 5]
    assert is_pivot_low(lows, 2, 1) is True


def test_pivot_high_rejects_ties():
    highs = [7, 6, 6, 5]
    assert is_pivot_high(highs, 1, 1) is False
    assert is_pivot_high(highs, 2, 1) is False


def test_pivot_high_unique_max_accepts():
    highs = [5, 7, 6, 4]
    assert is_pivot_high(highs, 1, 1) is True
