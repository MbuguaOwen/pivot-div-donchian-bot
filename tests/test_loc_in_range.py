from div_donchian_bot.strategy.indicators import loc_in_range


def test_loc_zero_range_returns_midpoint():
    assert loc_in_range(close=10.0, lo=10.0, hi=10.0) == 0.5


def test_loc_normal_range():
    assert loc_in_range(close=15.0, lo=10.0, hi=20.0) == 0.5
    assert loc_in_range(close=12.0, lo=10.0, hi=20.0) == 0.2
