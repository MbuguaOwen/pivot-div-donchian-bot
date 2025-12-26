from div_donchian_bot.direction import DirectionGate


def test_direction_gate_long_only_blocks_short():
    gate = DirectionGate("long_only")
    assert gate.allow("LONG") is True
    assert gate.allow("SHORT") is False


def test_direction_gate_short_only_blocks_long():
    gate = DirectionGate("short_only")
    assert gate.allow("SHORT") is True
    assert gate.allow("LONG") is False


def test_direction_gate_both_allows_both():
    gate = DirectionGate("both")
    assert gate.allow("LONG") is True
    assert gate.allow("SHORT") is True
