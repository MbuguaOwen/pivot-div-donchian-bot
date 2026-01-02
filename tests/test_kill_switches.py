from div_donchian_bot.filters.kill_switches import KillDecision, parse_kill_switches, evaluate_kill_switch


class DummySig:
    def __init__(self, side: str) -> None:
        self.side = side


def test_parse_defaults_and_feature_config() -> None:
    cfg = parse_kill_switches({})
    assert cfg.enabled is False
    assert cfg.mode == "off"
    assert cfg.missing_policy == "pass"
    assert cfg.feature_config["atr_len_15m"] == 24
    assert cfg.feature_config["liq_lookback_bars"] == 240
    assert cfg.feature_config["cache_size"] == 50


def test_all_any_semantics() -> None:
    raw = {
        "enabled": True,
        "mode": "filter",
        "rules": [
            {
                "name": "combo",
                "side": "BOTH",
                "all": [{"feature": "f1", "op": ">", "value": 1}],
                "any": [{"feature": "f2", "op": "<", "value": 0}, {"feature": "f3", "op": "==", "value": 10}],
            }
        ],
    }
    cfg = parse_kill_switches(raw)
    sig = DummySig("LONG")

    dec = evaluate_kill_switch(sig, {"f1": 2, "f2": 1, "f3": 5}, cfg)
    assert dec.passed is True

    dec = evaluate_kill_switch(sig, {"f1": 2, "f2": -1, "f3": 5}, cfg)
    assert dec.passed is False
    assert "combo" in dec.hit_rules


def test_missing_policy_fail_vs_pass() -> None:
    raw = {
        "enabled": True,
        "mode": "filter",
        "missing_policy": "fail",
        "rules": [
            {"name": "need", "side": "LONG", "all": [{"feature": "need_this", "op": ">", "value": 0}]}
        ],
    }
    cfg = parse_kill_switches(raw)
    sig = DummySig("LONG")
    dec = evaluate_kill_switch(sig, {}, cfg)
    assert dec.passed is False
    assert "need" in dec.hit_rules
    assert "need_this" in dec.missing_features

    cfg_pass = parse_kill_switches({**raw, "missing_policy": "pass"})
    dec2 = evaluate_kill_switch(sig, {}, cfg_pass)
    assert dec2.passed is True
    assert dec2.hit_rules == []


def test_operator_matrix() -> None:
    def run(op: str, value, value2, feature_val, expect_hit: bool) -> None:
        raw = {
            "enabled": True,
            "mode": "filter",
            "rules": [{"name": op, "side": "BOTH", "all": [{"feature": "x", "op": op, "value": value, "value2": value2}]}],
        }
        cfg = parse_kill_switches(raw)
        dec = evaluate_kill_switch(DummySig("SHORT"), {"x": feature_val}, cfg)
        assert (not dec.passed) is expect_hit

    run(">", 1, None, 2, True)
    run(">=", 1, None, 1, True)
    run("<", 2, None, 1, True)
    run("<=", 2, None, 2, True)
    run("==", 5, None, 5, True)
    run("!=", 5, None, 4, True)
    run("between", 1, 3, 2, True)
    run("in", [1, 2, 3], None, 2, True)
    run("not_in", [1, 2, 3], None, 4, True)
    run("abs_gt", 1, None, -2, True)
    run("abs_lt", 2, None, -1, True)


def test_side_matching() -> None:
    raw = {
        "enabled": True,
        "mode": "filter",
        "rules": [{"name": "short_only", "side": "SHORT", "all": [{"feature": "x", "op": ">", "value": 0}]}],
    }
    cfg = parse_kill_switches(raw)
    long_dec = evaluate_kill_switch(DummySig("LONG"), {"x": 5}, cfg)
    assert long_dec.passed is True
    short_dec = evaluate_kill_switch(DummySig("SHORT"), {"x": 5}, cfg)
    assert short_dec.passed is False
    assert "short_only" in short_dec.hit_rules


def test_shadow_mode_allows() -> None:
    raw = {
        "enabled": True,
        "mode": "shadow",
        "rules": [{"name": "shadow", "side": "BOTH", "all": [{"feature": "x", "op": "abs_gt", "value": 1}]}],
    }
    cfg = parse_kill_switches(raw)
    dec = evaluate_kill_switch(DummySig("SHORT"), {"x": -2}, cfg)
    assert isinstance(dec, KillDecision)
    assert dec.passed is False
    assert "shadow" in dec.hit_rules
