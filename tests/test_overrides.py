from pathlib import Path

from div_donchian_bot.engine import BotEngine


def base_cfg(tmp_path: Path) -> dict:
    return {
        "_config_dir": str(tmp_path),
        "run": {"mode": "paper", "timeframe": "15m", "heartbeat_seconds": 30},
        "exchange": {"market_type": "futures", "testnet": True, "recv_window_ms": 5000},
        "universe": {"dynamic": True, "quote_asset": "USDT", "pair_overrides_dir": "pairs", "symbols": []},
        "strategy": {
            "donchian": {"length": 120, "extreme_band_pct": 0.10},
            "pivot": {"length": 5},
            "oscillator": {"ema_length": 14},
            "divergence": {"mode": "pine"},
            "fire_on_close_only": True,
        },
        "risk": {"atr": {"enabled": False, "length": 14, "sl_mult": 2.0, "tp_mult": 3.0}},
        "orders": {"notional_usdt": 25.0, "leverage": 3, "order_type": "MARKET"},
        "position": {"one_position_per_symbol": True, "allow_flip": False},
        "limits": {"max_positions_total": 20, "cooldown_minutes_per_symbol": 30},
        "telegram": {"enabled": False},
    }


def test_pair_override_dynamic_universe(tmp_path):
    cfg = base_cfg(tmp_path)
    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir()
    # Override notional for BTCUSDT
    (pairs_dir / "BTCUSDT.yaml").write_text(
        "orders:\n  notional_usdt: 99.0\nstrategy:\n  divergence:\n    mode: cvd\n", encoding="utf-8"
    )

    engine = BotEngine(cfg)
    scfg = engine._build_symbol_config("BTCUSDT")
    assert scfg.orders_cfg["notional_usdt"] == 99.0
    assert scfg.divergence_mode == "cvd"


def test_stream_selection_cvd_only(tmp_path):
    cfg = base_cfg(tmp_path)
    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir()
    (pairs_dir / "BUSDT.yaml").write_text("strategy:\n  divergence:\n    mode: cvd\n", encoding="utf-8")
    (pairs_dir / "CUSDT.yaml").write_text("strategy:\n  divergence:\n    mode: both\n", encoding="utf-8")

    engine = BotEngine(cfg)
    symbols = ["AUSDT", "BUSDT", "CUSDT"]
    engine.symbol_cfgs = {s: engine._build_symbol_config(s) for s in symbols}
    trade_streams = [f"{s.lower()}@aggTrade" for s in symbols if engine.symbol_cfgs[s].enable_cvd]

    assert set(trade_streams) == {"busdt@aggTrade", "cusdt@aggTrade"}
