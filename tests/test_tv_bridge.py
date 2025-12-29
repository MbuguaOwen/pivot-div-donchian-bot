from div_donchian_bot.tv_bridge import normalize_tv_symbol, normalize_tv_tf, TvBridgeConfig


def test_normalize_tv_symbol():
    assert normalize_tv_symbol("BTCUSDT") == "BTCUSDT"
    assert normalize_tv_symbol("BINANCE:BTCUSDT") == "BTCUSDT"
    assert normalize_tv_symbol("BINANCE:BTCUSDT.P") == "BTCUSDT"


def test_normalize_tv_tf():
    assert normalize_tv_tf("15") == "15m"
    assert normalize_tv_tf("15m") == "15m"
    assert normalize_tv_tf("1H") == "1h"
    assert normalize_tv_tf("60") == "60m"


def test_tv_bridge_config_from_dict():
    cfg = TvBridgeConfig.from_dict({"enabled": True, "mode": "tv_and_bot", "secret": "x"})
    assert cfg.enabled is True
    assert cfg.mode == "tv_and_bot"
    assert cfg.secret == "x"
