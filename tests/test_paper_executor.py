import asyncio

from div_donchian_bot.execution.paper import PaperExecutor, PaperConfig
from div_donchian_bot.execution.base import ExecFill


def _mk_cfg():
    return PaperConfig(
        fee_bps=4.0,
        slippage_bps=1.5,
        max_wait_ms=1000,
        fill_policy="next_tick",
        use_mark_price=False,
        log_trades=False,
    )


def test_next_tick_fill_long_applies_slippage_and_fee(monkeypatch):
    fills = []
    exits = []
    cfg = _mk_cfg()
    exe = PaperExecutor(cfg, on_exit=lambda *args: exits.append(args), on_fill=lambda f: fills.append(f))
    monkeypatch.setattr("time.time", lambda: 0)

    async def run():
        await exe.place_entry("BTCUSDT", "LONG", 1.0, ref="t")
        await exe.on_trade_tick("BTCUSDT", 100.0, 0)
    asyncio.run(run())

    assert len(fills) == 1
    fill: ExecFill = fills[0]
    assert abs(fill.price - (100.0 * (1 + 0.00015))) < 1e-9
    assert abs(fill.fee_paid - (fill.price * 1.0 * 0.0004)) < 1e-9
    assert exits == []


def test_next_tick_fill_short_applies_slippage_and_fee(monkeypatch):
    fills = []
    cfg = _mk_cfg()
    exe = PaperExecutor(cfg, on_exit=lambda *args: None, on_fill=lambda f: fills.append(f))
    monkeypatch.setattr("time.time", lambda: 0)

    async def run():
        await exe.place_entry("ETHUSDT", "SHORT", 2.0, ref="t")
        await exe.on_trade_tick("ETHUSDT", 50.0, 0)
    asyncio.run(run())

    fill = fills[0]
    assert abs(fill.price - (50.0 * (1 - 0.00015))) < 1e-9
    assert abs(fill.fee_paid - (fill.price * 2.0 * 0.0004)) < 1e-9


def test_stop_exit_long(monkeypatch):
    exits = []
    cfg = _mk_cfg()
    exe = PaperExecutor(cfg, on_exit=lambda *args: exits.append(args), on_fill=lambda f: None)
    monkeypatch.setattr("time.time", lambda: 0)

    async def run():
        await exe.place_entry("SOLUSDT", "LONG", 1.0, ref="t")
        await exe.on_trade_tick("SOLUSDT", 10.0, 0)
        await exe.place_or_replace_stop("SOLUSDT", "LONG", 1.0, 9.5, reason="init", tick_size=0.01)
        await exe.on_trade_tick("SOLUSDT", 9.4, 1)
    asyncio.run(run())

    assert exits
    sym, reason, exit_px, pnl, ts = exits[0]
    assert sym == "SOLUSDT"
    assert reason == "STOP"
    assert exit_px < 9.5  # slippage against trader


def test_tp_exit_long(monkeypatch):
    exits = []
    cfg = _mk_cfg()
    exe = PaperExecutor(cfg, on_exit=lambda *args: exits.append(args), on_fill=lambda f: None)
    monkeypatch.setattr("time.time", lambda: 0)

    async def run():
        await exe.place_entry("ADAUSDT", "LONG", 1.0, ref="t")
        await exe.on_trade_tick("ADAUSDT", 1.0, 0)
        await exe.place_or_replace_tp("ADAUSDT", "LONG", 1.0, 1.2, tick_size=0.01)
        await exe.on_trade_tick("ADAUSDT", 1.21, 1)
    asyncio.run(run())

    assert exits
    sym, reason, exit_px, pnl, ts = exits[0]
    assert reason == "TP"
    assert exit_px < 1.21  # slippage against trader on TP too


def test_stop_engine_updates_stop_px_paper(monkeypatch):
    cfg = _mk_cfg()
    exe = PaperExecutor(cfg, on_exit=lambda *args: None, on_fill=lambda f: None)
    monkeypatch.setattr("time.time", lambda: 0)

    async def run():
        await exe.place_entry("XRPUSDT", "LONG", 1.0, ref="t")
        await exe.on_trade_tick("XRPUSDT", 0.5, 0)
        res = await exe.place_or_replace_stop("XRPUSDT", "LONG", 1.0, 0.45, reason="init", tick_size=0.001)
        pos = await exe.get_position("XRPUSDT")
        return res, pos

    res, pos = asyncio.run(run())
    assert res.ok
    assert pos["stop_px"] == 0.45


def test_pending_fill_timeout_bar_close_fallback(monkeypatch):
    fills = []
    cfg = _mk_cfg()
    cfg.fill_policy = "bar_close_fallback"
    exe = PaperExecutor(cfg, on_exit=lambda *args: None, on_fill=lambda f: fills.append(f))
    monkeypatch.setattr("time.time", lambda: 0)

    async def run():
        await exe.place_entry("DOTUSDT", "LONG", 1.0, ref="t")
        await exe.on_bar_close("DOTUSDT", 6.0, ts_ms=2000)
    asyncio.run(run())

    assert fills
    assert abs(fills[0].price - (6.0 * (1 + 0.00015))) < 1e-9
