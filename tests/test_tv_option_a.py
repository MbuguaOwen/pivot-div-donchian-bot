import asyncio
import time
from collections import deque

from div_donchian_bot.binance.rest import SymbolFilters
from div_donchian_bot.engine import BotEngine, SymbolRuntime
from div_donchian_bot.models import Bar
from div_donchian_bot.execution.base import ExecutionAdapter, ExecFill, ExecResult
from div_donchian_bot.strategy.pivot_div_donchian import SymbolStrategyState
from div_donchian_bot.tv_bridge import TvSignal


class DummyRest:
    def __init__(self, filters: SymbolFilters):
        self.filters = filters
        self.leverage_calls = []

    async def get_symbol_filters(self, symbol: str) -> SymbolFilters:
        return self.filters

    async def set_leverage(self, symbol: str, lev: int) -> None:
        self.leverage_calls.append((symbol, lev))


class DummyExecutor(ExecutionAdapter):
    def __init__(self, price: float = 100.0):
        self.price = price
        self.entries = []
        self.stops = []
        self.tps = []
        self.cancels = []
        self.closed = []

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def place_entry(self, symbol: str, side: str, qty: float, ref: str) -> ExecResult:
        self.entries.append((symbol, side, qty, ref))
        fill = ExecFill(symbol=symbol, side=side, qty=qty, price=self.price, fee_paid=0.0, ts_ms=int(time.time() * 1000), mode="binance")
        return ExecResult(ok=True, msg="FILLED", fill=fill, order_ids={"entry": f"e{len(self.entries)}"})

    async def place_or_replace_stop(self, symbol: str, side: str, qty: float, stop_px: float, reason: str, tick_size: float) -> ExecResult:
        self.stops.append((symbol, side, qty, stop_px, reason))
        return ExecResult(ok=True, msg="stop_set", order_ids={"stop": f"sl{len(self.stops)}"})

    async def place_or_replace_tp(self, symbol: str, side: str, qty: float, tp_px: float, tick_size: float) -> ExecResult:
        self.tps.append((symbol, side, qty, tp_px))
        return ExecResult(ok=True, msg="tp_set", order_ids={"tp": f"tp{len(self.tps)}"})

    async def close_position_market(self, symbol: str, entry_side: str, qty: float, price_hint: float | None = None) -> ExecResult:
        self.closed.append((symbol, entry_side, qty, price_hint))
        return ExecResult(ok=True, msg="closed")

    async def cancel_protection(self, symbol: str) -> None:
        self.cancels.append(symbol)

    async def on_trade_tick(self, symbol: str, price: float, ts_ms: int) -> None:
        return None

    async def on_bar_close(self, symbol: str, close: float, ts_ms: int) -> None:
        return None

    async def get_position(self, symbol: str) -> dict:
        return {}

    async def has_position(self, symbol: str) -> bool:
        return False


class DummyNotifier:
    def __init__(self):
        self.messages = []

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def send(self, text: str, reply_markup=None) -> None:
        self.messages.append(text)


class DummyStore:
    def __getattr__(self, name):
        # allow any log_* calls without side-effects
        def _noop(*args, **kwargs):
            return None

        return _noop


def _base_cfg():
    return {
        "run": {"mode": "live", "timeframe": "15m"},
        "exchange": {"market_type": "futures", "testnet": True},
        "execution": {"mode": "binance"},
        "strategy": {
            "direction": "both",
            "donchian": {"length": 20, "extreme_band_pct": 0.1},
            "pivot": {"length": 2},
            "oscillator": {"ema_length": 3},
            "divergence": {"mode": "pine"},
        },
        "risk": {
            "engine": {
                "enabled": True,
                "htf_interval": "1h",
                "htf_lookback_bars": 2,
                "atr_len": 2,
                "atr_floor_bps": 0,
                "buffer_bps": 10,
                "k_init": 1.0,
                "tp_r_mult": 2.0,
                "be_trigger_r": 1.0,
                "be_buffer_bps": 5,
                "trailing": {"enabled": False},
                "min_stop_replace_interval_sec": 0,
                "min_stop_tick_improvement": 0,
            }
        },
        "orders": {"notional_usdt": 50.0, "leverage": 1},
        "position": {"one_position_per_symbol": True, "allow_flip": False},
        "limits": {"max_positions_total": 5, "cooldown_minutes_per_symbol": 0},
        "tv_bridge": {"enabled": True, "mode": "tv_only", "require_tf_match": False},
        "telegram": {"enabled": False},
        "cvd": {"mode": "off"},
    }


def _setup_engine(symbol: str = "BTCUSDT", last_close_ms: int | None = None):
    cfg = _base_cfg()
    engine = BotEngine(cfg)
    scfg = engine._build_symbol_config(symbol)
    rt = SymbolRuntime(
        cfg=scfg,
        strat=SymbolStrategyState(symbol, scfg.strategy_params, enable_cvd=scfg.enable_cvd),
        h1_bars=deque(maxlen=10),
    )
    rt.stop_ready = True
    rt.h1_atr_value = 1.0
    rt.h1_bars.extend(
        [
            Bar(open_time_ms=0, open=100, high=105, low=95, close=102, volume=1, close_time_ms=3_599_999),
            Bar(open_time_ms=3_600_000, open=102, high=107, low=99, close=104, volume=1, close_time_ms=7_199_999),
        ]
    )
    rt.filters = SymbolFilters(step_size=0.001, min_qty=0.001, tick_size=0.1)
    rt.last_price = 100.0
    rt.last_kline_close_ms = last_close_ms
    engine.symbol_cfgs[symbol] = scfg
    engine.symbols_rt[symbol] = rt
    engine.symbols = [symbol]
    executor = DummyExecutor(price=100.0)
    engine.executor = executor
    engine.rest = DummyRest(rt.filters)
    engine.notifier = DummyNotifier()
    engine.store = DummyStore()
    engine.cooldown_minutes = 0
    engine.max_positions_total = cfg["limits"]["max_positions_total"]
    return engine, rt, executor


def test_tv_webhook_dedupe_option_a():
    engine, rt, executor = _setup_engine()
    sig = TvSignal(symbol="BTCUSDT", side="LONG", confirm_time_ms=1_000, entry_price=100.0)
    asyncio.run(engine._on_tv_signal(sig))
    asyncio.run(engine._on_tv_signal(sig))
    assert len(executor.entries) == 1


def test_tv_time_parity_tracks_canonical_close():
    engine, rt, executor = _setup_engine()
    asyncio.run(engine._on_ws_message({"data": {"e": "kline", "k": {"x": True, "s": "BTCUSDT", "i": "15m", "t": 0, "T": 60_000, "o": "1", "c": "1", "h": "1", "l": "1", "v": "1"}}}))
    assert rt.last_kline_close_ms == 59_999
    sig = TvSignal(symbol="BTCUSDT", side="SHORT", confirm_time_ms=59_999, entry_price=100.0)
    asyncio.run(engine._on_tv_signal(sig))
    assert len(executor.entries) == 1


def test_tv_executes_without_waiting_for_kline():
    engine, rt, executor = _setup_engine()
    sig = TvSignal(symbol="BTCUSDT", side="LONG", confirm_time_ms=123_000, entry_price=100.0)
    asyncio.run(engine._on_tv_signal(sig))
    assert len(executor.entries) == 1


def test_tv_idempotent_when_already_in_position():
    engine, rt, executor = _setup_engine()
    rt.in_position = True
    sig = TvSignal(symbol="BTCUSDT", side="LONG", confirm_time_ms=200_000, entry_price=100.0)
    asyncio.run(engine._on_tv_signal(sig))
    assert len(executor.entries) == 0


def test_tv_cancel_protection_before_replacing():
    engine, rt, executor = _setup_engine()
    first = TvSignal(symbol="BTCUSDT", side="LONG", confirm_time_ms=300_000, entry_price=100.0)
    asyncio.run(engine._on_tv_signal(first))
    rt.in_position = False
    engine._positions_total = 0
    second = TvSignal(symbol="BTCUSDT", side="LONG", confirm_time_ms=900_000, entry_price=101.0)
    asyncio.run(engine._on_tv_signal(second))
    assert len(executor.cancels) == 2
    assert len(executor.stops) == 2
