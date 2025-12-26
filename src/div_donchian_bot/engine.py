from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

from dotenv import load_dotenv

from .models import Bar, Signal
from .notifier import TelegramNotifier
from .storage.csv_store import CsvStore
from .strategy.pivot_div_donchian import StrategyParams, SymbolStrategyState
from .risk.atr_risk import AtrRiskParams, atr_levels
from .strategy.indicators import AtrState
from .binance.rest import BinanceRest, quantize
from .binance.ws import BinanceWsManager
from .config import deep_merge, load_pair_override
from .alerts import formatters
from .direction import DirectionGate
from .telegram_controls import TelegramControls, control_keyboard

log = logging.getLogger("engine")

def _tf_to_minutes(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    raise ValueError(f"Unsupported timeframe: {tf}")

@dataclass
class SymbolConfig:
    symbol: str
    cfg: Dict[str, Any]
    strategy_params: StrategyParams
    atr_params: AtrRiskParams
    enable_cvd: bool
    orders_cfg: Dict[str, Any]
    position_cfg: Dict[str, Any]
    divergence_mode: str

@dataclass
class SymbolRuntime:
    cfg: SymbolConfig
    strat: SymbolStrategyState
    atr: AtrState
    in_position: bool = False
    last_signal_ts_ms: int = 0

class BotEngine:
    def __init__(self, cfg: Dict[str, Any]):
        load_dotenv()

        self.cfg = cfg
        self.config_dir = Path(cfg.get("_config_dir", "."))
        self.mode = cfg["run"]["mode"]
        self.timeframe = cfg["run"].get("timeframe", "15m")
        self.enable_telegram = bool(cfg.get("telegram", {}).get("enabled", True))
        ex_cfg = cfg.get("exchange", {}) or {}
        self.testnet = bool(ex_cfg.get("testnet", True))
        self.market_type = str(ex_cfg.get("market_type", "futures"))
        tg_cfg = cfg.get("telegram", {}) or {}
        self.parse_mode = tg_cfg.get("parse_mode", "HTML")
        self.branding = tg_cfg.get("branding", "Pivot Div + Donchian")
        self.heartbeat_sec = int(tg_cfg.get("heartbeat_seconds", cfg["run"].get("heartbeat_seconds", 30)))
        self.heartbeat_enabled = bool(tg_cfg.get("heartbeat_enabled", True))
        self.pair_overrides_dir = (cfg.get("universe", {}) or {}).get("pair_overrides_dir", "configs/pairs")
        strat_cfg = cfg.get("strategy", {}) or {}
        self.direction_default = str(strat_cfg.get("direction", "both")).lower()
        self.direction_gate = DirectionGate(self.direction_default)
        self.controls: Optional[TelegramControls] = None
        self.controls_enabled = bool(tg_cfg.get("controls_enabled", False))
        self.controls_allowed_chat_id = tg_cfg.get("controls_allowed_chat_id")
        self.controls_state_path = tg_cfg.get("controls_state_path", "state/telegram_controls.json")

        self.rest: Optional[BinanceRest] = None
        self.ws: Optional[BinanceWsManager] = None
        self.notifier = TelegramNotifier(enabled=self.enable_telegram, parse_mode=self.parse_mode)
        self.store = CsvStore(cfg.get("storage", {}).get("out_dir", "artifacts"))

        self.symbols: List[str] = []
        self.symbols_rt: Dict[str, SymbolRuntime] = {}
        self.symbol_cfgs: Dict[str, SymbolConfig] = {}

        self._positions_total = 0
        self._last_signal_ts_ms: Optional[int] = None
        self._start_time = time.time()
        self._live_enabled = True
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def start(self, rest: BinanceRest, ws_url: str, symbols: List[str]) -> None:
        self.rest = rest
        self.symbols = symbols
        self._start_time = time.time()
        limits = self.cfg.get("limits", {}) or {}
        self.max_positions_total = int(limits.get("max_positions_total", 20))
        self.cooldown_minutes = int(limits.get("cooldown_minutes_per_symbol", 30))

        self.orders_cfg = self.cfg.get("orders", {}) or {}
        self.position_cfg = self.cfg.get("position", {}) or {}

        # live safety: refuse trading without keys
        if self.mode == "live" and (not rest.key or not rest.secret):
            self._live_enabled = False
            await self.notifier.send(formatters.format_error(self.branding, "LIVE DISABLED: missing keys", RuntimeError("missing BINANCE_API_KEY/SECRET")))
            raise RuntimeError("LIVE DISABLED: missing BINANCE_API_KEY/SECRET")

        await self.notifier.start()
        if self.controls_enabled and self.enable_telegram:
            allowed_chat = self.controls_allowed_chat_id
            if allowed_chat is None:
                try:
                    allowed_chat = int(self.notifier.chat_id)
                except Exception:
                    allowed_chat = None
            if allowed_chat is None:
                log.warning("Telegram controls enabled but no allowed_chat_id; controls not started")
            else:
                self.controls = TelegramControls(
                    token=self.notifier.token,
                    allowed_chat_id=allowed_chat,
                    state_path=self.controls_state_path,
                    gate=self.direction_gate,
                    notifier_send=self.notifier.send,
                    parse_mode=self.parse_mode,
                )
                await self.controls.start()

        # Build per-symbol runtime/config
        cvd_symbols: List[str] = []
        for sym in symbols:
            scfg = self._build_symbol_config(sym)
            self.symbol_cfgs[sym] = scfg
            if scfg.enable_cvd:
                cvd_symbols.append(sym)
            self.symbols_rt[sym] = SymbolRuntime(
                cfg=scfg,
                strat=SymbolStrategyState(sym, scfg.strategy_params, enable_cvd=scfg.enable_cvd),
                atr=AtrState(scfg.atr_params.length),
            )

        # Build streams
        tf = self.timeframe
        # Binance expects lowercase, e.g. btcusdt@kline_15m
        kline_streams = [f"{s.lower()}@kline_{tf}" for s in symbols]
        trade_streams = [f"{s.lower()}@aggTrade" for s in symbols if self.symbol_cfgs[s].enable_cvd]
        all_streams = kline_streams + trade_streams
        log.info("Streams: kline=%d aggTrade=%d cvd_symbols=%d", len(kline_streams), len(trade_streams), len(cvd_symbols))

        # Chunk streams to avoid overloading a single connection.
        # Binance allows many streams, but be conservative: 200 per connection.
        chunk = 200
        groups = [all_streams[i:i+chunk] for i in range(0, len(all_streams), chunk)]

        self.ws = BinanceWsManager(ws_url)
        self.ws.start(groups, self._on_ws_message)

        # Heartbeat
        if self.heartbeat_enabled:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        startup_markup = control_keyboard() if self.controls_enabled else None
        await self.notifier.send(
            formatters.format_startup(
                branding=self.branding,
                mode=self.mode,
                testnet=self.testnet,
                exchange="Binance",
                market_type=self.market_type,
                timeframe=self.timeframe,
                symbols=symbols,
                cvd_symbols=cvd_symbols,
                atr_enabled=bool(self.cfg.get("risk", {}).get("atr", {}).get("enabled", False)),
                max_positions_total=self.max_positions_total,
                cooldown_minutes=self.cooldown_minutes,
                heartbeat_sec=self.heartbeat_sec,
                direction=self.direction_gate.get_direction(),
            ),
            reply_markup=startup_markup,
        )

    def _build_symbol_config(self, symbol: str) -> SymbolConfig:
        override = load_pair_override(self.config_dir, self.pair_overrides_dir, symbol)
        merged = deep_merge(self.cfg, override)
        s_cfg = merged.get("strategy", {}) or {}
        divergence_mode = str(s_cfg.get("divergence", {}).get("mode", "pine")).lower()
        p = StrategyParams(
            don_len=int(s_cfg["donchian"]["length"]),
            ext_band_pct=float(s_cfg["donchian"]["extreme_band_pct"]),
            pivot_len=int(s_cfg["pivot"]["length"]),
            osc_ema_len=int(s_cfg["oscillator"]["ema_length"]),
            divergence_mode=divergence_mode,
        )
        enable_cvd = divergence_mode in ("cvd", "both", "either")

        r_cfg = merged.get("risk", {}).get("atr", {})
        atr_params = AtrRiskParams(
            enabled=bool(r_cfg.get("enabled", False)),
            length=int(r_cfg.get("length", 14)),
            sl_mult=float(r_cfg.get("sl_mult", 2.0)),
            tp_mult=float(r_cfg.get("tp_mult", 3.0)),
        )

        orders_cfg = merged.get("orders", {}) or {}
        position_cfg = merged.get("position", {}) or {}

        return SymbolConfig(
            symbol=symbol,
            cfg=merged,
            strategy_params=p,
            atr_params=atr_params,
            enable_cvd=enable_cvd,
            orders_cfg=orders_cfg,
            position_cfg=position_cfg,
            divergence_mode=divergence_mode,
        )

    async def stop(self) -> None:
        if self.ws:
            await self.ws.stop()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        if self.controls:
            await self.controls.stop()
        await self.notifier.stop()
        if self.rest:
            await self.rest.stop()

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_sec)
            last_sig = self._last_signal_ts_ms
            uptime = int(time.time() - self._start_time)
            await self.notifier.send(
                formatters.format_heartbeat(
                    branding=self.branding,
                    mode=self.mode,
                    testnet=self.testnet,
                    positions=self._positions_total,
                    max_positions=self.max_positions_total,
                    symbols_count=len(self.symbols),
                    last_signal_ts_ms=last_sig,
                    uptime_seconds=uptime,
                )
            )

    async def _on_ws_message(self, msg: dict) -> None:
        # Combined stream format: {"stream":"btcusdt@kline_15m","data":{...}}
        data = msg.get("data") or {}
        e = data.get("e")
        if e == "kline":
            k = data.get("k", {})
            if not k.get("x"):  # only closed bars
                return
            sym = k.get("s")
            if not sym or sym not in self.symbols_rt:
                return
            bar = Bar(
                open_time_ms=int(k["t"]),
                open=float(k["o"]),
                high=float(k["h"]),
                low=float(k["l"]),
                close=float(k["c"]),
                volume=float(k["v"]),
                close_time_ms=int(k["T"]),
            )
            await self._on_bar_close(sym, bar)

        elif e == "aggTrade":
            sym = data.get("s")
            if not sym or sym not in self.symbols_rt:
                return
            ts = int(data.get("T"))
            qty = float(data.get("q"))
            is_buyer_maker = bool(data.get("m"))
            self.symbols_rt[sym].strat.on_agg_trade(ts, qty, is_buyer_maker)

    async def _on_bar_close(self, symbol: str, bar: Bar) -> None:
        rt = self.symbols_rt[symbol]
        scfg = rt.cfg
        # update ATR
        atr = rt.atr.update(bar.high, bar.low, bar.close)

        sig = rt.strat.on_bar_close(bar)
        if not sig:
            return
        # Direction filter
        direction = self.direction_gate.get_direction()
        if direction == "long_only" and sig.side == "SHORT":
            log.info("Blocked by direction filter: SHORT (direction=long_only)")
            return
        if direction == "short_only" and sig.side == "LONG":
            log.info("Blocked by direction filter: LONG (direction=short_only)")
            return

        now_ms = int(time.time() * 1000)
        cooldown_ms = self.cooldown_minutes * 60 * 1000
        if (now_ms - rt.last_signal_ts_ms) < cooldown_ms:
            return
        rt.last_signal_ts_ms = now_ms
        self._last_signal_ts_ms = now_ms

        # Log signal
        self.store.log_event({
            "ts_ms": now_ms,
            "symbol": sig.symbol,
            "side": sig.side,
            "entry_price": sig.entry_price,
            "pivot_price": sig.pivot_price,
            "pivot_osc": sig.pivot_osc_value,
            "pivot_cvd": sig.pivot_cvd_value,
            "slip_bps": sig.slip_bps,
            "loc_at_pivot": sig.loc_at_pivot,
            "oscillator": sig.oscillator_name,
            "pine_div": sig.pine_div,
            "cvd_div": sig.cvd_div,
            "pivot_time_ms": sig.pivot_time_ms,
            "confirm_time_ms": sig.confirm_time_ms,
        })

        orders_cfg = scfg.orders_cfg
        notional = float(orders_cfg.get("notional_usdt", 25.0))
        planned_qty = None
        filters = None
        try:
            if self.rest:
                filters = await self.rest.get_symbol_filters(sig.symbol)
                planned_qty = quantize(notional / sig.entry_price, filters.step_size)
        except Exception:
            planned_qty = None
            filters = None

        # Telegram alert ALWAYS (pre-trade)
        await self.notifier.send(
            formatters.format_entry(
                branding=self.branding,
                sig=sig,
                atr=atr,
                strat_params=scfg.strategy_params,
                atr_params=scfg.atr_params,
                divergence_mode=scfg.divergence_mode,
                timeframe=self.timeframe,
                testnet=self.testnet,
                mode=self.mode,
                cooldown_minutes=self.cooldown_minutes,
                max_positions_total=self.max_positions_total,
                one_pos_per_symbol=scfg.position_cfg.get("one_position_per_symbol", True),
                notional_usdt=notional,
                planned_qty=planned_qty,
            )
        )

        # Position rules
        if scfg.position_cfg.get("one_position_per_symbol", True) and rt.in_position:
            return
        if self._positions_total >= self.max_positions_total:
            return

        if self.mode != "live" or not self._live_enabled:
            return

        assert self.rest is not None
        # Leverage (futures)
        lev = int(orders_cfg.get("leverage", 1))
        await self.rest.set_leverage(sig.symbol, lev)

        # Determine quantity from notional
        if filters is None:
            filters = await self.rest.get_symbol_filters(sig.symbol)
        qty_raw = notional / sig.entry_price
        qty = quantize(qty_raw, filters.step_size)
        levels = atr_levels(sig.side, sig.entry_price, atr, scfg.atr_params)
        if qty < filters.min_qty:
            err = RuntimeError(f"Qty too small for {sig.symbol}: {qty} < min {filters.min_qty}")
            await self.notifier.send(
                formatters.format_execution(
                    branding=self.branding,
                    sig=sig,
                    order={"orderId": "n/a", "status": "REJECTED"},
                    levels=levels,
                    qty=qty,
                    leverage=lev,
                    notional=notional,
                    atr_params=scfg.atr_params,
                    atr_value=atr,
                    error=err,
                )
            )
            return

        # Place market entry
        side = "BUY" if sig.side == "LONG" else "SELL"
        order = {}
        sl = tp = None
        try:
            order = await self.rest.order_market(sig.symbol, side=side, qty=qty, reduce_only=False)

            # Risk orders (optional)
            if scfg.atr_params.enabled and levels.sl and levels.tp:
                # For futures, exit sides are opposite
                exit_side = "SELL" if sig.side == "LONG" else "BUY"
                # Quantize prices to tick
                sl = round(levels.sl / filters.tick_size) * filters.tick_size
                tp = round(levels.tp / filters.tick_size) * filters.tick_size
                await self.rest.order_stop_market(sig.symbol, side=exit_side, qty=qty, stop_price=sl, reduce_only=True)
                await self.rest.order_take_profit_market(sig.symbol, side=exit_side, qty=qty, stop_price=tp, reduce_only=True)

            rt.in_position = True
            self._positions_total += 1
        except Exception as e:
            log.warning("Execution failed for %s: %s", sig.symbol, e)
            await self.notifier.send(
                formatters.format_execution(
                    branding=self.branding,
                    sig=sig,
                    order=order or {"orderId": "n/a", "status": "ERROR"},
                    levels=levels,
                    qty=qty,
                    leverage=lev,
                    notional=notional,
                    atr_params=scfg.atr_params,
                    atr_value=atr,
                    error=e,
                )
            )
            return

        self.store.log_trade({
            "ts_ms": now_ms,
            "symbol": sig.symbol,
            "side": sig.side,
            "order_id": order.get("orderId") if isinstance(order, dict) else None,
            "status": order.get("status") if isinstance(order, dict) else None,
            "qty": qty,
            "entry_price": sig.entry_price,
            "sl": sl,
            "tp": tp,
            "mode": self.mode,
        })

        await self.notifier.send(
            formatters.format_execution(
                branding=self.branding,
                sig=sig,
                order=order,
                levels=levels,
                qty=qty,
                leverage=lev,
                notional=notional,
                atr_params=scfg.atr_params,
                atr_value=atr,
            )
        )
