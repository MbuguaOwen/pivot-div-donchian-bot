from __future__ import annotations

import asyncio
import math
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List, Deque

from dotenv import load_dotenv

from .models import Bar, Signal, Side
from .notifier import TelegramNotifier
from .storage.csv_store import CsvStore
from .strategy.pivot_div_donchian import StrategyParams, SymbolStrategyState
from .strategy.indicators import AtrState
from .risk.structure_atr_trailing import (
    StructureAtrTrailParams,
    TrailingParams,
    StopState,
    compute_initial_sl,
    compute_tp_from_R,
    update_extremes,
    update_stop,
)
from .binance.rest import BinanceRest, SymbolFilters, quantize
from .binance.ws import BinanceWsManager
from .execution.base import ExecutionAdapter, ExecFill
from .execution.paper import PaperExecutor, PaperConfig
from .execution.binance_exec import BinanceExecutor
from .config import deep_merge, load_pair_override
from .alerts import formatters
from .direction import DirectionGate
from .telegram_controls import TelegramControls, control_keyboard
from .tv_bridge import TvBridgeConfig, TvWebhookServer, TvSignal

log = logging.getLogger("engine")

def floor_to_tick(px: float, tick: float) -> float:
    return math.floor(px / tick) * tick if tick > 0 else px


def ceil_to_tick(px: float, tick: float) -> float:
    return math.ceil(px / tick) * tick if tick > 0 else px


def quantize_sl(px: float, side: Side, tick: float) -> float:
    return floor_to_tick(px, tick) if side == "LONG" else ceil_to_tick(px, tick)


def quantize_tp(px: float, side: Side, tick: float) -> float:
    return ceil_to_tick(px, tick) if side == "LONG" else floor_to_tick(px, tick)

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
    stop_engine_params: StructureAtrTrailParams
    enable_cvd: bool
    orders_cfg: Dict[str, Any]
    position_cfg: Dict[str, Any]
    divergence_mode: str
    parity: ParityConfig

@dataclass
class ParityConfig:
    mode: bool = False
    ema_seed_mode: str = "first"
    pivot_tie_break: str = "strict"
    warmup_bars: int = 0
    gap_heal: bool = True
    desync_pause: bool = True

@dataclass
class SymbolRuntime:
    cfg: SymbolConfig
    strat: SymbolStrategyState
    h1_atr: Optional[AtrState] = None
    h1_bars: Optional[Deque[Bar]] = None
    h1_atr_value: Optional[float] = None
    stop_state: Optional[StopState] = None
    stop_ready: bool = False
    position_side: Optional[Side] = None
    position_qty: Optional[float] = None
    stop_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    filters: Optional[SymbolFilters] = None
    last_stop_replace_ms: Optional[int] = None
    pending_signal: Optional[Signal] = None
    in_position: bool = False
    last_signal_ts_ms: int = 0
    last_bar_open_ms: Optional[int] = None
    last_price: Optional[float] = None
    desynced: bool = False
    desync_warned: bool = False

class BotEngine:
    def __init__(self, cfg: Dict[str, Any]):
        load_dotenv()

        self.cfg = cfg
        self.config_dir = Path(cfg.get("_config_dir", "."))
        exec_cfg = cfg.get("execution", {}) or {}
        self.exec_mode = str(exec_cfg.get("mode", "binance")).lower()
        self.paper_cfg = exec_cfg.get("paper", {}) or {}
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
        parity_cfg = cfg.get("parity", {}) or {}
        self.parity_cfg = ParityConfig(
            mode=bool(parity_cfg.get("mode", False)),
            ema_seed_mode=str(parity_cfg.get("ema_seed_mode", "first")).lower(),
            pivot_tie_break=str(parity_cfg.get("pivot_tie_break", "strict")).lower(),
            warmup_bars=int(parity_cfg.get("warmup_bars", 0) or 0),
            gap_heal=bool(parity_cfg.get("gap_heal", True)),
            desync_pause=bool(parity_cfg.get("desync_pause", True)),
        )

        health_cfg = cfg.get("healthcheck", {}) or {}
        tv_cfg_raw = cfg.get("tv_bridge", {}) or {}
        host_for_health = tv_cfg_raw.get("host", "127.0.0.1")
        if host_for_health in ("0.0.0.0", "::"):
            host_for_health = "127.0.0.1"
        default_health_url = f"http://{host_for_health}:{tv_cfg_raw.get('port', 9001)}/health" if tv_cfg_raw.get("enabled", False) else ""
        self.health_enabled = bool(health_cfg.get("enabled", False))
        self.health_url = str(health_cfg.get("url", default_health_url or "") or default_health_url or "")
        self.health_interval_sec = int(health_cfg.get("interval_seconds", 60))
        self.health_timeout_sec = float(health_cfg.get("timeout_seconds", 5.0))
        self.health_alert_on_failure = bool(health_cfg.get("alert_on_failure", True))
        self.health_alert_on_recovery = bool(health_cfg.get("alert_on_recovery", True))
        self._last_healthcheck_ms = 0
        self._health_failed = False

        # TradingView bridge (optional). When enabled, TradingView can become
        # the signal engine (tv_only) or enforce a two-man rule (tv_and_bot).
        self.tv_cfg = TvBridgeConfig.from_dict(cfg.get("tv_bridge", {}) or {})
        if self.tv_cfg.mode not in ("tv_only", "tv_and_bot"):
            self.tv_cfg.mode = "tv_only"
        if not self.health_url and self.tv_cfg.enabled:
            host = self.tv_cfg.host
            if host in ("0.0.0.0", "::"):
                host = "127.0.0.1"
            self.health_url = f"http://{host}:{self.tv_cfg.port}/health"
        self.tv_server: Optional[TvWebhookServer] = None
        self._pending_tv: Dict[tuple[str, str, int, str], TvSignal] = {}
        self._pending_bot: Dict[tuple[str, str, int, str], Signal] = {}
        self.controls: Optional[TelegramControls] = None
        self.controls_enabled = bool(tg_cfg.get("controls_enabled", False))
        self.controls_allowed_chat_id = tg_cfg.get("controls_allowed_chat_id")
        self.controls_state_path = tg_cfg.get("controls_state_path", "state/telegram_controls.json")

        self.rest: Optional[BinanceRest] = None
        self.ws: Optional[BinanceWsManager] = None
        self.executor: Optional[ExecutionAdapter] = None
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
        self._position_watchdog_task: Optional[asyncio.Task] = None
        self._healthcheck_task: Optional[asyncio.Task] = None
        self._sent_alert_keys: set[tuple[str, str, int, str]] = set()

    async def start(self, rest: BinanceRest, ws_url: str, symbols: List[str]) -> None:
        self.rest = rest
        self.symbols = symbols
        self._start_time = time.time()
        limits = self.cfg.get("limits", {}) or {}
        self.max_positions_total = int(limits.get("max_positions_total", 20))
        default_cooldown = int(limits.get("cooldown_minutes_per_symbol", 30))
        # Parity mode must not suppress alerts via cooldown
        self.cooldown_minutes = 0 if self.parity_cfg.mode else default_cooldown

        self.orders_cfg = self.cfg.get("orders", {}) or {}
        self.position_cfg = self.cfg.get("position", {}) or {}

        # live safety: refuse trading without keys only when executing on binance
        if self.exec_mode == "binance" and self.mode == "live" and (not rest.key or not rest.secret):
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

        # Optional TradingView webhook server (runs inside the bot)
        if self.tv_cfg.enabled:
            self.tv_server = TvWebhookServer(self.tv_cfg, on_signal=self._on_tv_signal)
            await self.tv_server.start()
            if not self.tv_cfg.secret:
                log.warning("tv_bridge_enabled_but_no_secret")
            try:
                await self.notifier.send(
                    f"TV bridge: ON | mode={self.tv_cfg.mode} | listen=http://{self.tv_cfg.host}:{self.tv_cfg.port}{self.tv_cfg.path}"
                )
            except Exception:
                log.exception("Failed to send TV bridge startup notice")

        # Build per-symbol runtime/config
        cvd_symbols: List[str] = []
        stop_engine_symbols: List[str] = []
        for sym in symbols:
            scfg = self._build_symbol_config(sym)
            self.symbol_cfgs[sym] = scfg
            if scfg.enable_cvd:
                cvd_symbols.append(sym)
            if scfg.stop_engine_params.enabled:
                stop_engine_symbols.append(sym)
                h1_maxlen = scfg.stop_engine_params.htf_lookback_bars + scfg.stop_engine_params.atr_len + 5
                h1_bars: Optional[Deque[Bar]] = deque(maxlen=h1_maxlen)
                h1_atr = AtrState(scfg.stop_engine_params.atr_len, ema_seed_mode=self.parity_cfg.ema_seed_mode if self.parity_cfg.mode else "first")
            else:
                h1_bars = None
                h1_atr = None
            self.symbols_rt[sym] = SymbolRuntime(
                cfg=scfg,
                strat=SymbolStrategyState(sym, scfg.strategy_params, enable_cvd=scfg.enable_cvd),
                h1_atr=h1_atr,
                h1_bars=h1_bars,
            )

        for sym in symbols:
            try:
                await self._warmup_symbol(sym, self.symbols_rt[sym])
            except Exception as e:
                log.warning("Warmup failed for %s: %s", sym, e)
            try:
                await self._warmup_stop_engine(sym, self.symbols_rt[sym])
            except Exception as e:
                log.warning("Stop-engine warmup failed for %s: %s", sym, e)

        # Execution adapter
        if self.exec_mode == "paper":
            pcfg = PaperConfig(
                fee_bps=float(self.paper_cfg.get("fee_bps", 4.0)),
                slippage_bps=float(self.paper_cfg.get("slippage_bps", 1.5)),
                fill_policy=str(self.paper_cfg.get("fill_policy", "next_tick")),
                max_wait_ms=int(self.paper_cfg.get("max_wait_ms", 3000)),
                use_mark_price=bool(self.paper_cfg.get("use_mark_price", False)),
                log_trades=bool(self.paper_cfg.get("log_trades", True)),
            )
            self.executor = PaperExecutor(pcfg, on_exit=self._on_paper_exit, on_fill=self._on_paper_fill)
        else:
            self.executor = BinanceExecutor(self.rest)
        await self.executor.start()

        # Build streams
        tf = self.timeframe
        # Binance expects lowercase, e.g. btcusdt@kline_15m
        kline_streams = [f"{s.lower()}@kline_{tf}" for s in symbols]
        h1_streams = [
            f"{s.lower()}@kline_{self.symbol_cfgs[s].stop_engine_params.htf_interval}"
            for s in symbols
            if self.symbol_cfgs[s].stop_engine_params.enabled
        ]
        trade_streams = [
            f"{s.lower()}@aggTrade"
            for s in symbols
            if self.symbol_cfgs[s].enable_cvd or self.symbol_cfgs[s].stop_engine_params.enabled or self.exec_mode == "paper"
        ]
        # Deduplicate while preserving order
        all_streams = list(dict.fromkeys(kline_streams + h1_streams + trade_streams))
        log.info(
            "Streams: kline=%d htf=%d aggTrade=%d cvd_symbols=%d stop_engine=%d",
            len(kline_streams),
            len(h1_streams),
            len(trade_streams),
            len(cvd_symbols),
            len(stop_engine_symbols),
        )

        # Chunk streams to avoid overloading a single connection.
        # Binance allows many streams, but be conservative: 200 per connection.
        chunk = 200
        groups = [all_streams[i:i+chunk] for i in range(0, len(all_streams), chunk)]

        self.ws = BinanceWsManager(ws_url)
        self.ws.start(groups, self._on_ws_message)

        # Heartbeat
        if self.heartbeat_enabled:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        if self.health_enabled and self.health_url:
            self._healthcheck_task = asyncio.create_task(self._healthcheck_loop())
        if self.exec_mode == "binance" and self.mode == "live":
            self._position_watchdog_task = asyncio.create_task(self._position_watchdog_loop())

        startup_markup = control_keyboard() if self.controls_enabled else None
        await self.notifier.send(
            formatters.format_startup(
                branding=self.branding,
                mode=self.mode,
                exec_mode=self.exec_mode,
                testnet=self.testnet,
                exchange="Binance",
                market_type=self.market_type,
                timeframe=self.timeframe,
                symbols=symbols,
                cvd_symbols=cvd_symbols,
                stop_engine_enabled=bool(self.cfg.get("risk", {}).get("engine", {}).get("enabled", False)),
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
        ema_seed_mode = self.parity_cfg.ema_seed_mode if self.parity_cfg.mode else "first"
        pivot_tie_break = self.parity_cfg.pivot_tie_break if self.parity_cfg.mode else "strict"
        don_len = int(s_cfg["donchian"]["length"])
        pivot_len = int(s_cfg["pivot"]["length"])
        osc_len = int(s_cfg["oscillator"]["ema_length"])
        base_warmup = max(don_len, 5 * osc_len, 3 * (2 * pivot_len + 1)) + 2
        warmup_bars = max(self.parity_cfg.warmup_bars, base_warmup) if self.parity_cfg.mode else 0
        p = StrategyParams(
            don_len=don_len,
            ext_band_pct=float(s_cfg["donchian"]["extreme_band_pct"]),
            pivot_len=pivot_len,
            osc_ema_len=osc_len,
            divergence_mode=divergence_mode,
            ema_seed_mode=ema_seed_mode,
            pivot_tie_break=pivot_tie_break,
            warmup_bars=warmup_bars,
        )
        enable_cvd = divergence_mode in ("cvd", "both", "either")

        # risk.engine config block
        risk_cfg = merged.get("risk", {}).get("engine", {}) or {}
        trailing_cfg = risk_cfg.get("trailing", {}) or {}
        stop_engine_params = StructureAtrTrailParams(
            enabled=bool(risk_cfg.get("enabled", False)),
            htf_interval=str(risk_cfg.get("htf_interval", "1h")).lower(),
            htf_lookback_bars=int(risk_cfg.get("htf_lookback_bars", 72)),
            atr_len=int(risk_cfg.get("atr_len", 24)),
            k_init=float(risk_cfg.get("k_init", 1.8)),
            tp_r_mult=float(risk_cfg.get("tp_r_mult", 2.0)),
            buffer_bps=float(risk_cfg.get("buffer_bps", 10.0)),
            be_trigger_r=float(risk_cfg.get("be_trigger_r", 1.0)),
            be_buffer_bps=float(risk_cfg.get("be_buffer_bps", 10.0)),
            trailing=TrailingParams(
                enabled=bool(trailing_cfg.get("enabled", False)),
                trigger_r=float(trailing_cfg.get("trigger_r", 2.5)),
                k_trail=float(trailing_cfg.get("k_trail", 1.6)),
                lock_r=float(trailing_cfg.get("lock_r", 1.0)),
            ),
            min_stop_replace_interval_sec=float(risk_cfg.get("min_stop_replace_interval_sec", 2.0)),
            min_stop_tick_improvement=int(risk_cfg.get("min_stop_tick_improvement", 2)),
        )

        orders_cfg = merged.get("orders", {}) or {}
        position_cfg = merged.get("position", {}) or {}

        return SymbolConfig(
            symbol=symbol,
            cfg=merged,
            strategy_params=p,
            stop_engine_params=stop_engine_params,
            enable_cvd=enable_cvd,
            orders_cfg=orders_cfg,
            position_cfg=position_cfg,
            divergence_mode=divergence_mode,
            parity=self.parity_cfg,
        )

    def _suggest_warmup(self, params: StrategyParams) -> int:
        base = max(
            params.don_len,
            5 * params.osc_ema_len,
            3 * (2 * params.pivot_len + 1),
        )
        return base + 2  # small safety margin

    async def _warmup_symbol(self, symbol: str, rt: SymbolRuntime) -> None:
        if not self.parity_cfg.mode:
            return
        if self.rest is None:
            return
        target = max(rt.cfg.strategy_params.warmup_bars, self._suggest_warmup(rt.cfg.strategy_params))
        if target <= 0:
            return
        try:
            bars = await self.rest.fetch_last_n_bars(symbol, self.timeframe, target)
        except Exception as e:
            log.warning("Warmup fetch failed for %s: %s", symbol, e)
            return
        now_ms = int(time.time() * 1000)
        closed = [b for b in bars if b.close_time_ms <= now_ms]
        for b in closed[-target:]:
            await self._on_bar_close(symbol, b, heal_gaps=False, emit_alerts=False)
            rt.last_bar_open_ms = b.open_time_ms
        if closed:
            rt.last_bar_open_ms = closed[-1].open_time_ms

    async def _warmup_stop_engine(self, symbol: str, rt: SymbolRuntime) -> None:
        params = rt.cfg.stop_engine_params
        if not params.enabled:
            return
        if self.rest is None or rt.h1_atr is None or rt.h1_bars is None:
            return
        target = params.htf_lookback_bars + params.atr_len + 5
        try:
            bars = await self.rest.fetch_last_n_bars(symbol, params.htf_interval, target)
        except Exception as e:
            log.warning("Stop-engine warmup fetch failed for %s: %s", symbol, e)
            return
        now_ms = int(time.time() * 1000)
        closed = [b for b in bars if b.close_time_ms <= now_ms]
        for b in closed[-target:]:
            await self._on_h1_close(symbol, b, emit_alerts=False)
        if len(rt.h1_bars) >= params.htf_lookback_bars and rt.h1_atr_value is not None:
            rt.stop_ready = True

    async def stop(self) -> None:
        if self.ws:
            await self.ws.stop()
        if self.tv_server:
            try:
                await self.tv_server.stop()
            except Exception:
                log.exception("tv_bridge_stop_failed")
            self.tv_server = None
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        if self._position_watchdog_task:
            self._position_watchdog_task.cancel()
            self._position_watchdog_task = None
        if self._healthcheck_task:
            self._healthcheck_task.cancel()
            self._healthcheck_task = None
        if self.controls:
            await self.controls.stop()
        await self.notifier.stop()
        if self.rest:
            await self.rest.stop()
        if self.executor:
            await self.executor.stop()

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

    async def _healthcheck_loop(self) -> None:
        while True:
            await asyncio.sleep(max(self.health_interval_sec, 1))
            try:
                await self._maybe_healthcheck(int(time.time() * 1000))
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("healthcheck_error %s", e)

    def _signal_key(self, sig: Signal) -> tuple[str, str, int, str]:
        return (sig.symbol, self.timeframe, sig.confirm_time_ms, sig.side)

    def _tv_key(self, symbol: str, side: Side, confirm_time_ms: int) -> tuple[str, str, int, str]:
        return (symbol, self.timeframe, confirm_time_ms, side)

    async def _maybe_healthcheck(self, now_ms: int) -> None:
        if not self.health_enabled or not self.health_url:
            return
        if (now_ms - self._last_healthcheck_ms) < max(self.health_interval_sec, 1) * 1000:
            return
        self._last_healthcheck_ms = now_ms

        ok = False
        err_text = ""
        try:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=self.health_timeout_sec)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self.health_url) as resp:
                    if resp.status == 200:
                        ok = True
                    else:
                        err_text = f"HTTP {resp.status}"
        except Exception as e:
            err_text = str(e)

        if ok:
            if self._health_failed and self.health_alert_on_recovery:
                await self.notifier.send(f"✅ Webhook healthcheck recovered: {self.health_url}")
            self._health_failed = False
            return

        if not self._health_failed and self.health_alert_on_failure:
            await self.notifier.send(f"⚠️ Webhook healthcheck FAILED: {self.health_url} err={err_text or 'unknown'}")
        self._health_failed = True

    def _log_parity(self, key: tuple[str, str, int, str], tv: bool, bot: bool, action: str, details: str = "") -> None:
        now_ms = int(time.time() * 1000)
        symbol, _tf, confirm_ms, side = key
        try:
            self.store.log_parity({
                "ts_ms": now_ms,
                "mode": self.tv_cfg.mode,
                "symbol": symbol,
                "side": side,
                "confirm_time_ms": confirm_ms,
                "tv": int(tv),
                "bot": int(bot),
                "action": action,
                "details": details,
            })
        except Exception:
            log.exception("parity_csv_write_failed")

    async def _parity_timeout_tv(self, key: tuple[str, str, int, str]) -> None:
        await asyncio.sleep(max(self.tv_cfg.match_window_ms, 1000) / 1000)
        tv_sig = self._pending_tv.pop(key, None)
        if not tv_sig:
            return
        # Still unmatched
        log.warning("parity_breach_tv_only key=%s symbol=%s side=%s confirm=%s", key, tv_sig.symbol, tv_sig.side, tv_sig.confirm_time_ms)
        self._log_parity(key, tv=True, bot=False, action="mismatch", details="tv_only")
        if self.tv_cfg.alert_on_mismatch:
            try:
                await self.notifier.send(
                    f"⚠️ PARITY BREACH (TV only): {tv_sig.symbol} {tv_sig.side} confirm={tv_sig.confirm_time_ms}"
                )
            except Exception:
                log.warning("Failed to send TV-only parity breach")

    async def _parity_timeout_bot(self, key: tuple[str, str, int, str]) -> None:
        await asyncio.sleep(max(self.tv_cfg.match_window_ms, 1000) / 1000)
        bot_sig = self._pending_bot.pop(key, None)
        if not bot_sig:
            return
        log.warning("parity_breach_bot_only key=%s symbol=%s side=%s confirm=%s", key, bot_sig.symbol, bot_sig.side, bot_sig.confirm_time_ms)
        self._log_parity(key, tv=False, bot=True, action="mismatch", details="bot_only")
        if self.tv_cfg.alert_on_mismatch:
            try:
                await self.notifier.send(
                    f"⚠️ PARITY BREACH (BOT only): {bot_sig.symbol} {bot_sig.side} confirm={bot_sig.confirm_time_ms}"
                )
            except Exception:
                log.warning("Failed to send BOT-only parity breach")

    def _signal_from_tv(self, tv: TvSignal, rt: SymbolRuntime) -> Signal:
        # Entry price: prefer TV-provided, else our last known price.
        entry = tv.entry_price
        if entry is None:
            entry = rt.last_price
        if entry is None:
            entry = 0.0
        return Signal(
            symbol=tv.symbol,
            side=tv.side,
            entry_price=float(entry),
            pivot_price=float(tv.pivot_price) if tv.pivot_price is not None else float(entry),
            pivot_osc_value=float(tv.pivot_osc) if tv.pivot_osc is not None else 0.0,
            pivot_cvd_value=None,
            slip_bps=float(tv.slip_bps) if tv.slip_bps is not None else 0.0,
            loc_at_pivot=float(tv.loc_at_pivot) if tv.loc_at_pivot is not None else 0.5,
            oscillator_name="tv",
            pine_div=True,
            cvd_div=False,
            pivot_time_ms=int(tv.confirm_time_ms),
            confirm_time_ms=int(tv.confirm_time_ms),
        )

    async def _on_tv_signal(self, tv: TvSignal) -> None:
        """Handle an incoming TradingView webhook.

        The webhook contains a bar-confirmed signal. Depending on mode:
        - tv_only: trade on TV signal
        - tv_and_bot: trade only if bot's strategy produced the same key
        """
        if not self.tv_cfg.enabled:
            return
        symbol = tv.symbol
        if symbol not in self.symbols_rt:
            log.warning("tv_bridge_unknown_symbol %s", symbol)
            return
        if self.tv_cfg.require_tf_match and tv.tf:
            if tv.tf != self.timeframe:
                log.info("tv_bridge_tf_mismatch symbol=%s tv_tf=%s bot_tf=%s", symbol, tv.tf, self.timeframe)

        now_ms = int(time.time() * 1000)
        try:
            self.store.log_tv_signal({
                "ts_ms": now_ms,
                "symbol": tv.symbol,
                "side": tv.side,
                "confirm_time_ms": tv.confirm_time_ms,
                "tf": tv.tf or "",
                "entry_price": tv.entry_price if tv.entry_price is not None else "",
                "pivot_price": tv.pivot_price if tv.pivot_price is not None else "",
                "pivot_osc": tv.pivot_osc if tv.pivot_osc is not None else "",
                "slip_bps": tv.slip_bps if tv.slip_bps is not None else "",
                "loc_at_pivot": tv.loc_at_pivot if tv.loc_at_pivot is not None else "",
                "tickerid": tv.tickerid or "",
            })
        except Exception:
            log.exception("tv_signals_csv_write_failed")

        key = self._tv_key(tv.symbol, tv.side, tv.confirm_time_ms)
        rt = self.symbols_rt[tv.symbol]

        # tv_only: TV is boss
        if self.tv_cfg.mode == "tv_only":
            # log compare if bot had same signal
            if key in self._pending_bot:
                self._pending_bot.pop(key, None)
                self._log_parity(key, tv=True, bot=True, action="match", details="tv_only")
            else:
                # schedule mismatch check for TV-only event
                self._pending_tv[key] = tv
                asyncio.create_task(self._parity_timeout_tv(key))

            sig = self._signal_from_tv(tv, rt)
            await self._execute_tradable_signal(sig, rt, source="TV")
            return

        # tv_and_bot: two-man rule
        if self.tv_cfg.mode == "tv_and_bot":
            self._pending_tv[key] = tv
            bot_sig = self._pending_bot.pop(key, None)
            if bot_sig is None:
                asyncio.create_task(self._parity_timeout_tv(key))
                return

            # matched
            self._pending_tv.pop(key, None)
            self._log_parity(key, tv=True, bot=True, action="match", details="tv_and_bot")
            await self._execute_tradable_signal(bot_sig, rt, source="TV+BOT")
            return

    async def _on_bot_signal_tv_mode(self, sig: Signal, rt: SymbolRuntime) -> None:
        """Register bot strategy signal for parity/matching under TV modes."""
        key = self._signal_key(sig)
        self._pending_bot[key] = sig

        # For parity debugging, still write bot signal to events.csv (as-is)
        now_ms = int(time.time() * 1000)
        try:
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
        except Exception:
            log.exception("bot_events_csv_write_failed")

        # tv_only: just compare
        if self.tv_cfg.mode == "tv_only":
            if key in self._pending_tv:
                self._pending_tv.pop(key, None)
                self._pending_bot.pop(key, None)
                self._log_parity(key, tv=True, bot=True, action="match", details="tv_only")
            else:
                asyncio.create_task(self._parity_timeout_bot(key))
            return

        # tv_and_bot: trade only on match
        if self.tv_cfg.mode == "tv_and_bot":
            tv_sig = self._pending_tv.pop(key, None)
            if tv_sig is None:
                asyncio.create_task(self._parity_timeout_bot(key))
                return
            self._pending_bot.pop(key, None)
            self._log_parity(key, tv=True, bot=True, action="match", details="tv_and_bot")
            await self._execute_tradable_signal(sig, rt, source="TV+BOT")

    async def _execute_tradable_signal(self, sig: Signal, rt: SymbolRuntime, source: str) -> None:
        """Execute the shared signal pipeline for TV-driven trades.

        This mirrors the live path in _on_bar_close, but:
        - no cooldown gating (TV is already 1/bar)
        - uses idempotency key to ensure once-per-confirm-bar
        """
        scfg = rt.cfg
        if rt.desynced and self.parity_cfg.desync_pause:
            self._log_signal_suppressed(sig, "desynced_state", {"source": source})
            return

        key = self._signal_key(sig)
        if key in self._sent_alert_keys:
            self._log_signal_suppressed(sig, "duplicate", {"idempotency_key": key, "source": source})
            return

        # Direction filter
        direction = self.direction_gate.get_direction()
        if direction == "long_only" and sig.side == "SHORT":
            self._log_signal_suppressed(sig, "direction_filter", {"direction": direction, "source": source})
            return
        if direction == "short_only" and sig.side == "LONG":
            self._log_signal_suppressed(sig, "direction_filter", {"direction": direction, "source": source})
            return

        now_ms = int(time.time() * 1000)
        self._last_signal_ts_ms = now_ms
        self._sent_alert_keys.add(key)

        stop_preview = {
            "enabled": scfg.stop_engine_params.enabled,
            "ready": rt.stop_ready,
            "htf_interval": scfg.stop_engine_params.htf_interval,
            "htf_lookback_bars": scfg.stop_engine_params.htf_lookback_bars,
            "atr_len": scfg.stop_engine_params.atr_len,
            "atr": rt.h1_atr_value,
            "buffer_bps": scfg.stop_engine_params.buffer_bps,
            "k_init": scfg.stop_engine_params.k_init,
            "tp_r_mult": scfg.stop_engine_params.tp_r_mult,
            "trailing_enabled": scfg.stop_engine_params.trailing.enabled,
            "trail_trigger_r": scfg.stop_engine_params.trailing.trigger_r,
            "k_trail": scfg.stop_engine_params.trailing.k_trail,
            "lock_r": scfg.stop_engine_params.trailing.lock_r,
            "be_trigger_r": scfg.stop_engine_params.be_trigger_r,
            "be_buffer_bps": scfg.stop_engine_params.be_buffer_bps,
        }

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

        # Preview SL/TP if stop engine is ready
        if (
            scfg.stop_engine_params.enabled
            and rt.stop_ready
            and rt.h1_bars
            and rt.h1_atr_value is not None
        ):
            try:
                recent = list(rt.h1_bars)[-scfg.stop_engine_params.htf_lookback_bars:]
                struct_low = min(b.low for b in recent)
                struct_high = max(b.high for b in recent)
                tick_preview = filters.tick_size if filters else 0.0
                sl0_raw, _ = compute_initial_sl(sig.entry_price, sig.side, struct_low, struct_high, rt.h1_atr_value, scfg.stop_engine_params)
                sl0_q = quantize_sl(sl0_raw, sig.side, tick_preview)
                R_q = abs(sig.entry_price - sl0_q)
                tp0 = compute_tp_from_R(sig.entry_price, sig.side, R_q, scfg.stop_engine_params.tp_r_mult)
                tp0_q = quantize_tp(tp0, sig.side, tick_preview)
                stop_preview.update({"sl0": sl0_q, "tp0": tp0_q, "R": R_q})
            except Exception as e:
                log.warning("stop_preview_failed %s: %s", sig.symbol, e)

        # Pre-trade alert
        await self.notifier.send(
            formatters.format_entry(
                branding=f"{self.branding} [{source}]",
                sig=sig,
                strat_params=scfg.strategy_params,
                stop_info=stop_preview,
                divergence_mode=scfg.divergence_mode,
                timeframe=self.timeframe,
                testnet=self.testnet,
                mode=self.mode,
                exec_mode=self.exec_mode,
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
        if scfg.stop_engine_params.enabled and not rt.stop_ready:
            self._log_signal_suppressed(sig, "stop_engine_not_ready", {"source": source})
            return

        if self.mode != "live" or not self._live_enabled:
            return

        if filters is None and self.rest:
            filters = await self.rest.get_symbol_filters(sig.symbol)
        if filters:
            rt.filters = filters
        if filters is None:
            return

        qty_raw = notional / sig.entry_price if sig.entry_price > 0 else 0.0
        qty = quantize(qty_raw, filters.step_size)
        lev = int(orders_cfg.get("leverage", 1))
        if self.exec_mode == "binance" and self.rest:
            await self.rest.set_leverage(sig.symbol, lev)
        if qty < filters.min_qty:
            err = RuntimeError(f"Qty too small for {sig.symbol}: {qty} < min {filters.min_qty}")
            await self.notifier.send(
                formatters.format_execution(
                    branding=f"{self.branding} [{source}]",
                    sig=sig,
                    order={"orderId": "n/a", "status": "REJECTED"},
                    qty=qty,
                    leverage=lev,
                    notional=notional,
                    exec_mode=self.exec_mode,
                    stop_info=stop_preview,
                    error=err,
                )
            )
            return

        if self.executor is None:
            return
        rt.pending_signal = sig
        res = await self.executor.place_entry(sig.symbol, sig.side, qty, ref=f"tv:{source}")
        if not res.ok:
            await self.notifier.send(
                formatters.format_execution(
                    branding=f"{self.branding} [{source}]",
                    sig=sig,
                    order={"orderId": "n/a", "status": res.msg},
                    qty=qty,
                    leverage=lev,
                    notional=notional,
                    exec_mode=self.exec_mode,
                    stop_info=stop_preview,
                    error=RuntimeError(res.msg),
                )
            )
            return
        fill_obj = res.fill
        if fill_obj is None and self.exec_mode == "binance":
            fill_obj = ExecFill(symbol=sig.symbol, side=sig.side, qty=qty, price=sig.entry_price, fee_paid=0.0, ts_ms=now_ms, mode="binance")
        if fill_obj:
            await self._on_fill(sig, fill_obj, filters)

    def _log_signal_suppressed(self, sig: Signal, reason: str, extra: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {
            "reason": reason,
            "symbol": sig.symbol,
            "side": sig.side,
            "confirm_time_ms": sig.confirm_time_ms,
            "pivot_time_ms": sig.pivot_time_ms,
        }
        if extra:
            payload.update(extra)
        log.info("signal_suppressed %s", payload)

    async def _heal_if_gap(self, symbol: str, rt: SymbolRuntime, incoming: Bar, emit_alerts: bool) -> None:
        """Backfill missing klines if websocket skipped bars."""
        if not self.parity_cfg.mode or not self.parity_cfg.gap_heal:
            return
        last_open = rt.last_bar_open_ms
        if last_open is None:
            return
        tf_ms = _tf_to_minutes(self.timeframe) * 60 * 1000
        expected_open = last_open + tf_ms
        if incoming.open_time_ms <= expected_open:
            return
        missing = int((incoming.open_time_ms - expected_open) // tf_ms)
        if missing <= 0:
            return
        fillers: List[Bar] = []
        try:
            if self.rest:
                history = await self.rest.fetch_last_n_bars(symbol, self.timeframe, missing + 2)
                fillers = [b for b in history if expected_open <= b.open_time_ms < incoming.open_time_ms]
                fillers.sort(key=lambda x: x.open_time_ms)
        except Exception as e:
            log.warning("Gap heal fetch failed for %s: %s", symbol, e)
        if len(fillers) < missing:
            rt.desynced = True
            if self.parity_cfg.desync_pause and emit_alerts and not rt.desync_warned:
                warn = f"Desynced for {symbol}: missing {missing} bars before {incoming.open_time_ms}"
                try:
                    await self.notifier.send(warn)
                except Exception:
                    log.warning("Failed to send desync notice for %s", symbol)
                rt.desync_warned = True
            return
        for fb in fillers:
            await self._on_bar_close(symbol, fb, heal_gaps=False, emit_alerts=emit_alerts)
        rt.desynced = False
        rt.desync_warned = False

    async def _on_h1_close(self, symbol: str, bar: Bar, emit_alerts: bool = True) -> None:
        rt = self.symbols_rt[symbol]
        params = rt.cfg.stop_engine_params
        if not params.enabled or rt.h1_atr is None or rt.h1_bars is None:
            return
        rt.h1_bars.append(bar)
        rt.h1_atr_value = rt.h1_atr.update(bar.high, bar.low, bar.close)
        if rt.stop_state:
            update_extremes(rt.stop_state, high=bar.high, low=bar.low)
        if len(rt.h1_bars) >= params.htf_lookback_bars and rt.h1_atr_value is not None:
            rt.stop_ready = True
        await self._maybe_update_stop(symbol, reason="ATR_UPDATE")

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
            interval = str(k.get("i"))
            bar = Bar(
                open_time_ms=int(k["t"]),
                open=float(k["o"]),
                high=float(k["h"]),
                low=float(k["l"]),
                close=float(k["c"]),
                volume=float(k["v"]),
                close_time_ms=int(k["T"]),
            )
            # keep a best-effort last price for sizing / external triggers
            self.symbols_rt[sym].last_price = bar.close
            if interval == self.timeframe:
                await self._on_bar_close(sym, bar)
            elif interval == self.symbol_cfgs[sym].stop_engine_params.htf_interval:
                await self._on_h1_close(sym, bar)

        elif e == "aggTrade":
            sym = data.get("s")
            if not sym or sym not in self.symbols_rt:
                return
            ts = int(data.get("T"))
            qty = float(data.get("q"))
            is_buyer_maker = bool(data.get("m"))
            price = float(data.get("p"))
            rt = self.symbols_rt[sym]
            rt.last_price = price
            if self.executor:
                await self.executor.on_trade_tick(sym, price, ts)
            if rt.cfg.stop_engine_params.enabled and rt.stop_state:
                update_extremes(rt.stop_state, price=price)
                await self._maybe_update_stop(sym, reason="TICK")
            if rt.cfg.enable_cvd:
                rt.strat.on_agg_trade(ts, qty, is_buyer_maker)

    async def _replace_stop_order(self, symbol: str, rt: SymbolRuntime, new_sl: float, reason: str, old_sl: float) -> None:
        if rt.stop_state is None or rt.position_qty is None or self.executor is None:
            return
        tick = 0.0
        if rt.filters:
            tick = rt.filters.tick_size
        elif self.rest:
            try:
                rt.filters = await self.rest.get_symbol_filters(symbol)
                tick = rt.filters.tick_size
            except Exception as e:
                log.warning("Failed to fetch filters for stop replace %s: %s", symbol, e)
                return
        stop_price = quantize_sl(new_sl, rt.stop_state.side, tick)
        if rt.stop_state.side == "LONG" and stop_price <= old_sl:
            return
        if rt.stop_state.side == "SHORT" and stop_price >= old_sl:
            return
        min_dt_ms = int(rt.cfg.stop_engine_params.min_stop_replace_interval_sec * 1000)
        now_ms = int(time.time() * 1000)
        if rt.last_stop_replace_ms and (now_ms - rt.last_stop_replace_ms) < min_dt_ms:
            return
        if rt.cfg.stop_engine_params.min_stop_tick_improvement > 0 and tick > 0:
            ticks_moved = abs(stop_price - old_sl) / tick
            if ticks_moved < rt.cfg.stop_engine_params.min_stop_tick_improvement:
                return
        try:
            res = await self.executor.place_or_replace_stop(symbol, rt.stop_state.side, rt.position_qty, stop_price, reason=reason, tick_size=tick)
            if res and res.ok:
                if res.order_ids:
                    new_stop_id = res.order_ids.get("stop")
                else:
                    new_stop_id = None
                # cancel old stop only after new stop is confirmed
                if self.exec_mode == "binance" and rt.stop_order_id and self.rest:
                    try:
                        await self.rest.cancel_order(symbol, rt.stop_order_id)
                    except Exception as e:
                        log.warning("stop_cancel_failed %s: %s", symbol, e)
                rt.stop_order_id = new_stop_id or rt.stop_order_id
                rt.stop_state.sl = stop_price
                log.info(
                    "stop_update symbol=%s side=%s reason=%s old_sl=%.8f new_sl=%.8f atr=%.8f peak=%.8f trough=%.8f",
                    symbol,
                    rt.stop_state.side,
                    reason,
                    old_sl,
                    stop_price,
                    rt.h1_atr_value or 0.0,
                    rt.stop_state.peak,
                    rt.stop_state.trough,
                )
                rt.last_stop_replace_ms = now_ms
            else:
                log.warning("stop_replace_failed %s: %s", symbol, res.msg if res else "no_result")
        except Exception as e:
            log.warning("stop_replace_failed %s: %s", symbol, e)

    async def _maybe_update_stop(self, symbol: str, reason: str) -> None:
        rt = self.symbols_rt[symbol]
        params = rt.cfg.stop_engine_params
        state = rt.stop_state
        if not params.enabled or state is None or not rt.stop_ready:
            return
        if rt.h1_atr_value is None:
            return
        prev_be = state.be_done
        prev_trailing = state.trailing_active
        old_sl = state.sl
        new_sl = update_stop(state, rt.h1_atr_value, params)
        tightened = (state.side == "LONG" and new_sl > old_sl) or (state.side == "SHORT" and new_sl < old_sl)
        reason_flag = None
        if not prev_trailing and state.trailing_active and tightened:
            reason_flag = "LOCK"
        elif not prev_be and state.be_done and tightened:
            reason_flag = "BE"
        elif tightened:
            reason_flag = "TRAIL"
        if tightened:
            await self._replace_stop_order(symbol, rt, new_sl, reason_flag or reason, old_sl)

    async def _handle_stop_failure(self, sig: Signal, fill: ExecFill, scfg: SymbolConfig, stop_info: Dict[str, Any], reason: str) -> None:
        err = RuntimeError(reason)
        try:
            await self.notifier.send(
                formatters.format_execution(
                    branding=self.branding,
                    sig=sig,
                    order={"orderId": "n/a", "status": "STOP_FAILED"},
                    qty=fill.qty,
                    leverage=scfg.orders_cfg.get("leverage", 1),
                    notional=scfg.orders_cfg.get("notional_usdt", 25.0),
                    exec_mode=self.exec_mode,
                    error=err,
                    stop_info=stop_info,
                    fill=fill,
                )
            )
        except Exception as e:
            log.warning("stop_failure_notify_error %s: %s", sig.symbol, e)

        try:
            if self.executor:
                await self.executor.close_position_market(sig.symbol, sig.side, fill.qty, price_hint=fill.price)
        except Exception as e:
            log.warning("close_after_stop_fail_error %s: %s", sig.symbol, e)

        if self.exec_mode != "paper":
            await self._handle_position_closed(sig.symbol)

    async def _on_fill(self, sig: Signal, fill: ExecFill, filters: SymbolFilters) -> None:
        rt = self.symbols_rt[fill.symbol]
        scfg = rt.cfg
        rt.in_position = True
        rt.position_side = fill.side
        rt.position_qty = fill.qty
        rt.pending_signal = None
        self._positions_total += 1

        stop_info: Dict[str, Any] = {
            "enabled": scfg.stop_engine_params.enabled,
            "ready": rt.stop_ready,
            "be_trigger_r": scfg.stop_engine_params.be_trigger_r,
            "be_buffer_bps": scfg.stop_engine_params.be_buffer_bps,
            "atr_len": scfg.stop_engine_params.atr_len,
            "buffer_bps": scfg.stop_engine_params.buffer_bps,
            "k_init": scfg.stop_engine_params.k_init,
            "tp_r_mult": scfg.stop_engine_params.tp_r_mult,
            "trailing_enabled": scfg.stop_engine_params.trailing.enabled,
            "trail_trigger_r": scfg.stop_engine_params.trailing.trigger_r,
            "k_trail": scfg.stop_engine_params.trailing.k_trail,
            "lock_r": scfg.stop_engine_params.trailing.lock_r,
        }
        sl = tp = None
        R = None
        tick = filters.tick_size if filters else 0.0
        if scfg.stop_engine_params.enabled and (not rt.stop_ready or not rt.h1_bars or rt.h1_atr_value is None):
            await self._handle_stop_failure(sig, fill, scfg, stop_info, "stop_engine_not_ready")
            return
        # Initial stop via structure + ATR
        if scfg.stop_engine_params.enabled and rt.stop_ready and rt.h1_bars and rt.h1_atr_value:
            try:
                recent = list(rt.h1_bars)[-scfg.stop_engine_params.htf_lookback_bars:]
                struct_low = min(b.low for b in recent)
                struct_high = max(b.high for b in recent)
                sl0, _ = compute_initial_sl(fill.price, sig.side, struct_low, struct_high, rt.h1_atr_value, scfg.stop_engine_params)
                sl0_q = quantize_sl(sl0, sig.side, tick)
                R = abs(fill.price - sl0_q)
                stop_res = await self.executor.place_or_replace_stop(sig.symbol, sig.side, fill.qty, sl0_q, reason="INIT", tick_size=tick)
                if stop_res and stop_res.ok:
                    sl = sl0_q
                    if stop_res.order_ids:
                        rt.stop_order_id = stop_res.order_ids.get("stop", rt.stop_order_id)
                    rt.stop_state = StopState(
                        side=sig.side,
                        entry=fill.price,
                        sl=sl0_q,
                        R=R,
                        peak=fill.price,
                        trough=fill.price,
                        be_done=False,
                        trailing_active=False,
                    )
                    stop_info.update({"sl": sl0_q, "sl0": sl0_q, "R": R, "atr": rt.h1_atr_value})
                    rt.last_stop_replace_ms = int(time.time() * 1000)
                else:
                    msg = stop_res.msg if stop_res else "stop_failed"
                    await self._handle_stop_failure(sig, fill, scfg, stop_info, msg)
                    return
            except Exception as e:
                log.warning("Stop-engine placement failed for %s: %s", sig.symbol, e)
                await self._handle_stop_failure(sig, fill, scfg, stop_info, str(e))
                return

        if sl is not None and R is not None:
            tp_target = compute_tp_from_R(fill.price, sig.side, R, scfg.stop_engine_params.tp_r_mult)
            tp_px = quantize_tp(tp_target, sig.side, tick)
            try:
                tp_res = await self.executor.place_or_replace_tp(sig.symbol, sig.side, fill.qty, tp_px, tick)
                if tp_res and tp_res.ok:
                    tp = tp_px
                    if tp_res.order_ids:
                        rt.tp_order_id = tp_res.order_ids.get("tp", rt.tp_order_id)
            except Exception as e:
                log.warning("TP placement failed for %s: %s", sig.symbol, e)
            stop_info.update({"tp": tp or tp_target})

        self.store.log_trade({
            "ts_ms": fill.ts_ms,
            "symbol": sig.symbol,
            "side": sig.side,
            "order_id": None,
            "status": "FILLED",
            "qty": fill.qty,
            "entry_price": fill.price,
            "sl": sl,
            "tp": tp,
            "mode": self.exec_mode,
        })

        stop_info.setdefault("atr", rt.h1_atr_value)

        await self.notifier.send(
            formatters.format_execution(
                branding=self.branding,
                sig=sig,
                order={},
                qty=fill.qty,
                leverage=scfg.orders_cfg.get("leverage", 1),
                notional=scfg.orders_cfg.get("notional_usdt", 25.0),
                stop_info=stop_info,
                exec_mode=self.exec_mode,
                fill=fill,
            )
        )

    def _on_paper_fill(self, fill: ExecFill) -> None:
        rt = self.symbols_rt.get(fill.symbol)
        if not rt or rt.pending_signal is None or rt.filters is None:
            return
        # Schedule async handling
        asyncio.create_task(self._on_fill(rt.pending_signal, fill, rt.filters))

    def _on_paper_exit(self, symbol: str, reason: str, exit_px: float, pnl: float, ts_ms: int) -> None:
        async def _notify():
            rt = self.symbols_rt[symbol]
            rt.in_position = False
            rt.position_side = None
            rt.position_qty = None
            rt.stop_state = None
            rt.stop_order_id = None
            rt.tp_order_id = None
            if self._positions_total > 0:
                self._positions_total -= 1
            try:
                await self.notifier.send(f"EXIT detected for {symbol} (paper) reason={reason} pnl={pnl:.6f} px={exit_px:.8f}")
            except Exception:
                log.warning("Failed to send paper exit for %s", symbol)
        asyncio.create_task(_notify())

    async def _on_bar_close(self, symbol: str, bar: Bar, heal_gaps: bool = True, emit_alerts: bool = True) -> None:
        rt = self.symbols_rt[symbol]
        if heal_gaps:
            await self._heal_if_gap(symbol, rt, bar, emit_alerts)
        rt.last_bar_open_ms = bar.open_time_ms
        if self.executor:
            await self.executor.on_bar_close(symbol, bar.close, bar.close_time_ms)
        scfg = rt.cfg
        if scfg.stop_engine_params.enabled and rt.stop_state:
            update_extremes(rt.stop_state, high=bar.high, low=bar.low)

        # trailing update on bar close
        await self._maybe_update_stop(symbol, reason="BAR_CLOSE")

        sig = rt.strat.on_bar_close(bar)
        if not sig:
            return

        # If TradingView bridge is enabled, internal strategy signals are used
        # only for parity/matching (depending on mode). Trading decisions can be
        # driven by TV (tv_only) or by a TV+BOT handshake (tv_and_bot).
        if self.tv_cfg.enabled and self.tv_cfg.mode in ("tv_only", "tv_and_bot"):
            if emit_alerts:
                await self._on_bot_signal_tv_mode(sig, rt)
            return
        if rt.desynced and self.parity_cfg.desync_pause and emit_alerts:
            self._log_signal_suppressed(sig, "desynced_state")
            return
        if not emit_alerts:
            return
        key = self._signal_key(sig)
        if key in self._sent_alert_keys:
            self._log_signal_suppressed(sig, "duplicate", {"idempotency_key": key})
            return

        # Direction filter
        direction = self.direction_gate.get_direction()
        if direction == "long_only" and sig.side == "SHORT":
            self._log_signal_suppressed(sig, "direction_filter", {"direction": direction})
            return
        if direction == "short_only" and sig.side == "LONG":
            self._log_signal_suppressed(sig, "direction_filter", {"direction": direction})
            return

        now_ms = int(time.time() * 1000)
        cooldown_ms = self.cooldown_minutes * 60 * 1000
        if (now_ms - rt.last_signal_ts_ms) < cooldown_ms:
            self._log_signal_suppressed(sig, "cooldown", {"cooldown_ms": cooldown_ms})
            return
        rt.last_signal_ts_ms = now_ms
        self._last_signal_ts_ms = now_ms
        self._sent_alert_keys.add(key)

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

        stop_preview = {
            "enabled": scfg.stop_engine_params.enabled,
            "ready": rt.stop_ready,
            "htf_interval": scfg.stop_engine_params.htf_interval,
            "htf_lookback_bars": scfg.stop_engine_params.htf_lookback_bars,
            "atr_len": scfg.stop_engine_params.atr_len,
            "atr": rt.h1_atr_value,
            "buffer_bps": scfg.stop_engine_params.buffer_bps,
            "k_init": scfg.stop_engine_params.k_init,
            "tp_r_mult": scfg.stop_engine_params.tp_r_mult,
            "trailing_enabled": scfg.stop_engine_params.trailing.enabled,
            "trail_trigger_r": scfg.stop_engine_params.trailing.trigger_r,
            "k_trail": scfg.stop_engine_params.trailing.k_trail,
            "lock_r": scfg.stop_engine_params.trailing.lock_r,
            "be_trigger_r": scfg.stop_engine_params.be_trigger_r,
            "be_buffer_bps": scfg.stop_engine_params.be_buffer_bps,
        }

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

        # Preview SL/TP if stop engine is ready
        if (
            scfg.stop_engine_params.enabled
            and rt.stop_ready
            and rt.h1_bars
            and rt.h1_atr_value is not None
        ):
            try:
                recent = list(rt.h1_bars)[-scfg.stop_engine_params.htf_lookback_bars:]
                struct_low = min(b.low for b in recent)
                struct_high = max(b.high for b in recent)
                tick_preview = filters.tick_size if filters else 0.0
                sl0_raw, _ = compute_initial_sl(sig.entry_price, sig.side, struct_low, struct_high, rt.h1_atr_value, scfg.stop_engine_params)
                sl0_q = quantize_sl(sl0_raw, sig.side, tick_preview)
                R_q = abs(sig.entry_price - sl0_q)
                tp0 = compute_tp_from_R(sig.entry_price, sig.side, R_q, scfg.stop_engine_params.tp_r_mult)
                tp0_q = quantize_tp(tp0, sig.side, tick_preview)
                stop_preview.update({"sl0": sl0_q, "tp0": tp0_q, "R": R_q})
            except Exception as e:
                log.warning("stop_preview_failed %s: %s", sig.symbol, e)

        # Telegram alert ALWAYS (pre-trade)
        await self.notifier.send(
            formatters.format_entry(
                branding=self.branding,
                sig=sig,
                strat_params=scfg.strategy_params,
                stop_info=stop_preview,
                divergence_mode=scfg.divergence_mode,
                timeframe=self.timeframe,
                testnet=self.testnet,
                mode=self.mode,
                exec_mode=self.exec_mode,
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
        if scfg.stop_engine_params.enabled and not rt.stop_ready:
            self._log_signal_suppressed(sig, "stop_engine_not_ready")
            return

        if self.mode != "live" or not self._live_enabled:
            return

        if filters is None and self.rest:
            filters = await self.rest.get_symbol_filters(sig.symbol)
        if filters:
            rt.filters = filters
        if filters is None:
            return

        qty_raw = notional / sig.entry_price
        qty = quantize(qty_raw, filters.step_size)
        lev = int(orders_cfg.get("leverage", 1))
        if self.exec_mode == "binance" and self.rest:
            await self.rest.set_leverage(sig.symbol, lev)
        if qty < filters.min_qty:
            err = RuntimeError(f"Qty too small for {sig.symbol}: {qty} < min {filters.min_qty}")
            await self.notifier.send(
                formatters.format_execution(
                    branding=self.branding,
                    sig=sig,
                    order={"orderId": "n/a", "status": "REJECTED"},
                    qty=qty,
                    leverage=lev,
                    notional=notional,
                    exec_mode=self.exec_mode,
                    stop_info=stop_preview,
                    error=err,
                )
            )
            return

        if self.executor is None:
            return
        rt.pending_signal = sig
        res = await self.executor.place_entry(sig.symbol, sig.side, qty, ref="signal")
        if not res.ok:
            await self.notifier.send(
                formatters.format_execution(
                    branding=self.branding,
                    sig=sig,
                    order={"orderId": "n/a", "status": res.msg},
                    qty=qty,
                    leverage=lev,
                    notional=notional,
                    exec_mode=self.exec_mode,
                    stop_info=stop_preview,
                    error=RuntimeError(res.msg),
                )
            )
            return
        fill_obj = res.fill
        if fill_obj is None and self.exec_mode == "binance":
            fill_obj = ExecFill(symbol=sig.symbol, side=sig.side, qty=qty, price=sig.entry_price, fee_paid=0.0, ts_ms=now_ms, mode="binance")
        if fill_obj:
            await self._on_fill(sig, fill_obj, filters)

    async def _handle_position_closed(self, symbol: str) -> None:
        rt = self.symbols_rt[symbol]
        if not rt.in_position:
            return
        rt.in_position = False
        rt.position_side = None
        rt.position_qty = None
        rt.stop_state = None
        if self._positions_total > 0:
            self._positions_total -= 1
        if self.rest:
            for oid in (rt.stop_order_id, rt.tp_order_id):
                if not oid:
                    continue
                try:
                    await self.rest.cancel_order(symbol, oid)
                except Exception as e:
                    log.warning("Cancel failed for %s order %s: %s", symbol, oid, e)
        rt.stop_order_id = None
        rt.tp_order_id = None
        try:
            await self.notifier.send(f"EXIT detected for {symbol} ({self.exec_mode})")
        except Exception:
            log.warning("Failed to send exit notice for %s", symbol)

    async def _position_watchdog_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(5)
                if self.rest is None:
                    continue
                positions = []
                try:
                    positions = await self.rest.get_position_risk()
                except Exception as e:
                    log.warning("Position check batch failed: %s", e)
                    continue
                pos_map: Dict[str, float] = {}
                if isinstance(positions, list):
                    for row in positions:
                        sym = row.get("symbol")
                        try:
                            pos_map[sym] = float(row.get("positionAmt", 0))
                        except Exception:
                            pos_map[sym] = 0.0
                for sym, rt in self.symbols_rt.items():
                    if not rt.in_position:
                        continue
                    amt = pos_map.get(sym, 0.0)
                    if abs(amt) <= 1e-9:
                        await self._handle_position_closed(sym)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("Position watchdog error: %s", e)
