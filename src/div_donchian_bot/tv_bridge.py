from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

from aiohttp import web

from .models import Side

log = logging.getLogger("tv_bridge")


def normalize_tv_symbol(sym: str) -> str:
    """Best-effort normalization of TradingView ticker IDs.

    Common inputs:
      - "BTCUSDT"
      - "BINANCE:BTCUSDT"
      - "BINANCE:BTCUSDT.P"  (perps)
      - "BTCUSDT.P"
    """
    s = (sym or "").strip()
    if not s:
        return ""
    if ":" in s:
        s = s.split(":", 1)[1]
    s = s.upper()
    # common perp suffixes
    for suf in (".P", ".PERP", "PERP"):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return s


def normalize_tv_tf(tf: str) -> str:
    """Normalize timeframe strings to bot-style, e.g. "15" -> "15m", "1H" -> "1h"."""
    t = (tf or "").strip()
    if not t:
        return ""
    tl = t.lower()
    if tl.isdigit():
        return f"{int(tl)}m"
    # already like 15m / 1h
    if tl.endswith("m") or tl.endswith("h"):
        return tl
    # TradingView can send 1H, 4H, 1D in some contexts
    if tl.endswith("h"):
        return tl
    if tl.endswith("d"):
        return tl
    # last resort
    return tl


@dataclass
class TvSignal:
    symbol: str
    side: Side
    confirm_time_ms: int
    entry_price: Optional[float] = None
    pivot_price: Optional[float] = None
    pivot_osc: Optional[float] = None
    slip_bps: Optional[float] = None
    loc_at_pivot: Optional[float] = None
    tf: Optional[str] = None
    tickerid: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class TvBridgeConfig:
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 9010
    path: str = "/tv"
    # security
    secret: str = ""
    secret_env: str = "TV_WEBHOOK_SECRET"
    # modes: "tv_only" | "tv_and_bot"
    mode: str = "tv_only"
    require_tf_match: bool = True
    match_window_ms: int = 120_000
    alert_on_mismatch: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TvBridgeConfig":
        dd = d or {}
        secret = str(dd.get("secret") or "").strip()
        secret_env = str(dd.get("secret_env") or "TV_WEBHOOK_SECRET").strip()
        if not secret:
            secret = os.getenv(secret_env, "").strip()
        return cls(
            enabled=bool(dd.get("enabled", False)),
            host=str(dd.get("host", "0.0.0.0")),
            port=int(dd.get("port", 9010)),
            path=str(dd.get("path", "/tv")),
            secret=secret,
            secret_env=secret_env,
            mode=str(dd.get("mode", "tv_only")).lower(),
            require_tf_match=bool(dd.get("require_tf_match", True)),
            match_window_ms=int(dd.get("match_window_ms", 120_000)),
            alert_on_mismatch=bool(dd.get("alert_on_mismatch", True)),
        )


class TvWebhookServer:
    """Small aiohttp webhook server that receives TradingView alerts."""

    def __init__(
        self,
        cfg: TvBridgeConfig,
        on_signal: Callable[[TvSignal], Awaitable[None]],
    ):
        self.cfg = cfg
        self.on_signal = on_signal
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

    async def start(self) -> None:
        if not self.cfg.enabled:
            return
        self._app = web.Application()
        self._app.add_routes([web.post(self.cfg.path, self._handle)])
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, host=self.cfg.host, port=self.cfg.port)
        await self._site.start()
        log.info("tv_bridge_listening host=%s port=%s path=%s mode=%s", self.cfg.host, self.cfg.port, self.cfg.path, self.cfg.mode)

    async def stop(self) -> None:
        if self._site:
            await self._site.stop()
            self._site = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._app = None

    async def _handle(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception:
            txt = (await request.text()) if request.can_read_body else ""
            return web.json_response({"ok": False, "error": "invalid_json", "body": txt[:200]}, status=400)

        # Secret (optional but strongly recommended)
        if self.cfg.secret:
            got = str(payload.get("secret") or "")
            if got != self.cfg.secret:
                return web.json_response({"ok": False, "error": "bad_secret"}, status=401)

        raw_symbol = str(payload.get("symbol") or payload.get("ticker") or payload.get("tickerid") or "")
        symbol = normalize_tv_symbol(raw_symbol)
        side = str(payload.get("side") or "").upper()
        if side not in ("LONG", "SHORT"):
            return web.json_response({"ok": False, "error": "bad_side"}, status=400)

        try:
            confirm_time_ms = int(payload.get("confirm_time_ms") or payload.get("time_close_ms") or payload.get("t_close") or 0)
        except Exception:
            confirm_time_ms = 0
        if confirm_time_ms <= 0:
            # allow TV sending seconds; convert if needed
            try:
                maybe_s = int(payload.get("confirm_time") or 0)
                if 1_000_000_000 < maybe_s < 10_000_000_000:
                    confirm_time_ms = maybe_s * 1000
            except Exception:
                confirm_time_ms = 0
        if confirm_time_ms <= 0:
            return web.json_response({"ok": False, "error": "missing_confirm_time_ms"}, status=400)

        tf = payload.get("tf") or payload.get("timeframe")
        tf_norm = normalize_tv_tf(str(tf)) if tf is not None else None

        def fval(k: str) -> Optional[float]:
            if k not in payload or payload[k] is None:
                return None
            try:
                return float(payload[k])
            except Exception:
                return None

        sig = TvSignal(
            symbol=symbol,
            side=side,  # type: ignore[assignment]
            confirm_time_ms=confirm_time_ms,
            entry_price=fval("entry_price"),
            pivot_price=fval("pivot_price"),
            pivot_osc=fval("pivot_osc"),
            slip_bps=fval("slip_bps"),
            loc_at_pivot=fval("loc_at_pivot"),
            tf=tf_norm,
            tickerid=str(payload.get("tickerid") or payload.get("ticker") or "") or None,
            raw=payload,
        )

        # Fire and forget (don't block webhook response on execution)
        try:
            import asyncio

            asyncio.get_running_loop().create_task(self.on_signal(sig))
        except Exception as e:
            log.warning("tv_bridge_task_create_failed %s", e)

        return web.json_response({"ok": True, "symbol": symbol, "side": side, "confirm_time_ms": confirm_time_ms})
