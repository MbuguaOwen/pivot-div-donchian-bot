from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode

import aiohttp

log = logging.getLogger("binance.rest")

@dataclass
class SymbolFilters:
    step_size: float
    min_qty: float
    tick_size: float
    min_notional: Optional[float] = None

class BinanceRest:
    def __init__(self, base_url: str, market_type: str, recv_window_ms: int = 5000):
        self.base_url = base_url.rstrip("/")
        self.market_type = market_type.lower()
        self.recv_window_ms = recv_window_ms
        self.key = os.getenv("BINANCE_API_KEY", "")
        self.secret = os.getenv("BINANCE_API_SECRET", "")
        self._session: Optional[aiohttp.ClientSession] = None
        self._filters_cache: Dict[str, SymbolFilters] = {}

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(headers={"X-MBX-APIKEY": self.key} if self.key else None)

    async def stop(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    def _sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.secret:
            raise RuntimeError("BINANCE_API_SECRET missing (set .env)")
        query = urlencode(params, doseq=True)
        sig = hmac.new(self.secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
        params["signature"] = sig
        return params

    async def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None, signed: bool = False) -> Any:
        if self._session is None:
            await self.start()
        assert self._session is not None
        params = dict(params or {})
        url = f"{self.base_url}{path}"

        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = self.recv_window_ms
            params = self._sign(params)

        if method.upper() == "GET":
            async with self._session.get(url, params=params, timeout=20) as r:
                data = await r.json(content_type=None)
                if r.status != 200:
                    raise RuntimeError(f"Binance GET {path} failed {r.status}: {data}")
                return data

        async with self._session.request(method.upper(), url, params=params, timeout=20) as r:
            data = await r.json(content_type=None)
            if r.status != 200:
                raise RuntimeError(f"Binance {method} {path} failed {r.status}: {data}")
            return data

    async def exchange_info(self) -> Any:
        if self.market_type == "futures":
            return await self._request("GET", "/fapi/v1/exchangeInfo")
        return await self._request("GET", "/api/v3/exchangeInfo")

    async def get_symbols(self) -> list[str]:
        info = await self.exchange_info()
        syms = []
        for s in info.get("symbols", []):
            if s.get("status") != "TRADING":
                continue
            syms.append(s["symbol"])
        return syms

    async def get_symbol_filters(self, symbol: str) -> SymbolFilters:
        if symbol in self._filters_cache:
            return self._filters_cache[symbol]

        info = await self.exchange_info()
        target = None
        for s in info.get("symbols", []):
            if s.get("symbol") == symbol:
                target = s
                break
        if target is None:
            raise ValueError(f"Symbol not found: {symbol}")

        step = min_qty = tick = None
        min_notional = None
        for f in target.get("filters", []):
            t = f.get("filterType")
            if t == "LOT_SIZE":
                step = float(f["stepSize"])
                min_qty = float(f["minQty"])
            if t in ("PRICE_FILTER","PERCENT_PRICE"):
                if "tickSize" in f:
                    tick = float(f["tickSize"])
            if t in ("MIN_NOTIONAL",):
                if "notional" in f:
                    min_notional = float(f["notional"])
                elif "minNotional" in f:
                    min_notional = float(f["minNotional"])

        if step is None or min_qty is None:
            # Futures uses MARKET_LOT_SIZE sometimes; try that
            for f in target.get("filters", []):
                if f.get("filterType") == "MARKET_LOT_SIZE":
                    step = float(f["stepSize"])
                    min_qty = float(f["minQty"])
                    break

        if step is None or min_qty is None or tick is None:
            # Tick might be absent for futures in some responses; default safe
            tick = tick or 0.01
            step = step or 0.001
            min_qty = min_qty or 0.001

        out = SymbolFilters(step_size=step, min_qty=min_qty, tick_size=tick, min_notional=min_notional)
        self._filters_cache[symbol] = out
        return out

    async def price(self, symbol: str) -> float:
        if self.market_type == "futures":
            data = await self._request("GET", "/fapi/v1/ticker/price", {"symbol": symbol})
            return float(data["price"])
        data = await self._request("GET", "/api/v3/ticker/price", {"symbol": symbol})
        return float(data["price"])

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        if self.market_type != "futures":
            return
        try:
            await self._request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}, signed=True)
        except Exception as e:
            log.warning("Failed to set leverage for %s: %s", symbol, e)

    async def order_market(self, symbol: str, side: str, qty: float, reduce_only: bool = False) -> Any:
        if self.market_type == "futures":
            params = {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": qty,
            }
            if reduce_only:
                params["reduceOnly"] = "true"
            return await self._request("POST", "/fapi/v1/order", params, signed=True)
        # spot
        params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": qty}
        return await self._request("POST", "/api/v3/order", params, signed=True)

    async def order_stop_market(self, symbol: str, side: str, qty: float, stop_price: float, reduce_only: bool = True) -> Any:
        if self.market_type != "futures":
            raise RuntimeError("STOP_MARKET supported only for futures in this bot.")
        params = {
            "symbol": symbol,
            "side": side,
            "type": "STOP_MARKET",
            "stopPrice": stop_price,
            "closePosition": "false",
            "quantity": qty,
            "reduceOnly": "true" if reduce_only else "false",
            "timeInForce": "GTC",
            "workingType": "CONTRACT_PRICE",
        }
        return await self._request("POST", "/fapi/v1/order", params, signed=True)

    async def order_take_profit_market(self, symbol: str, side: str, qty: float, stop_price: float, reduce_only: bool = True) -> Any:
        if self.market_type != "futures":
            raise RuntimeError("TAKE_PROFIT_MARKET supported only for futures in this bot.")
        params = {
            "symbol": symbol,
            "side": side,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": stop_price,
            "closePosition": "false",
            "quantity": qty,
            "reduceOnly": "true" if reduce_only else "false",
            "timeInForce": "GTC",
            "workingType": "CONTRACT_PRICE",
        }
        return await self._request("POST", "/fapi/v1/order", params, signed=True)

def quantize(x: float, step: float) -> float:
    if step <= 0:
        return x
    # floor to step
    return (int(x / step)) * step
