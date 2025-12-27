from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from ..binance.rest import BinanceRest
from .base import ExecutionAdapter, ExecFill, ExecResult, Side

log = logging.getLogger("execution.binance")


class BinanceExecutor(ExecutionAdapter):
    def __init__(self, rest: BinanceRest):
        self.rest = rest

    def _client_oid(self, prefix: str, symbol: str) -> str:
        # Keep within Binance 36-char limit, avoid hyphens
        ts = int(time.time() * 1000)
        token = uuid.uuid4().hex[:8]
        base = f"{prefix}_{symbol}_{token}_{ts}"
        return base[:32]

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def place_entry(self, symbol: str, side: Side, qty: float, ref: str) -> ExecResult:
        order_side = "BUY" if side == "LONG" else "SELL"
        try:
            order = await self.rest.order_market(symbol, side=order_side, qty=qty, reduce_only=False)
            fill_price = order.get("avgPrice") or order.get("price")
            fill = None
            try:
                fill_px = float(fill_price)
                fill = ExecFill(symbol=symbol, side=side, qty=qty, price=fill_px, fee_paid=0.0, ts_ms=int(time.time() * 1000), mode="binance")
            except Exception:
                fill = None
            return ExecResult(ok=True, msg="FILLED", fill=fill, order_ids={"entry": order.get("orderId") if isinstance(order, dict) else None})
        except Exception as e:
            return ExecResult(ok=False, msg=str(e))

    async def place_or_replace_stop(self, symbol: str, side: Side, qty: float, stop_px: float, reason: str, tick_size: float) -> ExecResult:
        exit_side = "SELL" if side == "LONG" else "BUY"
        try:
            oid = self._client_oid("sl", symbol)
            order = await self.rest.order_stop_market(symbol, side=exit_side, qty=qty, stop_price=stop_px, reduce_only=True, client_order_id=oid)
            return ExecResult(ok=True, msg="stop_set", order_ids={"stop": order.get("orderId") if isinstance(order, dict) else None})
        except Exception as e:
            return ExecResult(ok=False, msg=str(e))

    async def place_or_replace_tp(self, symbol: str, side: Side, qty: float, tp_px: float, tick_size: float) -> ExecResult:
        exit_side = "SELL" if side == "LONG" else "BUY"
        try:
            oid = self._client_oid("tp", symbol)
            order = await self.rest.order_take_profit_market(symbol, side=exit_side, qty=qty, stop_price=tp_px, reduce_only=True, client_order_id=oid)
            return ExecResult(ok=True, msg="tp_set", order_ids={"tp": order.get("orderId") if isinstance(order, dict) else None})
        except Exception as e:
            return ExecResult(ok=False, msg=str(e))

    async def close_position_market(self, symbol: str, entry_side: Side, qty: float, price_hint: float | None = None) -> ExecResult:
        exit_side = "SELL" if entry_side == "LONG" else "BUY"
        try:
            order = await self.rest.order_market(symbol, side=exit_side, qty=qty, reduce_only=True)
            return ExecResult(ok=True, msg="closed", order_ids={"close": order.get("orderId") if isinstance(order, dict) else None})
        except Exception as e:
            return ExecResult(ok=False, msg=str(e))

    async def cancel_protection(self, symbol: str) -> None:
        try:
            orders = await self.rest.get_open_orders(symbol)
            for o in orders or []:
                t = o.get("origType") or o.get("type")
                if str(o.get("reduceOnly", "")).lower() not in ("true", "1", "yes"):
                    continue
                if t not in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
                    continue
                oid = o.get("orderId")
                if oid:
                    try:
                        await self.rest.cancel_order(symbol, oid)
                    except Exception as e:
                        log.warning("cancel_protection_failed %s %s: %s", symbol, oid, e)
        except Exception as e:
            log.warning("cancel_protection_error %s: %s", symbol, e)

    async def on_trade_tick(self, symbol: str, price: float, ts_ms: int) -> None:
        return None

    async def get_position(self, symbol: str) -> dict:
        data = await self.rest.get_position_risk(symbol)
        if isinstance(data, list):
            for row in data:
                if row.get("symbol") == symbol:
                    return row
        if isinstance(data, dict) and data.get("symbol") == symbol:
            return data
        return {}

    async def has_position(self, symbol: str) -> bool:
        pos = await self.get_position(symbol)
        try:
            amt = float(pos.get("positionAmt", 0))
            return abs(amt) > 1e-9
        except Exception:
            return False
