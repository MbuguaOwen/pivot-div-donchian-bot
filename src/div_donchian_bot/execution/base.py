from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Dict

Side = Literal["LONG", "SHORT"]


@dataclass
class ExecFill:
    symbol: str
    side: Side
    qty: float
    price: float
    fee_paid: float
    ts_ms: int
    mode: str  # paper | binance


@dataclass
class ExecResult:
    ok: bool
    msg: str
    fill: Optional[ExecFill] = None
    order_ids: Optional[Dict] = None


class ExecutionAdapter:
    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    async def place_entry(self, symbol: str, side: Side, qty: float, ref: str) -> ExecResult:
        raise NotImplementedError

    async def place_or_replace_stop(self, symbol: str, side: Side, qty: float, stop_px: float, reason: str, tick_size: float) -> ExecResult:
        raise NotImplementedError

    async def place_or_replace_tp(self, symbol: str, side: Side, qty: float, tp_px: float, tick_size: float) -> ExecResult:
        raise NotImplementedError

    async def close_position_market(self, symbol: str, entry_side: Side, qty: float, price_hint: Optional[float] = None) -> ExecResult:
        raise NotImplementedError

    async def cancel_protection(self, symbol: str) -> None:
        raise NotImplementedError

    async def on_trade_tick(self, symbol: str, price: float, ts_ms: int) -> None:
        raise NotImplementedError

    async def on_bar_close(self, symbol: str, close: float, ts_ms: int) -> None:
        return None

    async def get_position(self, symbol: str) -> dict:
        raise NotImplementedError

    async def has_position(self, symbol: str) -> bool:
        raise NotImplementedError
