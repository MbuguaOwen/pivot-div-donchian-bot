from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from .base import ExecutionAdapter, ExecFill, ExecResult, Side

log = logging.getLogger("execution.paper")


@dataclass
class PaperPosition:
    symbol: str
    side: Side
    qty: float
    entry_px: float
    entry_ts: int
    stop_px: Optional[float] = None
    tp_px: Optional[float] = None
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    open: bool = True


@dataclass
class PendingOrder:
    symbol: str
    side: Side
    qty: float
    created_ms: int
    deadline_ms: int
    ref: str


@dataclass
class PaperConfig:
    fee_bps: float
    slippage_bps: float
    max_wait_ms: int
    fill_policy: str = "next_tick"
    use_mark_price: bool = False
    log_trades: bool = True


class PaperExecutor(ExecutionAdapter):
    def __init__(self, cfg: PaperConfig, on_exit: Callable[[str, str, float, float, int], None], on_fill: Optional[Callable[[ExecFill], None]] = None):
        self.cfg = cfg
        self.on_exit = on_exit
        self.on_fill = on_fill
        self.positions: Dict[str, PaperPosition] = {}
        self.pending: Dict[str, PendingOrder] = {}
        self.last_close: Dict[str, float] = {}

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        self.pending.clear()

    def _slip_mult(self, side: Side, is_entry: bool, price: float) -> float:
        slip = self.cfg.slippage_bps / 10000.0
        if side == "LONG":
            return price * (1.0 + slip if is_entry else 1.0 - slip)
        return price * (1.0 - slip if is_entry else 1.0 + slip)

    def _fee(self, notional: float) -> float:
        return notional * (self.cfg.fee_bps / 10000.0)

    async def place_entry(self, symbol: str, side: Side, qty: float, ref: str) -> ExecResult:
        now_ms = int(time.time() * 1000)
        po = PendingOrder(
            symbol=symbol,
            side=side,
            qty=qty,
            created_ms=now_ms,
            deadline_ms=now_ms + self.cfg.max_wait_ms,
            ref=ref,
        )
        self.pending[symbol] = po
        return ExecResult(ok=True, msg="PENDING")

    def _fill_pending(self, symbol: str, price: float, ts_ms: int) -> Optional[ExecResult]:
        po = self.pending.pop(symbol, None)
        if not po:
            return None
        fill_px = self._slip_mult(po.side, True, price)
        fee = self._fee(fill_px * po.qty)
        pos = PaperPosition(symbol=po.symbol, side=po.side, qty=po.qty, entry_px=fill_px, entry_ts=ts_ms, fees_paid=fee)
        self.positions[symbol] = pos
        fill = ExecFill(symbol=symbol, side=po.side, qty=po.qty, price=fill_px, fee_paid=fee, ts_ms=ts_ms, mode="paper")
        if self.cfg.log_trades:
            log.info("paper_fill symbol=%s side=%s qty=%.8f px=%.8f fee=%.8f", symbol, po.side, po.qty, fill_px, fee)
        if self.on_fill:
            try:
                self.on_fill(fill)
            except Exception as e:
                log.warning("paper_fill_callback_error %s: %s", symbol, e)
        return ExecResult(ok=True, msg="FILLED", fill=fill)

    async def on_trade_tick(self, symbol: str, price: float, ts_ms: int) -> None:
        try:
            if symbol in self.pending:
                res = self._fill_pending(symbol, price, ts_ms)
                if res:
                    return
            pos = self.positions.get(symbol)
            if not pos or not pos.open:
                return
            slip_price_stop = self._slip_mult(pos.side, False, price)
            # stop/TP checks
            if pos.side == "LONG":
                if pos.stop_px is not None and price <= pos.stop_px:
                    await self._exit(symbol, "STOP", slip_price_stop, ts_ms)
                    return
                if pos.tp_px is not None and price >= pos.tp_px:
                    await self._exit(symbol, "TP", slip_price_stop, ts_ms)
            else:
                if pos.stop_px is not None and price >= pos.stop_px:
                    await self._exit(symbol, "STOP", slip_price_stop, ts_ms)
                    return
                if pos.tp_px is not None and price <= pos.tp_px:
                    await self._exit(symbol, "TP", slip_price_stop, ts_ms)
        except Exception as e:
            log.warning("paper_tick_error %s: %s", symbol, e)

    async def on_bar_close(self, symbol: str, close: float, ts_ms: int) -> None:
        self.last_close[symbol] = close
        po = self.pending.get(symbol)
        if not po:
            return
        if ts_ms >= po.deadline_ms and self.cfg.fill_policy == "bar_close_fallback":
            res = self._fill_pending(symbol, close, ts_ms)
            if res:
                return

    async def place_or_replace_stop(self, symbol: str, side: Side, qty: float, stop_px: float, reason: str, tick_size: float) -> ExecResult:
        pos = self.positions.get(symbol)
        if not pos or not pos.open:
            return ExecResult(ok=False, msg="no_position")
        pos.stop_px = stop_px
        return ExecResult(ok=True, msg="stop_set")

    async def place_or_replace_tp(self, symbol: str, side: Side, qty: float, tp_px: float, tick_size: float) -> ExecResult:
        pos = self.positions.get(symbol)
        if not pos or not pos.open:
            return ExecResult(ok=False, msg="no_position")
        pos.tp_px = tp_px
        return ExecResult(ok=True, msg="tp_set")

    async def close_position_market(self, symbol: str, entry_side: Side, qty: float, price_hint: float | None = None) -> ExecResult:
        pos = self.positions.get(symbol)
        if not pos or not pos.open:
            return ExecResult(ok=False, msg="no_position")
        px = price_hint or self.last_close.get(symbol) or pos.entry_px
        exit_px = self._slip_mult(entry_side, False, px)
        ts_ms = int(time.time() * 1000)
        await self._exit(symbol, "FORCE_EXIT", exit_px, ts_ms)
        return ExecResult(ok=True, msg="closed")

    async def cancel_protection(self, symbol: str) -> None:
        pos = self.positions.get(symbol)
        if pos:
            pos.stop_px = None
            pos.tp_px = None

    async def get_position(self, symbol: str) -> dict:
        pos = self.positions.get(symbol)
        if not pos or not pos.open:
            return {}
        return {
            "symbol": pos.symbol,
            "side": pos.side,
            "qty": pos.qty,
            "entry_px": pos.entry_px,
            "stop_px": pos.stop_px,
            "tp_px": pos.tp_px,
            "fees_paid": pos.fees_paid,
            "entry_ts": pos.entry_ts,
        }

    async def has_position(self, symbol: str) -> bool:
        pos = self.positions.get(symbol)
        return bool(pos and pos.open)

    async def _exit(self, symbol: str, reason: str, exit_px: float, ts_ms: int) -> None:
        pos = self.positions.get(symbol)
        if not pos or not pos.open:
            return
        exit_fee = self._fee(exit_px * pos.qty)
        pos.fees_paid += exit_fee
        if pos.side == "LONG":
            pnl = (exit_px - pos.entry_px) * pos.qty - pos.fees_paid
        else:
            pnl = (pos.entry_px - exit_px) * pos.qty - pos.fees_paid
        pos.realized_pnl = pnl
        pos.open = False
        if self.cfg.log_trades:
            log.info("paper_exit symbol=%s reason=%s exit_px=%.8f pnl=%.8f fees=%.8f", symbol, reason, exit_px, pnl, pos.fees_paid)
        try:
            self.on_exit(symbol, reason, exit_px, pnl, ts_ms)
        except Exception as e:
            log.warning("paper_exit_callback_error %s: %s", symbol, e)
