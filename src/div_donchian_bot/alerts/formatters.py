from __future__ import annotations

import datetime as dt
import html
from typing import Any, Dict, List, Optional

from ..models import Signal
from ..risk.atr_risk import AtrRiskParams, RiskLevels
from ..strategy.pivot_div_donchian import StrategyParams

def escape(val: Any) -> str:
    return html.escape(str(val)) if val is not None else ""

def _ts_utc(ts_ms: Optional[int]) -> str:
    if not ts_ms:
        return "n/a"
    return dt.datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d %H:%M:%SZ")

def _footer(branding: str) -> str:
    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    return f"<pre>Time: {now} | Bot: {escape(branding)}</pre>"

def _pre(block: List[str]) -> str:
    return "<pre>\n" + "\n".join(block) + "\n</pre>"

def format_startup(
    branding: str,
    mode: str,
    testnet: bool,
    exchange: str,
    market_type: str,
    timeframe: str,
    symbols: List[str],
    cvd_symbols: List[str],
    atr_enabled: bool,
    max_positions_total: int,
    cooldown_minutes: int,
    heartbeat_sec: int,
) -> str:
    sym_count = len(symbols)
    cvd_count = len(cvd_symbols)
    cvd_preview = ", ".join(cvd_symbols[:10]) if cvd_symbols else "none"
    header = f"<b>✅ SYSTEM ONLINE — {escape(branding)}</b>"
    ctx = _pre([
        f"Mode: {mode.upper():<6}  Testnet: {str(testnet):<5}  Exchange: {exchange} {market_type}",
        f"TF:   {timeframe:<6}  Symbols: {sym_count:<4}  CVD: {cvd_count:<4}",
        f"CVD Symbols: {escape(cvd_preview)}",
    ])
    risk = _pre([
        f"ATR SL/TP: {str(atr_enabled):<5}",
        f"Limits: max_pos={max_positions_total}  cooldown={cooldown_minutes}m",
        f"Heartbeat: {heartbeat_sec}s",
    ])
    return "\n".join([header, ctx, "<b>Controls</b>", risk, _footer(branding)])

def format_heartbeat(
    branding: str,
    mode: str,
    testnet: bool,
    positions: int,
    max_positions: int,
    symbols_count: int,
    last_signal_ts_ms: Optional[int],
    uptime_seconds: int,
) -> str:
    header = f"<b>⏱ HEARTBEAT — {escape(branding)}</b>"
    last_sig = _ts_utc(last_signal_ts_ms)
    ctx = _pre([
        f"Mode: {mode.upper():<6}  Testnet: {str(testnet):<5}  Symbols: {symbols_count}",
        f"Positions: {positions}/{max_positions}  LastSignal: {last_sig}",
        f"Uptime: {uptime_seconds}s",
    ])
    return "\n".join([header, ctx, _footer(branding)])

def format_entry_signal(
    branding: str,
    sig: Signal,
    atr: float,
    strat_params: StrategyParams,
    divergence_mode: str,
    timeframe: str,
    mode: str,
    cooldown_minutes: int,
    max_positions_total: int,
    one_pos_per_symbol: bool,
) -> str:
    header = "<b>📌 TRADE SIGNAL — ENTRY</b>"
    ctx = _pre([
        f"Symbol: {escape(sig.symbol):<12} TF: {escape(timeframe):<6} Mode: {escape(mode.upper())}",
        f"Side:   {escape(sig.side):<12} Div: {escape(divergence_mode.upper()):<6} Time: {_ts_utc(sig.confirm_time_ms)}",
    ])
    sig_block = _pre([
        f"Entry:  {sig.entry_price:.8f}",
        f"Pivot:  {sig.pivot_price:.8f}      Slip: {sig.slip_bps:+.1f} bps",
        f"Loc@P:  {sig.loc_at_pivot:.3f}      DonLen: {strat_params.don_len:<3}   PivotLen: {strat_params.pivot_len:<3}   ExtBand: {strat_params.ext_band_pct:.2f}",
    ])
    risk_block = _pre([
        f"ATR:  {atr:.8f}",
        "SL/TP: DISABLED",  # ATR disabled by default; informational only here
    ])
    ctrls = _pre([
        f"Cooldown: {cooldown_minutes}m   MaxPos: {max_positions_total}   OnePos/Symbol: {one_pos_per_symbol}",
    ])
    return "\n".join([header, ctx, "<b>Signal</b>", sig_block, "<b>Risk (ATR)</b>", risk_block, "<b>Controls</b>", ctrls, _footer(branding)])

def format_execution_result(
    branding: str,
    sig: Signal,
    order: Dict[str, Any],
    levels: RiskLevels,
    qty: float,
    leverage: Any,
    notional: Any,
    atr_enabled: bool,
    error: Optional[Exception] = None,
) -> str:
    header = "<b>🤝 EXECUTION RESULT</b>"
    sig_block = _pre([
        f"Symbol: {escape(sig.symbol):<12} Side: {escape(sig.side):<6} Qty: {qty}",
        f"Notional: ${notional}  Lev: {leverage}",
    ])
    ord_block = _pre([
        f"OrderId: {order.get('orderId', 'n/a')}  Status: {order.get('status', 'n/a')}",
        f"AvgFill: {order.get('avgPrice', order.get('price', 'n/a'))}",
    ])
    if atr_enabled and levels.sl and levels.tp:
        risk_line = f"SL: {levels.sl:.8f}  TP: {levels.tp:.8f}"
    else:
        risk_line = "SL/TP: DISABLED"
    risk_block = _pre([risk_line])
    err_block = []
    if error is not None:
        err_block = ["<b>Error</b>", _pre([f"{escape(error)}"])]
    parts = [header, sig_block, "<b>Order</b>", ord_block, "<b>Risk</b>", risk_block]
    parts.extend(err_block)
    parts.append(_footer(branding))
    return "\n".join(parts)

def format_error(branding: str, context: str, exc: Exception) -> str:
    header = "<b>⚠️ ERROR</b>"
    ctx = _pre([
        f"Context: {escape(context)}",
        f"Error:   {escape(exc)}",
    ])
    return "\n".join([header, ctx, _footer(branding)])
