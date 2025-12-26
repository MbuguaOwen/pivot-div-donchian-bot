from __future__ import annotations

import datetime as dt
import html
from typing import Any, Dict, List, Optional

from ..models import Signal
from ..risk.atr_risk import AtrRiskParams, RiskLevels, atr_levels
from ..strategy.pivot_div_donchian import StrategyParams

MAX_TELEGRAM_CHARS = 4096

def escape(val: Any) -> str:
    return html.escape(str(val)) if val is not None else ""

def _ts_utc(ts_ms: Optional[int]) -> str:
    if not ts_ms:
        return "n/a"
    return dt.datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d %H:%M:%SZ")

def _footer(branding: str) -> str:
    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    return f"<pre>Time: {now} | Bot: {escape(branding)}</pre>"

def _pre(lines: List[str]) -> str:
    return "<pre>\n" + "\n".join(lines) + "\n</pre>"

def _truncate(text: str) -> str:
    if len(text) <= MAX_TELEGRAM_CHARS:
        return text
    suffix = "... [truncated]"
    return text[: MAX_TELEGRAM_CHARS - len(suffix)] + suffix

# --- Formatters ---

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
    direction: str,
) -> str:
    sym_count = len(symbols)
    sym_preview = ", ".join(symbols[:5]) + (" ..." if sym_count > 5 else "")
    cvd_count = len(cvd_symbols)
    cvd_preview = ", ".join(cvd_symbols[:5]) + (" ..." if cvd_count > 5 else "")
    net = "Testnet" if testnet else "Mainnet"
    header = f"<b>🏛 {escape(branding)}</b>"
    block = _pre([
        f"MODE: {mode.upper():<6} | TF: {escape(timeframe):<5} | MKT: {market_type} | NET: {net} | DIR: {direction}",
        f"SYMS: {sym_count:<4} ({escape(sym_preview)}) | CVD: {cvd_count:<4} ({escape(cvd_preview)})",
        f"ATR: {atr_enabled} | Limits: max_pos={max_positions_total}, cooldown={cooldown_minutes}m | Heartbeat: {heartbeat_sec}s",
    ])
    return _truncate("\n".join([header, block, _footer(branding)]))


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
    block = _pre([
        f"MODE: {mode.upper():<6} | NET: {'Testnet' if testnet else 'Mainnet'} | SYMS: {symbols_count}",
        f"POS: {positions}/{max_positions} | LastSignal: {last_sig} | Uptime: {uptime_seconds}s",
    ])
    return _truncate("\n".join([header, block, _footer(branding)]))


def format_entry(
    branding: str,
    sig: Signal,
    atr: float,
    strat_params: StrategyParams,
    atr_params: AtrRiskParams,
    divergence_mode: str,
    timeframe: str,
    testnet: bool,
    mode: str,
    cooldown_minutes: int,
    max_positions_total: int,
    one_pos_per_symbol: bool,
    notional_usdt: Any,
    planned_qty: Optional[float] = None,
) -> str:
    header = "<b>📌 TRADE SIGNAL — ENTRY</b>"
    net_label = "Testnet" if testnet else "Mainnet"
    mode_line = f"MODE: {mode.upper():<6} | TF: {escape(timeframe):<5} | NET: {net_label}"
    ctx = _pre([
        f"SYM: {escape(sig.symbol):<12} | SIDE: {escape(sig.side):<5} | TIME: {_ts_utc(sig.confirm_time_ms)}",
        f"{mode_line}",
    ])
    div_line = f"DivMode: {escape(divergence_mode.upper())} | Pine: {sig.pine_div} | CVD: {sig.cvd_div}"
    osc_line = f"PivotOsc: {sig.pivot_osc_value:.6f}"
    if sig.pivot_cvd_value is not None:
        osc_line += f" | PivotCVD: {sig.pivot_cvd_value:.6f}"
    sig_block = _pre([
        f"Entry: {sig.entry_price:.8f} | Pivot: {sig.pivot_price:.8f} | Slip: {sig.slip_bps:+.1f} bps",
        f"{div_line}",
        f"{osc_line}",
        f"Loc@Pivot: {sig.loc_at_pivot:.3f} | DonLen: {strat_params.don_len} | PivotLen: {strat_params.pivot_len} | ExtBand: {strat_params.ext_band_pct:.2f}",
    ])
    if atr_params.enabled:
        levels = atr_levels(sig.side, sig.entry_price, atr, atr_params)
        sl_txt = f"{levels.sl:.8f}" if levels.sl else "n/a"
        tp_txt = f"{levels.tp:.8f}" if levels.tp else "n/a"
        risk_block = _pre([
            f"ATR{atr_params.length}: {atr:.8f}",
            f"SL: {sl_txt} | TP: {tp_txt} | SLx: {atr_params.sl_mult} | TPx: {atr_params.tp_mult}",
        ])
    else:
        risk_block = _pre([
            f"ATR{atr_params.length}: {atr:.8f}",
            "SL/TP: DISABLED",
        ])
    exec_block = _pre([
        f"Mode: {mode.upper()} | Notional: ${notional_usdt} | Qty: {planned_qty if planned_qty is not None else 'n/a'}",
    ])
    ctrls = _pre([
        f"Cooldown: {cooldown_minutes}m | MaxPos: {max_positions_total} | OnePos/Symbol: {one_pos_per_symbol}",
    ])
    return _truncate("\n".join([header, ctx, "<b>Signal</b>", sig_block, "<b>Risk</b>", risk_block, "<b>Execution</b>", exec_block, "<b>Controls</b>", ctrls, _footer(branding)]))


def format_execution(
    branding: str,
    sig: Signal,
    order: Dict[str, Any],
    levels: RiskLevels,
    qty: float,
    leverage: Any,
    notional: Any,
    atr_params: AtrRiskParams,
    atr_value: float,
    error: Optional[Exception] = None,
) -> str:
    header = "<b>🤝 EXECUTION RESULT</b>"
    sig_block = _pre([
        f"SYM: {escape(sig.symbol):<12} | SIDE: {escape(sig.side):<5} | QTY: {qty}",
        f"Notional: ${notional} | Lev: {leverage}",
    ])
    ord_block = _pre([
        f"OrderId: {order.get('orderId', 'n/a')} | Status: {order.get('status', 'n/a')}",
        f"AvgFill: {order.get('avgPrice', order.get('price', 'n/a'))}",
    ])
    if atr_params.enabled and levels.sl and levels.tp:
        risk_line = f"SL: {levels.sl:.8f} | TP: {levels.tp:.8f} | ATR{atr_params.length}: {atr_value:.8f}"
    else:
        risk_line = f"SL/TP: DISABLED | ATR{atr_params.length}: {atr_value:.8f}"
    risk_block = _pre([risk_line])
    err_block: List[str] = []
    if error is not None:
        err_block = ["<b>Error</b>", _pre([f"{escape(error)}"])]
    parts = [header, sig_block, "<b>Order</b>", ord_block, "<b>Risk</b>", risk_block]
    parts.extend(err_block)
    parts.append(_footer(branding))
    return _truncate("\n".join(parts))


def format_error(branding: str, context: str, exc: Exception) -> str:
    header = "<b>⚠️ ERROR</b>"
    ctx = _pre([
        f"Context: {escape(context)}",
        f"Error:   {escape(exc)}",
    ])
    return _truncate("\n".join([header, ctx, _footer(branding)]))
