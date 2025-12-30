from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..config import load_config, deep_merge
from ..models import Bar
from ..strategy.pivot_div_donchian import StrategyParams, SymbolStrategyState
from ..binance.rest import SymbolFilters
from ..engine import BotEngine, SymbolRuntime
from ..execution.paper import PaperConfig, PaperExecutor
from ..data.resample import resample_bars


@dataclass
class AggTrade:
    ts_ms: int
    price: float
    qty: float
    is_buyer_maker: bool
    symbol: Optional[str] = None


def _parse_time_arg(val: Optional[str]) -> Optional[int]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        num = int(s)
        return num
    except Exception:
        pass
    from datetime import datetime, timezone
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception as e:
        raise ValueError(f"Invalid time value: {val}") from e


def _load_bars(path: Path, default_symbol: str) -> Tuple[List[Bar], str]:
    bars: List[Bar] = []
    symbol = default_symbol
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row.get("symbol", symbol)
            bars.append(Bar(
                open_time_ms=int(row["open_time_ms"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                close_time_ms=int(row["close_time_ms"]),
            ))
    bars.sort(key=lambda b: b.open_time_ms)
    return bars, symbol


def _load_agg_trades(path: Optional[Path], default_symbol: str) -> List[AggTrade]:
    if path is None:
        return []
    trades: List[AggTrade] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append(AggTrade(
                ts_ms=int(row["ts_ms"]),
                price=float(row.get("price", 0.0)),
                qty=float(row["qty"]),
                is_buyer_maker=str(row["is_buyer_maker"]).lower() in ("true", "1", "yes"),
                symbol=row.get("symbol", default_symbol),
            ))
    trades.sort(key=lambda t: t.ts_ms)
    return trades


def _load_tv_signals(path: Optional[Path], default_symbol: str) -> Set[Tuple[str, str, int]]:
    if path is None:
        return set()
    expected: Set[Tuple[str, str, int]] = set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = row.get("symbol", default_symbol)
            expected.add((sym, row["side"], int(row["confirm_time_ms"])))
    return expected


def _build_params(cfg: Dict[str, Any]) -> Tuple[StrategyParams, bool]:
    strat_cfg = cfg.get("strategy", {}) or {}
    parity_cfg = cfg.get("parity", {}) or {}
    divergence_mode = str(strat_cfg.get("divergence", {}).get("mode", "pine")).lower()
    don_len = int(strat_cfg["donchian"]["length"])
    pivot_len = int(strat_cfg["pivot"]["length"])
    osc_len = int(strat_cfg["oscillator"]["ema_length"])
    ext_band = float(strat_cfg["donchian"]["extreme_band_pct"])
    ema_seed_mode = str(parity_cfg.get("ema_seed_mode", "first")).lower() if parity_cfg.get("mode", False) else "first"
    pivot_tie_break = str(parity_cfg.get("pivot_tie_break", "strict")).lower() if parity_cfg.get("mode", False) else "strict"
    base_warmup = max(don_len, 5 * osc_len, 3 * (2 * pivot_len + 1)) + 2
    warmup_bars = max(int(parity_cfg.get("warmup_bars", 0)), base_warmup) if parity_cfg.get("mode", False) else 0
    params = StrategyParams(
        don_len=don_len,
        ext_band_pct=ext_band,
        pivot_len=pivot_len,
        osc_ema_len=osc_len,
        divergence_mode=divergence_mode,
        ema_seed_mode=ema_seed_mode,
        pivot_tie_break=pivot_tie_break,
        warmup_bars=warmup_bars,
    )
    enable_cvd = divergence_mode in ("cvd", "both", "either")
    return params, enable_cvd


def _filter_bars(bars: List[Bar], start_ms: Optional[int], end_ms: Optional[int]) -> List[Bar]:
    out = []
    for b in bars:
        if start_ms is not None and b.close_time_ms < start_ms:
            continue
        if end_ms is not None and b.close_time_ms > end_ms:
            continue
        out.append(b)
    return out


def _calc_metrics(actual: List[Tuple[str, str, int]], expected: Set[Tuple[str, str, int]]) -> Dict[str, Any]:
    actual_set = set(actual)
    overlap = actual_set & expected
    precision = 1.0 if not actual_set and not expected else (len(overlap) / len(actual_set)) if actual_set else 0.0
    recall = 1.0 if not expected and not actual_set else (len(overlap) / len(expected)) if expected else 0.0
    missing = sorted(expected - actual_set, key=lambda x: x[2])
    extra = sorted(actual_set - expected, key=lambda x: x[2])
    first_mismatch = missing[0] if missing else (extra[0] if extra else None)
    return {
        "precision": precision,
        "recall": recall,
        "actual_count": len(actual_set),
        "expected_count": len(expected),
        "missing": missing,
        "extra": extra,
        "first_mismatch": first_mismatch,
    }

def _filter_expected_signals(expected: Set[Tuple[str, str, int]], start_ms: Optional[int], end_ms: Optional[int]) -> Set[Tuple[str, str, int]]:
    if not expected:
        return expected
    out: Set[Tuple[str, str, int]] = set()
    for sym, side, ts in expected:
        if start_ms is not None and ts < start_ms:
            continue
        if end_ms is not None and ts > end_ms:
            continue
        out.add((sym, side, ts))
    return out


def _replay_strategy(bars: List[Bar], trades: List[AggTrade], params: StrategyParams, enable_cvd: bool, symbol: str, expected: Set[Tuple[str, str, int]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    strat = SymbolStrategyState(symbol=symbol, params=params, enable_cvd=enable_cvd)
    trades_sorted = sorted(trades, key=lambda t: t.ts_ms)
    t_idx = 0
    signals: List[Dict[str, Any]] = []
    actual_keys: List[Tuple[str, str, int]] = []
    for b in bars:
        while t_idx < len(trades_sorted) and trades_sorted[t_idx].ts_ms < b.open_time_ms:
            t_idx += 1
        j = t_idx
        while j < len(trades_sorted) and trades_sorted[j].ts_ms <= b.close_time_ms:
            strat.on_agg_trade(trades_sorted[j].ts_ms, trades_sorted[j].qty, trades_sorted[j].is_buyer_maker)
            j += 1
        t_idx = j
        sig = strat.on_bar_close(b)
        if not sig:
            continue
        actual_keys.append((sig.symbol, sig.side, sig.confirm_time_ms))
        signals.append({
            "symbol": sig.symbol,
            "side": sig.side,
            "confirm_time_ms": sig.confirm_time_ms,
            "entry_price": sig.entry_price,
            "pivot_time_ms": sig.pivot_time_ms,
            "pivot_price": sig.pivot_price,
            "pine_div": sig.pine_div,
            "cvd_div": sig.cvd_div,
            "loc_at_pivot": sig.loc_at_pivot,
        })
    metrics = _calc_metrics(actual_keys, expected) if expected else {
        "precision": None,
        "recall": None,
        "actual_count": len(actual_keys),
        "expected_count": 0,
        "missing": [],
        "extra": [],
        "first_mismatch": None,
    }
    return signals, metrics


class _StaticRest:
    def __init__(self, filters: SymbolFilters):
        self.filters = filters

    async def get_symbol_filters(self, symbol: str) -> SymbolFilters:
        return self.filters

    async def stop(self) -> None:  # pragma: no cover - placeholder for interface symmetry
        return None


async def _replay_engine_for_tf(cfg: Dict[str, Any], tf: str, bars: List[Bar], trades: List[AggTrade], htf_bars: List[Bar], out_dir: Path, symbol: str, force_paper: bool, telegram_enabled: bool) -> None:
    cfg_base = copy.deepcopy(cfg)
    cfg_local = deep_merge(cfg_base, {
        "storage": {"out_dir": str(out_dir)},
        "run": {"timeframe": tf},
        "tv_bridge": {"enabled": False},
        "telegram": {"enabled": telegram_enabled},
    })
    if force_paper:
        cfg_local.setdefault("execution", {})
        cfg_local["execution"]["mode"] = "paper"

    engine = BotEngine(cfg_local)
    engine.cooldown_minutes = 0 if engine.parity_cfg.mode else int((cfg_local.get("limits", {}) or {}).get("cooldown_minutes_per_symbol", 30))
    engine.max_positions_total = int((cfg_local.get("limits", {}) or {}).get("max_positions_total", 20))
    engine.orders_cfg = cfg_local.get("orders", {}) or {}
    engine.position_cfg = cfg_local.get("position", {}) or {}
    engine.symbols = [symbol]
    engine.timeframe = tf

    scfg = engine._build_symbol_config(symbol)
    engine.symbol_cfgs[symbol] = scfg

    h1_bars = None
    h1_atr = None
    if scfg.stop_engine_params.enabled:
        from collections import deque
        from ..strategy.indicators import AtrState
        h1_bars = deque(maxlen=scfg.stop_engine_params.htf_lookback_bars + scfg.stop_engine_params.atr_len + 5)
        h1_atr = AtrState(scfg.stop_engine_params.atr_len, ema_seed_mode=engine.parity_cfg.ema_seed_mode if engine.parity_cfg.mode else "first")

    rt = SymbolRuntime(
        cfg=scfg,
        strat=SymbolStrategyState(symbol, scfg.strategy_params, enable_cvd=scfg.enable_cvd),
        h1_atr=h1_atr,
        h1_bars=h1_bars,
    )
    engine.symbols_rt[symbol] = rt

    # Static filters for paper replay (no REST calls)
    static_filters = SymbolFilters(step_size=0.001, min_qty=0.0, tick_size=0.01)
    engine.rest = _StaticRest(static_filters)

    pcfg = PaperConfig(
        fee_bps=float(engine.paper_cfg.get("fee_bps", 4.0)),
        slippage_bps=float(engine.paper_cfg.get("slippage_bps", 1.5)),
        fill_policy=str(engine.paper_cfg.get("fill_policy", "next_tick")),
        max_wait_ms=int(engine.paper_cfg.get("max_wait_ms", 3000)),
        use_mark_price=bool(engine.paper_cfg.get("use_mark_price", False)),
        log_trades=bool(engine.paper_cfg.get("log_trades", True)),
    )
    engine.exec_mode = "paper"
    engine.mode = cfg_local.get("run", {}).get("mode", engine.mode)
    engine.executor = PaperExecutor(pcfg, on_exit=engine._on_paper_exit, on_fill=engine._on_paper_fill)
    await engine.executor.start()

    trades_sorted = sorted(trades, key=lambda t: t.ts_ms)
    htf_sorted = sorted(htf_bars, key=lambda b: b.close_time_ms)
    trade_idx = 0
    h_idx = 0
    from ..risk.structure_atr_trailing import update_extremes

    for b in bars:
        while h_idx < len(htf_sorted) and htf_sorted[h_idx].close_time_ms <= b.close_time_ms:
            await engine._on_h1_close(symbol, htf_sorted[h_idx], emit_alerts=False)
            h_idx += 1

        while trade_idx < len(trades_sorted) and trades_sorted[trade_idx].ts_ms < b.open_time_ms:
            trade_idx += 1
        while trade_idx < len(trades_sorted) and trades_sorted[trade_idx].ts_ms <= b.close_time_ms:
            t = trades_sorted[trade_idx]
            rt.last_price = t.price
            if engine.executor:
                await engine.executor.on_trade_tick(symbol, t.price, t.ts_ms)
            if rt.cfg.stop_engine_params.enabled and rt.stop_state:
                update_extremes(rt.stop_state, price=t.price)
                await engine._maybe_update_stop(symbol, reason="TICK")
            if rt.cfg.enable_cvd:
                rt.strat.on_agg_trade(t.ts_ms, t.qty, t.is_buyer_maker)
            trade_idx += 1

        await engine._on_bar_close(symbol, b, heal_gaps=False, emit_alerts=True)

    await engine.executor.stop()
    await engine.notifier.stop()


def _build_run_id(symbol: str, tfs: Sequence[str], start_ms: Optional[int], end_ms: Optional[int], divergence_mode: str) -> str:
    start_tag = start_ms if start_ms is not None else "start"
    end_tag = end_ms if end_ms is not None else "end"
    tf_tag = "-".join(tfs)
    return f"{symbol}_{start_tag}_{end_tag}_{tf_tag}_{divergence_mode}"


def _write_signals(path: Path, signals: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "symbol", "side", "confirm_time_ms", "entry_price", "pivot_time_ms", "pivot_price", "pine_div", "cvd_div", "loc_at_pivot"
        ])
        w.writeheader()
        for row in signals:
            w.writerow(row)


def _resample_timeframes(raw_bars: List[Bar], tfs: Sequence[str], start_ms: Optional[int], end_ms: Optional[int]) -> Dict[str, List[Bar]]:
    out: Dict[str, List[Bar]] = {}
    for tf in tfs:
        resampled = resample_bars(raw_bars, tf)
        out[tf] = _filter_bars(resampled, start_ms, end_ms)
    return out


def _filter_trades(trades: List[AggTrade], start_ms: Optional[int], end_ms: Optional[int], symbol: Optional[str]) -> List[AggTrade]:
    out = []
    for t in trades:
        if symbol and t.symbol and t.symbol != symbol:
            continue
        if start_ms is not None and t.ts_ms < start_ms:
            continue
        if end_ms is not None and t.ts_ms > end_ms:
            continue
        out.append(t)
    return out


def _htf_bars_for_stop(raw_bars: List[Bar], htf_interval: str, start_ms: Optional[int], end_ms: Optional[int]) -> List[Bar]:
    try:
        bars = resample_bars(raw_bars, htf_interval)
    except Exception:
        return []
    return _filter_bars(bars, start_ms, end_ms)


def _parse_timeframes(tf_arg: str) -> List[str]:
    tfs = [t.strip() for t in tf_arg.split(",") if t.strip()]
    if not tfs:
        raise ValueError("No timeframes provided")
    return tfs


def _strategy_mode(tf_bars: Dict[str, List[Bar]], trades: List[AggTrade], symbol: str, expected: Set[Tuple[str, str, int]], out_root: Path, params: StrategyParams, enable_cvd: bool) -> None:
    for tf, bars in tf_bars.items():
        tf_dir = out_root / tf
        signals, metrics = _replay_strategy(bars, trades, params, enable_cvd, symbol, expected)
        _write_signals(tf_dir / "bot_signals.csv", signals)
        (tf_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


async def _engine_mode(cfg: Dict[str, Any], tf_bars: Dict[str, List[Bar]], trades: List[AggTrade], raw_bars: List[Bar], symbol: str, out_root: Path, force_paper: bool, telegram_enabled: bool, start_ms: Optional[int], end_ms: Optional[int]) -> None:
    # Single stop interval from config (per-symbol)
    scfg = BotEngine(cfg)._build_symbol_config(symbol)
    htf_interval = scfg.stop_engine_params.htf_interval
    htf_bars = _htf_bars_for_stop(raw_bars, htf_interval, start_ms, end_ms) if scfg.stop_engine_params.enabled else []
    for tf, bars in tf_bars.items():
        await _replay_engine_for_tf(cfg, tf, bars, trades, htf_bars, out_root / tf, symbol, force_paper, telegram_enabled)


def main() -> None:
    ap = argparse.ArgumentParser(description="Historical replay harness for Pivot Div + Donchian bot")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--bars_csv", required=True, help="OHLCV CSV input (open_time_ms,open,high,low,close,volume,close_time_ms[,symbol])")
    ap.add_argument("--agg_csv", help="Optional aggTrades CSV (ts_ms,price,qty,is_buyer_maker[,symbol])")
    ap.add_argument("--tv_signals_csv", help="Optional TradingView signals CSV for metrics")
    ap.add_argument("--timeframes", required=True, help="Comma separated timeframes, e.g. 15m,1h,4h")
    ap.add_argument("--mode", choices=["strategy_only", "engine_paper"], default="strategy_only")
    ap.add_argument("--start", help="Start timestamp (ms or ISO8601)")
    ap.add_argument("--end", help="End timestamp (ms or ISO8601)")
    ap.add_argument("--out_dir", default="artifacts/replay", help="Output root directory")
    ap.add_argument("--run_id", help="Optional run id override")
    ap.add_argument("--symbol", help="Override symbol (defaults to bars/config)")
    ap.add_argument("--force_paper", dest="force_paper", action="store_true", help="Force execution.mode=paper for engine mode")
    ap.add_argument("--no_force_paper", dest="force_paper", action="store_false", help="Do not override execution.mode")
    ap.set_defaults(force_paper=True)
    ap.add_argument("--telegram", dest="telegram", action="store_true", help="Enable telegram during replay (default off)")
    ap.set_defaults(telegram=False)
    args = ap.parse_args()

    cfg = load_config(args.config)
    params, enable_cvd = _build_params(cfg)
    divergence_mode = params.divergence_mode

    tfs = _parse_timeframes(args.timeframes)
    start_ms = _parse_time_arg(args.start)
    end_ms = _parse_time_arg(args.end)

    bars_raw, symbol_from_bars = _load_bars(Path(args.bars_csv), (cfg.get("universe", {}).get("symbols") or ["UNKNOWN"])[0])
    symbol = args.symbol or symbol_from_bars or (cfg.get("universe", {}).get("symbols") or ["UNKNOWN"])[0]
    tf_bars = _resample_timeframes(bars_raw, tfs, start_ms, end_ms)

    trades = _load_agg_trades(Path(args.agg_csv), symbol) if args.agg_csv else []
    trades = _filter_trades(trades, start_ms, end_ms, symbol)
    if enable_cvd and not trades:
        raise RuntimeError("CVD divergence requested but no aggTrades provided (use --agg_csv)")

    expected = _load_tv_signals(Path(args.tv_signals_csv), symbol) if args.tv_signals_csv else set()
    expected = _filter_expected_signals(expected, start_ms, end_ms)
    run_id = args.run_id or _build_run_id(symbol, tfs, start_ms, end_ms, divergence_mode)
    out_root = Path(args.out_dir) / run_id

    if args.mode == "strategy_only":
        _strategy_mode(tf_bars, trades, symbol, expected, out_root, params, enable_cvd)
    else:
        asyncio.run(_engine_mode(cfg, tf_bars, trades, bars_raw, symbol, out_root, args.force_paper, args.telegram, start_ms, end_ms))


if __name__ == "__main__":
    main()
