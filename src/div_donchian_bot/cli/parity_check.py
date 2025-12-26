from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from ..config import load_config
from ..models import Bar
from ..strategy.pivot_div_donchian import StrategyParams, SymbolStrategyState


SignalKey = Tuple[str, str, int]


def _build_params(cfg: Dict[str, Any]) -> Tuple[StrategyParams, bool]:
    strat_cfg = cfg.get("strategy", {}) or {}
    parity_cfg = cfg.get("parity", {}) or {}
    divergence_mode = str(strat_cfg.get("divergence", {}).get("mode", "pine")).lower()
    parity_mode = bool(parity_cfg.get("mode", False))

    don_len = int(strat_cfg["donchian"]["length"])
    pivot_len = int(strat_cfg["pivot"]["length"])
    osc_len = int(strat_cfg["oscillator"]["ema_length"])
    ext_band = float(strat_cfg["donchian"]["extreme_band_pct"])

    ema_seed_mode = str(parity_cfg.get("ema_seed_mode", "first")).lower() if parity_mode else "first"
    pivot_tie_break = str(parity_cfg.get("pivot_tie_break", "strict")).lower() if parity_mode else "strict"
    base_warmup = max(don_len, 5 * osc_len, 3 * (2 * pivot_len + 1)) + 2
    warmup_bars = max(int(parity_cfg.get("warmup_bars", 0)), base_warmup) if parity_mode else 0

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
    return sorted(bars, key=lambda b: b.close_time_ms), symbol


def _load_expected(path: Path, default_symbol: str) -> Set[SignalKey]:
    expected: Set[SignalKey] = set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = row.get("symbol", default_symbol)
            expected.add((sym, row["side"], int(row["confirm_time_ms"])))
    return expected


def run_parity_check(bars_csv: Path, tv_signals_csv: Path, cfg_path: Path) -> Dict[str, Any]:
    cfg = load_config(str(cfg_path))
    params, enable_cvd = _build_params(cfg)
    symbol = (cfg.get("universe", {}).get("symbols") or ["UNKNOWN"])[0]

    bars, symbol_from_bars = _load_bars(bars_csv, symbol)
    symbol = symbol_from_bars or symbol

    strat = SymbolStrategyState(symbol=symbol, params=params, enable_cvd=enable_cvd)
    actual: List[SignalKey] = []
    for b in bars:
        sig = strat.on_bar_close(b)
        if sig:
            actual.append((sig.symbol, sig.side, sig.confirm_time_ms))

    actual_set = set(actual)
    expected_set = _load_expected(tv_signals_csv, symbol)
    overlap = actual_set & expected_set

    precision = 1.0 if not actual_set and not expected_set else (len(overlap) / len(actual_set)) if actual_set else 0.0
    recall = 1.0 if not expected_set and not actual_set else (len(overlap) / len(expected_set)) if expected_set else 0.0

    missing = sorted(expected_set - actual_set, key=lambda x: x[2])
    extra = sorted(actual_set - expected_set, key=lambda x: x[2])
    first_mismatch = missing[0] if missing else (extra[0] if extra else None)

    return {
        "precision": precision,
        "recall": recall,
        "actual_count": len(actual_set),
        "expected_count": len(expected_set),
        "missing": missing,
        "extra": extra,
        "first_mismatch": first_mismatch,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars_csv", required=True, help="OHLCV CSV with columns open_time_ms,open,high,low,close,volume,close_time_ms[,symbol]")
    ap.add_argument("--tv_signals_csv", required=True, help="CSV of expected signals with columns symbol,side,confirm_time_ms")
    ap.add_argument("--config", required=True, help="Path to config YAML")
    args = ap.parse_args()

    res = run_parity_check(Path(args.bars_csv), Path(args.tv_signals_csv), Path(args.config))
    print(json.dumps(res, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
