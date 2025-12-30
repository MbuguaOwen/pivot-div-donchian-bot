from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple


TIME_COLUMNS = ["time", "entry time", "entry_time", "close time", "close_time", "timestamp", "date", "open time", "open_time"]
SIDE_COLUMNS = ["side", "direction", "order", "type", "position"]
SYMBOL_COLUMNS = ["symbol", "ticker", "tickerid"]


def _parse_time_to_ms(val: str) -> int:
    s = str(val).strip()
    if not s:
        raise ValueError("Empty time value")
    try:
        # numeric epoch (sec or ms)
        num = float(s)
        if num > 10_000_000_000:  # already ms
            return int(num)
        return int(num * 1000)
    except Exception:
        pass
    try:
        iso = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception as e:
        raise ValueError(f"Cannot parse time value: {s}") from e


def _find_col(headers: List[str], candidates: List[str]) -> str:
    lower_map = {h.lower(): h for h in headers}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    raise ValueError(f"Missing required column (tried {candidates})")


def _normalize_side(raw: str) -> str:
    s = str(raw).strip().lower()
    if not s:
        raise ValueError("Empty side value")
    if s in ("long", "buy", "b", "entry long", "open long"):
        return "LONG"
    if s in ("short", "sell", "s", "entry short", "open short"):
        return "SHORT"
    raise ValueError(f"Unrecognized side value: {raw}")


def convert_tv_export(in_path: Path, out_path: Path, symbol: str, side_map: str, confirm_time_mode: str) -> Tuple[int, int]:
    if side_map != "auto":
        raise ValueError(f"Unsupported side_map: {side_map} (only 'auto' is supported)")
    total_rows = 0
    with in_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        time_col = _find_col(headers, TIME_COLUMNS)
        side_col = _find_col(headers, SIDE_COLUMNS)
        sym_col = None
        try:
            sym_col = _find_col(headers, SYMBOL_COLUMNS)
        except Exception:
            sym_col = None

        rows = []
        for row in reader:
            total_rows += 1
            try:
                ts_ms = _parse_time_to_ms(row[time_col])
            except Exception:
                continue
            if confirm_time_mode == "binance_end_minus_1":
                ts_ms -= 1
            side = _normalize_side(row[side_col])
            sym_val = row.get(sym_col) if sym_col else None
            sym = sym_val.strip() if sym_val else symbol
            rows.append({"symbol": sym, "side": side, "confirm_time_ms": ts_ms})

    rows.sort(key=lambda r: r["confirm_time_ms"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "side", "confirm_time_ms"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return total_rows, len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert TradingView Strategy Tester export to tv_signals.csv")
    ap.add_argument("--in", dest="in_path", required=True, help="Path to TradingView exported trades CSV")
    ap.add_argument("--out", dest="out_path", required=True, help="Output tv_signals.csv path")
    ap.add_argument("--symbol", required=True, help="Symbol to fill when export lacks a symbol column")
    ap.add_argument("--side_map", default="auto", help="Side mapping mode (auto only for now)")
    ap.add_argument("--confirm_time_mode", choices=["tv_end", "binance_end_minus_1"], default="binance_end_minus_1", help="Interpret TV time as end-of-bar or end-1ms (Binance style)")
    args = ap.parse_args()

    convert_tv_export(Path(args.in_path), Path(args.out_path), args.symbol, args.side_map, args.confirm_time_mode)


if __name__ == "__main__":
    main()
