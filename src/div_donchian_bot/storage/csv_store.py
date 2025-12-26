from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Any

class CsvStore:
    def __init__(self, out_dir: str):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.events_csv = self.out / "events.csv"
        self.trades_csv = self.out / "trades.csv"
        self._init_files()

    def _init_files(self) -> None:
        if not self.events_csv.exists():
            with self.events_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "ts_ms","symbol","side","entry_price","pivot_price","slip_bps","loc_at_pivot","oscillator","pivot_time_ms","confirm_time_ms"
                ])
                w.writeheader()
        if not self.trades_csv.exists():
            with self.trades_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "ts_ms","symbol","side","order_id","status","qty","entry_price","sl","tp","mode"
                ])
                w.writeheader()

    def log_event(self, d: Dict[str, Any]) -> None:
        with self.events_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "ts_ms","symbol","side","entry_price","pivot_price","slip_bps","loc_at_pivot","oscillator","pivot_time_ms","confirm_time_ms"
            ])
            w.writerow(d)

    def log_trade(self, d: Dict[str, Any]) -> None:
        with self.trades_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "ts_ms","symbol","side","order_id","status","qty","entry_price","sl","tp","mode"
            ])
            w.writerow(d)
