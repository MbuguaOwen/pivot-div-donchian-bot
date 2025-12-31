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
        self.tv_signals_csv = self.out / "tv_signals.csv"
        self.parity_csv = self.out / "parity.csv"
        self.cvd_filter_csv = self.out / "cvd_filter.csv"
        self._init_files()

    def _init_files(self) -> None:
        if not self.events_csv.exists():
            with self.events_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "ts_ms","symbol","side","entry_price","pivot_price","pivot_osc","pivot_cvd","slip_bps","loc_at_pivot","oscillator","pine_div","cvd_div","pivot_time_ms","confirm_time_ms"
                ])
                w.writeheader()
        if not self.trades_csv.exists():
            with self.trades_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "ts_ms","symbol","side","order_id","status","qty","entry_price","sl","tp","mode"
                ])
                w.writeheader()

        if not self.tv_signals_csv.exists():
            with self.tv_signals_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "ts_ms","symbol","side","confirm_time_ms","tf","entry_price","pivot_price","pivot_osc","slip_bps","loc_at_pivot","tickerid"
                ])
                w.writeheader()

        if not self.parity_csv.exists():
            with self.parity_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "ts_ms","mode","symbol","side","confirm_time_ms","tv","bot","action","details"
                ])
                w.writeheader()

        if not self.cvd_filter_csv.exists():
            with self.cvd_filter_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "ts_ms","symbol","side","confirm_time_ms","end_ms","delta_window_m","delta_sum","signed_delta","thresh","pass","ok","reason","align","gaps","mode","source"
                ])
                w.writeheader()

    def log_event(self, d: Dict[str, Any]) -> None:
        with self.events_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "ts_ms","symbol","side","entry_price","pivot_price","pivot_osc","pivot_cvd","slip_bps","loc_at_pivot","oscillator","pine_div","cvd_div","pivot_time_ms","confirm_time_ms"
            ])
            w.writerow(d)

    def log_trade(self, d: Dict[str, Any]) -> None:
        with self.trades_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "ts_ms","symbol","side","order_id","status","qty","entry_price","sl","tp","mode"
            ])
            w.writerow(d)

    def log_tv_signal(self, d: Dict[str, Any]) -> None:
        with self.tv_signals_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "ts_ms","symbol","side","confirm_time_ms","tf","entry_price","pivot_price","pivot_osc","slip_bps","loc_at_pivot","tickerid"
            ])
            w.writerow(d)

    def log_parity(self, d: Dict[str, Any]) -> None:
        with self.parity_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "ts_ms","mode","symbol","side","confirm_time_ms","tv","bot","action","details"
            ])
            w.writerow(d)

    def log_cvd_filter(self, d: Dict[str, Any]) -> None:
        with self.cvd_filter_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "ts_ms","symbol","side","confirm_time_ms","end_ms","delta_window_m","delta_sum","signed_delta","thresh","pass","ok","reason","align","gaps","mode","source"
            ])
            w.writerow(d)
