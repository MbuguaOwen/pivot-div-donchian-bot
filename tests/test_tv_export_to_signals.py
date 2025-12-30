import csv
from datetime import datetime, timezone
from pathlib import Path

from div_donchian_bot.cli.tv_export_to_signals import convert_tv_export


def test_tv_export_handles_multiple_column_names(tmp_path: Path):
    src = tmp_path / "tv_export.csv"
    out = tmp_path / "tv_signals.csv"
    with src.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Entry Time", "Direction", "Ticker"])
        w.writerow(["2024-01-01T00:00:00Z", "Long", "BTCUSDT"])
        w.writerow(["2024-01-01T01:00:00Z", "Short", "BTCUSDT"])

    total, written = convert_tv_export(src, out, "BTCUSDT", "auto", "binance_end_minus_1")
    assert total == 2
    assert written == 2

    rows = list(csv.DictReader(out.open()))
    assert rows[0]["side"] == "LONG"
    assert rows[1]["side"] == "SHORT"
    dt = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    expected_ms = int(dt.timestamp() * 1000) - 1
    assert int(rows[0]["confirm_time_ms"]) == expected_ms
