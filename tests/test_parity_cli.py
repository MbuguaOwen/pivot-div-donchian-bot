from pathlib import Path

from div_donchian_bot.cli.parity_check import run_parity_check


def test_parity_check_handles_empty_signals():
    base = Path("tests/data")
    res = run_parity_check(
        bars_csv=base / "bars_sample.csv",
        tv_signals_csv=base / "tv_signals_empty.csv",
        cfg_path=Path("configs/default.yaml"),
    )
    assert res["missing"] == []
    assert res["extra"] == []
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
