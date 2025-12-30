from __future__ import annotations

import argparse
import asyncio
import csv
import os
from pathlib import Path
from typing import Dict, List

import aiohttp


def _load_signals(path: Path) -> List[Dict[str, str]]:
    signals: List[Dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = int(row["confirm_time_ms"])
            except Exception:
                continue
            signals.append({
                "symbol": row.get("symbol", "").strip(),
                "side": row.get("side", "").strip(),
                "confirm_time_ms": ts,
            })
    signals.sort(key=lambda r: r["confirm_time_ms"])
    return signals


async def _post_with_retry(session: aiohttp.ClientSession, url: str, payload: Dict[str, object], retries: int = 3) -> None:
    delay = 1.0
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            async with session.post(url, json=payload, timeout=10) as resp:
                if resp.status < 300:
                    return
                body = await resp.text()
                last_err = RuntimeError(f"HTTP {resp.status}: {body[:200]}")
        except Exception as e:
            last_err = e
        if attempt >= retries:
            break
        await asyncio.sleep(delay)
        delay = min(delay * 2, 5.0)
    if last_err:
        raise last_err


async def _replay(tv_signals: List[Dict[str, str]], url: str, secret: str, sleep_sec: float) -> None:
    async with aiohttp.ClientSession() as session:
        for row in tv_signals:
            payload = {
                "secret": secret,
                "symbol": row["symbol"],
                "side": row["side"],
                "confirm_time_ms": int(row["confirm_time_ms"]),
            }
            await _post_with_retry(session, url, payload)
            if sleep_sec > 0:
                await asyncio.sleep(sleep_sec)


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay TradingView webhook payloads from a tv_signals.csv file")
    ap.add_argument("--tv_signals_csv", required=True, help="Path to tv_signals.csv")
    ap.add_argument("--url", required=True, help="Webhook URL (e.g., http://127.0.0.1:9001/tv)")
    ap.add_argument("--secret_env", default="TV_WEBHOOK_SECRET", help="Env var containing the webhook secret")
    ap.add_argument("--sleep", type=float, default=0.0, help="Optional sleep between posts (seconds)")
    args = ap.parse_args()

    secret = os.getenv(args.secret_env, "")
    if secret is None:
        secret = ""
    if not secret:
        raise RuntimeError(f"Secret missing in env var {args.secret_env}")

    signals = _load_signals(Path(args.tv_signals_csv))
    if not signals:
        raise RuntimeError("No signals found to replay")
    asyncio.run(_replay(signals, args.url, secret, float(args.sleep)))


if __name__ == "__main__":
    main()
