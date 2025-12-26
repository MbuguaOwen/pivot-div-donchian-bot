from __future__ import annotations

import argparse
import asyncio
import logging

from dotenv import load_dotenv

from ..config import load_config
from ..logging_setup import setup_logging
from ..binance.endpoints import endpoints
from ..binance.rest import BinanceRest
from ..universe import build_symbol_universe
from ..engine import BotEngine

log = logging.getLogger("cli")


async def main_async(cfg_path: str) -> None:
    load_dotenv()
    cfg = load_config(cfg_path)

    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    testnet = bool(cfg.get("exchange", {}).get("testnet", True))
    market_type = str(cfg.get("exchange", {}).get("market_type", "futures")).lower()

    ep = endpoints(market_type=market_type, testnet=testnet)

    rest = BinanceRest(ep["rest"], market_type=market_type, recv_window_ms=int(cfg.get("exchange", {}).get("recv_window_ms", 5000)))
    await rest.start()

    symbols = await build_symbol_universe(cfg, rest)
    if not symbols:
        raise RuntimeError("No symbols selected. Update configs/default.yaml universe.*")

    engine = BotEngine(cfg)
    await engine.start(rest=rest, ws_url=ep["ws"], symbols=symbols)

    # Run forever
    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        log.info("Stopping...")
    finally:
        await engine.stop()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config (e.g. configs/default.yaml)")
    args = ap.parse_args()
    asyncio.run(main_async(args.config))


if __name__ == "__main__":
    main()
