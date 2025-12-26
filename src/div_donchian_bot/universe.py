from __future__ import annotations

from typing import List, Dict, Any
import logging

from .binance.rest import BinanceRest

log = logging.getLogger("universe")

async def build_symbol_universe(cfg: Dict[str, Any], rest: BinanceRest) -> List[str]:
    u = cfg.get("universe", {}) or {}
    symbols = u.get("symbols", []) or []
    if symbols:
        log.info("Universe: using explicit symbols (%d)", len(symbols))
        return [s.upper() for s in symbols]

    if not u.get("dynamic", True):
        raise ValueError("Universe has no explicit symbols and dynamic=false.")

    allow_bases = [x.upper() for x in (u.get("allow_bases", []) or [])]
    quote_asset = (u.get("quote_asset", "USDT") or "USDT").upper()

    info = await rest.exchange_info()
    out = []
    for s in info.get("symbols", []):
        if s.get("status") != "TRADING":
            continue
        sym = s.get("symbol", "")
        base = (s.get("baseAsset") or "")
        quote = (s.get("quoteAsset") or "")
        if quote.upper() != quote_asset:
            continue
        if allow_bases and base.upper() not in allow_bases:
            continue
        # For futures, avoid BUSD or weird contracts; filter by contractType if present
        if "contractType" in s and s.get("contractType") not in (None, "PERPETUAL"):
            continue
        out.append(sym)

    log.info("Universe: dynamic symbols=%d (quote=%s allow_bases=%s)", len(out), quote_asset, allow_bases or "ALL")
    return out
