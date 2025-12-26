from __future__ import annotations

def endpoints(market_type: str, testnet: bool) -> dict:
    market_type = market_type.lower()
    if market_type == "futures":
        if testnet:
            return {
                "rest": "https://testnet.binancefuture.com",
                # Combined streams base (append /stream?streams=...)
                "ws": "wss://stream.binancefuture.com",
            }
        return {
            "rest": "https://fapi.binance.com",
            "ws": "wss://fstream.binance.com",
        }
    # spot
    if testnet:
        return {
            "rest": "https://testnet.binance.vision",
            "ws": "wss://testnet.binance.vision",
        }
    return {
        "rest": "https://api.binance.com",
        "ws": "wss://stream.binance.com:9443",
    }
