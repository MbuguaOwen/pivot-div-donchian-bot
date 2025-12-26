from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, Awaitable, Dict, List, Optional

import websockets

log = logging.getLogger("binance.ws")

class BinanceWsManager:
    def __init__(self, ws_base_url: str):
        # Expects base host; we append combined-stream path (/stream?streams=...)
        self.ws_base = ws_base_url.rstrip("/")
        self._tasks: List[asyncio.Task] = []
        self._stop = asyncio.Event()

    async def stop(self) -> None:
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        self._tasks.clear()

    def _mk_url(self, streams: List[str]) -> str:
        return f"{self.ws_base}/stream?streams=" + "/".join(streams)

    async def run_stream_group(self, streams: List[str], on_msg: Callable[[dict], Awaitable[None]]) -> None:
        url = self._mk_url(streams)
        backoff = 1
        while not self._stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20, close_timeout=10) as ws:
                    backoff = 1
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        try:
                            msg = json.loads(raw)
                            await on_msg(msg)
                        except Exception:
                            log.exception("WS message handling error")
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("WS connection error (%s). Reconnecting in %ss", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def start(self, stream_groups: List[List[str]], on_msg: Callable[[dict], Awaitable[None]]) -> None:
        for g in stream_groups:
            self._tasks.append(asyncio.create_task(self.run_stream_group(g, on_msg)))
