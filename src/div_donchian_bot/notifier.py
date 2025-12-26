from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import aiohttp

log = logging.getLogger("notifier")

class TelegramNotifier:
    def __init__(self, enabled: bool = True, parse_mode: str = "HTML"):
        self.enabled = enabled
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.parse_mode = parse_mode
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if not self.enabled:
            return
        if not self.token or not self.chat_id:
            log.warning("Telegram enabled but TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing.")
            self.enabled = False
            return
        self._session = aiohttp.ClientSession()

    async def stop(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def send(self, text: str) -> None:
        if not self.enabled:
            return
        if not self._session:
            await self.start()
            if not self.enabled:
                return
        # Telegram limit ~4096 chars
        if len(text) > 4096:
            suffix = "... [truncated]"
            text = text[: 4096 - len(suffix)] + suffix
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": True,
            "parse_mode": self.parse_mode,
        }
        # Basic rate limit: serialize sends
        async with self._lock:
            try:
                async with self._session.post(url, json=payload, timeout=20) as r:
                    if r.status != 200:
                        body = await r.text()
                        log.warning("Telegram send failed: %s %s", r.status, body[:500])
            except Exception as e:
                log.exception("Telegram send error: %s", e)
