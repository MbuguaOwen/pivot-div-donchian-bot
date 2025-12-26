from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp

from .direction import DirectionGate, Direction

log = logging.getLogger("telegram.controls")


class TelegramControls:
    def __init__(
        self,
        token: str,
        allowed_chat_id: Optional[int],
        state_path: str,
        gate: DirectionGate,
        notifier_send,
        parse_mode: str = "HTML",
    ):
        self.token = token
        self.allowed_chat_id = allowed_chat_id
        self.state_path = Path(state_path)
        self.gate = gate
        self.notifier_send = notifier_send
        self.parse_mode = parse_mode
        self._session: Optional[aiohttp.ClientSession] = None
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._last_update_id: int = 0

    async def start(self) -> None:
        if not self.token:
            log.warning("Telegram controls enabled but token missing")
            return
        self._load_state()
        self._session = aiohttp.ClientSession()
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            self._task = None
        if self._session:
            await self._session.close()
            self._session = None

    def _load_state(self) -> None:
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                self._last_update_id = int(data.get("last_update_id", 0))
                dir_val = str(data.get("direction", self.gate.get_direction()))
                self.gate.set_direction(dir_val)  # type: ignore[arg-type]
            except Exception as e:
                log.warning("Failed to load telegram control state: %s", e)

    def _save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "last_update_id": self._last_update_id,
                "direction": self.gate.get_direction(),
            }
            self.state_path.write_text(json.dumps(data), encoding="utf-8")
        except Exception as e:
            log.warning("Failed to persist telegram control state: %s", e)

    async def _poll_loop(self) -> None:
        assert self._session is not None
        url = f"https://api.telegram.org/bot{self.token}/getUpdates"
        while not self._stop.is_set():
            try:
                params = {"timeout": 30, "offset": self._last_update_id + 1, "allowed_updates": ["callback_query"]}
                async with self._session.get(url, params=params, timeout=35) as resp:
                    data = await resp.json(content_type=None)
                    if not data.get("ok"):
                        await asyncio.sleep(5)
                        continue
                    for update in data.get("result", []):
                        self._last_update_id = max(self._last_update_id, int(update.get("update_id", 0)))
                        await self._handle_update(update)
                    self._save_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("Telegram controls polling error: %s", e)
                await asyncio.sleep(5)

    async def _handle_update(self, update: Dict[str, Any]) -> None:
        cq = update.get("callback_query")
        if not cq:
            return
        from_obj = cq.get("from", {})
        chat_id = from_obj.get("id")
        if self.allowed_chat_id is not None and chat_id != self.allowed_chat_id:
            return
        data = str(cq.get("data") or "")
        if not data.startswith("dir:"):
            return
        dir_val = data.split(":", 1)[1]
        if dir_val not in ("both", "long_only", "short_only"):
            return
        self.gate.set_direction(dir_val)  # type: ignore[arg-type]
        self._save_state()
        try:
            await self.notifier_send(f"Direction set to {dir_val.upper()}")
        except Exception:
            log.exception("Failed to send direction ack")


def control_keyboard() -> Dict[str, Any]:
    return {
        "inline_keyboard": [[
            {"text": "LONG ONLY", "callback_data": "dir:long_only"},
            {"text": "SHORT ONLY", "callback_data": "dir:short_only"},
            {"text": "BOTH", "callback_data": "dir:both"},
        ]]
    }
