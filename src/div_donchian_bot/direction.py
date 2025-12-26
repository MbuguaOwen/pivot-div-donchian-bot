from __future__ import annotations

from typing import Literal

Direction = Literal["both", "long_only", "short_only"]


class DirectionGate:
    def __init__(self, default: Direction = "both"):
        self._direction: Direction = default

    def set_direction(self, direction: Direction) -> None:
        self._direction = direction

    def get_direction(self) -> Direction:
        return self._direction

    def allow(self, side: str) -> bool:
        """Return True if the given side ('LONG'/'SHORT') is permitted."""
        if self._direction == "long_only" and side == "SHORT":
            return False
        if self._direction == "short_only" and side == "LONG":
            return False
        return True
