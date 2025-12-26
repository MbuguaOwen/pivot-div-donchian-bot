import logging
import os

def setup_logging(level: str = "INFO") -> None:
    lvl = os.getenv("LOG_LEVEL", level).upper()
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
