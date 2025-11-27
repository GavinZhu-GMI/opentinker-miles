#!/usr/bin/env python3
"""Standalone entrypoint for the Miles training API."""
import asyncio
import logging
import os

from training.server import serve

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("training-main")


def main() -> None:
    logger.info("Starting Miles training API server")
    asyncio.run(serve())


if __name__ == "__main__":
    main()
