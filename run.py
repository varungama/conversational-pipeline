"""Entry point: score every conversation in an input directory and write
durable JSON results.

Usage:
    python run.py --input data/conversations --output results

First run downloads the sentiment model (~256MB) and caches it; later
runs are offline. Falls back to VADER on download failure - see
src/sentiment_model.py.
"""

import argparse
import json
import logging
import pathlib
from datetime import datetime, timezone

from src.queue_store import FileQueue
from src.sentiment_model import SentimentScorer
from src.worker import Worker

logger = logging.getLogger(__name__)


def configure_logging(log_dir: pathlib.Path) -> pathlib.Path:
    """Logs to console and to a file under log_dir named by UTC start time,
    so concurrent/repeated runs never overwrite each other's history."""
    log_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"pipeline_{started_at}.log"

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return log_path


