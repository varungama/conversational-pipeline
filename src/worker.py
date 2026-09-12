"""The worker: claims one conversation file at a time from the queue,
scores it, and writes a durable result. Safe for multiple concurrent
processes - see queue_store.py for how claiming avoids double-scoring.
"""

import json
import logging
import pathlib
from datetime import datetime, timezone

from src import conversation as conv
from src.metrics import compute_overall_sentiment, compute_sentiment_trajectory
from src.queue_store import FileQueue
from src.result_index import ResultIndex
from src.sentiment_model import SentimentScorer

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"


def result_key(tenant_id: str, conversation_id: str) -> str:
    return f"{tenant_id}::{conversation_id}"


class Worker:
    def __init__(self, queue: FileQueue, results_dir: pathlib.Path, scorer: SentimentScorer):
        self.queue = queue
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.scorer = scorer
        self.index = ResultIndex(results_dir / "index.json")

    