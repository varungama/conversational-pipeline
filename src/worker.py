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

    def run_once(self) -> dict:
        """Processes everything currently pending. Returns counts by outcome."""
        summary = {"processed": 0, "skipped_duplicate": 0, "dead_lettered": 0}
        for path in self.queue.list_pending():
            claimed = self.queue.claim(path)
            if claimed is None:
                continue  # another worker claimed this one first
            outcome = self._process_one(claimed)
            summary[outcome] = summary.get(outcome, 0) + 1
        return summary

    def _process_one(self, path: pathlib.Path) -> str:
        try:
            raw_doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.queue.mark_dead(path, f"unreadable_json: {exc}")
            return "dead_lettered"

        tenant_id = raw_doc.get("tenant_id") if isinstance(raw_doc, dict) else None
        conversation_id = raw_doc.get("conversation_id") if isinstance(raw_doc, dict) else None
        if not tenant_id or not conversation_id:
            self.queue.mark_dead(path, "missing_tenant_id_or_conversation_id")
            return "dead_lettered"

        key = result_key(tenant_id, conversation_id)
        if self.index.has(key):
            logger.info("Duplicate delivery for %s (source file %s) - skipping.", key, path.name)
            self.queue.mark_done(path)
            return "skipped_duplicate"

        try:
            turn_results = conv.load_conversation(raw_doc)
        except conv.ConversationRejected as exc:
            self.queue.mark_dead(path, str(exc))
            return "dead_lettered"

        # Scoring can throw on unanticipated input (encoding, tokenizer edge
        # case); dead-letter it rather than taking the whole batch down.
        try:
            record = self._score_conversation(raw_doc, tenant_id, conversation_id, turn_results, path.name)
        except Exception as exc:
            logger.exception("Unexpected error scoring %s", key)
            self.queue.mark_dead(path, f"scoring_failed: {exc}")
            return "dead_lettered"

        out_path = self.results_dir / f"{tenant_id}__{conversation_id}.json"
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        self.index.add(key, out_path.name)
        self.queue.mark_done(path)
        return "processed"

    def _score_conversation(self, raw_doc, tenant_id, conversation_id, turn_results, source_file) -> dict:
        scored_turns = []
        customer_scores_unordered = []   # (signed_score, confidence) - order does not matter for a mean
        customer_scores_by_time = []     # (parsed_ts, signed_score) - order matters for trajectory

        for turn in turn_results:
            if turn["status"] == "skipped":
                scored_turns.append({
                    "index": turn["index"],
                    "label": None,
                    "score": None,
                    "confidence": None,
                    "included_in_metrics": False,
                    "reason": turn["reason"],
                })
                continue

            entry = {
                "index": turn["index"],
                "seq": turn["seq"],
                "role": turn["role"],
                "ts": turn["ts"],
            }

            if not turn["has_content"]:
                entry.update(label=None, score=None, confidence=None,
                             included_in_metrics=False, reason="empty_text")
                scored_turns.append(entry)
                continue

            scored = self.scorer.score(turn["text"])
            is_customer = turn["role"] == "customer"
            entry.update(scored, included_in_metrics=is_customer)
            if not is_customer:
                entry["reason"] = "not_a_customer_turn"
            scored_turns.append(entry)

            if is_customer:
                customer_scores_unordered.append((scored["score"], scored["confidence"]))
                customer_scores_by_time.append((turn["parsed_ts"], scored["score"]))

        # Chronological order comes from `ts`, not `seq` (fixture c-000021 has
        # non-monotonic/duplicate `seq` but strictly increasing timestamps).
        customer_scores_by_time.sort(key=lambda pair: pair[0])
        ordered_scores = [score for _, score in customer_scores_by_time]

        return {
            "schema_version": SCHEMA_VERSION,
            "result_key": result_key(tenant_id, conversation_id),
            "tenant_id": tenant_id,
            "conversation_id": conversation_id,
            "channel": raw_doc.get("channel"),
            "source_file": source_file,
            "scored_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": {
                "backend": self.scorer.backend_name,
                "name": self.scorer.model_name,
                "revision": self.scorer.model_revision,
                "neutral_threshold": self.scorer.neutral_threshold,
            },
            "turns": scored_turns,
            "metrics": {
                "overall_sentiment": compute_overall_sentiment(customer_scores_unordered),
                "sentiment_trajectory": compute_sentiment_trajectory(ordered_scores),
            },
        }
