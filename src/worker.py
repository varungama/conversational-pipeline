"""Local worker: claims one file at a time, scores it, writes a result; safe with several processes (`--workers`).
Atomic rename stops double file claims; result_index.try_claim stops double scoring of one conversation_id."""

import json
import logging
import pathlib

from src.metrics import DEFAULT_WEIGHTING
from src.queue_store import FileQueue
from src.result_index import ResultIndex
from src.scoring import evaluate, result_key
from src.sentiment_model import SentimentScorer

logger = logging.getLogger(__name__)


class Worker:
    def __init__(self, queue: FileQueue, results_dir: pathlib.Path, scorer: SentimentScorer,
                 scored_at: str | None = None, weighting: str = DEFAULT_WEIGHTING):
        self.queue = queue
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.scorer = scorer
        self.index = ResultIndex(results_dir / "index.json")
        # Overrides every record's `scored_at` for byte-identical reruns (see src/scoring.py).
        self.scored_at = scored_at
        # "equal" or "confidence" - see src/metrics.py, compute_overall_sentiment.
        self.weighting = weighting

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

        # Peek at identity before scoring, purely so a duplicate delivery
        # can be recognised and skipped without paying for model inference.
        tenant_id = raw_doc.get("tenant_id") if isinstance(raw_doc, dict) else None
        conversation_id = raw_doc.get("conversation_id") if isinstance(raw_doc, dict) else None
        if tenant_id and conversation_id and self.index.has(result_key(tenant_id, conversation_id)):
            logger.info("Duplicate delivery for %s::%s (source file %s) - skipping.",
                        tenant_id, conversation_id, path.name)
            self.queue.mark_done(path)
            return "skipped_duplicate"

        # The model can throw on unforeseen input; dead-letter that conversation instead of failing the batch.
        try:
            outcome = evaluate(raw_doc, self.scorer, source_ref=path.name, scored_at=self.scored_at,
                                weighting=self.weighting)
        except Exception as exc:
            logger.exception("Unexpected error scoring %s", path.name)
            self.queue.mark_dead(path, f"scoring_failed: {exc}")
            return "dead_lettered"

        if outcome[0] == "dead_letter":
            _, reason = outcome
            self.queue.mark_dead(path, reason)
            return "dead_lettered"

        _, tenant_id, conversation_id, record = outcome
        key = result_key(tenant_id, conversation_id)
        out_path = self.results_dir / f"{tenant_id}__{conversation_id}.json"

        # The earlier check is only an optimisation; try_claim is the real mutual exclusion,
        # so when worker processes race only one persists a result.
        if not self.index.try_claim(key, out_path.name):
            logger.info("Duplicate delivery for %s::%s (source file %s) - skipping.",
                        tenant_id, conversation_id, path.name)
            self.queue.mark_done(path)
            return "skipped_duplicate"

        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.queue.mark_done(path)
        return "processed"
