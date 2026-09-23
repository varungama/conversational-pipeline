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
from src.timing import StageTimer

logger = logging.getLogger(__name__)


class Worker:
    def __init__(self, queue: FileQueue, results_dir: pathlib.Path, scorer: SentimentScorer,
                 scored_at: str | None = None, weighting: str = DEFAULT_WEIGHTING, timer: StageTimer | None = None):
        self.queue = queue
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.scorer = scorer
        self.index = ResultIndex(results_dir / "index.json")
        self.scored_at = scored_at  # overrides `scored_at` for byte-identical reruns (src/scoring.py)
        self.weighting = weighting  # "equal" or "confidence" - see src/metrics.py
        self.timer = timer or StageTimer()

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
        with self.timer.time("processing"):
            return self._process_one_timed(path)

    def _process_one_timed(self, path: pathlib.Path) -> str:
        try:
            with self.timer.time("read_input"):
                raw_doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Could not read %s as JSON: %s", path.name, exc)
            self.queue.mark_dead(path, f"unreadable_json: {exc}")
            return "dead_lettered"

        tenant_id = raw_doc.get("tenant_id") if isinstance(raw_doc, dict) else None  # peek at identity to skip duplicates cheaply
        conversation_id = raw_doc.get("conversation_id") if isinstance(raw_doc, dict) else None
        if tenant_id and conversation_id and self.index.has(result_key(tenant_id, conversation_id)):
            logger.info("Duplicate delivery for %s::%s (source file %s) - skipping.",
                        tenant_id, conversation_id, path.name)
            self.queue.mark_done(path)
            return "skipped_duplicate"

        try:  # the model can throw on unforeseen input; dead-letter it instead of failing the batch
            with self.timer.time("scoring"):
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

        if not self.index.try_claim(key, out_path.name):  # earlier check is an optimization; this is the real mutual exclusion
            logger.info("Duplicate delivery for %s::%s (source file %s) - skipping.",
                        tenant_id, conversation_id, path.name)
            self.queue.mark_done(path)
            return "skipped_duplicate"

        with self.timer.time("result_write"):
            out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.queue.mark_done(path)
        return "processed"
