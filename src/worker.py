"""The worker: claims one conversation file at a time from the queue,
scores it, and writes a durable result. Safe to run as more than one
process against the same queue directory (see run.py's `--workers`) - two
layers make that true:

  - src/queue_store.py's atomic rename stops two workers claiming the same
    *file* twice.
  - src/result_index.py's `try_claim` stops two workers both scoring the
    same *conversation_id* to completion when it arrives as two different
    files (the shipped duplicate-delivery fixture, or two files racing
    across two processes) - the file-level claim above cannot catch that on
    its own, since each file is a distinct claim target.

The actual validate-and-score logic lives in src/scoring.py, shared with
the AWS (SQS/S3/DynamoDB) worker in src/aws/worker.py, so the two
transports can never quietly diverge in what a "correct" result looks like.
"""

import json
import logging
import pathlib

from src.metrics import DEFAULT_WEIGHTING
from src.queue_store import FileQueue
from src.result_index import ResultIndex
from src.scoring import evaluate, extract_identity, result_key
from src.sentiment_model import SentimentScorer
from src.shard_store import ShardStore
from src.sharding import merge_partials, score_shard

logger = logging.getLogger(__name__)


class Worker:
    def __init__(self, queue: FileQueue, results_dir: pathlib.Path, scorer: SentimentScorer,
                 scored_at: str | None = None, weighting: str = DEFAULT_WEIGHTING):
        self.queue = queue
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.scorer = scorer
        self.index = ResultIndex(results_dir / "index.json")
        # Partial scores of sharded conversations live with the queue, not
        # the results, so they never show up in the results listing.
        self.shards = ShardStore(queue.root / "shards")
        # Overrides every record's `scored_at` when set - see src/scoring.py.
        # Used to make a rerun over unchanged input byte-identical to a
        # prior run, instead of differing only in real wall-clock time.
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

        if isinstance(raw_doc, dict) and "shard_task" in raw_doc:
            return self._process_shard(path, raw_doc["shard_task"])

        # Peek at identity before scoring, purely so a duplicate delivery
        # can be recognised and skipped without paying for model inference.
        tenant_id = raw_doc.get("tenant_id") if isinstance(raw_doc, dict) else None
        conversation_id = raw_doc.get("conversation_id") if isinstance(raw_doc, dict) else None
        if tenant_id and conversation_id and self.index.has(result_key(tenant_id, conversation_id)):
            logger.info("Duplicate delivery for %s::%s (source file %s) - skipping.",
                        tenant_id, conversation_id, path.name)
            self.queue.mark_done(path)
            return "skipped_duplicate"

        # A conversation that parses fine can still make the model itself
        # throw on some input we did not anticipate (an unusual encoding, a
        # tokenizer edge case). One conversation's bad luck should not take
        # the rest of the batch down with it, so this is caught here and
        # dead-lettered like any other unprocessable record, and the loop
        # in run_once keeps going.
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

        return self._persist(path, outcome)

    def _process_shard(self, path: pathlib.Path, task: dict) -> str:
        """Scores one slice of a large conversation and parks the partial.
        The worker that lands the last slice merges them into the normal
        result record (see src/sharding.py)."""
        doc, shard = task["doc"], task["shard"]
        tenant_id, conversation_id = extract_identity(doc)
        if self.index.has(result_key(tenant_id, conversation_id)):
            self.queue.mark_done(path)
            return "skipped_duplicate"

        try:
            self.shards.put(tenant_id, conversation_id, shard["index"], score_shard(doc, shard, self.scorer))
        except Exception as exc:
            logger.exception("Unexpected error scoring shard %s", path.name)
            self.queue.mark_dead(path, f"scoring_failed: {exc}")
            return "dead_lettered"

        if len(self.shards.indexes(tenant_id, conversation_id)) < shard["count"]:
            self.queue.mark_done(path)
            return "shards_scored"

        merged = merge_partials(self.shards.load(tenant_id, conversation_id))
        try:
            outcome = evaluate(doc, self.scorer, source_ref=task["source_file"], scored_at=self.scored_at,
                                weighting=self.weighting, turn_scores=merged)
        except Exception as exc:
            logger.exception("Unexpected error merging shards for %s", path.name)
            self.queue.mark_dead(path, f"scoring_failed: {exc}")
            return "dead_lettered"
        if outcome[0] == "dead_letter":
            self.queue.mark_dead(path, outcome[1])
            return "dead_lettered"
        return self._persist(path, outcome)

    def _persist(self, path: pathlib.Path, outcome) -> str:
        _, tenant_id, conversation_id, record = outcome
        key = result_key(tenant_id, conversation_id)
        out_path = self.results_dir / f"{tenant_id}__{conversation_id}.json"

        # The pre-check above is only an optimisation (skip re-inference for
        # an *obvious* duplicate); this is the real mutual-exclusion point -
        # with multiple worker processes racing (run.py --workers), two of
        # them can both fail the pre-check and both score the same
        # conversation, but only one of them wins try_claim and gets to
        # persist a result. See src/result_index.py.
        if not self.index.try_claim(key, out_path.name):
            logger.info("Duplicate delivery for %s::%s (source file %s) - skipping.",
                        tenant_id, conversation_id, path.name)
            self.queue.mark_done(path)
            return "skipped_duplicate"

        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.queue.mark_done(path)
        return "processed"
