"""AWS worker: same scoring as src/worker.py (via src/scoring.py) over SQS/S3/DynamoDB.
Failures are never acked: SQS redelivers them and moves them to the DLQ after max_receive_count."""

import logging
import time

from src.aws.config import RECEIVE_BATCH_SIZE
from src.aws.dynamo_ledger import DynamoResultIndex
from src.aws.s3_store import S3ConversationStore, S3ResultStore, S3ShardStore
from src.aws.sqs_queue import Message, SqsQueue
from src.metrics import DEFAULT_WEIGHTING
from src.scoring import evaluate, result_key
from src.sharding import is_chunk_shard, merge_partials, score_chunk_shard, score_shard
from src.timing import StageTimer

logger = logging.getLogger(__name__)


class AwsWorker:
    def __init__(
        self,
        queue: SqsQueue,
        conversation_store: S3ConversationStore,
        result_store: S3ResultStore,
        ledger: DynamoResultIndex,
        scorer,
        scored_at: str | None = None,
        weighting: str = DEFAULT_WEIGHTING,
        shard_store: S3ShardStore | None = None,
        timer: StageTimer | None = None,
    ):
        self.queue = queue
        self.shard_store = shard_store  # only needed when sharded messages are in play
        self.conversation_store = conversation_store
        self.result_store = result_store
        self.ledger = ledger
        self.scorer = scorer
        self.scored_at = scored_at  # byte-identical-rerun knob, same as the local Worker
        self.weighting = weighting  # equal-vs-confidence knob, same as the local Worker
        self.timer = timer or StageTimer()

    def run_once(self, max_messages: int = RECEIVE_BATCH_SIZE) -> dict:
        """Processes one receive batch; returns counts by outcome. `shards_scored` = parked partials, not the merge."""
        summary = {"processed": 0, "skipped_duplicate": 0, "left_for_retry": 0, "shards_scored": 0}
        with self.timer.time("sqs_receive"):
            messages = self.queue.receive_batch(max_messages=max_messages)
        for message in messages:
            if message.sent_timestamp is not None:
                self.timer.add("sqs_queue_wait", max(0.0, time.time() - message.sent_timestamp))
            outcome = self._process_one(message)
            summary[outcome] = summary.get(outcome, 0) + 1
        return summary

    def _process_one(self, message: Message) -> str:
        with self.queue.keep_alive(message):  # held invisible so a slow task is never redelivered mid-flight
            with self.timer.time("processing"):
                return self._handle(message)

    def _handle(self, message: Message) -> str:
        tenant_id = message.body.get("tenant_id")
        conversation_id = message.body.get("conversation_id")
        if not tenant_id or not conversation_id:
            logger.error("Malformed queue message, missing tenant_id/conversation_id: %s", message.body)
            return "left_for_retry"

        key = result_key(tenant_id, conversation_id)
        shard = message.body.get("shard")

        turn_scores = None
        if shard:
            if self.ledger.has(key):
                logger.info("Duplicate shard delivery for %s - already scored.", key)
                self.queue.ack(message)
                return "skipped_duplicate"
            try:
                # only this shard's slice is read; the merging shard fetches the whole doc below
                plan_hash = shard["plan_hash"]
                with self.timer.time("shard_fetch"):
                    slice_data = self.shard_store.get_slice(tenant_id, conversation_id, plan_hash, shard["index"])
                with self.timer.time("shard_score"):
                    partial = (score_chunk_shard(slice_data, shard, self.scorer) if is_chunk_shard(shard)
                              else score_shard(slice_data, shard, self.scorer))
                with self.timer.time("shard_store_put"):
                    self.shard_store.put(tenant_id, conversation_id, plan_hash, shard["index"], partial)
                # atomic: exactly one shard's call here returns True, however many land at once
                if not self.ledger.mark_shard_landed(key, plan_hash, shard["index"], shard["count"]):
                    self.queue.ack(message)  # partial is durable; the shard that completes the set does the merge
                    return "shards_scored"
                with self.timer.time("shard_merge"):
                    turn_scores = merge_partials(self.shard_store.load(tenant_id, conversation_id, plan_hash))
            except Exception:
                logger.exception("Unexpected error scoring shard %s of %s - will retry / eventually DLQ.",
                                 shard.get("index"), key)
                return "left_for_retry"

        return self._finish(message, tenant_id, conversation_id, turn_scores)

    def _finish(self, message: Message, tenant_id: str, conversation_id: str, turn_scores) -> str:
        """Builds and stores the result record; `turn_scores` is merged partials, or None to score it whole here."""
        key = result_key(tenant_id, conversation_id)
        source_ref = f"{tenant_id}/{conversation_id}.json"

        try:
            with self.timer.time("conversation_fetch"):
                raw_doc = self.conversation_store.get(tenant_id, conversation_id)
        except Exception:
            logger.exception("Could not fetch %s from S3 - will retry / eventually DLQ.", source_ref)
            return "left_for_retry"

        try:
            with self.timer.time("scoring"):
                outcome = evaluate(raw_doc, self.scorer, source_ref=source_ref, scored_at=self.scored_at,
                                    weighting=self.weighting, turn_scores=turn_scores)
        except Exception:
            logger.exception("Unexpected error scoring %s - will retry / eventually DLQ.", key)
            return "left_for_retry"

        if outcome[0] == "dead_letter":
            _, reason = outcome
            logger.error("Unprocessable conversation %s: %s - will retry / eventually DLQ.", key, reason)
            return "left_for_retry"

        _, tenant_id, conversation_id, record = outcome

        if not self.ledger.try_claim(key, result_pointer=source_ref):
            logger.info("Duplicate delivery for %s - skipping without re-scoring.", key)
            self.queue.ack(message)
            return "skipped_duplicate"

        with self.timer.time("result_write"):
            self.result_store.put(tenant_id, conversation_id, record)
        with self.timer.time("sqs_ack"):
            self.queue.ack(message)
        return "processed"

    def drain_dead_letters(self) -> list:
        """Harvests the DLQ; entries carry only the pointer, since the failure reasons are in the worker logs."""
        dead_letters = []
        while True:
            batch = self.queue.receive_dead_letters()
            if not batch:
                break
            for message in batch:
                shard = message.body.get("shard")
                detail = ""
                if shard:
                    detail = f"; shard {shard['index'] + 1} of {shard['count']}"
                    if is_chunk_shard(shard):
                        detail += f" (chunks {shard['chunk_start']}-{shard['chunk_end'] - 1} of turn {shard['start']})"
                dead_letters.append({
                    "source_file": f"{message.body.get('tenant_id')}/{message.body.get('conversation_id')}.json",
                    "reason": "exceeded_max_receive_count (see worker logs for per-attempt reasons)" + detail,
                })
                self.queue.ack_dead_letter(message)
        return dead_letters
