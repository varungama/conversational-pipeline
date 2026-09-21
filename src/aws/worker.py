"""The AWS-shaped worker: same scoring logic as src/worker.py (both call
into src/scoring.py), different plumbing - SQS instead of a directory,
S3 instead of local JSON files, DynamoDB instead of a JSON ledger.

Failure handling deliberately does *not* mirror the local worker's
"dead-letter immediately" behaviour. Here, an unprocessable message is
simply never deleted (never ack'd); SQS's own visibility timeout makes it
reappear for another attempt, and after `max_receive_count` attempts the
queue's RedrivePolicy moves it to the DLQ with no application code
involved. See sqs_queue.py for why.
"""

import logging

from src.aws.config import RECEIVE_BATCH_SIZE
from src.aws.dynamo_ledger import DynamoResultIndex
from src.aws.s3_store import S3ConversationStore, S3ResultStore, S3ShardStore
from src.aws.sqs_queue import Message, SqsQueue
from src.metrics import DEFAULT_WEIGHTING
from src.scoring import evaluate, result_key
from src.sharding import merge_partials, score_shard

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
    ):
        self.queue = queue
        # Only needed when sharded messages are in play.
        self.shard_store = shard_store
        self.conversation_store = conversation_store
        self.result_store = result_store
        self.ledger = ledger
        self.scorer = scorer
        # Same byte-identical-rerun knob as the local Worker - see
        # src/scoring.py and src/worker.py.
        self.scored_at = scored_at
        # Same "equal" vs "confidence" knob as the local Worker - kept in
        # sync so the two transports cannot silently diverge on metrics
        # either, same as everything else routed through src/scoring.py.
        self.weighting = weighting

    def run_once(self, max_messages: int = RECEIVE_BATCH_SIZE) -> dict:
        """Receives and processes one batch. Returns counts by outcome.

        `left_for_retry` messages are not a bug - they are unprocessable
        (or hit an unexpected exception) and were deliberately not deleted,
        so SQS will redeliver them. Call this repeatedly (see run.py) to
        drain a queue; a message only truly disappears (to done or to the
        DLQ) after enough passes.

        `max_messages` must be paired with a visibility timeout that covers
        that many worst-case tasks in a row - see src/aws/config.py.
        `shards_scored` counts shard messages that only parked a partial;
        the one that completes a conversation counts as `processed`.
        """
        summary = {"processed": 0, "skipped_duplicate": 0, "left_for_retry": 0, "shards_scored": 0}
        for message in self.queue.receive_batch(max_messages=max_messages):
            outcome = self._process_one(message)
            summary[outcome] = summary.get(outcome, 0) + 1
        return summary

    def _process_one(self, message: Message) -> str:
        tenant_id = message.body.get("tenant_id")
        conversation_id = message.body.get("conversation_id")
        if not tenant_id or not conversation_id:
            logger.error("Malformed queue message, missing tenant_id/conversation_id: %s", message.body)
            return "left_for_retry"

        key = result_key(tenant_id, conversation_id)
        source_ref = f"{tenant_id}/{conversation_id}.json"

        shard = message.body.get("shard")

        try:
            raw_doc = self.conversation_store.get(tenant_id, conversation_id)
        except Exception:
            logger.exception("Could not fetch %s from S3 - will retry / eventually DLQ.", source_ref)
            return "left_for_retry"

        turn_scores = None
        if shard:
            if self.ledger.has(key):
                logger.info("Duplicate shard delivery for %s - already scored.", key)
                self.queue.ack(message)
                return "skipped_duplicate"
            try:
                self.shard_store.put(tenant_id, conversation_id, shard["index"],
                                     score_shard(raw_doc, shard, self.scorer))
                if len(self.shard_store.indexes(tenant_id, conversation_id)) < shard["count"]:
                    self.queue.ack(message)  # partial is durable; the last shard to land does the merge
                    return "shards_scored"
                turn_scores = merge_partials(self.shard_store.load(tenant_id, conversation_id))
            except Exception:
                logger.exception("Unexpected error scoring shard %s of %s - will retry / eventually DLQ.",
                                 shard.get("index"), key)
                return "left_for_retry"

        try:
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

        self.result_store.put(tenant_id, conversation_id, record)
        self.queue.ack(message)
        return "processed"

    def drain_dead_letters(self) -> list:
        """Harvests whatever has landed in the DLQ so far. Each entry only
        carries what SQS actually knows (the pointer) - the specific reason
        lives in the worker logs from each failed attempt, not on the
        message itself. Contrast with the local dead_letter/*.reason.txt
        file, which the local worker writes by hand."""
        dead_letters = []
        while True:
            batch = self.queue.receive_dead_letters()
            if not batch:
                break
            for message in batch:
                shard = message.body.get("shard")
                dead_letters.append({
                    "source_file": f"{message.body.get('tenant_id')}/{message.body.get('conversation_id')}.json",
                    "reason": "exceeded_max_receive_count (see worker logs for per-attempt reasons)"
                              + (f"; shard {shard['index'] + 1} of {shard['count']}" if shard else ""),
                })
                self.queue.ack_dead_letter(message)
        return dead_letters
