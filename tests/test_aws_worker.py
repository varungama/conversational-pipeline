"""Proves the SQS/S3/DynamoDB path end to end without Docker or a network
call: moto intercepts boto3 and fakes each AWS service in-memory, so the
same client code that would talk to LocalStack (or real AWS - only
AwsConfig.endpoint_url changes) runs here against a fake. See NOTES.md for
why this is a substitute for testing against real LocalStack.
"""

import time
import unittest

import boto3

try:
    from moto import mock_aws
except ImportError:
    mock_aws = None
    _mock_aws_decorator = lambda cls: cls  # noqa: E731 - keeps class importable without moto
else:
    _mock_aws_decorator = mock_aws

from src.aws.bootstrap import ensure_infrastructure
from src.aws.clients import AwsClients
from src.aws.config import AwsConfig
from src.aws.dynamo_ledger import DynamoResultIndex
from src.aws.s3_store import S3ConversationStore, S3ResultStore, S3ShardStore
from src.aws.sqs_queue import SqsQueue
from src.aws.worker import AwsWorker
from src.scoring import evaluate, result_key
from src.sharding import plan_shards
from tests.stub_scorer import StubScorer


# moto is test-only (requirements-dev.txt) so requirements.txt stays lean
# for --backend local; this suite skips rather than failing without it
@unittest.skipUnless(mock_aws, "moto not installed - run `pip install -r requirements-dev.txt` "
                                "to also exercise the --backend aws path")
@_mock_aws_decorator
class TestAwsWorker(unittest.TestCase):
    def setUp(self):
        # mock_aws intercepts every boto3 call regardless of endpoint_url,
        # so this runs in-memory, skipping LocalStack entirely
        self.config = AwsConfig(
            endpoint_url=None,
            region="us-east-1",
            conversations_bucket="test-conversations",
            results_bucket="test-results",
            queue_name="test-work",
            dead_letter_queue_name="test-work-dlq",
            max_receive_count=1,
            visibility_timeout_seconds=1,  # short so DLQ tests stay fast
            ledger_table_name="test-result-index",
        )
        self.clients = AwsClients(self.config)
        infra = ensure_infrastructure(self.clients, self.config)

        self.conversations = S3ConversationStore(self.clients, self.config.conversations_bucket)
        self.results = S3ResultStore(self.clients, self.config.results_bucket)
        self.ledger = DynamoResultIndex(self.clients, self.config.ledger_table_name)
        self.queue = SqsQueue(self.clients, infra["queue_url"], infra["dlq_url"])
        self.shard_store = S3ShardStore(self.clients, self.config.conversations_bucket)
        self.worker = AwsWorker(self.queue, self.conversations, self.results, self.ledger, StubScorer(),
                                shard_store=self.shard_store)

    def _submit(self, tenant_id, conversation_id, doc):
        self.conversations.put(tenant_id, conversation_id, doc)
        self.queue.enqueue(tenant_id, conversation_id)

    def test_bootstrap_is_idempotent(self):
        # must not raise - a real deploy re-runs this on every worker start
        ensure_infrastructure(self.clients, self.config)

    def test_happy_path_scores_and_stores_result(self):
        doc = {
            "conversation_id": "c-1", "tenant_id": "acme", "channel": "chat",
            "turns": [{"seq": 0, "role": "customer", "ts": "2026-03-02T09:14:00Z", "text": "GOOD"}],
        }
        self._submit("acme", "c-1", doc)

        summary = self.worker.run_once()
        self.assertEqual(summary["processed"], 1)

        results = self.results.list_all()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["result_key"], result_key("acme", "c-1"))
        self.assertEqual(results[0]["metrics"]["overall_sentiment"]["label"], "positive")

        # The message is gone from the live queue - ack'd, not left in flight.
        self.assertEqual(self.queue.approximate_depth(), 0)

    def test_duplicate_delivery_is_scored_once(self):
        doc = {
            "conversation_id": "c-2", "tenant_id": "acme", "channel": "chat",
            "turns": [{"seq": 0, "role": "customer", "ts": "2026-03-02T09:14:00Z", "text": "GOOD"}],
        }
        self._submit("acme", "c-2", doc)
        self._submit("acme", "c-2", doc)  # same conversation, delivered twice

        summary = self.worker.run_once()
        self.assertEqual(summary["processed"], 1)
        self.assertEqual(summary["skipped_duplicate"], 1)
        self.assertEqual(len(self.results.list_all()), 1)

    def test_ledger_claim_is_atomic_and_rejects_second_claim(self):
        key = result_key("acme", "c-3")
        self.assertTrue(self.ledger.try_claim(key, "acme/c-3.json"))
        self.assertFalse(self.ledger.try_claim(key, "acme/c-3.json"))
        self.assertTrue(self.ledger.has(key))

    def test_unprocessable_message_is_left_for_retry_then_reaches_dlq(self):
        # no `turns` key - whole-record malformation, same shape as c-000029
        doc = {"conversation_id": "c-4", "tenant_id": "acme", "channel": "chat"}
        self._submit("acme", "c-4", doc)

        # fails validation, left un-ack'd (max_receive_count=1, so this is
        # the only delivery attempt)
        summary = self.worker.run_once()
        self.assertEqual(summary["left_for_retry"], 1)
        self.assertEqual(self.results.list_all(), [])

        time.sleep(self.config.visibility_timeout_seconds + 0.5)

        # receive count now meets max_receive_count, so SQS redrives to the
        # DLQ instead of delivering again - worker sees nothing this pass
        summary = self.worker.run_once()
        self.assertEqual(summary["processed"] + summary["skipped_duplicate"] + summary["left_for_retry"], 0)

        dead_letters = self.worker.drain_dead_letters()
        self.assertEqual(len(dead_letters), 1)
        self.assertEqual(dead_letters[0]["source_file"], "acme/c-4.json")
        self.assertEqual(self.queue.approximate_depth(), 0)

    def test_sharded_conversation_is_merged_into_one_result_equal_to_unsharded(self):
        turns = [{"seq": i, "role": "customer", "ts": f"2026-03-02T09:{i // 60:02d}:{i % 60:02d}Z",
                  "text": ["GOOD", "BAD", "MEH"][i % 3]} for i in range(120)]
        doc = {"conversation_id": "c-big", "tenant_id": "acme", "channel": "chat", "turns": turns}
        self.conversations.put("acme", "c-big", doc)
        shards = plan_shards(doc)
        self.assertEqual(len(shards), 3)
        for shard in shards:
            self.queue.enqueue("acme", "c-big", shard=shard)

        summary = {"processed": 0, "shards_scored": 0}
        while self.queue.approximate_depth() > 0:
            batch = self.worker.run_once()
            summary["processed"] += batch["processed"]
            summary["shards_scored"] += batch["shards_scored"]

        self.assertEqual(summary, {"processed": 1, "shards_scored": 2})
        results = self.results.list_all()
        self.assertEqual(len(results), 1)
        expected = evaluate(doc, StubScorer(), "acme/c-big.json")[3]
        expected["scored_at"] = results[0]["scored_at"]
        self.assertEqual(results[0], expected)

        # a redelivered shard after completion is a no-op
        self.queue.enqueue("acme", "c-big", shard=shards[0])
        again = self.worker.run_once()
        self.assertEqual(again["skipped_duplicate"], 1)
        self.assertEqual(len(self.results.list_all()), 1)

    def test_pointer_message_never_carries_the_transcript(self):
        doc = {
            "conversation_id": "c-5", "tenant_id": "acme", "channel": "chat",
            "turns": [{"seq": 0, "role": "customer", "ts": "2026-03-02T09:14:00Z", "text": "x" * 5000}],
        }
        self._submit("acme", "c-5", doc)

        sqs = boto3.client("sqs", region_name="us-east-1")
        raw = sqs.receive_message(QueueUrl=self.queue.queue_url, MaxNumberOfMessages=1)
        body = raw["Messages"][0]["Body"]
        self.assertLess(len(body), 200)  # a pointer, not a 5000-char transcript
        self.assertIn("conversation_id", body)


if __name__ == "__main__":
    unittest.main()
