"""End-to-end SQS/S3/DynamoDB tests on moto (in-memory AWS, no Docker); a stand-in for LocalStack (NOTES.md)."""

import threading
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
from src.sentiment_model import chunk_text
from src.sharding import is_chunk_shard, plan_shards
from tests.stub_scorer import FakeChunkScorer, StubScorer, word_count


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

        summary = self.worker.run_once(max_messages=2)  # both copies in one receive
        self.assertEqual(summary["processed"], 1)
        self.assertEqual(summary["skipped_duplicate"], 1)
        self.assertEqual(len(self.results.list_all()), 1)

    def test_ledger_claim_is_atomic_and_rejects_second_claim(self):
        key = result_key("acme", "c-3")
        self.assertTrue(self.ledger.try_claim(key, "acme/c-3.json"))
        self.assertFalse(self.ledger.try_claim(key, "acme/c-3.json"))
        self.assertTrue(self.ledger.has(key))

    def test_mark_shard_landed_is_true_only_once_the_set_is_complete(self):
        key = result_key("acme", "c-shardcount")
        self.assertFalse(self.ledger.mark_shard_landed(key, 0, total=3))
        self.assertFalse(self.ledger.mark_shard_landed(key, 1, total=3))
        self.assertTrue(self.ledger.mark_shard_landed(key, 2, total=3))
        # a completed conversation's own result claim is a separate item - unaffected
        self.assertFalse(self.ledger.has(key))

    def test_mark_shard_landed_is_idempotent_under_redelivery(self):
        # a shard message delivered twice must not let two different indices both look
        # "complete", and a redelivery of the completing shard must still say True (so SQS's
        # own redelivery - not a lock - is what retries a merge that crashed after winning)
        key = result_key("acme", "c-shardcount-2")
        self.assertFalse(self.ledger.mark_shard_landed(key, 0, total=2))
        self.assertFalse(self.ledger.mark_shard_landed(key, 0, total=2))  # redelivered shard 0
        self.assertTrue(self.ledger.mark_shard_landed(key, 1, total=2))
        self.assertTrue(self.ledger.mark_shard_landed(key, 1, total=2))  # redelivered shard 1 (the winner)

    def test_mark_shard_landed_has_exactly_one_winner_under_real_concurrency(self):
        # every shard "lands" from its own thread at once - the scenario in the finding this
        # fixes (two shards finishing simultaneously); exactly one must see completion
        key = result_key("acme", "c-shardcount-race")
        total = 8
        results = [None] * total

        def land(i):
            results[i] = self.ledger.mark_shard_landed(key, i, total)

        threads = [threading.Thread(target=land, args=(i,)) for i in range(total)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(sum(results), 1)

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
                  "text": ["GOOD", "BAD", "MEH"][i % 3] + " " * 40} for i in range(120)]
        doc = {"conversation_id": "c-big", "tenant_id": "acme", "channel": "chat", "turns": turns}
        self.conversations.put("acme", "c-big", doc)
        shards = plan_shards(doc)
        self.assertEqual(len(shards), 3)
        self.shard_store.put_slices("acme", "c-big", doc, shards)
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

    def test_shards_that_finish_simultaneously_still_merge_exactly_once(self):
        # every shard lands "at the same time" - separate worker instances, separate threads,
        # each doing its own S3 put + ledger update concurrently, the exact scenario the
        # mark_shard_landed fix targets (two-shards-finish-simultaneously)
        turns = [{"seq": i, "role": "customer", "ts": f"2026-03-02T09:{i // 60:02d}:{i % 60:02d}Z",
                  "text": ["GOOD", "BAD", "MEH"][i % 3] + " " * 40} for i in range(120)]
        doc = {"conversation_id": "c-race", "tenant_id": "acme", "channel": "chat", "turns": turns}
        self.conversations.put("acme", "c-race", doc)
        shards = plan_shards(doc)
        self.assertGreaterEqual(len(shards), 3)
        self.shard_store.put_slices("acme", "c-race", doc, shards)
        for shard in shards:
            self.queue.enqueue("acme", "c-race", shard=shard)

        outcomes = [None] * len(shards)

        def process(i):
            # its own AwsWorker instance, like a separate worker process would have
            worker = AwsWorker(self.queue, self.conversations, self.results, self.ledger, StubScorer(),
                               shard_store=self.shard_store)
            outcomes[i] = worker.run_once()

        threads = [threading.Thread(target=process, args=(i,)) for i in range(len(shards))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(sum(o["processed"] for o in outcomes), 1)  # exactly one merge, never zero or two
        self.assertEqual(sum(o["shards_scored"] for o in outcomes), len(shards) - 1)
        results = self.results.list_all()
        self.assertEqual(len(results), 1)
        expected = evaluate(doc, StubScorer(), "acme/c-race.json")[3]
        expected["scored_at"] = results[0]["scored_at"]
        self.assertEqual(results[0], expected)

    def test_shard_workers_read_only_their_slice_not_the_whole_document(self):
        turns = [{"seq": i, "role": "customer", "ts": f"2026-03-02T09:{i // 60:02d}:{i % 60:02d}Z",
                  "text": "GOOD" + " " * 40} for i in range(120)]
        doc = {"conversation_id": "c-slice", "tenant_id": "acme", "channel": "chat", "turns": turns}
        self.conversations.put("acme", "c-slice", doc)
        shards = plan_shards(doc)
        self.shard_store.put_slices("acme", "c-slice", doc, shards)
        for shard in shards:
            self.queue.enqueue("acme", "c-slice", shard=shard)

        full_reads = []
        original_get = self.conversations.get
        self.conversations.get = lambda *a: (full_reads.append(a), original_get(*a))[1]
        while self.queue.approximate_depth() > 0:
            self.worker.run_once()
        self.assertEqual(len(full_reads), 1)  # only the shard that merges reads the document
        self.assertEqual(len(self.results.list_all()), 1)

    def test_oversized_turn_is_chunk_sharded_and_merges_to_the_unsharded_result(self):
        import functools

        chunk_fn = functools.partial(chunk_text, count_tokens=word_count)
        long_text = "This is good. " * 200 + "This is bad. " * 200  # over CHUNK_TOKEN_LIMIT, needs >1 chunk
        turns = [{"seq": 0, "role": "customer", "ts": "2026-03-02T09:14:00Z", "text": long_text},
                 {"seq": 1, "role": "customer", "ts": "2026-03-02T09:15:00Z", "text": "GOOD"}]
        doc = {"conversation_id": "c-chunked", "tenant_id": "acme", "channel": "chat", "turns": turns}
        self.conversations.put("acme", "c-chunked", doc)

        scorer = FakeChunkScorer()
        shards = plan_shards(doc, max_cost=1000, count_tokens=word_count, chunk_text_fn=chunk_fn)
        self.assertGreater(len(shards), 2)  # the long turn split into several chunk shards
        self.assertTrue(any(is_chunk_shard(s) for s in shards))
        self.shard_store.put_slices("acme", "c-chunked", doc, shards, chunk_fn)
        for shard in shards:
            self.queue.enqueue("acme", "c-chunked", shard=shard)

        worker = AwsWorker(self.queue, self.conversations, self.results, self.ledger, scorer,
                           shard_store=self.shard_store)
        summary = {"processed": 0, "shards_scored": 0}
        while self.queue.approximate_depth() > 0:
            batch = worker.run_once()
            summary["processed"] += batch["processed"]
            summary["shards_scored"] += batch["shards_scored"]

        self.assertEqual(summary, {"processed": 1, "shards_scored": len(shards) - 1})
        results = self.results.list_all()
        self.assertEqual(len(results), 1)
        expected = evaluate(doc, scorer, "acme/c-chunked.json")[3]
        expected["scored_at"] = results[0]["scored_at"]
        self.assertEqual(results[0], expected)
        self.assertEqual(results[0]["turns"][0]["n_chunks"], len(chunk_fn(long_text)))

    def test_slow_task_is_not_redelivered_while_it_is_still_running(self):
        class SlowScorer(StubScorer):
            def score(self, text):
                time.sleep(2.5)  # > the 1s visibility timeout
                return super().score(text)

        queue = SqsQueue(self.clients, self.queue.queue_url, self.queue.dlq_url,
                         self.config.visibility_timeout_seconds)
        worker = AwsWorker(queue, self.conversations, self.results, self.ledger, SlowScorer())
        doc = {"conversation_id": "c-slow", "tenant_id": "acme", "channel": "chat",
               "turns": [{"seq": 0, "role": "customer", "ts": "2026-03-02T09:14:00Z", "text": "GOOD"}]}
        self._submit("acme", "c-slow", doc)

        stolen = []
        thread = threading.Thread(target=worker.run_once)
        thread.start()
        while True:  # wait until the worker really holds the message, or the poller could win the race
            attrs = self.queue.sqs.get_queue_attributes(
                QueueUrl=self.queue.queue_url, AttributeNames=["ApproximateNumberOfMessagesNotVisible"])
            if int(attrs["Attributes"]["ApproximateNumberOfMessagesNotVisible"]) == 1:
                break
            time.sleep(0.05)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:  # a second worker polling while the first is mid-task
            stolen += queue.receive_batch(max_messages=1, wait_time_seconds=0)
            time.sleep(0.2)
        thread.join()
        self.assertEqual(stolen, [])
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
