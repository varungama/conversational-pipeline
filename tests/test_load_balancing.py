import json
import pathlib
import shutil
import tempfile
import unittest

from src.aws.config import AwsConfig, worst_case_visibility_timeout
from src.queue_store import FileQueue
from src.scoring import evaluate
from src.sentiment_model import SentimentScorer, length_sorted_batches
from src.sharding import SHARD_MAX_TURNS, estimate_cost, plan_shards
from src.worker import Worker
from tests.stub_scorer import BatchStubScorer, StubScorer


def make_doc(conversation_id, n_turns, tenant="acme"):
    words = ["GOOD", "BAD", "MEH", "other"]
    return {
        "conversation_id": conversation_id, "tenant_id": tenant, "channel": "chat",
        "turns": [{"seq": i, "role": "customer" if i % 2 == 0 else "agent",
                   "ts": f"2026-03-02T09:{i // 60:02d}:{i % 60:02d}Z", "text": words[i % 4]}
                  for i in range(n_turns)],
    }


class TestBatching(unittest.TestCase):
    def test_batches_are_length_sorted_and_cover_everything_once(self):
        lengths = [5, 500, 7, 480, 6, 490]
        batches = length_sorted_batches(lengths, batch_size=3)
        self.assertEqual(batches, [[1, 5, 3], [2, 4, 0]])  # long ones together, short ones together
        self.assertEqual(sorted(i for b in batches for i in b), list(range(len(lengths))))

    def test_batching_is_deterministic_on_ties(self):
        self.assertEqual(length_sorted_batches([4, 4, 4, 4], 2), [[0, 1], [2, 3]])

    def test_conversation_is_scored_in_one_batch_and_matches_unbatched(self):
        doc = make_doc("c-b", 40)
        batch_scorer = BatchStubScorer()
        batched = evaluate(doc, batch_scorer, "x", scored_at="2026-03-02T10:00:00Z")
        plain = evaluate(doc, StubScorer(), "x", scored_at="2026-03-02T10:00:00Z")
        self.assertEqual(batch_scorer.batch_sizes, [40])
        self.assertEqual(batched[3], plain[3])  # identical record

    def test_transformer_batch_regroups_chunks_per_text_in_order(self):
        class Fake(SentimentScorer):
            def __init__(self):
                self.backend_name = "transformer"
                self.neutral_threshold = 0.2
                self.calls = []

            def _count_tokens(self, text):
                return len(text.split())

            def _class_probs_batch(self, chunks, lengths):
                self.calls.append(len(chunks))
                return [{"positive": 0.9, "negative": 0.1} if "good" in c else {"positive": 0.1, "negative": 0.9}
                        for c in chunks]

        scorer = Fake()
        results = scorer.score_batch(["good", "bad", "good"])
        self.assertEqual([r["label"] for r in results], ["positive", "negative", "positive"])
        self.assertEqual(scorer.calls, [3])


class TestSharding(unittest.TestCase):
    def test_small_conversation_is_not_sharded(self):
        self.assertEqual(plan_shards(make_doc("c", SHARD_MAX_TURNS)), [])

    def test_shards_cover_every_turn_once_and_stay_small(self):
        shards = plan_shards(make_doc("c", 200))
        self.assertEqual([(s["start"], s["end"]) for s in shards], [(0, 50), (50, 100), (100, 150), (150, 200)])
        odd = plan_shards(make_doc("c", 101))
        self.assertEqual(odd[0]["start"], 0)
        self.assertEqual(odd[-1]["end"], 101)
        self.assertTrue(all(b["start"] == a["end"] for a, b in zip(odd, odd[1:])))
        self.assertTrue(all(s["end"] - s["start"] <= SHARD_MAX_TURNS for s in odd))

    def test_unidentifiable_or_malformed_docs_are_not_sharded(self):
        self.assertEqual(plan_shards({"turns": [1] * 500}), [])
        self.assertEqual(plan_shards({"tenant_id": "a", "conversation_id": "b", "turns": "nope"}), [])
        self.assertEqual(plan_shards(None), [])

    def test_estimate_cost(self):
        self.assertEqual(estimate_cost(make_doc("c", 200)), 200)
        self.assertEqual(estimate_cost({"turns": "x"}), 1)
        self.assertEqual(estimate_cost(None), 1)


class TestLocalDispatch(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.queue = FileQueue(self.tmp / "queue")
        self.results = self.tmp / "results"

    def _enqueue(self, doc):
        """Mirrors the local enqueue loop in run.py."""
        source = self.tmp / f"{doc['tenant_id']}__{doc['conversation_id']}.json"
        source.write_text(json.dumps(doc), encoding="utf-8")
        shards = plan_shards(doc)
        if not shards:
            self.queue.enqueue(source, cost=estimate_cost(doc))
            return
        for shard in shards:
            self.queue.enqueue_task(
                f"{source.stem}.shard-{shard['index']:04d}-of-{shard['count']:04d}.json",
                {"shard_task": {"source_file": source.name, "shard": shard, "doc": doc}},
                cost=shard["end"] - shard["start"])

    def test_pending_is_longest_first_with_name_tiebreak(self):
        for doc in [make_doc("c-a", 3), make_doc("c-b", 30), make_doc("c-c", 3), make_doc("c-d", 12)]:
            self._enqueue(doc)
        self.assertEqual([p.name for p in self.queue.list_pending()],
                         ["acme__c-b.json", "acme__c-d.json", "acme__c-a.json", "acme__c-c.json"])

    def test_claimed_file_leaves_no_cost_sidecar_behind(self):
        self._enqueue(make_doc("c-a", 3))
        self.queue.claim(self.queue.list_pending()[0])
        self.assertEqual(list(self.queue.pending.iterdir()), [])

    def test_sharded_result_equals_unsharded_result(self):
        doc = make_doc("c-big", 130)
        whole = evaluate(doc, StubScorer(), "acme__c-big.json", scored_at="2026-03-02T10:00:00Z")[3]

        self._enqueue(doc)
        self.assertEqual(len(self.queue.list_pending()), 3)  # 130 turns -> 3 shards
        worker = Worker(self.queue, self.results, BatchStubScorer(), scored_at="2026-03-02T10:00:00Z")
        summary = worker.run_once()

        self.assertEqual(summary["processed"], 1)
        self.assertEqual(summary["shards_scored"], 2)
        self.assertEqual(summary["dead_lettered"], 0)
        record = json.loads((self.results / "acme__c-big.json").read_text(encoding="utf-8"))
        self.assertEqual(record, whole)

    def test_two_workers_sharing_a_sharded_conversation_write_one_result(self):
        self._enqueue(make_doc("c-big", 200))
        workers = [Worker(self.queue, self.results, StubScorer()), Worker(self.queue, self.results, StubScorer())]
        # interleave: each takes shards in turn, as concurrent workers would
        for i, path in enumerate(self.queue.list_pending()):
            workers[i % 2]._process_one(self.queue.claim(path))
        self.assertEqual(len(list(self.results.glob("acme__c-big.json"))), 1)

    def test_redelivered_shards_after_completion_are_skipped(self):
        doc = make_doc("c-big", 100)
        self._enqueue(doc)
        worker = Worker(self.queue, self.results, StubScorer())
        worker.run_once()
        self._enqueue(doc)  # same conversation delivered again
        summary = worker.run_once()
        self.assertEqual(summary["processed"], 0)
        self.assertEqual(summary["skipped_duplicate"], 2)


class TestVisibilityTimeout(unittest.TestCase):
    def test_covers_a_full_batch_of_worst_case_tasks(self):
        self.assertEqual(worst_case_visibility_timeout(5, 6), 30)
        self.assertEqual(worst_case_visibility_timeout(1, 0.2), 1)

    def test_default_config_is_derived_not_a_fixed_5_seconds(self):
        config = AwsConfig(endpoint_url=None)
        self.assertEqual(config.visibility_timeout_seconds,
                         worst_case_visibility_timeout(config.receive_batch_size, config.worst_case_task_seconds))
        self.assertGreaterEqual(config.visibility_timeout_seconds,
                                config.receive_batch_size * config.worst_case_task_seconds)

    def test_explicit_value_wins(self):
        self.assertEqual(AwsConfig(endpoint_url=None, visibility_timeout_seconds=1).visibility_timeout_seconds, 1)


if __name__ == "__main__":
    unittest.main()
