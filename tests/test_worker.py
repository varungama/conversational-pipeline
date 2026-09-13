import json
import pathlib
import shutil
import tempfile
import unittest

from src.queue_store import FileQueue
from src.worker import Worker
from tests.stub_scorer import StubScorer


class TestWorker(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.queue = FileQueue(self.tmp / "queue")
        self.results = self.tmp / "results"
        self.worker = Worker(self.queue, self.results, StubScorer())

    def _write_source(self, name, doc):
        path = self.tmp / name
        path.write_text(json.dumps(doc), encoding="utf-8")
        return path

    def test_malformed_record_is_dead_lettered_not_scored(self):
        doc = {"conversation_id": "c-1", "tenant_id": "acme", "channel": "chat"}  # no turns key
        self.queue.enqueue(self._write_source("bad.json", doc))
        summary = self.worker.run_once()
        self.assertEqual(summary["dead_lettered"], 1)
        self.assertEqual(list(self.results.glob("acme__c-1.json")), [])

    def test_duplicate_delivery_is_scored_once(self):
        doc = {
            "conversation_id": "c-2", "tenant_id": "acme", "channel": "chat",
            "turns": [{"seq": 0, "role": "customer", "ts": "2026-03-02T09:14:00Z", "text": "GOOD"}],
        }
        self.queue.enqueue(self._write_source("first.json", doc))
        self.queue.enqueue(self._write_source("first.duplicate.json", doc))
        summary = self.worker.run_once()
        self.assertEqual(summary["processed"], 1)
        self.assertEqual(summary["skipped_duplicate"], 1)

    def test_trajectory_orders_by_timestamp_not_seq(self):
        # seq is scrambled (2, 0, 1) but ts strictly increases - true order is BAD, then GOOD
        doc = {
            "conversation_id": "c-3", "tenant_id": "acme", "channel": "chat",
            "turns": [
                {"seq": 2, "role": "customer", "ts": "2026-03-02T09:14:00Z", "text": "BAD"},
                {"seq": 0, "role": "agent", "ts": "2026-03-02T09:14:40Z", "text": "sorry"},
                {"seq": 1, "role": "customer", "ts": "2026-03-02T09:15:20Z", "text": "GOOD"},
            ],
        }
        self.queue.enqueue(self._write_source("ordering.json", doc))
        self.worker.run_once()
        record = json.loads((self.results / "acme__c-3.json").read_text(encoding="utf-8"))
        trajectory = record["metrics"]["sentiment_trajectory"]
        self.assertEqual(trajectory["label"], "improving")  # BAD -> GOOD, not GOOD -> BAD

    def test_no_customer_turns_gives_null_metrics(self):
        doc = {
            "conversation_id": "c-4", "tenant_id": "acme", "channel": "chat",
            "turns": [{"seq": 0, "role": "agent", "ts": "2026-03-02T09:14:00Z", "text": "hello"}],
        }
        self.queue.enqueue(self._write_source("agent_only.json", doc))
        self.worker.run_once()
        record = json.loads((self.results / "acme__c-4.json").read_text(encoding="utf-8"))
        self.assertIsNone(record["metrics"]["overall_sentiment"])
        self.assertIsNone(record["metrics"]["sentiment_trajectory"])


if __name__ == "__main__":
    unittest.main()
