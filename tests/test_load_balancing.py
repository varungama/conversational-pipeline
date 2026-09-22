import unittest

from src.aws.config import AwsConfig, worst_case_visibility_timeout
from src.scoring import evaluate
from src.sentiment_model import SentimentScorer, length_sorted_batches
from src.sharding import SHARD_MAX_COST, _greedy_ranges, lpt_interleaved, estimate_cost, plan_shards, turn_cost
from tests.stub_scorer import BatchStubScorer, StubScorer


def make_doc(conversation_id, n_turns, tenant="acme", pad=0):
    words = ["GOOD", "BAD", "MEH", "other"]
    return {
        "conversation_id": conversation_id, "tenant_id": tenant, "channel": "chat",
        "turns": [{"seq": i, "role": "customer" if i % 2 == 0 else "agent",
                   "ts": f"2026-03-02T09:{i // 60:02d}:{i % 60:02d}Z", "text": words[i % 4] + " " * pad}
                  for i in range(n_turns)],
    }


class TestBatching(unittest.TestCase):
    def test_batches_are_length_sorted_and_cover_everything_once(self):
        lengths = [5, 500, 7, 480, 6, 490]
        batches = length_sorted_batches(lengths, batch_size=3, max_tokens=10**6)
        self.assertEqual(batches, [[1, 5, 3], [2, 4, 0]])  # long ones together, short ones together
        self.assertEqual(sorted(i for b in batches for i in b), list(range(len(lengths))))

    def test_batching_is_deterministic_on_ties(self):
        self.assertEqual(length_sorted_batches([4, 4, 4, 4], 2), [[0, 1], [2, 3]])

    def test_token_budget_keeps_a_long_chunk_from_dragging_short_ones(self):
        # one 400-token chunk among 30 tiny ones: without a budget they would share a batch padded to 400
        lengths = [400] + [8] * 30
        batches = length_sorted_batches(lengths, batch_size=32, max_tokens=512)
        self.assertEqual(batches[0], [0])  # a second item would pad to 2 x 400 > 512, so the long chunk goes alone
        self.assertTrue(all(len(b) * lengths[b[0]] <= 512 for b in batches))
        self.assertEqual(sorted(i for b in batches for i in b), list(range(31)))

    def test_item_over_budget_still_gets_its_own_batch(self):
        self.assertEqual(length_sorted_batches([600, 5], batch_size=32, max_tokens=512), [[0], [1]])

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
        self.assertEqual(plan_shards(make_doc("c", 20)), [])  # cheap: 20 short turns

    def test_shards_cover_every_turn_once_and_stay_under_the_budget(self):
        doc = make_doc("c", 200, pad=30)  # ~40-turn-cost each -> big enough to shard
        shards = plan_shards(doc)
        self.assertGreater(len(shards), 1)
        self.assertEqual(shards[0]["start"], 0)
        self.assertEqual(shards[-1]["end"], 200)
        self.assertTrue(all(b["start"] == a["end"] for a, b in zip(shards, shards[1:])))
        self.assertEqual({s["count"] for s in shards}, {len(shards)})
        self.assertEqual(sum(s["cost"] for s in shards), estimate_cost(doc))
        self.assertTrue(all(s["cost"] <= SHARD_MAX_COST * 1.25 for s in shards))  # near-equal, not exactly capped

    def test_cap_is_hard_and_shards_are_balanced(self):
        # turns of 3,000 against a 5,000 cap: closing *after* passing the target used to give 6,000+ shards
        doc = make_doc("c", 40, pad=2990)
        cap = 5000
        shards = plan_shards(doc, max_cost=cap)
        self.assertTrue(all(s["cost"] <= cap for s in shards))
        # mixed sizes: never over the cap, and no more shards than the fewest possible
        doc = make_doc("c", 300, pad=0)
        for i, t in enumerate(doc["turns"]):
            t["text"] = "x" * (50 + (i * 37) % 900)
        shards = plan_shards(doc, max_cost=cap)
        self.assertTrue(all(s["cost"] <= cap for s in shards))
        self.assertEqual(len(shards), len(_greedy_ranges([turn_cost(t) for t in doc["turns"]], cap)))  # fewest possible
        costs = [s["cost"] for s in shards]
        self.assertLess(max(costs) - min(costs[:-1]), 1000)  # near-equal, not full-full-sliver

    def test_oversized_turn_gets_its_own_shard_and_nothing_else_exceeds_the_cap(self):
        doc = make_doc("c", 20, pad=100)
        doc["turns"][10]["text"] = "x" * 9000
        shards = plan_shards(doc, max_cost=2500)
        over = [s for s in shards if s["cost"] > 2500]
        self.assertEqual([(s["start"], s["end"]) for s in over], [(10, 11)])

    def test_few_long_turns_are_sharded_even_though_turn_count_is_tiny(self):
        # the measured case: 3 turns, one ~4,000 characters, took 3.7s
        doc = make_doc("c", 3)
        doc["turns"][1]["text"] = "x" * 4000
        shards = plan_shards(doc)
        self.assertGreaterEqual(len(shards), 2)
        self.assertEqual(sum(s["end"] - s["start"] for s in shards), 3)

    def test_one_enormous_turn_cannot_be_split(self):
        doc = make_doc("c", 1)
        doc["turns"][0]["text"] = "x" * 50000
        self.assertEqual(plan_shards(doc), [])

    def test_unidentifiable_or_malformed_docs_are_not_sharded(self):
        self.assertEqual(plan_shards({"turns": [{"text": "x" * 9000}] * 5}), [])
        self.assertEqual(plan_shards({"tenant_id": "a", "conversation_id": "b", "turns": "nope"}), [])
        self.assertEqual(plan_shards(None), [])

    def test_cost_counts_turns_and_characters(self):
        self.assertEqual(turn_cost({"text": "abcde"}), turn_cost({"text": ""}) + 5)
        self.assertGreater(estimate_cost(make_doc("c", 3, pad=4000)), estimate_cost(make_doc("c", 30)))
        self.assertEqual(estimate_cost({"turns": "x"}), 1)
        self.assertEqual(estimate_cost(None), 1)


class TestTokenCost(unittest.TestCase):
    words = staticmethod(lambda text: len(text.split()))  # stand-in tokenizer

    def test_token_count_replaces_character_count(self):
        # same characters, very different token counts (a CJK-like text has many tokens per character)
        dense, sparse = {"text": "a b c d e f g h"}, {"text": "abcdefgh" * 2}
        self.assertLess(turn_cost(dense), turn_cost(sparse))  # by characters (fallback): 15 < 16
        self.assertGreater(turn_cost(dense, self.words), turn_cost(sparse, self.words))  # by tokens: 8 vs 1

    def test_empty_turn_costs_only_the_fixed_part(self):
        self.assertEqual(turn_cost({"text": "  "}, self.words), turn_cost({"text": ""}, self.words))
        self.assertEqual(turn_cost({}, self.words), turn_cost({"text": None}, self.words))

    def test_plan_uses_the_counter_and_still_covers_every_turn(self):
        doc = make_doc("c", 200, pad=0)
        for t in doc["turns"]:
            t["text"] = "w " * 30  # 30 tokens, only 60 characters
        shards = plan_shards(doc, count_tokens=self.words)
        self.assertGreater(len(shards), 1)
        self.assertEqual(sum(s["cost"] for s in shards), estimate_cost(doc, self.words))
        self.assertEqual((shards[0]["start"], shards[-1]["end"]), (0, 200))


class TestDispatchOrder(unittest.TestCase):
    def test_lanes_1_is_plain_longest_first(self):
        tasks = [(3, "c"), (9, "a"), (1, "d"), (6, "b")]
        self.assertEqual(lpt_interleaved(tasks, lanes=1), sorted(tasks, key=lambda t: -t[0]))

    def test_no_receive_batch_window_gets_more_than_one_of_the_biggest_tasks(self):
        # 5 big (cost 2500) shards and 20 small (cost 100) ones - the shape a real
        # run produces after sharding: a receive batch of 5 must not grab several big ones
        tasks = [(2500, f"big{i}") for i in range(5)] + [(100, f"small{i}") for i in range(20)]
        ordered = lpt_interleaved(tasks, lanes=5)
        for i in range(0, len(ordered) - 4, 5):
            window = ordered[i:i + 5]
            self.assertLessEqual(sum(1 for cost, _ in window if cost == 2500), 1)

    def test_covers_every_task_exactly_once(self):
        tasks = [(i, i) for i in range(17)]
        self.assertEqual(sorted(t for _, t in lpt_interleaved(tasks, lanes=4)), list(range(17)))


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
