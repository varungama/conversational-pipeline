"""Chunk-level sharding: an oversized single turn split at the model's own chunk boundaries
(src/sentiment_model.chunk_text) instead of staying one unsplittable shard. The promise under
test is the same one whole-turn sharding makes: sharded and unsharded scoring must agree exactly."""

import functools
import unittest

from src.sentiment_model import chunk_text
from src.sharding import (is_chunk_shard, merge_partials, plan_shards, score_chunk_shard,
                          slice_chunk_texts, slice_turns)
from tests.stub_scorer import FakeChunkScorer, word_count


def make_doc(turn_text, conversation_id="c-1", tenant="acme"):
    return {
        "conversation_id": conversation_id, "tenant_id": tenant, "channel": "chat",
        "turns": [{"seq": 0, "role": "customer", "ts": "2026-03-02T09:00:00Z", "text": turn_text}],
    }


CHUNK_FN = functools.partial(chunk_text, count_tokens=word_count)
# 1,200 words: over CHUNK_TOKEN_LIMIT (510), so this actually needs >1 model-window
# chunk - not just "over the shard cost cap" (a much lower bar, see test_chunking.py).
LONG_TEXT = "This is good. " * 200 + "This is bad. " * 200


class TestPlanning(unittest.TestCase):
    def test_without_chunk_text_fn_a_lone_oversized_turn_is_left_unsharded(self):
        # matches the pre-chunk-splitting contract: nothing to shard at turn granularity
        # for a single-turn document, so plan_shards defers to the normal whole-doc path
        self.assertEqual(plan_shards(make_doc(LONG_TEXT), max_cost=1000, count_tokens=word_count), [])

    def test_oversized_turn_becomes_several_chunk_shards(self):
        shards = plan_shards(make_doc(LONG_TEXT), max_cost=1000, count_tokens=word_count, chunk_text_fn=CHUNK_FN)
        self.assertGreater(len(shards), 1)
        self.assertTrue(all(is_chunk_shard(s) for s in shards))
        self.assertTrue(all(s["start"] == 0 and s["end"] == 1 for s in shards))  # all one turn
        self.assertTrue(all(s["count"] == len(shards) for s in shards))
        self.assertEqual([s["index"] for s in shards], list(range(len(shards))))

    def test_chunk_shards_cover_every_chunk_once_in_order(self):
        shards = plan_shards(make_doc(LONG_TEXT), max_cost=1000, count_tokens=word_count, chunk_text_fn=CHUNK_FN)
        n_chunks = len(CHUNK_FN(LONG_TEXT))
        self.assertGreater(n_chunks, 1)
        covered = [c for s in shards for c in range(s["chunk_start"], s["chunk_end"])]
        self.assertEqual(covered, list(range(n_chunks)))

    def test_normal_conversations_are_unaffected(self):
        # plenty of small turns, no oversized one - chunk_text_fn present but never triggers
        doc = {"conversation_id": "c", "tenant_id": "acme", "channel": "chat",
              "turns": [{"seq": i, "role": "customer", "ts": "2026-03-02T09:00:00Z", "text": "hi there"}
                        for i in range(50)]}
        shards = plan_shards(doc, max_cost=20, count_tokens=word_count, chunk_text_fn=CHUNK_FN)
        self.assertTrue(shards)
        self.assertFalse(any(is_chunk_shard(s) for s in shards))

    def test_a_turn_that_fits_the_model_window_but_not_the_shard_cap_stays_whole(self):
        # over the (much lower) shard cost cap but under CHUNK_TOKEN_LIMIT: chunk_text_fn
        # returns exactly 1 chunk, so there's nothing to gain by chunk-splitting it
        short_but_costly = make_doc("bad. " * 100)  # 200 words, well under 510
        shards = plan_shards(short_but_costly, max_cost=100, count_tokens=word_count, chunk_text_fn=CHUNK_FN)
        self.assertEqual(shards, [])  # falls back exactly like the no-chunk_text_fn case above


class TestSlicing(unittest.TestCase):
    def test_slice_chunk_texts_matches_the_planned_range(self):
        doc = make_doc(LONG_TEXT)
        shards = plan_shards(doc, max_cost=1000, count_tokens=word_count, chunk_text_fn=CHUNK_FN)
        all_chunks = CHUNK_FN(LONG_TEXT)
        for shard in shards:
            self.assertEqual(slice_chunk_texts(doc, shard, CHUNK_FN),
                             all_chunks[shard["chunk_start"]:shard["chunk_end"]])

    def test_whole_turn_slicing_is_untouched(self):
        doc = make_doc(LONG_TEXT, conversation_id="c-2")
        doc["turns"].append({"seq": 1, "role": "agent", "ts": "2026-03-02T09:01:00Z", "text": "ok"})
        shard = {"start": 1, "end": 2}
        self.assertEqual(slice_turns(doc, shard), [doc["turns"][1]])


class TestScoreAndMerge(unittest.TestCase):
    def setUp(self):
        self.scorer = FakeChunkScorer()

    def test_chunk_sharded_score_equals_scoring_the_turn_whole(self):
        """The core promise: split a turn into chunk-level shards, score each shard
        separately, merge - and get exactly what scoring it in one call would give."""
        whole = self.scorer.score(LONG_TEXT)

        doc = make_doc(LONG_TEXT)
        shards = plan_shards(doc, max_cost=1000, count_tokens=word_count, chunk_text_fn=CHUNK_FN)
        self.assertGreater(len(shards), 2)  # meaningfully split, not a trivial case

        partials = {}
        for shard in shards:
            chunks = slice_chunk_texts(doc, shard, CHUNK_FN)
            partials[shard["index"]] = score_chunk_shard(chunks, shard, self.scorer)

        merged = merge_partials(partials)
        self.assertEqual(merged, {0: whole})

    def test_merge_still_handles_a_mix_of_whole_turn_and_chunk_shards(self):
        # turn 0: normal, scored whole. turn 1: oversized, split into 2 chunk-shards.
        whole_score = self.scorer.score("short and good")
        long_score = self.scorer.score(LONG_TEXT)
        partials = {
            0: {0: whole_score},
            1: {1: {"_chunk_probs": [[self.scorer._class_probs(c), n] for c, n in CHUNK_FN(LONG_TEXT)[:1]]}},
            2: {1: {"_chunk_probs": [[self.scorer._class_probs(c), n] for c, n in CHUNK_FN(LONG_TEXT)[1:]]}},
        }
        merged = merge_partials(partials)
        self.assertEqual(merged, {0: whole_score, 1: long_score})

    def test_non_transformer_scorer_is_rejected_clearly(self):
        from tests.stub_scorer import StubScorer
        with self.assertRaises(ValueError):
            score_chunk_shard([("x", 1)], {"start": 0}, StubScorer())


if __name__ == "__main__":
    unittest.main()
