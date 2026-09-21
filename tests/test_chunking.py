import unittest

from src.sentiment_model import SentimentScorer, aggregate_probs, chunk_text, split_sentences


def word_count(text: str) -> int:
    """Stand-in tokenizer: one token per whitespace-separated word."""
    return len(text.split())


class FakeChunkScorer(SentimentScorer):
    """Real `score` / chunking / aggregation, fake model: a chunk is positive
    if it contains 'good' and not 'not good', else negative."""

    def __init__(self):  # skip model loading
        self.backend_name = "transformer"
        self.neutral_threshold = 0.2

    def _count_tokens(self, text):
        return word_count(text)

    def _class_probs_batch(self, chunks, lengths):
        return [self._class_probs(c) for c in chunks]

    def _class_probs(self, text):
        positive = "good" in text and "not good" not in text
        return {"positive": 0.9, "negative": 0.1} if positive else {"positive": 0.1, "negative": 0.9}


class TestSplitting(unittest.TestCase):
    def test_split_loses_nothing(self):
        for text in ["It was fine. Then it broke! Really?\nYes.", "最悪です。二度と使いません。よろしく！"]:
            self.assertEqual("".join(split_sentences(text)), text)

    def test_japanese_splits_on_full_stop(self):
        self.assertEqual(split_sentences("最悪です。二度と使いません。"), ["最悪です。", "二度と使いません。"])

    def test_negation_stays_with_its_sentence(self):
        text = "Hello there. This is not good. Thanks."
        chunks = chunk_text(text, word_count, limit=5)
        self.assertTrue(any("not good" in c for c, _ in chunks))

    def test_chunking_is_deterministic(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(50))
        self.assertEqual(chunk_text(text, word_count, limit=12), chunk_text(text, word_count, limit=12))

    def test_every_chunk_fits(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(50))
        self.assertTrue(all(n <= 12 for _, n in chunk_text(text, word_count, limit=12)))

    def test_oversize_single_sentence_is_hard_split(self):
        chunks = chunk_text("a" * 100, lambda t: len(t), limit=30)
        self.assertTrue(all(n <= 30 for _, n in chunks))
        self.assertEqual("".join(c for c, _ in chunks), "a" * 100)


class TestScoreThroughScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = FakeChunkScorer()

    def test_short_text_is_one_chunk_and_unchanged(self):
        result = self.scorer.score("This is good.")
        self.assertEqual(result, {"label": "positive", "score": 0.8, "confidence": 0.9, "n_chunks": 1})

    def test_repeats_score_same_as_one_sentence(self):
        one = self.scorer.score("This is good.")
        many = self.scorer.score(" ".join(["This is good."] * 400))
        self.assertGreater(many["n_chunks"], 1)
        for key in ("label", "score", "confidence"):
            self.assertEqual(many[key], one[key])

    def test_late_text_is_no_longer_dropped(self):
        # Positive opening, long filler, negative ending: the old truncation
        # would only ever see the start. Both halves must now count.
        text = "This is good. " * 300 + "This is bad. " * 300
        result = self.scorer.score(text)
        self.assertGreater(result["n_chunks"], 1)
        # Truncation would give +0.8; the mixed chunk at the seam keeps this
        # from being exactly 0, but it must land near the middle.
        self.assertLess(abs(result["score"]), 0.3)


class TestAggregate(unittest.TestCase):
    def test_token_weighted_mean(self):
        probs = aggregate_probs([({"positive": 1.0}, 3), ({"positive": 0.0}, 1)])
        self.assertAlmostEqual(probs["positive"], 0.75)


if __name__ == "__main__":
    unittest.main()
