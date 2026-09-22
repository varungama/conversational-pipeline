"""Fake scorer shared by both test suites: fixed marker words give deterministic sentiment."""

from src.sentiment_model import SentimentScorer


class StubScorer:
    backend_name = "stub"
    model_name = "stub-scorer"
    model_revision = "test"
    neutral_threshold = 0.2

    _known = {"GOOD": 0.9, "BAD": -0.9, "MEH": 0.0}

    def score(self, text: str) -> dict:
        signed_score = self._known.get(text.strip(), 0.5)
        label = "positive" if signed_score > 0.2 else "negative" if signed_score < -0.2 else "neutral"
        return {"label": label, "score": signed_score, "confidence": abs(signed_score) or 0.5}


class BatchStubScorer(StubScorer):
    """StubScorer with `score_batch`, recording each batch's size so tests
    can assert a conversation is scored in one call."""

    def __init__(self):
        self.batch_sizes = []

    def score_batch(self, texts):
        self.batch_sizes.append(len(texts))
        return [self.score(t) for t in texts]


def word_count(text: str) -> int:
    """Stand-in tokenizer: one token per whitespace-separated word."""
    return len(text.split())


class FakeChunkScorer(SentimentScorer):
    """Real `score` / chunking / aggregation (SentimentScorer's own), fake model: a chunk is
    positive if it contains 'good' and not 'not good', else negative. `backend_name` is
    "transformer" and `_class_probs_batch` is implemented, so this also exercises the
    chunk-level sharding path (src/sharding.py's score_chunk_shard/merge_partials), not just
    SentimentScorer.score_batch."""

    def __init__(self):  # skip model loading
        self.backend_name = "transformer"
        self.model_name = "fake-chunk-scorer"
        self.model_revision = "test"
        self.neutral_threshold = 0.2

    def _count_tokens(self, text):
        return word_count(text)

    def _class_probs_batch(self, chunks, lengths):
        return [self._class_probs(c) for c in chunks]

    def _class_probs(self, text):
        positive = "good" in text and "not good" not in text
        return {"positive": 0.9, "negative": 0.1} if positive else {"positive": 0.1, "negative": 0.9}
