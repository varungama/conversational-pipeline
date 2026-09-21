"""Fake scorer shared by both test suites: fixed marker words give deterministic sentiment."""


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
