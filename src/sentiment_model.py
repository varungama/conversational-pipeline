"""Turn-level sentiment scoring: a transformer model, with a VADER lexicon
fallback if the transformer can't be loaded (offline, missing packages, etc).
Both backends return the same shape:

    {"label": "positive" | "neutral" | "negative",
     "score": signed value in [-1, 1],
     "confidence": value in [0, 1]}

The transformer only outputs POSITIVE/NEGATIVE, so NEUTRAL is recovered by
thresholding: a signed score near zero (within NEUTRAL_THRESHOLD) means the
model was unsure, which we report as neutral.
"""

import logging

logger = logging.getLogger(__name__)

TRANSFORMER_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
TRANSFORMER_REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"  # pinned commit, not "main"
NEUTRAL_THRESHOLD = 0.2


class SentimentScorer:
    """Loads a backend once, then scores as many turns as you like."""

    def __init__(self, neutral_threshold: float = NEUTRAL_THRESHOLD):
        self.neutral_threshold = neutral_threshold
        self.backend_name, self.model_name, self.model_revision = self._load_backend()

    def _load_backend(self):
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                task="sentiment-analysis",
                model=TRANSFORMER_MODEL_NAME,
                revision=TRANSFORMER_REVISION,
                truncation=True,
                max_length=512,
            )
            resolved_revision = getattr(self._pipeline.model.config, "_commit_hash", None)
            revision = resolved_revision or TRANSFORMER_REVISION
            logger.info("Loaded transformer model %s @ %s", TRANSFORMER_MODEL_NAME, revision)
            return "transformer", TRANSFORMER_MODEL_NAME, revision

        except Exception as exc:  # anything at all - missing package, no network, bad cache
            logger.warning("Transformer model unavailable (%s). Falling back to VADER lexicon.", exc)
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            import vaderSentiment

            self._vader = SentimentIntensityAnalyzer()
            version = getattr(vaderSentiment, "__version__", "unknown")
            return "vader_lexicon", "vaderSentiment", version

    def score(self, text: str) -> dict:
        """Scores one piece of text. Assumes non-empty text; the caller
        decides whether a turn has content worth scoring."""
        if self.backend_name == "transformer":
            signed_score, confidence = self._score_with_transformer(text)
        else:
            signed_score, confidence = self._score_with_vader(text)

        return {
            "label": self._label(signed_score),
            "score": round(signed_score, 4),
            "confidence": round(confidence, 4),
        }

    def _score_with_transformer(self, text: str):
        result = self._pipeline(text)[0]
        confidence = float(result["score"])
        signed_score = confidence if result["label"] == "POSITIVE" else -confidence
        return signed_score, confidence

    def _score_with_vader(self, text: str):
        polarity = self._vader.polarity_scores(text)
        signed_score = float(polarity["compound"])  # already bounded to [-1, 1]
        confidence = abs(signed_score)
        return signed_score, confidence

    def _label(self, signed_score: float) -> str:
        if signed_score > self.neutral_threshold:
            return "positive"
        if signed_score < -self.neutral_threshold:
            return "negative"
        return "neutral"
