"""Turn-level sentiment scoring: a transformer model, with a rule-based lexicon VADER fallback.

Every score is reported the same shape regardless of which backend produced
it, so the rest of the pipeline never has to know which one ran:

    {"label": "positive" | "neutral" | "negative",
     "score": signed value in [-1, 1],
     "confidence": value in [0, 1]}

Two transformer models are registered in MODEL_REGISTRY, selectable via
`--model` on run.py (NOTES.md Section 5.1 has a measured cost/throughput
comparison between them):

- `roberta` (clapAI/roberta-base-multilingual-sentiment, XLM-RoBERTa-base,
  278M params) - the current default. A genuine 3-way classifier -
  positive/neutral/negative - trained across 16+ languages including French
  and Japanese, so its per-turn `label` comes directly from the model's own
  prediction rather than a threshold we impose.
- `distilbert` (distilbert-base-uncased-finetuned-sst-2-english, 66M params)
  - the original backend. English-only and strictly binary
  (positive/negative, no native neutral class) - see NOTES.md Section 5.2
  for the measured French/Japanese failure mode this model has and
  `roberta` fixes, at a throughput cost (NOTES.md Section 5.1).
"""

import logging

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "roberta": {
        "model_name": "clapAI/roberta-base-multilingual-sentiment",
        "revision": "5986405bd6f2fbaf98b2a5568fdbc02d74546694",
        "params": "278M",
    },
    "distilbert": {
        "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
        "revision": "714eb0fa89d2f80546fda750413ed43d93601a13",
        "params": "66M",
    },
}
DEFAULT_MODEL_KEY = "roberta"
NEUTRAL_THRESHOLD = 0.2


class SentimentScorer:
    """Loads a backend once, then scores as many turns as you like."""

    def __init__(self, model_key: str = DEFAULT_MODEL_KEY, neutral_threshold: float = NEUTRAL_THRESHOLD):
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model {model_key!r}. Choose one of {sorted(MODEL_REGISTRY)}.")
        self.model_key = model_key
        self.neutral_threshold = neutral_threshold
        self.backend_name, self.model_name, self.model_revision = self._load_backend()

    def _load_backend(self):
        target = MODEL_REGISTRY[self.model_key]
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                task="sentiment-analysis",
                model=target["model_name"],
                revision=target["revision"],
                truncation=True,
                max_length=512,
            )
            resolved_revision = getattr(self._pipeline.model.config, "_commit_hash", None)
            revision = resolved_revision or target["revision"]
            logger.info("Loaded transformer model %s @ %s", target["model_name"], revision)
            return "transformer", target["model_name"], revision

        except Exception as exc:  # anything at all - missing package, no network, bad cache
            logger.warning("Transformer model unavailable (%s). Falling back to VADER lexicon.", exc)
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            import vaderSentiment

            self._vader = SentimentIntensityAnalyzer()
            version = getattr(vaderSentiment, "__version__", "unknown")
            return "vader_lexicon", "vaderSentiment", version

    def score(self, text: str) -> dict:
        """Scores one piece of text. Assumes text is a non-empty string -
        the caller is responsible for deciding whether a turn has content
        worth scoring at all."""
        if self.backend_name == "transformer":
            signed_score, confidence, label = self._score_with_transformer(text)
        else:
            signed_score, confidence = self._score_with_vader(text)
            label = self._label(signed_score)

        return {
            "label": label,
            "score": round(signed_score, 4),
            "confidence": round(confidence, 4),
        }

    def _score_with_transformer(self, text: str):
        results = self._pipeline(text, top_k=None)
        # A single string input returns a flat list of {label, score} dicts,
        # one per class - but guard against the nested [[...]] shape some
        # pipeline versions use for single-example batches.
        if results and isinstance(results[0], list):
            results = results[0]
        # Lower-cased because the two registered models disagree on casing
        # (roberta: "positive"/"neutral"/"negative", distilbert:
        # "POSITIVE"/"NEGATIVE") - normalising here is what lets
        # `probs.get("positive")` below work for both.
        probs = {r["label"].lower(): float(r["score"]) for r in results}
        predicted_label, confidence = max(probs.items(), key=lambda kv: kv[1])
        signed_score = probs.get("positive", 0.0) - probs.get("negative", 0.0)
        return signed_score, confidence, predicted_label

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
