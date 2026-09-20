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
import re

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

# The model's window is 512 tokens including the <s>/</s> special tokens.
# Text longer than this used to be silently truncated, dropping whatever
# came after the cutoff - see `SentimentScorer.score` for the chunking that
# replaces that.
MAX_MODEL_TOKENS = 512
SPECIAL_TOKEN_BUDGET = 2
CHUNK_TOKEN_LIMIT = MAX_MODEL_TOKENS - SPECIAL_TOKEN_BUDGET

# Sentence boundaries, zero-width so no characters are lost: after a
# Japanese/fullwidth terminator (no space follows in Japanese), after an
# ASCII terminator that is followed by whitespace, and after a newline.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？])|(?<=[.!?])(?=\s)|(?<=\n)")


def split_sentences(text: str) -> list[str]:
    """Splits on sentence boundaries without dropping or altering any
    character: `"".join(split_sentences(t)) == t`."""
    return [part for part in _SENTENCE_BOUNDARY.split(text) if part]


def _hard_split(piece: str, count_tokens, limit: int) -> list[str]:
    """Last resort for a single sentence that alone exceeds `limit`: halve it
    by characters until every part fits. Deterministic; may cut mid-word."""
    if len(piece) <= 1 or count_tokens(piece) <= limit:
        return [piece]
    mid = len(piece) // 2
    return _hard_split(piece[:mid], count_tokens, limit) + _hard_split(piece[mid:], count_tokens, limit)


def chunk_text(text: str, count_tokens, limit: int = CHUNK_TOKEN_LIMIT) -> list[tuple[str, int]]:
    """Packs whole sentences greedily into chunks of at most `limit` tokens.

    Returns [(chunk_text, token_count), ...]. Sentence-aware so a negation
    is never separated from what it negates ("not" / "good"); greedy
    left-to-right with no randomness so the same text always chunks the same
    way (byte-identical reruns). Text that already fits returns one chunk
    that is the original text, untouched.
    """
    total = count_tokens(text)
    if total <= limit:
        return [(text, total)]

    pieces = []
    for sentence in split_sentences(text):
        pieces.extend(_hard_split(sentence, count_tokens, limit))

    chunks, current, current_tokens = [], "", 0
    for piece in pieces:
        n = count_tokens(piece)
        # Token counts of adjacent pieces are near-additive; re-check the
        # joined text for exactness instead of trusting the sum.
        if current and count_tokens(current + piece) > limit:
            chunks.append((current, current_tokens))
            current = ""
        current += piece
        current_tokens = count_tokens(current) if current != piece else n
    if current:
        chunks.append((current, current_tokens))
    return [(c.strip(), n) for c, n in chunks if c.strip()]


def aggregate_probs(chunk_probs: list[tuple[dict, int]]) -> dict:
    """Token-weighted mean of per-class probabilities across chunks. A
    longer chunk says more of the text, so it counts for more. Repeating one
    sentence N times leaves the mean unchanged."""
    total = sum(weight for _, weight in chunk_probs) or 1
    classes = {label for probs, _ in chunk_probs for label in probs}
    return {c: sum(p.get(c, 0.0) * w for p, w in chunk_probs) / total for c in classes}


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
        worth scoring at all.

        Text over the model's window is split into sentence-aware chunks,
        each chunk scored, and the per-class probabilities averaged
        (token-weighted) before the label / signed score / confidence are
        derived - so those keep exactly the meaning they have for short
        text. `n_chunks` reports how many pieces were scored (1 for text
        that fits, and always 1 for the VADER fallback, which has no
        window limit). This lives here, not in src/scoring.py, so both
        workers pick it up unchanged."""
        if self.backend_name == "transformer":
            signed_score, confidence, label, n_chunks = self._score_with_transformer(text)
        else:
            signed_score, confidence = self._score_with_vader(text)
            label = self._label(signed_score)
            n_chunks = 1

        return {
            "label": label,
            "score": round(signed_score, 4),
            "confidence": round(confidence, 4),
            "n_chunks": n_chunks,
        }

    def _count_tokens(self, text: str) -> int:
        return len(self._pipeline.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _class_probs(self, text: str) -> dict:
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
        return {r["label"].lower(): float(r["score"]) for r in results}

    def _score_with_transformer(self, text: str):
        chunks = chunk_text(text, self._count_tokens)
        if len(chunks) == 1:
            probs = self._class_probs(chunks[0][0])
        else:
            probs = aggregate_probs([(self._class_probs(c), n) for c, n in chunks])
        predicted_label, confidence = max(probs.items(), key=lambda kv: kv[1])
        signed_score = probs.get("positive", 0.0) - probs.get("negative", 0.0)
        return signed_score, confidence, predicted_label, len(chunks)

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
