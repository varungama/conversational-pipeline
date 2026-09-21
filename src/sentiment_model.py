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


# Chunks per forward pass. Batches are cut from a length-sorted list (see
# `length_sorted_batches`), so padding to the longest item in a batch wastes
# little; unsorted, one long turn pads all 31 of its neighbours to its length.
BATCH_SIZE = 32


def length_sorted_batches(lengths: list[int], batch_size: int = BATCH_SIZE) -> list[list[int]]:
    """Groups item indices into batches of similar length: longest first,
    ties broken by original index so the grouping is deterministic. Callers
    scatter results back by index, so input order is preserved."""
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


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
        """Scores one piece of text. See `score_batch`."""
        return self.score_batch([text])[0]

    def score_batch(self, texts: list[str]) -> list[dict]:
        """Scores many texts at once; returns one result per text, in order.
        Assumes each text is a non-empty string - the caller is responsible
        for deciding whether a turn has content worth scoring at all.

        Text over the model's window is split into sentence-aware chunks,
        each chunk scored, and the per-class probabilities averaged
        (token-weighted) before the label / signed score / confidence are
        derived - so those keep exactly the meaning they have for short
        text. `n_chunks` reports how many pieces were scored (1 for text
        that fits, and always 1 for the VADER fallback, which has no
        window limit).

        All chunks from all texts are pooled and run through the model in
        length-sorted batches (`length_sorted_batches`), then regrouped per
        text. This lives here, not in src/scoring.py, so both workers pick
        it up unchanged."""
        if self.backend_name == "transformer":
            outcomes = self._score_batch_with_transformer(texts)
        else:
            outcomes = []
            for text in texts:
                signed_score, confidence = self._score_with_vader(text)
                outcomes.append((signed_score, confidence, self._label(signed_score), 1))

        return [
            {"label": label, "score": round(signed_score, 4), "confidence": round(confidence, 4),
             "n_chunks": n_chunks}
            for signed_score, confidence, label, n_chunks in outcomes
        ]

    def _count_tokens(self, text: str) -> int:
        return len(self._pipeline.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _class_probs_batch(self, chunks: list[str], lengths: list[int]) -> list[dict]:
        """Class probabilities for each chunk, in input order. `lengths` are
        the chunks' token counts, used only to group similar lengths."""
        probs = [None] * len(chunks)
        for batch in length_sorted_batches(lengths):
            results = self._pipeline([chunks[i] for i in batch], top_k=None, batch_size=len(batch))
            # A list input returns one list of {label, score} per item, but
            # guard against the flat single-example shape some pipeline
            # versions use.
            if results and isinstance(results[0], dict):
                results = [results]
            for i, result in zip(batch, results):
                # Lower-cased because the two registered models disagree on
                # casing (roberta: "positive"/"neutral"/"negative",
                # distilbert: "POSITIVE"/"NEGATIVE") - normalising here is
                # what lets `probs.get("positive")` below work for both.
                probs[i] = {r["label"].lower(): float(r["score"]) for r in result}
        return probs

    def _score_batch_with_transformer(self, texts: list[str]):
        chunked = [chunk_text(text, self._count_tokens) for text in texts]
        flat = [(text_index, chunk, n) for text_index, chunks in enumerate(chunked) for chunk, n in chunks]
        flat_probs = self._class_probs_batch([chunk for _, chunk, _ in flat], [n for _, _, n in flat])

        per_text = [[] for _ in texts]
        for (text_index, _, n), probs in zip(flat, flat_probs):
            per_text[text_index].append((probs, n))

        outcomes = []
        for chunk_probs in per_text:
            probs = chunk_probs[0][0] if len(chunk_probs) == 1 else aggregate_probs(chunk_probs)
            predicted_label, confidence = max(probs.items(), key=lambda kv: kv[1])
            signed_score = probs.get("positive", 0.0) - probs.get("negative", 0.0)
            outcomes.append((signed_score, confidence, predicted_label, len(chunk_probs)))
        return outcomes

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
