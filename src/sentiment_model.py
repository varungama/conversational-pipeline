"""Turn-level sentiment: a transformer (default `roberta`, 3-way multilingual; `distilbert`, English binary) with a VADER fallback.
Every backend returns {"label", "score" in [-1, 1], "confidence" in [0, 1]}; model comparison in NOTES.md 5.1-5.2."""

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

# Model window is 512 tokens including special tokens; longer text is chunked, not truncated.
MAX_MODEL_TOKENS = 512
SPECIAL_TOKEN_BUDGET = 2
CHUNK_TOKEN_LIMIT = MAX_MODEL_TOKENS - SPECIAL_TOKEN_BUDGET

# Zero-width sentence boundaries: after 。！？, after .!? followed by whitespace, and after a newline.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？])|(?<=[.!?])(?=\s)|(?<=\n)")


def split_sentences(text: str) -> list[str]:
    """Splits on sentence boundaries, lossless: `"".join(split_sentences(t)) == t`."""
    return [part for part in _SENTENCE_BOUNDARY.split(text) if part]


def _hard_split(piece: str, count_tokens, limit: int) -> list[str]:
    """Last resort for a sentence over `limit`: halve by characters until every part fits; may cut mid-word."""
    if len(piece) <= 1 or count_tokens(piece) <= limit:
        return [piece]
    mid = len(piece) // 2
    return _hard_split(piece[:mid], count_tokens, limit) + _hard_split(piece[mid:], count_tokens, limit)


def chunk_text(text: str, count_tokens, limit: int = CHUNK_TOKEN_LIMIT) -> list[tuple[str, int]]:
    """Packs whole sentences greedily into chunks of at most `limit` tokens; returns [(chunk, token_count)].
    Sentence-aware and deterministic; text that fits comes back as one untouched chunk."""
    total = count_tokens(text)
    if total <= limit:
        return [(text, total)]

    pieces = []
    for sentence in split_sentences(text):
        pieces.extend(_hard_split(sentence, count_tokens, limit))

    chunks, current, current_tokens = [], "", 0
    for piece in pieces:
        n = count_tokens(piece)
        if current and count_tokens(current + piece) > limit:  # re-check joined text, don't trust the sum
            chunks.append((current, current_tokens))
            current = ""
        current += piece
        current_tokens = count_tokens(current) if current != piece else n
    if current:
        chunks.append((current, current_tokens))
    return [(c.strip(), n) for c, n in chunks if c.strip()]


def aggregate_probs(chunk_probs: list[tuple[dict, int]]) -> dict:
    """Token-weighted mean of per-class probabilities across chunks (longer chunks count for more)."""
    total = sum(weight for _, weight in chunk_probs) or 1
    classes = {label for probs, _ in chunk_probs for label in probs}
    return {c: sum(p.get(c, 0.0) * w for p, w in chunk_probs) / total for c in classes}


def finalize_chunk_probs(chunk_probs: list[tuple[dict, int]]):
    """(signed_score, confidence, label, n_chunks) from chunks' (class_probs, token_count); shared by score_batch and merge_partials."""
    probs = chunk_probs[0][0] if len(chunk_probs) == 1 else aggregate_probs(chunk_probs)
    label, confidence = max(probs.items(), key=lambda kv: kv[1])
    signed_score = probs.get("positive", 0.0) - probs.get("negative", 0.0)
    return signed_score, confidence, label, len(chunk_probs)


# Caps on one forward pass: batches limited by padded tokens (items x longest) as well as item count
BATCH_SIZE = 32
MAX_BATCH_TOKENS = 512


def length_sorted_batches(lengths: list[int], batch_size: int = BATCH_SIZE,
                          max_tokens: int = MAX_BATCH_TOKENS) -> list[list[int]]:
    """Length-sorted batches capped by `batch_size` and padded size `max_tokens`; an over-budget item gets its own batch."""
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    batches, current = [], []
    for i in order:
        if current and (len(current) >= batch_size or (len(current) + 1) * lengths[current[0]] > max_tokens):  # longest-first: current[0] sets padded width
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches


def load_token_counter(model_key: str = DEFAULT_MODEL_KEY):
    """A str -> token-count function from the model's own tokenizer (model not loaded), or None so callers use characters."""
    target = MODEL_REGISTRY[model_key]
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(target["model_name"], revision=target["revision"])
    except Exception as exc:
        logger.warning("Tokenizer unavailable (%s); estimating cost from characters instead.", exc)
        return None
    return lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])


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
        """Scores many texts at once, one result per text in order (`n_chunks` = pieces scored; texts must be non-empty).
        Long text is chunked, chunk probabilities averaged, and all chunks batched together by length."""
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
        """Class probabilities for each chunk, in input order; `lengths` groups similar-length chunks together."""
        probs = [None] * len(chunks)
        for batch in length_sorted_batches(lengths, BATCH_SIZE, MAX_BATCH_TOKENS):  # module globals, read at call time
            results = self._pipeline([chunks[i] for i in batch], top_k=None, batch_size=len(batch))
            # Some pipeline versions return a flat single-example list; wrap it.
            if results and isinstance(results[0], dict):
                results = [results]
            for i, result in zip(batch, results):
                # Lower-cased: roberta returns "positive", distilbert "POSITIVE".
                probs[i] = {r["label"].lower(): float(r["score"]) for r in result}
        return probs

    def _score_batch_with_transformer(self, texts: list[str]):
        chunked = [chunk_text(text, self._count_tokens) for text in texts]
        flat = [(text_index, chunk, n) for text_index, chunks in enumerate(chunked) for chunk, n in chunks]
        flat_probs = self._class_probs_batch([chunk for _, chunk, _ in flat], [n for _, _, n in flat])

        per_text = [[] for _ in texts]
        for (text_index, _, n), probs in zip(flat, flat_probs):
            per_text[text_index].append((probs, n))

        return [finalize_chunk_probs(chunk_probs) for chunk_probs in per_text]

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
