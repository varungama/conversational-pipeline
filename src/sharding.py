"""Splits a huge conversation into shards scored independently (turns are independent) and merged by scoring.evaluate.
Cost (~ms) = 8/turn + 4/token (fit: scripts/fit_cost_model.py, R2 98.8%), or 12/turn + 1/char without a tokenizer."""

from src import conversation as conv
from src.scoring import extract_identity, score_turns
from src.sentiment_model import finalize_chunk_probs

COST_PER_TURN = 8    # ~ms of fixed work per turn (fitted: 7.9 ms)
COST_PER_TOKEN = 4   # ~ms per token of text      (fitted: 4.05 ms)
# Fallback when no tokenizer is available (fitted with a 12.5 ms per turn: 12.5 + 0.88/char)
COST_PER_TURN_CHARS = 12
COST_PER_CHAR = 1

# Max cost per shard (~2.5s), also what the visibility timeout is sized against;
# a single turn bigger than this cannot be split and gets its own shard.
SHARD_MAX_COST = 2500


def _token_cost(tokens: int) -> int:
    return COST_PER_TURN + COST_PER_TOKEN * tokens


def turn_cost(raw_turn, count_tokens=None) -> int:
    """`count_tokens`, if given, is a str -> int function (the model's
    tokenizer); otherwise the text's length in characters is used."""
    text = raw_turn.get("text") if isinstance(raw_turn, dict) else None
    if not isinstance(text, str) or not text.strip():
        return COST_PER_TURN if count_tokens else COST_PER_TURN_CHARS
    if count_tokens:
        return _token_cost(count_tokens(text))
    return COST_PER_TURN_CHARS + COST_PER_CHAR * len(text)


def estimate_cost(raw_doc, count_tokens=None) -> int:
    """Relative cost of scoring this document. Malformed docs cost 1 - they
    are rejected quickly."""
    turns = raw_doc.get("turns") if isinstance(raw_doc, dict) else None
    if not isinstance(turns, list):
        return 1
    return max(1, sum(turn_cost(t, count_tokens) for t in turns))


def _greedy_ranges(costs: list, cap: int) -> list:
    """Contiguous (start, end, cost) ranges closed before passing `cap`; an oversized turn gets its own (fewest ranges)."""
    ranges, start, acc = [], 0, 0
    for i, cost in enumerate(costs):
        if acc and acc + cost > cap:
            ranges.append((start, i, acc))
            start, acc = i, 0
        acc += cost
    ranges.append((start, len(costs), acc))
    return ranges


def _pack(costs: list, cap: int) -> list:
    """Fewest contiguous ranges under `cap` (via `_greedy_ranges`), then the cap is
    lowered as far as that count allows, so ranges come out near-equal instead of
    full-full-...-sliver. Shared by whole-turn and chunk-level packing."""
    count = len(_greedy_ranges(costs, cap))
    if count == 1:
        return _greedy_ranges(costs, cap)
    lo, hi = min(max(costs), cap), cap  # smallest cap that still yields `count` ranges
    while lo < hi:
        mid = (lo + hi) // 2
        if len(_greedy_ranges(costs, mid)) <= count:
            hi = mid
        else:
            lo = mid + 1
    return _greedy_ranges(costs, hi)


def plan_shards(raw_doc, max_cost: int = SHARD_MAX_COST, count_tokens=None, chunk_text_fn=None) -> list[dict]:
    """Returns [] to process whole, else shard descriptors {"index", "count", "start", "end", "cost"} covering every turn once.
    `max_cost` is a hard cap; fewest shards, then rebalanced near-equal. Malformed docs are not sharded.

    A single turn bigger than `max_cost` on its own cannot be split at turn granularity. If `chunk_text_fn`
    (`text -> [(chunk, token_count), ...]`, normally `sentiment_model.chunk_text` bound to the model's
    tokenizer) is given, such a turn is instead split into chunk-level shards - the extra keys
    `"chunk_start"`/`"chunk_end"` index into that turn's own chunk list, and `"start"`/`"end"` both name
    that one turn. Without `chunk_text_fn` (the old behaviour), it still gets one oversized shard."""
    tenant_id, conversation_id = extract_identity(raw_doc)
    if not tenant_id or not conversation_id:
        return []
    turns = raw_doc.get("turns")
    if not isinstance(turns, list):
        return []
    costs = [turn_cost(t, count_tokens) for t in turns]
    if sum(costs) <= max_cost:
        return []

    ranges = _pack(costs, max_cost)

    shards = []
    for start, end, cost in ranges:
        chunk_shards = None
        if chunk_text_fn and end - start == 1 and cost > max_cost:
            text = turns[start].get("text") if isinstance(turns[start], dict) else None
            chunks = chunk_text_fn(text) if isinstance(text, str) else []
            if len(chunks) > 1:
                chunk_costs = [_token_cost(n) for _, n in chunks]
                chunk_shards = [{"start": start, "end": end, "cost": c,
                                 "chunk_start": a, "chunk_end": b}
                                for a, b, c in _pack(chunk_costs, max_cost)]
        shards.extend(chunk_shards or [{"start": start, "end": end, "cost": cost}])

    # One range and it couldn't be chunk-split either (no chunk_text_fn, or one
    # unsplittable chunk): truly nothing to shard, e.g. one enormous turn alone.
    if len(shards) == 1:
        return []

    for i, shard in enumerate(shards):
        shard["index"], shard["count"] = i, len(shards)
    return shards


def is_chunk_shard(shard: dict) -> bool:
    return "chunk_start" in shard


def lpt_interleaved(tasks: list, lanes: int) -> list:
    """Longest-first order, dealt round-robin into `lanes` buckets and
    concatenated, so any `lanes`-sized window mixes big and small costs -
    safe to enqueue before a receive batch > 1 (one ReceiveMessage call can
    otherwise return several of the biggest tasks to a single worker).
    `tasks`: anything sortable by `tasks[i][0]` (cost first). `lanes <= 1`
    is plain longest-first, unchanged from before this existed."""
    ordered = sorted(tasks, key=lambda t: -t[0])
    if lanes <= 1:
        return ordered
    lanes_of = [ordered[i::lanes] for i in range(lanes)]
    return [task for lane in lanes_of for task in lane]


def slice_turns(raw_doc, shard: dict) -> list:
    """The raw turns a whole-turn shard covers - what gets stored (once) per
    shard so a worker never has to fetch the whole document to score its slice."""
    return raw_doc["turns"][shard["start"]:shard["end"]]


def slice_chunk_texts(raw_doc, shard: dict, chunk_text_fn) -> list:
    """The (chunk, token_count) pairs one chunk-level shard covers: re-chunks
    that one turn (same `chunk_text_fn` used at planning time, so boundaries
    match exactly) and slices `[chunk_start:chunk_end]`."""
    text = raw_doc["turns"][shard["start"]]["text"]
    return chunk_text_fn(text)[shard["chunk_start"]:shard["chunk_end"]]


def score_shard(turns: list, shard: dict, scorer) -> dict:
    """Scores one whole-turn shard's raw turns; returns {turn index in the whole conversation: score}."""
    turn_results = [conv.validate_turn(raw, shard["start"] + i) for i, raw in enumerate(turns)]
    return score_turns(turn_results, scorer)


def score_chunk_shard(chunks: list, shard: dict, scorer) -> dict:
    """Scores one chunk-level shard's (chunk, token_count) pairs (transformer backend only - `scorer`
    must have `_class_probs_batch`). Returns a partial that is NOT a finished score:
    {turn index: {"_chunk_probs": [[class_probs, token_count], ...]}} - `merge_partials` pools every
    shard's list for that turn and finalises them together once all have landed."""
    if getattr(scorer, "backend_name", None) != "transformer":
        raise ValueError(f"Chunk-level shards need the transformer backend, got {scorer.backend_name!r}.")
    class_probs = scorer._class_probs_batch([c for c, _ in chunks], [n for _, n in chunks])
    return {shard["start"]: {"_chunk_probs": [[p, n] for p, (_, n) in zip(class_probs, chunks)]}}


def encode_partial(scores: dict) -> dict:
    """JSON-safe form of a shard's scores (JSON object keys are strings)."""
    return {str(index): scored for index, scored in scores.items()}


def decode_partial(payload: dict) -> dict:
    return {int(index): scored for index, scored in payload.items()}


def merge_partials(partials: dict) -> dict:
    """Unions {shard index: {turn index: payload}} into one {turn index: score}. A payload is
    normally a finished score; for an oversized turn split into chunk-level shards it is instead
    `{"_chunk_probs": [...]}` - pooled across every shard covering that turn and finalised together
    (same formula as scoring it whole, via `sentiment_model.finalize_chunk_probs`) once all have landed."""
    merged, raw_by_turn = {}, {}
    for shard_scores in partials.values():
        for turn_index, payload in shard_scores.items():
            if isinstance(payload, dict) and "_chunk_probs" in payload:
                raw_by_turn.setdefault(turn_index, []).extend(payload["_chunk_probs"])
            else:
                merged[turn_index] = payload
    for turn_index, chunk_probs in raw_by_turn.items():
        signed_score, confidence, label, n_chunks = finalize_chunk_probs(chunk_probs)
        merged[turn_index] = {"label": label, "score": round(signed_score, 4),
                              "confidence": round(confidence, 4), "n_chunks": n_chunks}
    return merged
