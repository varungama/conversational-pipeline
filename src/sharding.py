"""Splits a huge conversation into shards scored independently (turns are independent) and merged by scoring.evaluate.
Cost (~ms) = 8/turn + 4/token (fit: scripts/fit_cost_model.py, R2 98.8%), or 12/turn + 1/char without a tokenizer."""

from src import conversation as conv
from src.scoring import extract_identity, score_turns

COST_PER_TURN = 8    # ~ms of fixed work per turn (fitted: 7.9 ms)
COST_PER_TOKEN = 4   # ~ms per token of text      (fitted: 4.05 ms)
# Fallback when no tokenizer is available (fitted with a 12.5 ms per turn: 12.5 + 0.88/char)
COST_PER_TURN_CHARS = 12
COST_PER_CHAR = 1

# Max cost per shard (~2.5s), also what the visibility timeout is sized against;
# a single turn bigger than this cannot be split and gets its own shard.
SHARD_MAX_COST = 2500


def turn_cost(raw_turn, count_tokens=None) -> int:
    """`count_tokens`, if given, is a str -> int function (the model's
    tokenizer); otherwise the text's length in characters is used."""
    text = raw_turn.get("text") if isinstance(raw_turn, dict) else None
    if not isinstance(text, str) or not text.strip():
        return COST_PER_TURN if count_tokens else COST_PER_TURN_CHARS
    if count_tokens:
        return COST_PER_TURN + COST_PER_TOKEN * count_tokens(text)
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


def plan_shards(raw_doc, max_cost: int = SHARD_MAX_COST, count_tokens=None) -> list[dict]:
    """Returns [] to process whole, else shard descriptors {"index", "count", "start", "end", "cost"} covering every turn once.
    `max_cost` is a hard cap (except a single oversized turn); fewest shards, then rebalanced near-equal. Malformed docs are not sharded."""
    tenant_id, conversation_id = extract_identity(raw_doc)
    if not tenant_id or not conversation_id:
        return []
    turns = raw_doc.get("turns")
    if not isinstance(turns, list):
        return []
    costs = [turn_cost(t, count_tokens) for t in turns]
    if sum(costs) <= max_cost:
        return []

    count = len(_greedy_ranges(costs, max_cost))
    if count == 1:
        return []  # e.g. one enormous turn: nothing to split
    lo, hi = max(costs), max_cost  # smallest cap that still yields `count` shards
    lo = min(lo, hi)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(_greedy_ranges(costs, mid)) <= count:
            hi = mid
        else:
            lo = mid + 1
    ranges = _greedy_ranges(costs, hi)
    return [{"index": i, "count": len(ranges), "start": a, "end": b, "cost": c}
            for i, (a, b, c) in enumerate(ranges)]


def slice_turns(raw_doc, shard: dict) -> list:
    """The raw turns a shard covers - what gets stored (once) per shard so a
    worker never has to fetch the whole document to score its slice."""
    return raw_doc["turns"][shard["start"]:shard["end"]]


def score_shard(turns: list, shard: dict, scorer) -> dict:
    """Scores one shard's raw turns; returns {turn index in the whole conversation: score}."""
    turn_results = [conv.validate_turn(raw, shard["start"] + i) for i, raw in enumerate(turns)]
    return score_turns(turn_results, scorer)


def encode_partial(scores: dict) -> dict:
    """JSON-safe form of a shard's scores (JSON object keys are strings)."""
    return {str(index): scored for index, scored in scores.items()}


def decode_partial(payload: dict) -> dict:
    return {int(index): scored for index, scored in payload.items()}


def merge_partials(partials: dict) -> dict:
    """Unions {shard index: {turn index: score}} into one {turn index: score}."""
    merged = {}
    for shard_scores in partials.values():
        merged.update(shard_scores)
    return merged
