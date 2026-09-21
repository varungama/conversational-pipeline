"""Splitting one huge conversation into independently schedulable shards.

Conversation cost is wildly uneven (a 200-turn conversation is ~200x a
1-turn one). Longest-first dispatch alone cannot fix that: if the single
biggest job is bigger than a worker's fair share of the whole run, one
worker is still the tail. Sharding caps the size of any one task.

Turns are scored independently of each other (a turn's score depends only on
its own text), so a shard is just "score turns [start, end)" and the result
is a partial map of `turn index -> score`. Once every shard's partial is
present, whoever finished last assembles the normal result record with
`scoring.evaluate(..., turn_scores=merged)` - the same code path, so a
sharded conversation produces exactly the record an unsharded one would.

`cost` everywhere in this module means turn count: a cheap proxy that needs
no tokenizer. It ignores turn length (a very long turn is chunked and
batched inside the scorer instead).
"""

import math

from src import conversation as conv
from src.scoring import extract_identity, score_turns

# One shard is at most this many turns. Also the worst-case task size that
# the SQS visibility timeout is sized against (src/aws/config.py).
SHARD_MAX_TURNS = 50


def estimate_cost(raw_doc) -> int:
    """Relative cost of scoring this document. Malformed docs cost 1 - they
    are rejected quickly."""
    turns = raw_doc.get("turns") if isinstance(raw_doc, dict) else None
    return max(1, len(turns)) if isinstance(turns, list) else 1


def plan_shards(raw_doc, max_turns: int = SHARD_MAX_TURNS) -> list[dict]:
    """Returns [] if the conversation should be processed whole, else one
    descriptor per shard: {"index", "count", "start", "end"} over the raw
    `turns` list, contiguous, near-equal size, covering every turn once.

    Only well-formed, identifiable conversations are sharded; anything else
    goes through the normal whole-record path so it dead-letters with the
    usual reason."""
    tenant_id, conversation_id = extract_identity(raw_doc)
    if not tenant_id or not conversation_id:
        return []
    turns = raw_doc.get("turns")
    if not isinstance(turns, list) or len(turns) <= max_turns:
        return []
    count = math.ceil(len(turns) / max_turns)
    bounds = [i * len(turns) // count for i in range(count + 1)]
    return [{"index": i, "count": count, "start": bounds[i], "end": bounds[i + 1]} for i in range(count)]


def score_shard(raw_doc, shard: dict, scorer) -> dict:
    """Scores the turns in one shard. Returns {turn index: score dict}.
    Raises conv.ConversationRejected if the doc is not scorable at all."""
    turn_results = conv.load_conversation(raw_doc)
    return score_turns(turn_results, scorer, indices=range(shard["start"], shard["end"]))


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
