"""Splits a huge conversation into shards scored independently and merged by scoring.evaluate."""

import hashlib
import json
import logging

from src import conversation as conv
from src.scoring import extract_identity, input_content_hash, score_turns
from src.sentiment_model import finalize_chunk_probs

logger = logging.getLogger(__name__)

COST_PER_TURN = 8    # ~ms fixed work per turn (fitted)
COST_PER_TOKEN = 4   # ~ms per token (fitted)
COST_PER_TURN_CHARS = 12  # fallback per-turn cost without a tokenizer
COST_PER_CHAR = 1

SHARD_MAX_COST = 2500  # hard cap per shard (~2.5s); an oversized turn gets its own shard


def _token_cost(tokens: int) -> int:
    return COST_PER_TURN + COST_PER_TOKEN * tokens


def turn_cost(raw_turn, count_tokens=None) -> int:
    """Cost by tokens if `count_tokens` (str -> int) is given, else by character length."""
    text = raw_turn.get("text") if isinstance(raw_turn, dict) else None
    if not isinstance(text, str) or not text.strip():
        return COST_PER_TURN if count_tokens else COST_PER_TURN_CHARS
    if count_tokens:
        return _token_cost(count_tokens(text))
    return COST_PER_TURN_CHARS + COST_PER_CHAR * len(text)


def estimate_cost(raw_doc, count_tokens=None) -> int:
    """Relative cost of scoring this document; malformed docs cost 1 (rejected quickly)."""
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
    """Fewest ranges under `cap`, then cap lowered as far as that count allows so ranges come out near-equal."""
    ranges = _greedy_ranges(costs, cap)
    count = len(ranges)
    if count == 1:
        return ranges
    lo, hi = min(max(costs), cap), cap  # smallest cap that still yields `count` ranges
    while lo < hi:
        mid = (lo + hi) // 2
        if len(_greedy_ranges(costs, mid)) <= count:
            hi = mid
        else:
            lo = mid + 1
    balanced = _greedy_ranges(costs, hi)
    logger.debug("_pack: %d item(s), cap=%d -> %d range(s) (balanced at cap=%d): %s",
                 len(costs), cap, count, hi, balanced)
    return balanced


def plan_shards(raw_doc, max_cost: int = SHARD_MAX_COST, count_tokens=None, chunk_text_fn=None) -> list[dict]:
    """[] to process whole, else shard descriptors covering every turn once, capped at `max_cost`.
    A turn over `max_cost` alone is chunk-split via `chunk_text_fn` if given, else kept as one oversized shard."""
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

    if len(shards) == 1:  # nothing to shard, e.g. one enormous unsplittable turn alone
        return []

    for i, shard in enumerate(shards):
        shard["index"], shard["count"] = i, len(shards)
    h = plan_hash(raw_doc, shards)
    for shard in shards:
        shard["plan_hash"] = h

    logger.info("plan_shards %s/%s: %d turn(s), cost=%d > cap=%d -> %d shard(s) (plan_hash=%s)",
                tenant_id, conversation_id, len(turns), sum(costs), max_cost, len(shards), h)
    for shard in shards:
        if is_chunk_shard(shard):
            logger.debug("  shard %d/%d: turn %d, chunks %d-%d, cost=%d",
                         shard["index"], shard["count"], shard["start"],
                         shard["chunk_start"], shard["chunk_end"] - 1, shard["cost"])
        else:
            logger.debug("  shard %d/%d: turns %d-%d, cost=%d",
                         shard["index"], shard["count"], shard["start"], shard["end"] - 1, shard["cost"])
    return shards


def plan_hash(raw_doc, shards: list[dict]) -> str:
    """Short id of this doc's content + shard plan; scopes S3/DynamoDB shard state so a replanned rerun can't merge stale data."""
    canonical = json.dumps({"content": input_content_hash(raw_doc), "shards": shards},
                            sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def is_chunk_shard(shard: dict) -> bool:
    return "chunk_start" in shard


def plan_shards_and_cost(raw_doc, max_cost: int = SHARD_MAX_COST, count_tokens=None, chunk_text_fn=None):
    """(shards, total_cost) in one pass; equivalent to plan_shards + estimate_cost but tokenizes each turn once."""
    tenant_id, conversation_id = extract_identity(raw_doc)
    turns = raw_doc.get("turns") if isinstance(raw_doc, dict) else None
    if not tenant_id or not conversation_id or not isinstance(turns, list):
        return [], estimate_cost(raw_doc, count_tokens)
    total = max(1, sum(turn_cost(t, count_tokens) for t in turns))
    if total <= max_cost:
        return [], total
    return plan_shards(raw_doc, max_cost=max_cost, count_tokens=count_tokens, chunk_text_fn=chunk_text_fn), total


def _task_label(task: tuple) -> str:
    """Human-readable label for a (cost, tenant_id, conversation_id, shard) task; falls back to the raw tuple."""
    if len(task) == 4:
        cost, tenant_id, conversation_id, shard = task
        suffix = f"#shard{shard['index']}" if shard else ""
        return f"{tenant_id}/{conversation_id}{suffix} (cost={cost})"
    return repr(task)


def lpt_interleaved(tasks: list, lanes: int) -> list:
    """Longest-first, dealt round-robin into `lanes` buckets so no `lanes`-sized receive-batch window is all-big."""
    ordered = sorted(tasks, key=lambda t: -t[0])
    lanes_of = [ordered] if lanes <= 1 else [ordered[i::lanes] for i in range(lanes)]
    if logger.isEnabledFor(logging.DEBUG):
        for i, lane in enumerate(lanes_of):
            logger.debug("lane %d: %d task(s), total cost=%d: %s",
                         i, len(lane), sum(t[0] for t in lane), [_task_label(t) for t in lane])
    if lanes <= 1:
        return ordered
    return [task for lane in lanes_of for task in lane]


def slice_turns(raw_doc, shard: dict) -> list:
    """The raw turns a whole-turn shard covers - stored once so a worker never fetches the whole doc."""
    return raw_doc["turns"][shard["start"]:shard["end"]]


def slice_chunk_texts(raw_doc, shard: dict, chunk_text_fn) -> list:
    """The (chunk, token_count) pairs one chunk-level shard covers: re-chunks the turn and slices by index."""
    text = raw_doc["turns"][shard["start"]]["text"]
    return chunk_text_fn(text)[shard["chunk_start"]:shard["chunk_end"]]


def score_shard(turns: list, shard: dict, scorer) -> dict:
    """Scores one whole-turn shard; returns {turn index in the whole conversation: score}."""
    turn_results = [conv.validate_turn(raw, shard["start"] + i) for i, raw in enumerate(turns)]
    return score_turns(turn_results, scorer)


def score_chunk_shard(chunks: list, shard: dict, scorer) -> dict:
    """Scores one chunk-level shard (transformer backend only); returns an unfinished partial for merge_partials to pool."""
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
    """Unions {shard index: {turn index: payload}} into {turn index: score}, finalizing pooled chunk partials."""
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
