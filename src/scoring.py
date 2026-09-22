"""Transport-agnostic core: raw conversation -> scored record or dead-letter reason.
Both workers call it so metrics and output shape cannot drift between transports."""

import hashlib
import json
from datetime import datetime, timezone

from src import conversation as conv
from src.metrics import DEFAULT_WEIGHTING, compute_overall_sentiment, compute_sentiment_trajectory, compute_volatility

SCHEMA_VERSION = "1.2"  # 1.1: per-turn `n_chunks`. 1.2: `metrics.volatility`


def result_key(tenant_id: str, conversation_id: str) -> str:
    return f"{tenant_id}::{conversation_id}"


def extract_identity(raw_doc):
    """(tenant_id, conversation_id), or (None, None) if either is missing."""
    if not isinstance(raw_doc, dict):
        return None, None
    return raw_doc.get("tenant_id"), raw_doc.get("conversation_id")


def input_content_hash(raw_doc) -> str:
    """SHA256 of the conversation content, independent of key order and whitespace."""
    canonical = json.dumps(raw_doc, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _score_texts(scorer, texts: list) -> list:
    """Scores all texts in one batched model call when the scorer has `score_batch`, else one call each."""
    if not texts:
        return []
    score_batch = getattr(scorer, "score_batch", None)
    if score_batch is not None:
        return score_batch(texts)
    return [scorer.score(text) for text in texts]


def score_turns(turn_results, scorer, indices=None) -> dict:
    """Scores every scorable turn in one batch; returns {turn index: score}, limited to `indices` if given (a shard)."""
    todo = [t for t in turn_results
            if t["status"] == "ok" and t["has_content"] and (indices is None or t["index"] in indices)]
    scored = _score_texts(scorer, [t["text"] for t in todo])
    return {t["index"]: s for t, s in zip(todo, scored)}


def score_conversation(raw_doc, tenant_id, conversation_id, turn_results, scorer, source_ref,
                        scored_at: str | None = None, weighting: str = DEFAULT_WEIGHTING,
                        turn_scores: dict | None = None) -> dict:
    """Builds the output record; `turn_scores` supplies precomputed shard scores, `source_ref` is the provenance pointer."""
    if turn_scores is None:
        turn_scores = score_turns(turn_results, scorer)

    scored_turns = []
    customer_scores_unordered = []   # (signed_score, confidence) - order does not matter for a mean
    customer_scores_by_time = []     # (parsed_ts, signed_score) - order matters for trajectory

    for turn in turn_results:
        if turn["status"] == "skipped":
            scored_turns.append({
                "index": turn["index"],
                "label": None,
                "score": None,
                "confidence": None,
                "included_in_metrics": False,
                "reason": turn["reason"],
            })
            continue

        entry = {
            "index": turn["index"],
            "seq": turn["seq"],
            "role": turn["role"],
            "ts": turn["ts"],
        }

        if not turn["has_content"]:
            entry.update(label=None, score=None, confidence=None,
                         included_in_metrics=False, reason="empty_text")
            scored_turns.append(entry)
            continue

        scored = turn_scores[turn["index"]]
        is_customer = turn["role"] == "customer"
        entry.update(scored, included_in_metrics=is_customer)
        if not is_customer:
            entry["reason"] = "not_a_customer_turn"
        scored_turns.append(entry)

        if is_customer:
            customer_scores_unordered.append((scored["score"], scored["confidence"]))
            customer_scores_by_time.append((turn["parsed_ts"], scored["score"]))

    # Order by `ts`, not `seq`: fixture c-000021 has non-monotonic, duplicate `seq` but increasing timestamps.
    customer_scores_by_time.sort(key=lambda pair: pair[0])
    ordered_scores = [score for _, score in customer_scores_by_time]

    return {
        "schema_version": SCHEMA_VERSION,
        "result_key": result_key(tenant_id, conversation_id),
        "tenant_id": tenant_id,
        "conversation_id": conversation_id,
        "channel": raw_doc.get("channel"),
        "source_file": source_ref,
        "input_sha256": input_content_hash(raw_doc),
        "scored_at": scored_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": {
            "backend": scorer.backend_name,
            "name": scorer.model_name,
            "revision": scorer.model_revision,
            "neutral_threshold": scorer.neutral_threshold,
            "weighting": weighting,  # recorded for provenance - see src/metrics.py
        },
        "turns": scored_turns,
        "metrics": {
            "overall_sentiment": compute_overall_sentiment(customer_scores_unordered, weighting=weighting),
            "sentiment_trajectory": compute_sentiment_trajectory(ordered_scores),
            "volatility": compute_volatility(ordered_scores),
        },
    }


def evaluate(raw_doc, scorer, source_ref, scored_at: str | None = None, weighting: str = DEFAULT_WEIGHTING,
             turn_scores: dict | None = None):
    """Validates then scores one raw document: ("dead_letter", reason) or ("ok", tenant_id, conversation_id, record)."""
    tenant_id, conversation_id = extract_identity(raw_doc)
    if not tenant_id or not conversation_id:
        return ("dead_letter", "missing_tenant_id_or_conversation_id")

    try:
        turn_results = conv.load_conversation(raw_doc)
    except conv.ConversationRejected as exc:
        return ("dead_letter", str(exc))

    record = score_conversation(raw_doc, tenant_id, conversation_id, turn_results, scorer, source_ref,
                                 scored_at=scored_at, weighting=weighting, turn_scores=turn_scores)
    return ("ok", tenant_id, conversation_id, record)
