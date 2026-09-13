"""The transport-agnostic core of the pipeline: turn a raw conversation
document into a scored result record, or a dead-letter reason.

This is deliberately the *only* place that knows how to go from
`raw_doc -> record`. Both `src/worker.py` (the local directory-queue path)
and `src/aws/worker.py` (the SQS/S3/DynamoDB path) call into this module so
that scoring behaviour - metric formulas, edge-case handling, output shape -
cannot drift between the two transports. Everything transport-specific
(where the bytes came from, where the result goes, how "already done" is
tracked) stays in the two worker modules.
"""

import hashlib
import json
from datetime import datetime, timezone

from src import conversation as conv
from src.metrics import DEFAULT_WEIGHTING, compute_overall_sentiment, compute_sentiment_trajectory

SCHEMA_VERSION = "1.0"


def result_key(tenant_id: str, conversation_id: str) -> str:
    return f"{tenant_id}::{conversation_id}"


def extract_identity(raw_doc):
    """Returns (tenant_id, conversation_id) or (None, None) if either is
    missing - a record we cannot even key, let alone score."""
    if not isinstance(raw_doc, dict):
        return None, None
    return raw_doc.get("tenant_id"), raw_doc.get("conversation_id")


def input_content_hash(raw_doc) -> str:
    """SHA256 of the conversation's content, independent of source
    formatting (key order, whitespace). Answers "did the underlying
    transcript actually change" between two scored results that otherwise
    look alike - e.g. the same conversation_id re-delivered after an
    upstream edit, not just re-delivered byte-for-byte. `sort_keys=True`
    is what makes it a content hash rather than a file hash: two JSON
    files with the same data in a different key order hash the same."""
    canonical = json.dumps(raw_doc, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def score_conversation(raw_doc, tenant_id, conversation_id, turn_results, scorer, source_ref,
                        scored_at: str | None = None, weighting: str = DEFAULT_WEIGHTING) -> dict:
    """Builds the full output record for one already-validated conversation.

    `turn_results` is the list returned by `conversation.load_conversation`.
    `source_ref` is a transport-specific pointer to where this conversation
    came from - a local filename for the directory queue, an S3 key for the
    AWS path - carried through into the output for provenance.

    `scored_at` defaults to the current UTC time. Passing a fixed value
    (e.g. from `run.py --scored-at`) is what makes two runs over the same
    input produce byte-identical output - every other field here is already
    deterministic given the same input and the same pinned model revision,
    so `scored_at` is the only source of run-to-run difference otherwise.
    """
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

        scored = scorer.score(turn["text"])
        is_customer = turn["role"] == "customer"
        entry.update(scored, included_in_metrics=is_customer)
        if not is_customer:
            entry["reason"] = "not_a_customer_turn"
        scored_turns.append(entry)

        if is_customer:
            customer_scores_unordered.append((scored["score"], scored["confidence"]))
            customer_scores_by_time.append((turn["parsed_ts"], scored["score"]))

    # Chronological order comes from `ts`, not `seq`: the sample data has a
    # fixture (c-000021) with non-monotonic and duplicate `seq` values but
    # strictly increasing timestamps, so `ts` is the field we trust for "what
    # happened in what order".
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
            # Recorded for provenance, same reasoning as neutral_threshold -
            # see src/metrics.py for what "equal" vs "confidence" means.
            "weighting": weighting,
        },
        "turns": scored_turns,
        "metrics": {
            "overall_sentiment": compute_overall_sentiment(customer_scores_unordered, weighting=weighting),
            "sentiment_trajectory": compute_sentiment_trajectory(ordered_scores),
        },
    }


def evaluate(raw_doc, scorer, source_ref, scored_at: str | None = None, weighting: str = DEFAULT_WEIGHTING):
    """Runs the full validate-then-score pipeline for one raw document.

    Returns a tuple:
      ("dead_letter", reason)                         - whole record unusable
      ("ok", tenant_id, conversation_id, record)       - scored successfully

    Does not decide idempotency (duplicate detection) or where to persist
    the result - that stays with the caller, since it differs by transport
    (a JSON ledger file locally, a DynamoDB conditional write on AWS).

    `scored_at`, if given, overrides the output record's timestamp - see
    `score_conversation` for why that is the byte-identical-reruns knob.
    """
    tenant_id, conversation_id = extract_identity(raw_doc)
    if not tenant_id or not conversation_id:
        return ("dead_letter", "missing_tenant_id_or_conversation_id")

    try:
        turn_results = conv.load_conversation(raw_doc)
    except conv.ConversationRejected as exc:
        return ("dead_letter", str(exc))

    record = score_conversation(raw_doc, tenant_id, conversation_id, turn_results, scorer, source_ref,
                                 scored_at=scored_at, weighting=weighting)
    return ("ok", tenant_id, conversation_id, record)
