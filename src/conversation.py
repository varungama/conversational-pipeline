"""Parsing and validation of one conversation: a bad `turns` rejects the record, a bad turn is skipped with a reason."""

from datetime import datetime

VALID_ROLES = {"customer", "agent", "system"}


class ConversationRejected(Exception):
    """Raised when the whole conversation record cannot be processed at all."""


def parse_timestamp(value):
    """Returns a datetime for a well-formed 'YYYY-MM-DDTHH:MM:SSZ' string, else None."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None


def validate_turn(raw_turn, index):
    """Checks one turn; returns {"status": "ok", index, seq, role, ts, parsed_ts, text, has_content}
    or {"status": "skipped", index, reason}."""
    if not isinstance(raw_turn, dict):
        return {"status": "skipped", "index": index, "reason": "turn_not_an_object"}

    seq = raw_turn.get("seq")
    role = raw_turn.get("role")
    ts_raw = raw_turn.get("ts")
    text = raw_turn.get("text")

    # bool is a subclass of int, so excluded explicitly
    if not isinstance(seq, int) or isinstance(seq, bool):
        return {"status": "skipped", "index": index, "reason": "seq_missing_or_wrong_type"}

    if role not in VALID_ROLES:
        return {"status": "skipped", "index": index, "reason": "role_missing_or_unrecognised"}

    parsed_ts = parse_timestamp(ts_raw)
    if parsed_ts is None:
        return {"status": "skipped", "index": index, "reason": "ts_missing_or_unparseable"}

    if not isinstance(text, str):
        return {"status": "skipped", "index": index, "reason": "text_missing_or_wrong_type"}

    return {
        "status": "ok",
        "index": index,
        "seq": seq,
        "role": role,
        "ts": ts_raw,
        "parsed_ts": parsed_ts,
        "text": text,
        "has_content": bool(text.strip()),
    }


def load_conversation(raw_doc):
    """Returns one validated result per turn; raises ConversationRejected if `turns` is missing or not a list."""
    if not isinstance(raw_doc, dict):
        raise ConversationRejected("record_not_an_object")

    turns = raw_doc.get("turns")
    if turns is None:
        raise ConversationRejected("turns_key_missing")
    if not isinstance(turns, list):
        raise ConversationRejected("turns_not_a_list")

    return [validate_turn(raw_turn, index) for index, raw_turn in enumerate(turns)]
