"""The two conversation-level metrics, computed over customer turns with
scoreable text only (empty/whitespace text is excluded, not treated as
neutral).

overall_sentiment: plain mean of signed per-turn scores - equal weighting,
no recency bias.

sentiment_trajectory: (last - first) / 2 in chronological order. Scores
are bounded to [-1, 1], so dividing by 2 keeps the delta bounded too,
regardless of conversation length - only the endpoints matter.

Both return None (not 0.0) when undefined, since "no data" and "measured
zero" are different facts.
"""

NEUTRAL_THRESHOLD = 0.2  # a signed score inside [-threshold, threshold] reads as neutral / flat


def sentiment_label(score, neutral_threshold=NEUTRAL_THRESHOLD):
    if score > neutral_threshold:
        return "positive"
    if score < -neutral_threshold:
        return "negative"
    return "neutral"


def trend_label(delta, neutral_threshold=NEUTRAL_THRESHOLD):
    if delta > neutral_threshold:
        return "improving"
    if delta < -neutral_threshold:
        return "declining"
    return "flat"


def compute_overall_sentiment(customer_scores, neutral_threshold=NEUTRAL_THRESHOLD):
    """customer_scores: list of (signed_score, confidence) tuples for
    scoreable customer turns, in any order. Returns None if the list is empty."""
    if not customer_scores:
        return None

    scores = [score for score, _ in customer_scores]
    confidences = [confidence for _, confidence in customer_scores]
    mean_score = sum(scores) / len(scores)
    mean_confidence = sum(confidences) / len(confidences)

    return {
        "label": sentiment_label(mean_score, neutral_threshold),
        "score": round(mean_score, 4),
        "confidence": round(mean_confidence, 4),
        "n_turns": len(scores),
    }


def compute_sentiment_trajectory(ordered_customer_scores, neutral_threshold=NEUTRAL_THRESHOLD):
    """ordered_customer_scores: signed scores for scoreable customer turns,
    already sorted chronologically. Returns None with fewer than 2 turns -
    a single point has no direction to report."""
    if len(ordered_customer_scores) < 2:
        return None

    first_score = ordered_customer_scores[0]
    last_score = ordered_customer_scores[-1]
    delta = (last_score - first_score) / 2

    return {
        "label": trend_label(delta, neutral_threshold),
        "score": round(delta, 4),
        "n_turns": len(ordered_customer_scores),
    }
