"""Conversation-level metrics: overall_sentiment (the equal- or confidence-weighted mean of signed per-turn scores) and sentiment_trajectory
((last - first customer score) / 2), each None when there aren't enough scoreable turns to define it."""

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


WEIGHTING_CHOICES = ("equal", "confidence")
DEFAULT_WEIGHTING = "equal"


def compute_overall_sentiment(customer_scores, neutral_threshold=NEUTRAL_THRESHOLD, weighting=DEFAULT_WEIGHTING):
    """Returns the equal- or confidence-weighted mean (per `weighting`) of customer_scores, a list of (signed_score, confidence) tuples, or None if empty."""
    if weighting not in WEIGHTING_CHOICES:
        raise ValueError(f"Unknown weighting {weighting!r}. Choose one of {WEIGHTING_CHOICES}.")
    if not customer_scores:
        return None

    scores = [score for score, _ in customer_scores]
    confidences = [confidence for _, confidence in customer_scores]
    total_confidence = sum(confidences)
    mean_confidence = total_confidence / len(confidences)

    if weighting == "confidence" and total_confidence > 0:
        mean_score = sum(score * confidence for score, confidence in customer_scores) / total_confidence
    else:
        mean_score = sum(scores) / len(scores)

    return {
        "label": sentiment_label(mean_score, neutral_threshold),
        "score": round(mean_score, 4),
        "confidence": round(mean_confidence, 4),
        "n_turns": len(scores),
        "weighting": weighting,
    }


def compute_sentiment_trajectory(ordered_customer_scores, neutral_threshold=NEUTRAL_THRESHOLD):
    """Returns the labeled sentiment shift across ordered_customer_scores (chronologically sorted signed scores), or None with fewer than 2 turns."""
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
