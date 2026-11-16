"""Conversation-level metrics: overall_sentiment, sentiment_trajectory, volatility - None when too few scoreable turns."""

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
DEFAULT_WEIGHTING = "confidence"


def compute_overall_sentiment(customer_scores, neutral_threshold=NEUTRAL_THRESHOLD, weighting=DEFAULT_WEIGHTING):
    """Equal- or confidence-weighted mean of customer_scores (signed_score, confidence) tuples, or None if empty."""
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
    """Labeled sentiment shift across chronologically ordered signed scores, or None with fewer than 2 turns."""
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


VOLATILITY_THRESHOLD = 0.5  # mean abs consecutive change above this = "chaotic"; wider than NEUTRAL_THRESHOLD since a diff of two scores carries more noise


def volatility_label(level_label: str, volatility: float, volatility_threshold=VOLATILITY_THRESHOLD) -> str:
    stability = "chaotic" if volatility > volatility_threshold else "stable"
    level = "flat" if level_label == "neutral" else level_label
    return f"{stability}_{level}"


def compute_volatility(ordered_customer_scores, neutral_threshold=NEUTRAL_THRESHOLD,
                        volatility_threshold=VOLATILITY_THRESHOLD):
    """Mean absolute consecutive change, labeled stability (stable/chaotic) x level (positive/flat/negative). None with fewer than 2 turns."""
    if len(ordered_customer_scores) < 2:
        return None

    diffs = [abs(b - a) for a, b in zip(ordered_customer_scores, ordered_customer_scores[1:])]
    mean_abs_change = sum(diffs) / len(diffs)
    mean_score = sum(ordered_customer_scores) / len(ordered_customer_scores)
    level_label = sentiment_label(mean_score, neutral_threshold)

    return {
        "label": volatility_label(level_label, mean_abs_change, volatility_threshold),
        "score": round(mean_abs_change, 4),
        "n_turns": len(ordered_customer_scores),
    }
