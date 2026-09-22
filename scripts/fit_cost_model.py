"""Fits src/sharding.py's cost constants by least squares on measured times (turns only / +chars / +tokens).
Usage: python scripts/fit_cost_model.py [--reps 2] [--out fit.json]; rerun after changing model, hardware or data."""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts.benchmark import load_valid_docs, scorable_texts, timed  # noqa: E402
from src.sentiment_model import SentimentScorer  # noqa: E402


def slices(doc):
    n = len(doc["turns"])
    yield 0, n
    for parts in (2, 4):
        if n >= parts * 2:
            step = n // parts
            for i in range(parts):
                yield i * step, n if i == parts - 1 else (i + 1) * step


def fit(features, times):
    """Least squares without intercept. Returns (coefficients, R^2)."""
    x, y = np.array(features, dtype=float), np.array(times, dtype=float)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = ((y - x @ coef) ** 2).sum()
    return coef, 1 - residual / ((y - y.mean()) ** 2).sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--out")
    args = parser.parse_args()

    scorer = SentimentScorer()
    if scorer.backend_name != "transformer":
        sys.exit("transformer model unavailable (fell back to VADER) - nothing meaningful to measure")
    scorer.score_batch(["warm up"] * 8)

    rows = []  # (turns, chars, tokens, ms)
    for doc in load_valid_docs():
        for start, end in slices(doc):
            texts = scorable_texts(doc, start, end)
            if not texts:
                continue
            raw = doc["turns"][start:end]
            rows.append((len(raw), sum(len(t) for t in texts),
                         sum(scorer._count_tokens(t) for t in texts),
                         timed(lambda t=texts: scorer.score_batch(t), args.reps) * 1000))  # ms
    print(f"{len(rows)} measurements\n")

    turns, chars, tokens, ms = (list(col) for col in zip(*rows))
    models = {
        "turns only": ([[t] for t in turns], ["per turn"]),
        "turns + chars": ([[t, c] for t, c in zip(turns, chars)], ["per turn", "per char"]),
        "turns + tokens": ([[t, k] for t, k in zip(turns, tokens)], ["per turn", "per token"]),
    }
    report = {}
    for name, (features, labels) in models.items():
        coef, r2 = fit(features, ms)
        report[name] = {"r2": round(r2, 4), **{l: round(float(c), 3) for l, c in zip(labels, coef)}}
        print(f"{name:<16} R^2 {r2:.4f}   " + "   ".join(f"{l} {c:.3f} ms" for l, c in zip(labels, coef)))
    print(f"\nchars per token here: {sum(chars) / sum(tokens):.2f}")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps({"rows": rows, "fits": report}, indent=2))


if __name__ == "__main__":
    main()
