# Conversational Sentiment Pipeline

A worker pipeline that scores the sentiment of support conversations (chat, email, ticketing) and derives conversation-level quality metrics from the sequence of per-turn scores.

Work is queued, picked up by one or more workers, scored by a transformer sentiment model, and written out as durable JSON. It runs fully locally with a file-based queue, or against AWS (SQS + S3 + DynamoDB) via LocalStack.

## Features

- **Per-turn scoring** — sentiment label, signed score in [-1, 1] and model confidence for every turn.
- **Conversation metrics** — `overall_sentiment`, `sentiment_trajectory` and `volatility`, computed over customer turns only. Degenerate inputs (one turn, no customer turns, empty transcripts) yield `null` rather than a misleading `0.0`.
- **Two backends** — a local directory queue with atomic claim-by-rename, or SQS/S3/DynamoDB (`--backend aws`).
- **Multi-process workers** — `--workers N` on either backend.
- **Idempotent** — results are keyed by `tenant_id::conversation_id`, so a conversation delivered twice is scored once.
- **Uneven-work handling** — long transcripts are sharded, batched and dispatched longest-first so a 200-turn conversation doesn't stall a lane.
- **Long turns** — split at sentence boundaries, scored per chunk and aggregated instead of truncated.
- **Multilingual** — defaults to a multilingual XLM-RoBERTa model; `distilbert` (English, faster) is available via `--model`.
- **Reproducible** — fixed `--scored-at` gives byte-identical reruns; model name, pinned revision, schema version, thresholds and weights are recorded in each output.

## Quick start

Requires Python 3.11+ and about 650 MB of disk (model weights download on first run).

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` elsewhere
pip install -r requirements.txt
python run.py --input data/conversations --output results
```

Scored conversations are written to `results/results.json`, and any that fail validation or scoring to `results/errors.json`.

### AWS backend (LocalStack)

Needs Docker.

```bash
docker compose up -d
python run.py --backend aws --input data/conversations
```

### Docker

```bash
docker build -t sentiment-pipeline .
docker run --rm sentiment-pipeline
```

### Common flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--backend {local,aws}` | `local` | |
| `--model {roberta,distilbert}` | `roberta` | |
| `--workers N` | `1` | Worker processes, on either backend |
| `--weighting {equal,confidence}` | `confidence` | How `overall_sentiment` weights turns |
| `--scored-at <ISO8601>` | fixed | Pin the timestamp for byte-identical reruns |

Run `python run.py --help` for the full list.

## Input format

One conversation per JSON file:

```json
{
  "conversation_id": "c-000123",
  "tenant_id": "acme",
  "channel": "chat",
  "started_at": "2026-03-02T09:14:00Z",
  "turns": [
    { "seq": 0, "role": "customer", "ts": "2026-03-02T09:14:00Z",
      "text": "My order still hasn't arrived and it's been two weeks." },
    { "seq": 1, "role": "agent", "ts": "2026-03-02T09:14:40Z",
      "text": "I'm sorry about that — let me check the tracking for you." }
  ]
}
```

`role` is `customer`, `agent` or `system`. Only customer turns feed the metrics.

## Project layout

```text
run.py              CLI entry point
src/                scoring, metrics, sharding, queue/index, AWS backend
tests/              unit and end-to-end tests (AWS tests run on moto)
scripts/            benchmarking and cost-model tools
data/               synthetic sample conversations and their generator
docker-compose.yml  LocalStack for the AWS backend
```

## Tests

```bash
pip install -r requirements-dev.txt   # moto, needed for the AWS tests only
python -m unittest discover -s tests
```

## Documentation

Design notes — metric definitions, edge-case behaviour, trade-offs, scale and cost measurements, and the AWS deployment design — are in [NOTES.md](NOTES.md). The sample dataset is described in [data/README.md](data/README.md).
