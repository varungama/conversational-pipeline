# Design notes

---

## Evolution — quick diff

| Area | Initial version | Now |
| --- | --- | --- |
| Backend | Local directory queue only | Local, or `--backend aws` (real SQS+S3+DynamoDB) |
| Model | distilbert, English only, 66M params | `roberta` multilingual, 278M params (default). `distilbert` still selectable via `--model` |
| Workers | Single process | `--workers N`, real multiprocess |
| Weighting | Equal mean only | `--weighting {equal,confidence}` |
| Docker | Written, never run  | Built and run for real |
| AWS backend | Design only, not built | Built. Verified with moto, then real LocalStack |
| Idempotency ledger | Single-process safe only | Fixed race across multiple processes (`try_claim`) |
| Reproducibility | `scored_at` was real time | Fixed `--scored-at`, plus `input_sha256` |

Everything below is the full picture, current state.

---

## 1. How to run it

Local backend (no AWS, no Docker needed):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py --input data/conversations --output results
```

- Python 3.11–3.14. ~650MB disk (model + deps).
- First run downloads the model (~550MB). Later runs are offline.
- ~9s model load. ~20s to score 27 sample conversations.
- Writes only to `queue/` and `results/`. No account, API key, or GPU needed.

AWS backend (needs Docker):

```bash
docker compose up -d
python run.py --backend aws --input data/conversations
```

- Runs the same logic against SQS + S3 + DynamoDB, via LocalStack.

Flags, all recorded in every output's `model` block:

- `--model {roberta, distilbert}` — default `roberta`.
- `--workers N` — local backend only. Default 1.
- `--weighting {equal, confidence}` — default `equal`.
- `--backend {local, aws}` — default `local`.
- `--scored-at <ISO8601>` — fixed default, for byte-identical output.

Tests:

```bash
pip install -r requirements-dev.txt   # adds moto, for AWS tests only
python -m unittest discover -s tests
```

Docker, run for real:
- `docker build` ~3 min. Image size 3.5GB.
- Cache-mount verified: second run dropped from ~40s to ~7s, 25+ HTTP calls to 6.

---

## 2. The pipeline

Two backends. One scoring path (`src/scoring.py`), shared by both. They can't silently disagree on what "correct" means.

**Local (default):**
- Directory tree: `queue/pending`, `processing`, `done`, `dead_letter`.
- Claim = `os.rename`. Atomic. That's the whole locking mechanism.
- Cut on purpose: a crash mid-claim leaves a file stuck in `processing/` forever. No retry.

**AWS (`--backend aws`):**
- SQS for the queue. S3 for transcripts and results. DynamoDB for the ledger.
- Claim = `ReceiveMessage`. SQS's visibility timeout hides the message from other workers.
- A crashed worker never deletes the message. It reappears automatically.
- Repeated failures go to a DLQ via SQS's own redrive policy. Platform handles it, not app code.

**Idempotency:** `tenant_id::conversation_id`, not filename.
- Confirmed on the shipped duplicate fixture. Only one result written.
- `ResultIndex.try_claim` (new): re-reads the ledger under a file lock before inserting.
- Fixes a real race: with `--workers N` > 1, two processes could both think a conversation was unseen, both score it, and one write could get lost. Now fixed.

---

## 3. Metric definitions

Both computed over customer turns with real content only. Empty/whitespace text excluded.

**`overall_sentiment`** — mean of signed per-turn scores.
- `--weighting equal` (default): plain mean.
- `--weighting confidence` (new): `sum(score × confidence) / sum(confidence)`.
- Measured effect: `c-000022` (noisy, low-confidence) moves from -0.14 to -0.08. `c-000007` moves from -0.74 to -0.84. Confidence-weighting pulls toward what the model was sure about, not toward zero.
- 0 scoreable turns → `null`. 1+ turns → `{label, score, confidence, n_turns, weighting}`.
- Label band: `score > 0.2` positive, `< -0.2` negative, else neutral.

**`sentiment_trajectory`** — `(last_score - first_score) / 2`. Unchanged.
- Bounded `[-1, 1]`. Endpoint-only, so comparable across short and long conversations.
- Confidence-weighting doesn't map cleanly onto a 2-point formula. Not added.
- A separate "volatility" metric was considered. Rejected again — one defendable formula beats four shaky ones.
- 0 or 1 turns → `null`. 2+ turns → `{label, score, n_turns}`.

**Ordering:** sorted by parsed `ts`, not `seq`. `seq` can be wrong; `ts` can't.

**Reproducibility, new:**
- `--scored-at` defaults to a fixed timestamp, not real time. Two runs now produce byte-identical `results.json` (checked via SHA256).
- `input_sha256`: content hash of the raw conversation. Flags if a transcript changed since it was last scored — `scored_at` alone can't tell you that.

---

## 4. Edge cases (`src/conversation.py`)

- **Missing/wrong-type `turns`** — whole conversation dead-lettered. Never scored.
- **One bad turn** — that turn is skipped with a reason. Rest of the conversation still scores.
- **Empty/whitespace text** — excluded from metrics, kept in output as `included_in_metrics: false`.
- **Zero customer turns** — both metrics `null`.
- **Empty `turns: []`** — same null result, no crash.
- **4,000-char turn** — truncated at 512 tokens. Not chunked. Stated behavior, not silent.
- **Unexpected scoring exception** — caught per-conversation. Dead-lettered like any other failure. One bad conversation can't take the batch down.

All confirmed against the shipped sample fixtures.

---

## 5. Trade-offs — status now

| Initial limitation | Status now |
| --- | --- |
| No real distributed queue | **Closed.** `--backend aws` is real. |
| No retry with backoff  | AWS backend gets it free from SQS. |
| Single worker process | **Closed.** `--workers N`, real processes. |
| Unweighted mean only | **Half-closed.** `overall_sentiment` has `--weighting`. `sentiment_trajectory` unchanged, by design. |
| — (new cost, from the model swap) | Roberta is 4.7x–6.4x slower than distilbert. Taken on deliberately, to fix non-English scoring. |

---

## 6. Scale & cost

Measured on one CPU core, no GPU, single-conversation-set benchmark.

| | distilbert (66M, English only) | roberta (278M, multilingual, default) |
| --- | --- | --- |
| Load time | ~10s | ~9s |
| Throughput | 77 turns/sec | 16.4 turns/sec (4.7x slower) |
| Backfill (500M convos, 7-day window) | ~129 cores | ~605 cores |
| Steady state (2M/day) | ~3.6 cores | ~16.9 cores |
| Cost / million conversations, on-demand | ~$1.73 | ~$8.13 |
| Cost / million conversations, spot | ~$0.52–0.69 | ~$2.45–3.25 |

- Bottleneck is model inference, not I/O. A forward pass is ~13ms (distilbert) / ~61ms (roberta).
- Batching (`batch_size=32`) measured *slower* than unbatched, on distilbert. Padding waste from mixed turn lengths. Not re-tested on roberta.
- A same-day, back-to-back comparison (separate ad-hoc script, not shipped in this repo) put the gap at 6.4x, not 4.7x — the two headline numbers above were measured on different days. Same-day table: distilbert 96.3 t/s, roberta 15.1 t/s.

**What breaks first at 10x** (2M/day → 20M/day):
1. Worker fleet size — solved by autoscaling.
2. The idempotency ledger. It rewrites the whole seen-set on every write. `O(n)`. This is the real bottleneck, before fleet size is.

**Scaling levers, by effort:** more workers (done) → length-bucketed batching (unverified) → ONNX + INT8 quantization (estimated 2–4x, not measured) → temporary GPU for the backfill only.

---

## 7. Non-English text — measured, before and after

Original model (distilbert) confidently mis-scored non-English and edge-case text:

| Fixture | Before (distilbert) | After (roberta) |
| --- | --- | --- |
| Japanese (`c-000025`), ends with a thank-you | Negative, 91–96% — wrong | Negative → positive close, correct |
| French (`c-000024`) | Mixed, partly lucky | All 3 turns correct, higher confidence |
| Emoji "👍" (`c-000022`) | Negative 70% | Positive 71% |
| "..." (`c-000022`) | Positive 96% | Neutral 58% — more honest |
| Sarcasm / double negation (`c-000023`) | Wrong | **Still wrong.** Pragmatics gap, not a language gap. |

The multilingual model fixes language coverage. It does not fix sarcasm or double negation.

---

## 8. Deploying on AWS

Local → AWS mapping (now actually implemented in `src/aws/`, not just designed):

| Local | AWS |
| --- | --- |
| `queue/pending` | SQS queue (pointer, not transcript, in the message) |
| `queue/processing` (rename) | SQS visibility timeout |
| `queue/dead_letter` + reason | SQS redrive policy → DLQ |
| `results/*.json` | S3 |
| `results/index.json` | DynamoDB, conditional write |

- **Worker:** ECS Fargate. Not Lambda — the model needs to stay loaded, not reload per invocation.
- **Why SQS over Kafka:** one consumer group, one job per message. No need for independent fan-out or full-stream replay. SQS is zero-ops here; Kafka would just be an always-on cost.
- **Scaling:** ECS autoscaling on SQS queue depth. Hard max task count, so a bad backfill re-queue can't run up an unbounded bill.
- **Cost:** same order of magnitude as Section 6, ~$8/million with the current model.
- **Bonus prototype — built for real:**
  - Verified with moto first (no real infra).
  - Then verified against a real LocalStack container: `25 processed, 1 skipped_duplicate, 2 dead_lettered`.
  - Re-run on the same warm container: `processed: 0, skipped_duplicate: 26` — proves the ledger survives across separate processes.
  - **Real bug found:** default `--aws-drain-timeout` (30s) was too short to let SQS redrive stuck messages. Bumped to 60s. Re-run confirmed the fix.
  - Not verified: LocalStack's fidelity to real AWS in rarer corners (IAM, edge-case API behavior).
- **Is AWS the right call?** Mostly yes. The 500M backfill is what justifies it — steady state (2M/day) alone would fit on a scheduled container.

---

## 9. Still open — next steps

1. Continuous poll loop instead of the current bounded drain-and-exit.
2. CloudWatch alarm on DLQ depth, wired to actually page someone.
3. A standalone "volatility" metric, if the product ever needs to tell "flat" apart from "chaotic but flat."
4. Length-bucketed batching, re-measured against current baselines.
5. ONNX Runtime + INT8 quantization — highest-value unmeasured lever, given roberta's cost.
6. A human-labeled, real support-transcript set, for an actual accuracy number.
7. Sarcasm / double-negation handling — still broken on both models tried so far.
8. Retries with backoff on the local backend (AWS backend already gets this from SQS).
