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
| Long turns | Truncated at 512 tokens, rest dropped | Sentence-aware chunking + aggregation (§4.1). Schema `1.1` |
| Uneven work | One model call per turn; whole conversation = one task | Length-sorted batching, token-cost sharding with S3 slices, longest-first dispatch, visibility heartbeat (§6.1) |

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
- Writes only to `queue/` and `results/`. No account or API key needed.

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
- Claim = `os.rename`. Atomic on POSIX. That's the whole locking mechanism.
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
- **Turn over 512 tokens** — chunked and aggregated, not truncated. See §4.1.
- **Unexpected scoring exception** — caught per-conversation. Dead-lettered like any other failure. One bad conversation can't take the batch down.

All confirmed against the shipped sample fixtures.

### 4.1 Long turns: chunk and aggregate

Previously anything past 512 tokens was silently dropped, so a resolution at the end of a long message never counted. Now handled inside `SentimentScorer.score` (`src/sentiment_model.py`). `src/scoring.py` logic and both workers are untouched, so both backends get it.

1. Count tokens with the model's own tokenizer. Limit is 510 (512 minus 2 special tokens).
2. Fits → one model call, identical to before.
3. Doesn't fit → split at sentence boundaries (`。！？`, `. ! ?` before whitespace, newlines), then pack sentences greedily into chunks under the limit.
4. Score each chunk. Take the **token-weighted mean of per-class probabilities**, then derive label, score and confidence with the same formulas as short text.
5. A single sentence over the limit is halved by characters until it fits. Last resort; can cut mid-word.

Why it's built this way:
- **Sentence-aware:** a hard cut can split "not" from "good".
- **Deterministic:** no randomness, so `--scored-at` reruns stay byte-identical.
- **Average probabilities, not labels:** keeps the meaning of the three output fields, and N repeats of a sentence score the same as one.
- **Schema `1.1`:** each turn gains `n_chunks` (1 for short text, and always 1 on the VADER fallback).

Tests: `tests/test_chunking.py`, using a fake scorer and tokenizer.

**Verified on the real model:**
- Full local run: 25 processed, 1 duplicate, 2 dead-lettered. All records are schema `1.1`. Two runs give the same SHA256.
- 319 short turns compared with the pre-change results: 0 differences in label, score or confidence.
- Only c-000027 turn 0 is over the window (854 tokens). It splits into 2 chunks: −0.9968 vs −0.9987 truncated.
- AWS backend on real LocalStack: 25 processed, 1 duplicate, 2 dead-lettered via SQS redrive. Re-run: 0 processed, 26 duplicates. Records identical to the local run except `source_file`.

Mixed sentiment, synthetic 600–781 token turns. Truncation would see only the opening:

| Text | Result | Chunks |
| --- | --- | --- |
| Angry, then happy | negative, −0.67 (all-angry ≈ −0.99) | 2 |
| Happy, then angry | negative, −0.82 | 2 |
| Japanese angry, then happy | **positive, +0.40** (truncation: negative) | 2 |

Limits:
- The late text now counts, but the result depends slightly on order. Angry-first and happy-first differ (−0.67 vs −0.82) because each chunk sees different context. It is not an exact average.
- Repeat invariance is approximate on the real model: one sentence −0.9977, 25 repeats −0.9987.
- Cost impact is unmeasured. Long turns cost one forward pass per chunk. Sample turns are short, so §6 numbers are unchanged.

---

## 5. Trade-offs — status now

| Initial limitation | Status now |
| --- | --- |
| No real distributed queue | **Closed.** `--backend aws` is real. |
| No retry with backoff  | AWS backend gets it free from SQS. |
| Single worker process | **Closed.** `--workers N`, real processes. |
| Unweighted mean only | **Half-closed.** `overall_sentiment` has `--weighting`. `sentiment_trajectory` unchanged, by design. |
| Long turns truncated | **Closed.** Chunked and aggregated (§4.1). |
| — (new cost, from the model swap) | Roberta is 4.7x–6.4x slower than distilbert. Taken on deliberately, to fix non-English scoring. |

---

## 6. Scale & cost

Measured on one CPU core, single-conversation-set benchmark.

| | distilbert (66M, English only) | roberta (278M, multilingual, default) |
| --- | --- | --- |
| Load time | ~10s | ~9s |
| Throughput | 77 turns/sec | 16.4 turns/sec (4.7x slower) |
| Backfill (500M convos, 7-day window) | ~129 cores | ~605 cores |
| Steady state (2M/day) | ~3.6 cores | ~16.9 cores |
| Cost / million conversations, on-demand | ~\$1.73 | ~\$8.13 |
| Cost / million conversations, spot | ~\$0.52–0.69 | ~\$2.45–3.25 |

- Bottleneck is model inference, not I/O. A forward pass is ~13ms (distilbert) / ~61ms (roberta).
- Batching (`batch_size=32`) measured *slower* than unbatched, on distilbert. Padding waste from mixed turn lengths. That measurement was of unsorted batches. Length-sorted, token-capped batching is now in and gives 1.1-1.2x on CPU (§6.1); the throughput numbers in this table are still the old unbatched ones.
- A same-day, back-to-back comparison (separate ad-hoc script, not shipped in this repo) put the gap at 6.4x, not 4.7x — the two headline numbers above were measured on different days. Same-day table: distilbert 96.3 t/s, roberta 15.1 t/s.

**What breaks first at 10x** (2M/day → 20M/day):
1. Worker fleet size — solved by autoscaling.
2. The idempotency ledger. It rewrites the whole seen-set on every write. `O(n)`. This is the real bottleneck, before fleet size is.

**Scaling levers, by effort:** more workers (done) → length-sorted batching + sharding (built, §6.1; batching measured 1.1-1.2x on CPU) → ONNX + INT8 quantization (estimated 2–4x, not measured).

**Cost, as \$/month and \$/backfill (assumptions stated, not measured):**

Pricing: Fargate on-demand \$0.04048/vCPU-hr. SQS \$0.40/million requests, 1M free. DynamoDB on-demand write \$1.25/million WRU. (The Fargate rate is reverse-solved from the cost/million row above, not a new assumption.)

Steady state, 2M/day = 60M conversations/month:

| | distilbert | roberta (default) |
| --- | --- | --- |
| Compute (3.6 / 16.9 cores, 24/7) | ~\$105/mo | ~\$493/mo |
| SQS (~2.1 req/convo: send + batched receive + delete) | ~\$48/mo | ~\$48/mo |
| DynamoDB (60M conditional-put writes) | ~\$75/mo | ~\$75/mo |
| **Total** | **~\$228/mo** | **~\$616/mo** |

- roberta is the shipped default: ~\$616/mo. That range is an 8-vCPU flat assumption — closer to distilbert's 3.6 cores than roberta's measured 16.9.
- S3 (transcript + result writes, growing storage) not in the table — order of \$20-30/mo, small next to compute either way.

Backfill, 500M conversations, one-time (cost/million × 500):

| | distilbert | roberta (default) |
| --- | --- | --- |
| On-demand | ~\$865 | ~\$4,065 |
| Spot | ~\$260-345 | ~\$1,225-1,625 |

- roberta's on-demand backfill (~\$4,065) is ≈6.6 months of its own steady-state bill, not a separate order of magnitude. It's a one-time bump worth sizing, not the number to fear.

---

### 6.1 Uneven work: 1 turn vs 200 turns (AWS backend)

Cost per conversation varies ~200x (c-000028 has 200 turns, the median 5). CPU only, no GPU.

| Step | What it does | Where |
| --- | --- | --- |
| Cost estimate | ms ≈ 8 per turn + 4 per token (model tokenizer); fallback 12 per turn + 1 per char. Fit R²: turns only 0.849, +chars 0.983, **+tokens 0.988** (82 points). | `sharding.py`, `scripts/fit_cost_model.py` |
| Shard plan | Cost over `SHARD_MAX_COST=2500` (~2.5s): fewest shards under a **hard cap**, rebalanced to near-equal. A single turn over the cap gets its own shard. | `sharding.plan_shards` |
| Slices | Each shard's turns are written once to S3 `slices/`; a shard worker reads only its slice (500 shards: ~1.1 GB of reads becomes ~2.3 MB). | `S3ShardStore` |
| Longest-first | `run.py` enqueues by descending cost (SQS is best-effort ordered, so a hint). | `run.py` |
| Batching | One `score_batch` per task: chunks sorted by length, at most 32 items and 512 padded tokens per batch. | `SentimentScorer.score_batch` |
| Merge | Each shard parks `{turn: score}` in S3 `shards/`; the last to land calls `evaluate(..., turn_scores=)`. Ledger `try_claim` and `has` cover duplicates. | `aws/worker.py` |
| One message per receive | `RECEIVE_BATCH_SIZE=1`; a worker holding several hoards the biggest tasks. | `aws/config.py` |
| Visibility heartbeat | `keep_alive` re-extends the timeout every third of it while a task runs. The 6s base timeout only bounds a crashed worker. | `SqsQueue.keep_alive` |

Env overrides: `SENTIMENT_VISIBILITY_TIMEOUT`, `SENTIMENT_RECEIVE_BATCH`, `SENTIMENT_WORST_TASK_SECONDS`.

**Measured** (real roberta, 12-core CPU, `scripts/benchmark.py`):

| Item | Result |
| --- | --- |
| Batching | 27.2 vs 21.8 turns/s on c-000028 (1.25x); 17.7 vs 15.8 on all turns (1.12x) |
| `MAX_BATCH_TOKENS` | 128-1024 within ~7%; 2048 → 14.7, 4096 → 11.0, unlimited → 4.1 turns/s. 512 chosen |
| Cost model | Measured 1.11 ms per cost unit on unsharded tasks (1.00 = perfect): ~11% underestimate |
| Worst task | 7.35s unsharded (c-000028) → **3.28s** at cap 2500 (c-000027, one unsplittable 854-token turn) |

| `SHARD_MAX_COST` | Tasks | Largest | Extra work | Makespan, 4 workers | 8 workers |
| --- | --- | --- | --- | --- | --- |
| 1200 | 32 | 3.28s | +12% | 5.7s | 3.3s |
| **2500** | 29 | 3.28s | +7% | 5.4s | 3.3s |
| 5000 | 26 | 3.94s | +2% | 5.3s | 3.9s |

**Shard plan on the shipped data** (cap 2500; costs in ~ms, tokens vs the character fallback):

| Conversation | Shard | Turns | Cost (tokens) | Cost (chars) |
| --- | --- | --- | --- | --- |
| c-000027 (3 turns) | 0 | 0 | **3,424** | **3,950** |
| | 1 | 1–2 | 148 | 158 |
| c-000028 (200 turns) | 0 | 0–49 | 1,928 | 2,090 |
| | 1 | 50–99 | 1,912 | 2,098 |
| | 2 | 100–149 | 1,928 | 2,090 |
| | 3 | 150–199 | 1,912 | 2,098 |

- c-000027 turn 0 (~4,000 characters) exceeds the cap alone and cannot be split, so it gets its own shard (the 3.7s case).
- c-000028 costs 7,680 (tokens): at least 4 shards, rebalanced to within ~1% of each other, all under the cap.
- The other 23 conversations cost under 2,500 and stay whole. 28 files → 26 whole tasks + 6 shards = **32 tasks**.

Simulated makespan from those task times:

| Workers | Unsharded | Longest-first | + shards ≤2500 | Lower bound |
| --- | --- | --- | --- | --- |
| 1 | 20.2s | 20.2s | 21.5s | 20.2s |
| 2 | 13.6s | 10.2s | 10.8s | 10.1s |
| 4 | 10.1s | 7.3s | 5.4s | 5.1s |
| 8 | 8.4s | 7.3s | 3.3s | 2.5s |

**Real 4-worker run on LocalStack** (3 torch threads each; measured before the token cost and heartbeat, not re-run):

| Receive batch | Unsharded | Longest-first | Sharded + longest-first |
| --- | --- | --- | --- |
| 1 | 23.9s | 17.4s (1.37x) | **15.8s (1.51x)** |
| 5 | 22.0s | 28.2s (0.78x) | 26.0s (0.84x) |

With a batch of 5, one worker grabs the 5 biggest tasks and becomes the tail, hence the default of 1 (cost: ~1 extra SQS request per task, negligible).

**Scores are not bit-identical to per-turn or unsharded scoring.** Batch composition changes padding: 73/319 turn scores differ from the pre-batching run (max 0.004), sharded c-000028 differs from unsharded by up to ~0.002. No labels changed. Deterministic for a fixed input and shard plan.

**Verified:** 65 tests pass (stub scorers, moto). Real model: local run 25 processed, 1 duplicate, 2 dead-lettered. Real LocalStack: same counts, 2 dead-lettered after 3 retries, 4 shard partials merged.

**Limits**
- One machine, 25 conversations, one big one: re-run `scripts/fit_cost_model.py` and `scripts/benchmark.py` on the production CPU type.
- A single turn over the cap cannot be split; the fix for very large turns is chunk-level shards.
- A dead-lettered shard reaches the DLQ; that conversation gets no result.
- `slices/` and `shards/` are never deleted: add an S3 lifecycle expiry on both.
- The local backend has batched scoring only: no sharding or cost ordering.

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

```text
                    ┌───────────────────────┐
  conversation ───▶ │  S3: conversations    │  key = "{tenant_id}/{conversation_id}.json"
    completed       │       bucket           │  (mirrors data/conversations/ on local)
                    └───────────┬────────────┘
                                │ pointer only: {tenant_id, conversation_id} —
                                │ never the transcript (SQS caps messages at
                                │ 256KB; see src/aws/sqs_queue.py)
                                ▼
                    ┌─────────────────┐  after maxReceiveCount   ┌────────────┐
                    │  SQS standard   │─────────────────────────▶│  SQS DLQ   │
                    │     queue       │      (redrive policy)    │            │
                    └────────┬────────┘                          └─────┬──────┘
                             │                                         │
        ECS autoscaling on ApproximateNumberOfMessagesVisible          ▼
                             │                              CloudWatch alarm → SNS
                             ▼                              → on-call (design only,
              ┌───────────────────────────┐                  not wired yet — §9)
              │  ECS on Fargate: worker    │  model loaded ONCE per task, then
              │  task (min 1-2 warm, no    │  loops: receive → fetch from S3 →
              │  per-invocation reload     │  score → try_claim → put → delete
              │  like Lambda would need)   │
              └──────────────┬─────────────┘
                              │
                 ┌────────────┴─────────────┐
                 ▼                          ▼
      ┌────────────────────────┐   ┌───────────────────────┐
      │       DynamoDB         │   │  S3: results bucket   │
      │  idempotency ledger    │   │  key = same           │
      │  ONLY — result_key =   │   │  "{tenant}/{conv_id}" │
      │  "tenant::conv_id",    │   │  .json — the scored   │
      │  conditional put       │   │  record itself lives  │
      │  (mirrors local        │   │  here, not in DynamoDB│
      │  results/index.json,   │   └───────────────────────┘
      │  but O(1)/write, not   │
      │  O(n) — see §6)        │
      └────────────────────────┘
```

- **Worker:** ECS Fargate. Not Lambda — the model needs to stay loaded, not reload per invocation.
- **Why SQS over Kafka:** one consumer group, one job per message. No need for independent fan-out or full-stream replay. SQS is zero-ops here; Kafka would just be an always-on cost.
- **Scaling:** ECS autoscaling on SQS queue depth. Hard max task count, so a bad backfill re-queue can't run up an unbounded bill.
- **Cost:** ~\$616/mo steady state, ~\$4,065 one-time backfill, roberta on-demand — see §6's cost table.
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
4. Re-run `scripts/fit_cost_model.py` and `scripts/benchmark.py` on the production CPU type (§6.1).
5. ONNX Runtime + INT8 quantization — highest-value unmeasured lever, given roberta's cost.
6. A human-labeled, real support-transcript set, for an actual accuracy number.
7. Sarcasm / double-negation handling — still broken on both models tried so far.
8. Retries with backoff on the local backend (AWS backend already gets this from SQS).
9. Measure chunking's throughput cost on a set of genuinely long, real transcripts (behavior is verified, §4.1; cost is not).
10. Chunk-level shards for very large single turns; S3 lifecycle expiry on `slices/` and `shards/`.
