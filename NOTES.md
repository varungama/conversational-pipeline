# Design notes

---

## Evolution — quick diff

| Area | Initial version | Now |
| --- | --- | --- |
| Backend | Local directory queue | Local or `--backend aws` (real SQS+S3+DynamoDB) |
| Model | distilbert, English, 66M | `roberta` multilingual 278M (default); `distilbert` via `--model` |
| Workers | Single process | `--workers N`, real multiprocess |
| Weighting | Equal mean only | `--weighting {equal,confidence}` |
| Docker | Written, never run | Built and run |
| AWS backend | Design only | Built, verified on moto + LocalStack |
| Idempotency | Single-process safe | Cross-process race fixed (`try_claim`) |
| Reproducibility | `scored_at` = real time | Fixed `--scored-at` + `input_sha256` |
| Long turns | Truncated at 512 tokens | Chunked + aggregated (§4.1), schema 1.1 |
| Uneven work | One task per conversation | Sharded, batched, interleaved dispatch (§6.1) |
| Metrics | overall_sentiment, sentiment_trajectory | + `volatility` (§3), schema 1.2 |
| Shard cleanup | N/A | Plan-scoped keys + S3 lifecycle expiry (§6.1.4) |

---

## 1. How to run it

Local (no AWS/Docker):
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py --input data/conversations --output results
```
Python 3.11–3.14, ~650MB disk, ~9s model load, ~20s for 28 sample conversations. Writes only to `queue/`/`results/`.

AWS (needs Docker):
```bash
docker compose up -d
python run.py --backend aws --input data/conversations
```
Same logic against SQS+S3+DynamoDB via LocalStack.

Flags (all recorded in each output's `model` block):

| Flag | Default | Notes |
| --- | --- | --- |
| `--model {roberta,distilbert}` | roberta | |
| `--workers N` | 1 | local or aws (§6.1.5); on aws also sizes the receive-batch/interleave lanes |
| `--weighting {equal,confidence}` | confidence | |
| `--backend {local,aws}` | local | |
| `--scored-at <ISO8601>` | fixed | byte-identical reruns |

Tests:
```bash
pip install -r requirements-dev.txt   # moto, AWS tests only
python -m unittest discover -s tests
```

Docker: build ~3 min, 3.5GB image; cache-mount verified (40s → 7s on second run).

---

## 2. The pipeline

One scoring path (`src/scoring.py`) shared by both backends.

- **Local:** `queue/{pending,processing,done,dead_letter}`; claim = atomic `os.rename`. Crash mid-claim leaves a file stuck in `processing/` — no retry, by design.
- **AWS:** SQS queue, S3 for transcripts/results, DynamoDB ledger. Claim = `ReceiveMessage` (visibility timeout hides it from others); un-acked messages redeliver, then DLQ via SQS redrive.
- **Idempotency:** keyed by `tenant_id::conversation_id`, not filename. `ResultIndex.try_claim` re-reads under a file lock, fixing a real race where `--workers N>1` could double-score and lose a write.

---

## 3. Metric definitions

Computed over customer turns with real content only (empty/whitespace excluded).

| Metric | Formula | Null when |
| --- | --- | --- |
| `overall_sentiment` | equal- or confidence-weighted mean of signed scores (`--weighting`) | 0 scoreable turns |
| `sentiment_trajectory` | `(last - first) / 2`, bounded [-1,1] | < 2 turns |
| `volatility` (new) | mean `\|score[i]-score[i-1]\|`, labeled stability × level | < 2 turns |

- Confidence-weighting pulls toward what the model was sure about: `c-000022` -0.14→-0.08, `c-000007` -0.74→-0.84.
- Label bands: `overall_sentiment`/`volatility` level `>0.2` positive, `<-0.2` negative, else neutral/flat.
- `volatility` crosses stability (`stable`/`chaotic`, threshold 0.3 — wider than 0.2 since a *diff* of two scores carries more noise) with level (from the *mean* score) → 6 labels. Tells a whiplash conversation (`+0.9,-0.9,+0.9,-0.9,+0.9`, trajectory="flat") apart from one that's actually flat throughout.
- Ordering: sorted by parsed `ts`, not `seq` (`seq` can be wrong).
- Reproducibility: `--scored-at` fixed (byte-identical reruns); `input_sha256` flags content drift.

---

## 4. Edge cases (`src/conversation.py`)

| Case | Behavior |
| --- | --- |
| Missing/wrong-type `turns` | Whole conversation dead-lettered |
| One bad turn | Skipped with a reason; rest still scores |
| Empty/whitespace text | Excluded from metrics, `included_in_metrics: false` |
| Zero customer turns | All metrics `null` |
| Empty `turns: []` | Same null result, no crash |
| Turn over 512 tokens | Chunked and aggregated, not truncated (§4.1) |
| Scoring exception | Caught per-conversation, dead-lettered |

Confirmed against shipped sample fixtures.

### 4.1 Long turns: chunk and aggregate

Previously truncated at 512 tokens, dropping late content. Now: split at sentence boundaries into ≤510-token chunks, score each, take the token-weighted mean of per-class probabilities. Deterministic; adds `n_chunks` (schema 1.1).

- Sentence-aware split; falls back to character-halving for a single over-limit sentence.
- Verified on the real model: 25 processed, 1 duplicate, 2 dead-lettered, byte-identical reruns; 319 short turns unchanged vs pre-change; c-000027 turn 0 (854 tokens) → 2 chunks.

| Text | Result (chunked) | Chunks |
| --- | --- | --- |
| Angry, then happy | negative −0.67 | 2 |
| Happy, then angry | negative −0.82 | 2 |
| Japanese angry, then happy | **positive +0.40** (truncation gave negative) | 2 |

Limits: chunk order affects the result slightly (each chunk sees different context); cost impact unmeasured (sample turns are short).

---

## 5. Trade-offs — status now

| Initial limitation | Status |
| --- | --- |
| No real distributed queue | Closed — `--backend aws` |
| No retry with backoff | Free from SQS |
| Single worker process | Closed — `--workers N` |
| Unweighted mean only | Half — `overall_sentiment` weighted, `sentiment_trajectory` unchanged by design |
| Long turns truncated | Closed — chunked (§4.1) |
| (new) Roberta cost | 4.7–6.4x slower than distilbert; taken on for non-English scoring |

---

## 6. Scale & cost

Measured on one CPU core.

| | distilbert (66M, English) | roberta (278M, multilingual, default) |
| --- | --- | --- |
| Load time | ~10s | ~9s |
| Throughput | 77 turns/s | 16.4 turns/s (4.7x slower) |
| Backfill (500M, 7d window) | ~129 cores | ~605 cores |
| Steady state (2M/day) | ~3.6 cores | ~16.9 cores |
| Cost/million, on-demand | ~$1.73 | ~$8.13 |
| Cost/million, spot | ~$0.52–0.69 | ~$2.45–3.25 |

- Bottleneck: model inference (~13ms/~61ms per forward pass), not I/O.
- Length-sorted, token-capped batching gives 1.1-1.2x on CPU (§6.1).
- A same-day comparison put the gap at 6.4x, not 4.7x (the two headline numbers were measured on different days).

**Breaks first at 10x (2M→20M/day):** the idempotency ledger's O(n) full-rewrite, before fleet size.
**Scaling levers:** more workers (done) → batching+sharding (done, §6.1) → ONNX+INT8 (est. 2-4x, unmeasured).

**Cost** (Fargate $0.04048/vCPU-hr, SQS $0.40/M requests, DynamoDB $1.25/M WRU):

Steady state, 2M/day = 60M conversations/month:

| | distilbert | roberta (default) |
| --- | --- | --- |
| Compute | ~$105/mo | ~$493/mo |
| SQS | ~$48/mo | ~$48/mo |
| DynamoDB | ~$75/mo | ~$75/mo |
| **Total** | **~$228/mo** | **~$616/mo** |

S3 not included (~$20-30/mo, small next to compute).

Backfill, 500M conversations, one-time:

| | distilbert | roberta (default) |
| --- | --- | --- |
| On-demand | ~$865 | ~$4,065 |
| Spot | ~$260-345 | ~$1,225-1,625 |

Roberta's on-demand backfill (~$4,065) ≈ 6.6 months of its own steady-state bill — a one-time bump, not a separate order of magnitude.

---

### 6.1 Uneven work: 1 turn vs 200 turns (AWS backend)

Cost per conversation varies ~200x (c-000028: 200 turns; median 5). CPU only, no GPU.

| Step | What it does | Where |
| --- | --- | --- |
| Cost estimate | ~8ms/turn + 4ms/token (fit R² 0.988); fallback 12ms/turn + 1ms/char | `sharding.py` |
| Shard plan | Fewest shards under `SHARD_MAX_COST=2500` hard cap, rebalanced near-equal; oversized turn → chunk-level shards (§6.1.1) | `plan_shards` |
| Slices | Each shard's turns/chunks written once to S3 `slices/`; a worker reads only its slice | `S3ShardStore` |
| Dispatch | Longest-first, interleaved into `receive_batch_size` lanes (§6.1.3) | `lpt_interleaved` |
| Batching | ≤32 items, ≤512 padded tokens per `score_batch` call | `SentimentScorer` |
| Merge | `mark_shard_landed` (atomic, §6.1.2) picks the one merging shard | `aws/worker.py` |
| Receive batch | `RECEIVE_BATCH_SIZE=1`; `--workers N` auto-matches it (§6.1.5) | `aws/config.py` |
| Heartbeat | `keep_alive` re-extends visibility every 1/3 timeout | `SqsQueue` |

Env overrides: `SENTIMENT_VISIBILITY_TIMEOUT`, `SENTIMENT_RECEIVE_BATCH`, `SENTIMENT_WORST_TASK_SECONDS`.

**Measured** (real roberta, 12-core CPU, `scripts/benchmark.py`):

| Item | Result |
| --- | --- |
| Batching | 27.0 turns/s on c-000028 (unbatched baseline noisy: 6.9-21.8 t/s) |
| `MAX_BATCH_TOKENS` | 128-1024 within ~4%; 512 chosen |
| Cost model | 1.11ms/cost unit (~11% underestimate) |
| Worst task | 7.37s unsharded → 2.21s at cap 2500 |

| `SHARD_MAX_COST` | Tasks | Largest | Extra work | 4 workers | 8 workers |
| --- | --- | --- | --- | --- | --- |
| 1200 | 33 | 1.87s | +10% | 5.7s | 3.0s |
| **2500** | 30 | 2.21s | +6% | 5.4s | 2.8s |
| 5000 | 26 | 3.95s | +3% | 5.3s | 4.0s |

Simulated makespan (shipped 28-conversation set):

| Workers | Unsharded | Longest-first | +shards≤2500 | Lower bound |
| --- | --- | --- | --- | --- |
| 1 | 20.2s | 20.2s | 21.5s | 20.2s |
| 2 | 13.7s | 10.1s | 10.8s | 10.1s |
| 4 | 10.0s | 7.4s | 5.4s | 5.1s |
| 8 | 8.4s | 7.4s | 2.8s | 2.5s |

#### 6.1.1 Chunk-level sharding

An oversized turn used to ship as one unsplittable shard. Now reuses the scorer's own chunker (§4.1) to shard by chunk, capping the worst case at one chunk (≤2,048) instead of one whole turn.

| Change | Detail |
| --- | --- |
| Shard descriptor | Optional `chunk_start`/`chunk_end` (`is_chunk_shard`) |
| S3 slice | Chunk texts, not the raw turn |
| Worker partial | `{turn: {"_chunk_probs": [...]}}`, pooled + finalized once all land |
| Backend | Transformer only; VADER has no window limit |

Verified: chunked vs whole scoring gives an identical dict; matches `evaluate()` byte-for-byte on the real model and via a real LocalStack run.

#### 6.1.2 Shard-merge race, fixed

**Finding:** two shards landing together could both pass a list-then-compare completion check and both attempt `try_claim` — correct but wasteful (a full S3 GET + record build for nothing).

**Fix:** `DynamoResultIndex.mark_shard_landed` — one atomic DynamoDB `ADD` to a set; exactly one caller's update ever completes it. Not a lock: a crashed winner's message redelivers, the no-op `ADD` still returns True, so SQS redelivery (not a hand-rolled lease) retries the merge.

| Verified | Result |
| --- | --- |
| Moto | 8 threads racing → exactly 1 winner; 3 concurrent workers → 1 `processed` |
| Real LocalStack | Same properties against real DynamoDB |
| Real 4-process | `bench_workers.py --workers 4`: `results: 25` both repeats |
| Bug found along the way | `ReturnValues="ALL_NEW"` needed — `UPDATED_NEW` omits a no-op `ADD`'s attribute |

#### 6.1.3 Receive-batch dispatch: interleaving

**Why batch 5 regressed:** one `ReceiveMessage` call doesn't round-robin — sorted biggest-first, the first poller could grab all 5 biggest tasks.

**Fix:** `lpt_interleaved` deals the sorted list round-robin into `receive_batch_size` lanes so no batch-sized window is all-big.

| Receive batch | Unsharded | Longest-first | Sharded+longest-first |
| --- | --- | --- | --- |
| 1 | 22.1s | 17.4s | **16.4s** |
| 5, before fix | 22.0s | 28.2s | 26.0s |
| 5, after fix | 20.7s | 19.9s | 17.7s |

Scores aren't bit-identical to unsharded (padding shifts with batch composition; max diff 0.004, no label changes). `RECEIVE_BATCH_SIZE` stays 1 by default.

**Limits:** one machine measured; a single chunk (≤2,048) is still the smallest unsplittable unit; a dead-lettered shard leaves its conversation unscored; local backend has no sharding/cost ordering.

#### 6.1.4 Plan-scoped shard keys + S3 lifecycle expiry

**Finding (review):** shard state was keyed only by `tenant::conversation`, never by which plan produced it, and `slices/`/`shards/` were never deleted — a rerun under a changed `--model`/`SHARD_MAX_COST` could in principle merge stale leftover partials. Not hit in practice.

**Fix:** `sharding.plan_hash` hashes content + the concrete shard plan into a short id, stamped on every shard; S3/DynamoDB keys nest under it, so two plans never share a merge. `bootstrap.ensure_shard_lifecycle` expires `slices/`/`shards/` after `SENTIMENT_SHARD_EXPIRY_DAYS` (default 7).

Verified: `test_mark_shard_landed_is_scoped_to_its_plan_hash`, `test_replanned_shards_never_merge_with_a_stale_plans_leftovers` (both moto); full suite (94 tests) green.

#### 6.1.5 Multi-process AWS workers

`--workers N` now spans the aws backend too: `run.py` spawns N processes against the same SQS queue, each loading its own model (mirrors `bench_workers.py`'s design; nothing survives a process spawn).

**Finding:** two bugs surfaced running the full 28-conversation set with real `--workers 3`. `approximate_depth()` counted only visible messages, so a message mid-redelivery-wait looked like an empty queue — the idle-exit could shut every worker down before SQS redrove it, silently dropping it from both `results.json` and `errors.json`. Separately, no process capped its torch thread pool, so 3 processes fought over every core, slowing inference enough to trigger spurious "duplicate delivery" redeliveries too.

**Fix:** `approximate_depth` now sums `ApproximateNumberOfMessages` + `...NotVisible`; each worker process calls `torch.set_num_threads(cores // workers)` before loading its model.

**Verified:** `--backend aws --model roberta --workers 3 --aws-drain-timeout 90` on the full set, fresh LocalStack: both bugs reproduced pre-fix (a message vanished; spurious duplicates), then 25 processed + 1 duplicate + 2 dead-lettered = 28 and an empty queue/DLQ post-fix.

---

## 7. Non-English text — measured, before and after

| Fixture | Before (distilbert) | After (roberta) |
| --- | --- | --- |
| Japanese (`c-000025`) | Negative 91–96% — wrong | Negative → positive, correct |
| French (`c-000024`) | Mixed, partly lucky | All 3 turns correct |
| Emoji "👍" (`c-000022`) | Negative 70% | Positive 71% |
| "..." (`c-000022`) | Positive 96% | Neutral 58% |
| Sarcasm (`c-000023`) | Wrong | Still wrong — pragmatics gap, not language |

Multilingual model fixes language coverage, not sarcasm/double negation.

---

## 8. Deploying on AWS

| Local | AWS |
| --- | --- |
| `queue/pending` | SQS queue (pointer only) |
| `queue/processing` | SQS visibility timeout |
| `queue/dead_letter` | SQS redrive → DLQ |
| `results/*.json` | S3 |
| `results/index.json` | DynamoDB conditional write (+ shard completion, §6.1.2) |
| — | S3 `slices/`/`shards/` for large conversations (§6.1, §6.1.1) |

```text
                    ┌───────────────────────┐
  conversation ───▶ │  S3: conversations    │  key = "{tenant_id}/{conversation_id}.json"
    completed       │       bucket           │
                    └───────────┬────────────┘
                                │ pointer only, never the transcript
                                ▼
                    ┌─────────────────┐  after maxReceiveCount   ┌────────────┐
                    │  SQS standard   │─────────────────────────▶│  SQS DLQ   │
                    │     queue       │      (redrive policy)    │            │
                    └────────┬────────┘                          └─────┬──────┘
                             │                                         │
        ECS autoscaling on queue depth                                 ▼
                             │                              CloudWatch alarm → SNS
                             ▼                              → on-call (design only, §9)
              ┌───────────────────────────┐
              │  ECS on Fargate: worker    │  model loaded once per task,
              │  task (min 1-2 warm)       │  loops receive → fetch → score → put
              └──────────────┬─────────────┘
                              │
                 ┌────────────┴─────────────┐
                 ▼                          ▼
      ┌────────────────────────┐   ┌───────────────────────┐
      │       DynamoDB         │   │  S3: results bucket   │
      │  idempotency ledger    │   │  scored record lives  │
      │  (conditional put,     │   │  here, not DynamoDB   │
      │  O(1)/write — see §6)  │   └───────────────────────┘
      └────────────────────────┘
```

Simplified: shows the unsharded path. A large conversation (§6.1) adds one hop — a worker fetches its shard's slice, parks a partial, and `mark_shard_landed` (§6.1.2) picks the merging shard.

- **Worker:** ECS Fargate, not Lambda — model stays loaded, no per-invocation reload.
- **SQS over Kafka:** one consumer group, no fan-out/replay need; zero-ops.
- **Scaling:** ECS autoscaling on queue depth, hard max task count caps a bad backfill's bill.
- **Cost:** ~$616/mo steady state, ~$4,065 one-time backfill (roberta, on-demand) — §6.
- **Verified:** moto → real LocalStack → real 4-process concurrency, re-run after each fix (sharding, chunking, dispatch, race). Real bug found: `--aws-drain-timeout` 30s too short, bumped to 60s.
- Not verified: LocalStack fidelity to real AWS on rarer corners (IAM, multi-partition SQS ordering).
- **AWS the right call?** Mostly yes — the 500M backfill justifies it; steady state alone would fit a scheduled container.

---

## 9. Still open — next steps

1. Continuous poll loop instead of the current bounded drain-and-exit.
2. CloudWatch alarm on DLQ depth, wired to page someone.
3. Re-run cost/benchmark scripts on the production CPU type.
4. ONNX Runtime + INT8 quantisation — est. 2-4x, unmeasured.
5. A human-labelled, real support-transcript set, for an accuracy number.
6. Sarcasm / double-negation handling — still broken on both models tried.
7. Retries with backoff on the local backend (AWS already gets this from SQS).
8. Measure chunking's throughput cost on real, genuinely long transcripts.
