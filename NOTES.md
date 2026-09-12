# Design notes

## 1. How to run it

- Python 3.11-3.14, ~350MB disk (model weights + dependencies).
- First run downloads `distilbert-base-uncased-finetuned-sst-2-english` (~256MB)
  from huggingface.co; every run after is fully offline via `~/.cache/huggingface`.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows. `source .venv/bin/activate` on macOS/Linux.
pip install -r requirements.txt
python run.py --input data/conversations --output results
```

- ~10s model load, then ~4-5s to score the 27 sample conversations (~330 turns).
  First run is slower - it's also downloading the model.
- No account/API key/GPU needed. Writes only to `queue/` and `results/`, both
  created next to `run.py`. 

## 2. The pipeline

- **Queue**: plain directory tree (`queue/pending`, `processing`, `done`,
  `dead_letter`). Claiming a file is one `os.rename` - atomic on the same
  filesystem, so two racing workers never double-process one file. That's the
  entire mutual-exclusion mechanism, no lock server needed.
- **Cut deliberately**: a worker that claims a file and then crashes leaves it
  stuck in `processing/` forever - no auto-redelivery. A real queue's visibility
  timeout would fix this; see Section 8 for how I'd map this onto SQS given more
  time.
- **Storage**: one JSON file per conversation, on local disk.
- **Idempotency**: `results/index.json`, keyed by `tenant_id::conversation_id`
  (not filename). Confirmed correct on the shipped duplicate fixture
  (`acme__c-000002.json` / its byte-identical `.duplicate.json` twin) -
  `skipped_duplicate: 1` in the summary, only one result file on disk.

## 3. Metric definitions

Both computed over **customer turns with actual content only** (empty/
whitespace text excluded, same as not being a customer turn at all).

**`overall_sentiment`** - plain mean of signed per-turn scores.
- Equal weighting, not recency- or confidence-weighted: the alternative needs a
  tunable parameter with no real data to justify it. Simpler, fully-specified
  choice.
- 0 scoreable turns -> `null` (not `0.0` - "nothing to say" is not "measured
  zero").
- 1+ turns -> `{label, score, confidence, n_turns}`. `label`: score `> 0.2`
  positive, `< -0.2` negative, else neutral (`NEUTRAL_THRESHOLD`, written to
  every output's `model` block).

**`sentiment_trajectory`** - `(last_score - first_score) / 2`, chronological.
- Bounded `[-1, 1]` (max swing 2, divided by 2); endpoint-only keeps it
  comparable across a 2-turn and a 200-turn conversation, at the cost of missing
  mid-conversation swings (`c-000005`: sharp swings, ends where it started ->
  correctly reads `flat`). A named limitation, not a bug.
- 0 or 1 turns -> `null` (undefined). 2+ turns -> `{label, score, n_turns}`, same
  0.2 band.

**Ordering**: sorted by parsed `ts`, not `seq` (`c-000021` has non-monotonic/
duplicate `seq` but strictly increasing `ts`).

**Reproducibility**: model revision pinned to a specific commit
(`714eb0fa89d2f80546fda750413ed43d93601a13`, captured from an actual run, not
copied from documentation). Every output record's `model` block carries backend
name, model name, revision, and neutral threshold. Given a fixed model and input,
inference here is a deterministic forward pass (no sampling), so re-running
reproduces the same scores - the only thing that changes run to run is
`scored_at`.

## 4. Edge cases (`src/conversation.py`)

- **Whole-record malformation** - `turns` missing entirely (`c-000029`) or the
  wrong type (`c-000030`) - the whole conversation is rejected to `dead_letter/`
  with a reason, never scored, never silently defaulted. Confirmed in
  `results/errors.json` (`turns_key_missing` / `turns_not_a_list`); neither file
  appears in `results/`.
- **Turn-level malformation** (`c-000031`) - one bad turn inside an otherwise
  fine list skips just that turn (reason: missing/wrong-type `seq`, unrecognised
  `role`, unparseable `ts`, or non-string `text`) and keeps scoring the rest.
  Confirmed: 5 turns in the fixture, exactly 1 valid, `overall_sentiment`
  reports `n_turns: 1`.
- **Empty/whitespace-only text** (`c-000019`) - excluded from both metrics, kept
  in the per-turn output with `included_in_metrics: false, reason: "empty_text"`
  rather than disappearing silently.
- **Zero customer turns** (`c-000017`, and `c-000032` - only an unrecognised
  `bot` role) - both metrics `null`, same code path for both.
- **Empty `turns: []`** (`c-000018`) - same null result, no crash, no
  `ZeroDivisionError`.
- **The 4,000-character turn** (`c-000027`) - truncated at 512 tokens
  (`truncation=True, max_length=512`) rather than chunked. A single, clearly stated
  behaviour beat a chunking scheme that couldn't yet be validated with confidence.
- **Any unexpected exception during scoring** (not in the sample data, a real
  gap I found and closed) - wrapped per-conversation in `worker.py` and
  dead-lettered like any other unprocessable record, so one bad conversation
  can't take the batch down with it.

## 5. Trade-offs - what got cut

- **No real distributed queue.** A directory tree instead of SQS - demonstrates
  the same claim/done/dead-letter semantics without external infrastructure.
  Section 8 has the SQS mapping.
- **No retry with backoff.** Dead-lettered on first failure; production would
  want N retries with backoff (SQS's `maxReceiveCount` + redrive gives this
  free).
- **Single worker process in `run.py`.** The queue's claim mechanism is
  multi-worker-safe by design (that was the point of the atomic rename), but
  only one `Worker` instance is wired up. Running `python run.py` twice
  concurrently against the same `--work-dir` is safe and splits the work -
  tested by hand - nothing auto-launches a second process.
- **Unweighted mean and endpoint-only trajectory**, as argued in Section 3,
  instead of confidence-weighting or a fuller trend model. Simpler, fully
  specified, and I'd rather defend one formula than half-defend four.
- **No model-quality evaluation against real support text.**
  distilbert-sst2 is trained on movie reviews; nothing here confirms it
  transfers to support conversations. See Section 6 for what I found when I
  looked.

## 6. Scale, cost & model choice

**Measured on this machine** (single CPU thread, no GPU, distilbert-base,
66M params):
- Model load: ~10s/worker (mostly HTTP calls to resolve the pinned revision
  against the Hub - near-instant with `HF_HUB_OFFLINE=1` once cached).
- Unbatched: **77 turns/sec**/core (331 non-empty-text turns, 4.3s total).
- Batched (`batch_size=32`): **46 turns/sec** - slower, not faster. Working
  theory: mixed turn lengths (single words next to the 4,000-char `c-000027`)
  waste compute padding to the batch's longest member. Not verified with
  length-bucketed batches - flagging as a measured, counter-intuitive result.
- Bottleneck is model inference (~13ms/forward pass), not I/O.

**Backfill** (500M convos x 12 turns = 6B turns): 6e9 / 77 ~= 78M core-seconds
~= 902 core-days -> **~129 CPU cores** for a 7-day window (150-200 in practice
with retries/headroom).

**Steady state** (2M/day x 12 turns = 24M turns/day): ~311,700 core-sec/day
~= **~3.6 cores** continuous, round to 4-8 for headroom/multi-AZ.

**Cost per million conversations**: 12M turns / 77 turns/sec/core ~= 43.3
core-hours. At 0.04 USD/vCPU-hour on-demand (c6i/c7g-class), **1.73 USD/million** in
compute; spot (60-70% cheaper than on-demand) brings it to **$0.52-0.69/million**.
Excludes S3/SQS/DynamoDB (small next to compute at this volume).

**Scaling levers, in order of effort**: 
- **More Workers, Unchanged Code** - cheapest, purely horizontal; 
- **ONNX Runtime + INT8 quantisation** - commonly cited 2-4x CPU speedup, **not measured here**, an
estimate from general knowledge; 
- **GPU for the backfill specifically** - spin up temporary, tear down after.

**What breaks first at 10x** (2M/day -> 20M/day, 36 cores instead of ~4): two
things.
- **Worker Fleet Size** - solved by autoscaling on queue depth; 
- The one I'd actually worry about first - **the idempotency ledger**. My local
`ResultIndex` rewrites the entire seen-set on every write (`O(n)`), fine at 25
conversations but a serialised bottleneck before the worker fleet is, at 10x
steady-state volume. This is exactly what a DynamoDB conditional-write would
replace it with (Section 8).

**How I'd know before a customer tells me**: alarm on (a) oldest-message-age in
the queue (rising = workers can't keep up, a leading indicator) and (b)
result-write latency/error rate (catches ledger contention before it's a
backlog).

**Model behavior on non-English text, measured against the shipped fixtures**:
- `c-000025` (Japanese, ends "ありがとうございます。助かりました" - thank you,
  that helped) - all three customer turns scored **negative at 91-96%
  confidence**, including the thank-you close. Confidently wrong, not
  uncertain.
- `c-000024` (French) was a mixed bag - got the opening complaint right and,
  more surprisingly, the closing "merci beaucoup" right too (positive, 72%) -
  so "confidently wrong" isn't universal, just common enough to be dangerous.
- Sarcasm (`c-000023`): "Oh fantastic, another outage" scored **positive at
  99.7%**; "I wouldn't say I'm not unhappy about this" (double negation) scored
  **positive at 99.8%**. Both are actually negative, and the model doesn't
  hedge when it's wrong.
- Emoji/punctuation-only (`c-000022`): a lone "👍" scored **negative at 70%**
  (tokenizer likely has no real emoji representation); a lone "..." scored
  **positive at 96%**. The strongest argument here for not trusting confidence
  blindly.

**How I'd know if the model is any good on support text** (trained on
tweets/movie reviews, not this): the bullets above are real, reproducible
failure modes found on the first sample set, not hypothetical. A proper answer
needs a small human-labeled set of actual support transcripts and a confusion
matrix against that - a next-day task, not a four-hour one.

## 7. Deploying on AWS

Paper design only - nothing in `src/` talks to AWS; this is what I'd build with
another day, not what's running today.

- **Worker runs on ECS Fargate** (or EC2 if reserved-instance pricing matters
  more than Fargate's operational simplicity), not Lambda - the model needs to
  load once (~10s) and stay resident across many turns; Lambda's per-invocation
  model means reloading it on every cold start.
- **Local -> AWS mapping**:

  | Local (this repo) | AWS |
  | --- | --- |
  | `queue/pending` | SQS queue (message = pointer, not the transcript) |
  | `queue/processing` (rename) | SQS's own visibility-timeout state |
  | `queue/dead_letter` + reason | SQS redrive policy -> DLQ |
  | `results/*.json` | S3 (per-conversation result, archival) |
  | `results/index.json` (ledger) | DynamoDB, conditional put on the key |

- **One real change**: the queue message carries a pointer, not the transcript. SQS caps messages at 256KB, and a 200-turn conversation (`c-000028`) is
  comfortably under that today but pointer-passing costs nothing extra to build
  now.
- **Why SQS over Kafka/MSK**: Kafka wins on independent fan-out (scoring +
  PII redaction + analytics reading the same stream at their own pace) or
  cheap full-backfill replay on every model retrain. Neither applies — one
  consumer group, one job per message — so SQS's zero-ops queue with free
  visibility-timeout/DLQ semantics beats paying for an always-on broker.
- **Scaling**: ECS target-tracking on SQS `ApproximateNumberOfMessagesVisible`,
  with a **hard max task count** so a mistaken backfill re-queue can't
  autoscale into an unbounded bill.
- **Failing messages**: SQS `maxReceiveCount` redrive -> DLQ, mirroring local
  `dead_letter/` + reason. CloudWatch alarm on DLQ depth > 0 -> SNS ->
  Slack/PagerDuty, so nobody has to notice a growing directory by hand.
- **Cost**: same order of magnitude as Section 6 - **~$1-2/million** compute on
  Fargate, spot pricing helping the batch-shaped backfill specifically.
- **If inference were the bottleneck**: ONNX Runtime + quantisation first
  (cheapest, no architecture change), a temporary GPU fleet second for the
  backfill specifically.
- **Is AWS even the right call?** Mostly yes, with an asterisk: steady state
  (2M/day, ~4-8 cores) alone wouldn't need much more than "a container on a
  schedule" - the real scaling problem here is the 500M backfill, a bounded,
  one-time batch job. If this were only the backfill and never repeated, "one
  big box (or a handful of spot GPU instances) and a shell script" would beat
  standing up a permanent autoscaling service. There is a steady state here
  (2M/day forever), so the always-on service earns its keep - it just doesn't
  need to be big.

## 8. Future scope - not implemented in this repo

Everything below is what I'd build with more time, not code that exists here
today. Called out explicitly so this NOTES.md doesn't claim more than the repo
does.

1. **Wire SQS + S3 + DynamoDB for real** (or LocalStack if I get a working
   Docker setup), replacing the local directory queue per the Section 7
   mapping - this is "build it," the design already accounts for it.
2. **Confidence-weight metrics** and try length-bucketed batching, then
   re-measure against the 77 turns/sec baseline instead of assuming either
   helps.
3. **Route non-English text to a multilingual model** (e.g. an XLM-R-based
   sentiment model) behind a `--model` flag, and re-run the Section 6 failure
   cases to see what actually improves - multilingual models are typically
   several times larger/slower than distilbert-sst2, so this trades cost for
   correctness on non-English traffic and needs measuring, not assuming.
4. **Multi-process `run.py`** (a `--workers N` flag launching real worker
   processes against the same `--work-dir`) instead of relying on two manual
   invocations - the queue's claim mechanism already supports it.
5. **Retries with backoff** before dead-lettering (or just the AWS move above,
   which gets this for free via SQS redrive).

