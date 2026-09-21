"""Measures batching/sharding constants on the real model: token-budget sweep, task times, worker-balance simulation.
Usage: python scripts/benchmark.py [--quick] [--out numbers.json] (a few minutes on CPU)."""

import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import src.sentiment_model as sm  # noqa: E402
from src import conversation as conv  # noqa: E402
from src.aws.config import RECEIVE_BATCH_SIZE, worst_case_visibility_timeout  # noqa: E402
from src.scoring import extract_identity  # noqa: E402
from src.sentiment_model import load_token_counter  # noqa: E402
from src.sharding import estimate_cost, plan_shards  # noqa: E402

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "conversations"


def load_valid_docs():
    """Scorable conversations only (no dead letters, no duplicate fixture)."""
    docs = []
    for path in sorted(DATA.glob("*.json")):
        if ".duplicate" in path.name:
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        tenant, cid = extract_identity(doc)
        if not (tenant and cid):
            continue
        try:
            conv.load_conversation(doc)
        except conv.ConversationRejected:
            continue
        docs.append(doc)
    return docs


def scorable_texts(doc, start=0, end=None):
    turns = conv.load_conversation(doc)[start:end]
    return [t["text"] for t in turns if t["status"] == "ok" and t["has_content"]]


def timed(fn, reps):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def simulate(task_times, workers, prefetch):
    """List-scheduling makespan: each free worker takes its next `prefetch` tasks (an SQS batch) and runs them serially."""
    free_at = [0.0] * workers
    queue = list(task_times)
    while queue:
        w = min(range(workers), key=lambda i: free_at[i])
        grabbed, queue = queue[:prefetch], queue[prefetch:]
        free_at[w] += sum(grabbed)
    return max(free_at)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="fewer repetitions and sweep points")
    parser.add_argument("--out", help="also write the raw numbers as JSON here")
    args = parser.parse_args()
    reps = 1 if args.quick else 2

    scorer = sm.SentimentScorer()
    if scorer.backend_name != "transformer":
        sys.exit("transformer model unavailable (fell back to VADER) - nothing meaningful to measure")
    docs = load_valid_docs()
    big = max(docs, key=lambda d: len(d["turns"]))
    all_texts = [t for d in docs for t in scorable_texts(d)]
    big_texts = scorable_texts(big)
    print(f"conversations={len(docs)}  turns={len(all_texts)}  "
          f"largest={big['conversation_id']} ({len(big['turns'])} turns)\n")
    scorer.score_batch(all_texts[:8])  # warm-up
    results = {}

    # 1. token budget --------------------------------------------------------------------------
    print("1. MAX_BATCH_TOKENS sweep (BATCH_SIZE=32), turns/s")
    per_turn = {name: len(t) / timed(lambda t=t: [scorer.score(x) for x in t], reps)
                for name, t in [("largest conv", big_texts), ("all turns", all_texts)]}
    print(f"   {'per-turn (no batching)':<24}" + "".join(f"{n}: {v:6.1f}   " for n, v in per_turn.items()))
    sweep = (256, 1024, 4096) if args.quick else (128, 256, 512, 1024, 2048, 4096, 16384)
    results["token_budget"] = {"per_turn": per_turn}
    original = sm.MAX_BATCH_TOKENS
    for budget in sweep:
        sm.MAX_BATCH_TOKENS = budget
        row = {name: len(t) / timed(lambda t=t: scorer.score_batch(t), reps)
               for name, t in [("largest conv", big_texts), ("all turns", all_texts)]}
        results["token_budget"][budget] = row
        print(f"   MAX_BATCH_TOKENS={budget:<6}    " + "".join(f"{n}: {v:6.1f}   " for n, v in row.items()))
    sm.MAX_BATCH_TOKENS = original

    # 2. task service times --------------------------------------------------------------------
    print("\n2. Task service times (seconds, batched scoring)")
    turn_counts = sorted((len(d["turns"]) for d in docs), reverse=True)
    print(f"   turns per conversation, largest first: {turn_counts[:6]} ... median {statistics.median(turn_counts):.0f}")
    count_tokens = load_token_counter()  # shards are planned with token counts, as run.py does
    memo = {}

    def measure(doc, start, end):
        key = (doc["conversation_id"], start, end)
        if key not in memo:
            memo[key] = timed(lambda: scorer.score_batch(scorable_texts(doc, start, end)), reps)
        return memo[key]

    def tasks_at(size):
        """What the queue would hold if conversations over `size` turns were sharded:
        [(cost, seconds, conversation_id, shard_index or None), ...] in file order."""
        out = []
        for d in docs:
            shards = plan_shards(d, max_cost=size, count_tokens=count_tokens)
            if shards:
                out += [(sh["cost"], measure(d, sh["start"], sh["end"]), d["conversation_id"], sh["index"])
                        for sh in shards]
            else:
                out.append((estimate_cost(d, count_tokens), measure(d, 0, None), d["conversation_id"], None))
        return out

    whole = tasks_at(10 ** 9)
    shard_sizes = (2500,) if args.quick else (1200, 2500, 5000)
    sharded = {size: tasks_at(size) for size in shard_sizes}
    print(f"   no sharding: {len(whole)} tasks, largest {max(t for _, t, _, _ in whole):.2f}s "
          f"(median {statistics.median(t for _, t, _, _ in whole):.2f}s), total {sum(t for _, t, _, _ in whole):.1f}s")
    for size in shard_sizes:
        times = [t for _, t, _, _ in sharded[size]]
        print(f"   SHARD_MAX_COST={size:<5}: {len(times)} tasks, largest {max(times):.2f}s, total {sum(times):.1f}s "
              f"({sum(times) / sum(t for _, t, _, _ in whole) - 1:+.0%} work vs unsharded: smaller batches)")
    est = [(c, s) for c, s, _, _ in whole]
    print(f"   cost model check: measured {1000 * sum(s for _, s in est) / sum(c for c, _ in est):.2f} ms per cost unit "
          f"over unsharded tasks (1.00 = perfect)")
    if 2500 in sharded:
        worst = max(t for _, t, _, _ in sharded[2500])
        print(f"   worst task with SHARD_MAX_COST=2500: {worst:.2f}s  (assumed {6}s -> visibility timeout "
              f"{worst_case_visibility_timeout(RECEIVE_BATCH_SIZE, 6)}s for a batch of {RECEIVE_BATCH_SIZE}; "
              f"the measured worst needs {worst_case_visibility_timeout(RECEIVE_BATCH_SIZE, worst)}s)")
    results["task_times"] = {"whole": [t[:3] for t in whole],
                             "sharded": {str(k): [t[:3] for t in v] for k, v in sharded.items()}}

    # 3. balance across workers (simulation from the measured times) --------------------------
    print("\n3. Makespan across workers (simulated from the times above; lower is better)")
    total = sum(t for _, t, _, _ in whole)
    results["simulation"] = {}
    print(f"   {'workers':<8}{'prefetch':<9}{'unsharded':>11}{'longest-first':>15}" + "".join(
        f"{'+shards<=' + str(s):>15}" for s in shard_sizes) + f"{'lower bound':>13}")
    by_cost = lambda tasks: [t for _, t, _, _ in sorted(tasks, key=lambda p: -p[0])]  # noqa: E731
    for workers in (1, 2, 4, 8):
        for prefetch in sorted({1, RECEIVE_BATCH_SIZE}):
            row = [simulate([t for _, t, _, _ in whole], workers, prefetch), simulate(by_cost(whole), workers, prefetch)]
            row += [simulate(by_cost(sharded[size]), workers, prefetch) for size in shard_sizes]
            print(f"   {workers:<8}{prefetch:<9}" + f"{row[0]:>10.1f}s{row[1]:>14.1f}s"
                  + "".join(f"{v:>14.1f}s" for v in row[2:]) + f"{total / workers:>12.1f}s")
            results["simulation"][f"{workers}x{prefetch}"] = row
    print("   (lower bound = unsharded total work / workers; ignores per-task S3/SQS overhead)")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
