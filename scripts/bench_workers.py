"""Real multi-worker AWS run on LocalStack: unsharded vs longest-first vs sharded. Usage: python scripts/bench_workers.py --workers 4
Reports the time from a shared barrier until the last worker finishes."""

import argparse
import functools
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts.benchmark import load_valid_docs  # noqa: E402
from src.aws.bootstrap import ensure_infrastructure  # noqa: E402
from src.aws.clients import AwsClients  # noqa: E402
from src.aws.config import AwsConfig, RECEIVE_BATCH_SIZE  # noqa: E402
from src.aws.dynamo_ledger import DynamoResultIndex  # noqa: E402
from src.aws.s3_store import S3ConversationStore, S3ResultStore, S3ShardStore  # noqa: E402
from src.aws.sqs_queue import SqsQueue  # noqa: E402
from src.aws.worker import AwsWorker  # noqa: E402
from src.sentiment_model import chunk_text, load_token_counter  # noqa: E402
from src.sharding import estimate_cost, lpt_interleaved, plan_shards  # noqa: E402

MODES = ("unsharded", "lpt", "sharded")


def make_config(run_id: str, visibility: int) -> AwsConfig:
    return AwsConfig(
        conversations_bucket=f"bench-conv-{run_id}", results_bucket=f"bench-res-{run_id}",
        queue_name=f"bench-q-{run_id}", dead_letter_queue_name=f"bench-dlq-{run_id}",
        ledger_table_name=f"bench-ledger-{run_id}", visibility_timeout_seconds=visibility)


def build_worker(config, infra):
    from src.sentiment_model import SentimentScorer

    clients = AwsClients(config)
    queue = SqsQueue(clients, infra["queue_url"], infra["dlq_url"], config.visibility_timeout_seconds)
    worker = AwsWorker(queue, S3ConversationStore(clients, config.conversations_bucket),
                       S3ResultStore(clients, config.results_bucket),
                       DynamoResultIndex(clients, config.ledger_table_name),
                       SentimentScorer(), shard_store=S3ShardStore(clients, config.conversations_bucket))
    return worker, queue


def worker_main(config, infra, threads, prefetch, barrier, out):
    import torch

    torch.set_num_threads(threads)
    worker, queue = build_worker(config, infra)
    barrier.wait()
    last_busy, idle, done = time.time(), 0, 0
    while idle < 2:
        batch = worker.run_once(max_messages=prefetch)
        if sum(batch.values()):
            last_busy, idle, done = time.time(), 0, done + sum(batch.values())
        else:
            idle += 1 if queue.approximate_depth() == 0 else 0
    out.put((last_busy, done))


def run_mode(mode, docs, workers, threads, prefetch):
    run_id = uuid.uuid4().hex[:8]
    config = make_config(run_id, visibility=120)  # generous: this measures balance, not redelivery
    clients = AwsClients(config)
    infra = ensure_infrastructure(clients, config)
    store = S3ConversationStore(clients, config.conversations_bucket)
    queue = SqsQueue(clients, infra["queue_url"], infra["dlq_url"])

    count_tokens = load_token_counter()
    chunk_text_fn = functools.partial(chunk_text, count_tokens=count_tokens) if count_tokens else None
    tasks = []  # (cost, tenant, conversation, shard)
    for doc in docs:
        store.put(doc["tenant_id"], doc["conversation_id"], doc)
        shards = plan_shards(doc, count_tokens=count_tokens, chunk_text_fn=chunk_text_fn) if mode == "sharded" else []
        if shards:
            S3ShardStore(clients, config.conversations_bucket).put_slices(
                doc["tenant_id"], doc["conversation_id"], doc, shards, chunk_text_fn)
            tasks += [(s["cost"], doc["tenant_id"], doc["conversation_id"], s) for s in shards]
        else:
            tasks.append((estimate_cost(doc, count_tokens), doc["tenant_id"], doc["conversation_id"], None))
    if mode != "unsharded":
        tasks = lpt_interleaved(tasks, prefetch)  # same interleave run.py uses
    for _, tenant, cid, shard in tasks:
        queue.enqueue(tenant, cid, shard=shard)

    ctx = mp.get_context("spawn")
    barrier, out = ctx.Barrier(workers + 1), ctx.Queue()
    procs = [ctx.Process(target=worker_main, args=(config, infra, threads, prefetch, barrier, out)) for _ in range(workers)]
    for p in procs:
        p.start()
    barrier.wait()
    t0 = time.time()
    finished = [out.get() for _ in procs]
    for p in procs:
        p.join()
    n_results = len(S3ResultStore(clients, config.results_bucket).list_all())
    per_worker = sorted(round(f[0] - t0, 1) for f in finished)
    return max(f[0] for f in finished) - t0, len(tasks), n_results, per_worker


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch", type=int, default=RECEIVE_BATCH_SIZE,
                        help="messages per ReceiveMessage; all go invisible together, then run serially")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    docs = load_valid_docs()
    threads = max(1, (os.cpu_count() or 1) // args.workers)
    print(f"{args.workers} workers x {threads} torch threads, {len(docs)} conversations "
          f"({sum(len(d['turns']) for d in docs)} turns), prefetch {args.prefetch}\n")
    print(f"{'mode':<11}{'tasks':>6}{'results':>9}{'makespan':>10}   each worker finished at (s)")
    summary = {}
    for mode in MODES:
        for _ in range(args.repeat):
            makespan, n_tasks, n_results, per_worker = run_mode(mode, docs, args.workers, threads, args.prefetch)
            summary.setdefault(mode, []).append(makespan)
            print(f"{mode:<11}{n_tasks:>6}{n_results:>9}{makespan:>9.1f}s   {per_worker}", flush=True)
    base = min(summary["unsharded"])
    print("\nbest of each mode, vs unsharded: " + ", ".join(f"{m} {base / min(v):.2f}x" for m, v in summary.items()))
    print(json.dumps({m: [round(x, 2) for x in v] for m, v in summary.items()}))


if __name__ == "__main__":
    main()
