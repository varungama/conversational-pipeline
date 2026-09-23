"""Score every conversation in an input directory with a sentiment model and write durable JSON results, via a local file-based queue or AWS (SQS + S3 + DynamoDB)."""

import argparse
import functools
import json
import logging
import multiprocessing as mp
import os
import pathlib
import time
from datetime import datetime, timezone

from src.conversation import parse_timestamp
from src.metrics import DEFAULT_WEIGHTING, WEIGHTING_CHOICES
from src.queue_store import FileQueue
from src.sentiment_model import DEFAULT_MODEL_KEY, MODEL_REGISTRY, SentimentScorer, chunk_text, load_token_counter
from src.sharding import lpt_interleaved, plan_shards_and_cost
from src.worker import Worker

logger = logging.getLogger(__name__)


def _write_run_output(output_dir: pathlib.Path, combined: list, dead_letters: list, summary: dict):
    """Shared tail of both backends: write results.json/errors.json and log the run summary."""
    (output_dir / "results.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "errors.json").write_text(
        json.dumps(dead_letters, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("Run summary: %s", json.dumps(summary))
    logger.info("Wrote %d scored conversation(s) to %s", len(combined), output_dir / "results.json")
    logger.info("Wrote %d dead-lettered conversation(s) to %s", len(dead_letters), output_dir / "errors.json")


def configure_logging(log_dir: pathlib.Path, suffix: str = "", level: int = logging.INFO) -> pathlib.Path:
    """Logs to the console and logs/YYYY/MM/DD/<worker>/pipeline_<timestamp>.log; `suffix` names concurrent workers."""
    now = datetime.now(timezone.utc)
    worker_name = suffix[1:] if suffix else "main"  # suffix is "" (coordinator) or "_workerN"
    run_dir = log_dir / now.strftime("%Y") / now.strftime("%m") / now.strftime("%d") / worker_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"pipeline_{now.strftime('%Y%m%dT%H%M%SZ')}.log"

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return log_path


def _limit_torch_threads(total_workers: int):
    """Caps this process's torch thread pool to its fair share, else N processes oversubscribe the CPU and stall."""
    try:
        import torch
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // total_workers))
    except ImportError:
        pass


def _run_one_worker_process(worker_index, total_workers, work_dir_str, output_dir_str, scored_at, model_key,
                             weighting, log_dir_str, result_conn):
    """Child-process entry for `--workers`; module-level because Windows "spawn" pickles the target by reference."""
    configure_logging(pathlib.Path(log_dir_str), suffix=f"_worker{worker_index}")
    try:
        _limit_torch_threads(total_workers)
        queue = FileQueue(pathlib.Path(work_dir_str))
        logger.info("[worker %d] Loading sentiment model %s...", worker_index, model_key)
        scorer = SentimentScorer(model_key=model_key)
        logger.info("[worker %d] Using scoring backend: %s (%s @ %s)", worker_index,
                    scorer.backend_name, scorer.model_name, scorer.model_revision)
        worker = Worker(queue, pathlib.Path(output_dir_str), scorer, scored_at=scored_at, weighting=weighting)
        summary = worker.run_once()
        logger.info("[worker %d] Done: %s", worker_index, json.dumps(summary))
        result_conn.send(summary)
    finally:
        result_conn.close()


def _drain_queue(worker, queue, config, drain_timeout: float) -> dict:
    """One AwsWorker's receive/process/ack loop. Normal exit is idle-with-empty-queue (queue depth truly 0);
    `drain_timeout` is only a safety ceiling for a genuinely stuck queue (e.g. broken infra), not the expected
    stopping point - a message mid-redelivery must be allowed to finish its retry cycle, however long that takes."""
    summary = {"processed": 0, "skipped_duplicate": 0, "retry_attempts": 0, "shards_scored": 0}
    idle_passes = 0
    deadline = time.monotonic() + drain_timeout
    while time.monotonic() < deadline:
        batch = worker.run_once(max_messages=config.receive_batch_size)
        summary["shards_scored"] += batch["shards_scored"]
        summary["processed"] += batch["processed"]
        summary["skipped_duplicate"] += batch["skipped_duplicate"]
        summary["retry_attempts"] += batch["left_for_retry"]
        if sum(batch.values()) == 0:
            idle_passes += 1
            if idle_passes >= 3 and queue.approximate_depth() == 0:
                break
            time.sleep(config.visibility_timeout_seconds / 2 + 0.5)
        else:
            idle_passes = 0
    return summary


def _run_one_aws_worker_process(worker_index, total_workers, config, infra, drain_timeout, model_key,
                                 weighting, scored_at, log_dir_str, result_conn):
    """Child-process entry for AWS `--workers`; module-level because Windows "spawn" pickles the target by reference.
    Builds its own boto3 clients/model (neither survives a spawn) against the infra the parent already created."""
    from src.aws.clients import AwsClients
    from src.aws.dynamo_ledger import DynamoResultIndex
    from src.aws.s3_store import S3ConversationStore, S3ResultStore, S3ShardStore
    from src.aws.sqs_queue import SqsQueue
    from src.aws.worker import AwsWorker

    configure_logging(pathlib.Path(log_dir_str), suffix=f"_worker{worker_index}")
    try:
        _limit_torch_threads(total_workers)
        clients = AwsClients(config)
        queue = SqsQueue(clients, infra["queue_url"], infra["dlq_url"], config.visibility_timeout_seconds)
        conversations = S3ConversationStore(clients, config.conversations_bucket)
        results_store = S3ResultStore(clients, config.results_bucket)
        ledger = DynamoResultIndex(clients, config.ledger_table_name)
        shard_store = S3ShardStore(clients, config.conversations_bucket)

        logger.info("[worker %d] Loading sentiment model %s...", worker_index, model_key)
        scorer = SentimentScorer(model_key=model_key)
        logger.info("[worker %d] Using scoring backend: %s (%s @ %s)", worker_index,
                    scorer.backend_name, scorer.model_name, scorer.model_revision)

        worker = AwsWorker(queue, conversations, results_store, ledger, scorer,
                            scored_at=scored_at, weighting=weighting, shard_store=shard_store)
        summary = _drain_queue(worker, queue, config, drain_timeout)
        logger.info("[worker %d] Done: %s", worker_index, json.dumps(summary))
        result_conn.send(summary)
    finally:
        result_conn.close()


def run_local(args):
    input_dir = pathlib.Path(args.input)
    work_dir = pathlib.Path(args.work_dir)
    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    queue = FileQueue(work_dir)
    enqueued = 0
    for source_file in sorted(input_dir.glob("*.json")):
        queue.enqueue(source_file)
        enqueued += 1
    logger.info("Enqueued %d conversation file(s) from %s", enqueued, input_dir)

    if args.workers <= 1:
        logger.info("Loading sentiment model %s (downloads on first run if not already cached)...",
                     args.model)
        scorer = SentimentScorer(model_key=args.model)
        logger.info("Using scoring backend: %s (%s @ %s)",
                    scorer.backend_name, scorer.model_name, scorer.model_revision)

        worker = Worker(queue, output_dir, scorer, scored_at=args.scored_at, weighting=args.weighting)
        summary = worker.run_once()
    else:
        logger.info("Starting %d worker processes against %s (model=%s, weighting=%s)",
                     args.workers, work_dir, args.model, args.weighting)
        processes = []
        parent_conns = []
        for worker_index in range(args.workers):
            parent_conn, child_conn = mp.Pipe(duplex=False)
            process = mp.Process(
                target=_run_one_worker_process,
                args=(worker_index, args.workers, str(work_dir), str(output_dir), args.scored_at, args.model,
                      args.weighting, args.log_dir, child_conn),
            )
            process.start()
            child_conn.close()  # the child's copy is enough; close the parent's duplicate
            processes.append(process)
            parent_conns.append(parent_conn)

        summary = {"processed": 0, "skipped_duplicate": 0, "dead_lettered": 0}
        for worker_index, (process, conn) in enumerate(zip(processes, parent_conns)):
            worker_summary = conn.recv()
            conn.close()
            process.join()
            for key, count in worker_summary.items():
                summary[key] = summary.get(key, 0) + count
            log = logger.info if process.exitcode == 0 else logger.warning
            log("worker %d exited with code %s", worker_index, process.exitcode)

    combined = []
    for result_file in sorted(output_dir.glob("*.json")):
        if result_file.name == "index.json":
            continue
        combined.append(json.loads(result_file.read_text(encoding="utf-8")))

    dead_letters = []
    for dead_file in sorted(queue.dead_letter.glob("*.json")):
        reason_file = dead_file.parent / (dead_file.stem + ".reason.txt")
        dead_letters.append({
            "source_file": dead_file.name,
            "reason": reason_file.read_text(encoding="utf-8") if reason_file.exists() else "unknown",
        })

    _write_run_output(output_dir, combined, dead_letters, summary)


def run_aws(args):
    from src.aws.bootstrap import ensure_infrastructure
    from src.aws.clients import AwsClients
    from src.aws.config import AwsConfig
    from src.aws.dynamo_ledger import DynamoResultIndex
    from src.aws.s3_store import S3ConversationStore, S3ResultStore, S3ShardStore
    from src.aws.sqs_queue import SqsQueue
    from src.aws.worker import AwsWorker

    input_dir = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = AwsConfig()
    if args.workers > 1 and "SENTIMENT_RECEIVE_BATCH" not in os.environ:
        # Interleave lanes only pay off with multiple concurrent receivers (NOTES.md 6.1.3);
        # match them to --workers unless the user pinned the batch size explicitly via the env var.
        config = AwsConfig(receive_batch_size=args.workers)
        logger.info("Matching receive-batch/interleave lanes to --workers=%d (override with SENTIMENT_RECEIVE_BATCH)",
                    args.workers)
    logger.info("Connecting to %s (%s)...", config.endpoint_url or "real AWS", config.region)
    clients = AwsClients(config)
    infra = ensure_infrastructure(clients, config)
    logger.info("Infrastructure ready: queue=%s, dlq=%s, buckets=%s/%s, table=%s",
                config.queue_name, config.dead_letter_queue_name,
                config.conversations_bucket, config.results_bucket, config.ledger_table_name)

    conversations = S3ConversationStore(clients, config.conversations_bucket)
    results_store = S3ResultStore(clients, config.results_bucket)
    ledger = DynamoResultIndex(clients, config.ledger_table_name)
    queue = SqsQueue(clients, infra["queue_url"], infra["dlq_url"], config.visibility_timeout_seconds)
    shard_store = S3ShardStore(clients, config.conversations_bucket)

    count_tokens = load_token_counter(args.model)  # None -> costs are estimated from characters
    logger.info("Estimating task cost from %s", "tokenizer token counts" if count_tokens else "character counts")
    chunk_text_fn = functools.partial(chunk_text, count_tokens=count_tokens) if count_tokens else None  # enables chunk-level shards

    tasks = []  # (cost, tenant_id, conversation_id, shard or None)
    expected_identities = set()  # every unique (tenant_id, conversation_id) enqueued; duplicates collapse naturally
    enqueued = 0
    for source_file in sorted(input_dir.glob("*.json")):
        raw_doc = json.loads(source_file.read_text(encoding="utf-8"))
        tenant_id = raw_doc.get("tenant_id") if isinstance(raw_doc, dict) else None
        conversation_id = raw_doc.get("conversation_id") if isinstance(raw_doc, dict) else None
        if not tenant_id or not conversation_id:
            # No identity for an S3 key: upload under a synthetic key so it dead-letters through normal validation.
            tenant_id, conversation_id = "unknown", source_file.stem
        expected_identities.add((tenant_id, conversation_id))
        conversations.put(tenant_id, conversation_id, raw_doc)
        shards, cost = plan_shards_and_cost(raw_doc, count_tokens=count_tokens, chunk_text_fn=chunk_text_fn)
        if shards:
            shard_store.put_slices(tenant_id, conversation_id, raw_doc, shards, chunk_text_fn)  # before enqueueing
            tasks.extend((shard["cost"], tenant_id, conversation_id, shard) for shard in shards)
        else:
            tasks.append((cost, tenant_id, conversation_id, None))
        enqueued += 1

    # Longest-first, interleaved by receive batch size so one ReceiveMessage can't grab several big tasks at once.
    tasks = lpt_interleaved(tasks, config.receive_batch_size)
    for _, tenant_id, conversation_id, shard in tasks:
        queue.enqueue(tenant_id, conversation_id, shard=shard)
    logger.info("Enqueued %d conversation file(s) from %s as %d task(s)", enqueued, input_dir, len(tasks))

    logger.info("Visibility timeout %ds for receive batches of %d (worst-case task %.0fs)",
                config.visibility_timeout_seconds, config.receive_batch_size, config.worst_case_task_seconds)

    if args.workers <= 1:
        logger.info("Loading sentiment model %s (downloads on first run if not already cached)...", args.model)
        scorer = SentimentScorer(model_key=args.model)
        logger.info("Using scoring backend: %s (%s @ %s)",
                    scorer.backend_name, scorer.model_name, scorer.model_revision)

        worker = AwsWorker(queue, conversations, results_store, ledger, scorer, scored_at=args.scored_at,
                            weighting=args.weighting, shard_store=shard_store)
        summary = _drain_queue(worker, queue, config, args.aws_drain_timeout)
        dead_letter_worker = worker
    else:
        logger.info("Starting %d worker processes against queue %s (model=%s, weighting=%s)",
                     args.workers, config.queue_name, args.model, args.weighting)
        processes = []
        parent_conns = []
        for worker_index in range(args.workers):
            parent_conn, child_conn = mp.Pipe(duplex=False)
            process = mp.Process(
                target=_run_one_aws_worker_process,
                args=(worker_index, args.workers, config, infra, args.aws_drain_timeout, args.model,
                      args.weighting, args.scored_at, args.log_dir, child_conn),
            )
            process.start()
            child_conn.close()  # the child's copy is enough; close the parent's duplicate
            processes.append(process)
            parent_conns.append(parent_conn)

        summary = {"processed": 0, "skipped_duplicate": 0, "retry_attempts": 0, "shards_scored": 0}
        for worker_index, (process, conn) in enumerate(zip(processes, parent_conns)):
            worker_summary = conn.recv()
            conn.close()
            process.join()
            for key, count in worker_summary.items():
                summary[key] = summary.get(key, 0) + count
            log = logger.info if process.exitcode == 0 else logger.warning
            log("worker %d exited with code %s", worker_index, process.exitcode)

        # drain_dead_letters only touches self.queue; skip loading a model just for this.
        dead_letter_worker = AwsWorker(queue, conversations, results_store, ledger, None, shard_store=shard_store)

    dead_letters = dead_letter_worker.drain_dead_letters()
    combined = results_store.list_all()

    accounted_for = {(r["tenant_id"], r["conversation_id"]) for r in combined}
    for dl in dead_letters:
        tenant_id, _, rest = dl["source_file"].partition("/")
        accounted_for.add((tenant_id, rest.removesuffix(".json")))
    missing = expected_identities - accounted_for
    if missing:
        logger.warning(
            "%d conversation(s) neither scored nor dead-lettered - still in-flight when the run ended "
            "(raise --aws-drain-timeout, or rerun to let their retry cycle finish): %s",
            len(missing), sorted(missing))

    _write_run_output(output_dir, combined, dead_letters, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["local", "aws"], default="local",
                         help="'local' (default, file-based queue) or 'aws' (SQS + S3 + DynamoDB via LocalStack).")
    parser.add_argument("--input", default="data/conversations",
                         help="Directory of conversation JSON files to score.")
    parser.add_argument("--work-dir", default="queue",
                         help="[local backend only] Where the durable queue directories live.")
    parser.add_argument("--output", default="results",
                         help="Where scored results are written.")
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default=DEFAULT_MODEL_KEY,
                         help=f"Transformer backend to score turns with (default: {DEFAULT_MODEL_KEY}); see src/sentiment_model.py's MODEL_REGISTRY.")
    parser.add_argument("--workers", type=int, default=1,
                         help="Number of worker processes to run concurrently against the same --work-dir "
                              "(local backend) or the same SQS queue (aws backend); on aws, also sizes the "
                              "receive-batch/interleave lanes unless SENTIMENT_RECEIVE_BATCH is set.")
    parser.add_argument("--weighting", choices=WEIGHTING_CHOICES, default=DEFAULT_WEIGHTING,
                         help=f"How overall_sentiment averages per-turn scores (default: {DEFAULT_WEIGHTING}); see src/metrics.py.")
    parser.add_argument("--aws-drain-timeout", type=float, default=900.0,
                         help="[aws backend only] Safety ceiling (seconds): normal exit is the queue actually going "
                              "idle and empty, however long that takes (e.g. a bad message's retry cycle before it "
                              "reaches the DLQ). This only fires if something is genuinely stuck - keep it generous.")
    parser.add_argument("--log-dir", default="logs",
                         help="Where per-run log files are written.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                         help="DEBUG shows per-shard packing detail and lane/task assignment from src.sharding.")
    parser.add_argument("--scored-at", default="2026-03-02T10:00:00Z", metavar="YYYY-MM-DDTHH:MM:SSZ",
                         help="Timestamp stamped on every output record's scored_at field (default: fixed, for reproducible runs).")
    args = parser.parse_args()

    if parse_timestamp(args.scored_at) is None:
        parser.error(f"--scored-at must look like 2026-03-02T10:00:00Z, got {args.scored_at!r}")
    if args.workers < 1:
        parser.error(f"--workers must be at least 1, got {args.workers}")

    log_path = configure_logging(pathlib.Path(args.log_dir), level=getattr(logging, args.log_level))
    logger.info("Logging to %s", log_path)

    run_start = datetime.now(timezone.utc)
    start_monotonic = time.monotonic()

    if args.backend == "aws":
        run_aws(args)
    else:
        run_local(args)

    elapsed_seconds = time.monotonic() - start_monotonic
    run_end = datetime.now(timezone.utc)
    logger.info("Run started at %s, finished at %s, total wall-clock time: %.2fs (%dm %.1fs)",
                run_start.isoformat(), run_end.isoformat(), elapsed_seconds,
                int(elapsed_seconds // 60), elapsed_seconds % 60)


if __name__ == "__main__":
    main()
