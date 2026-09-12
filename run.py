"""Entry point: score every conversation in an input directory and write
durable JSON results.

Usage:
    python run.py --input data/conversations --output results

First run downloads the sentiment model (~256MB) and caches it; later
runs are offline. Falls back to VADER on download failure - see
src/sentiment_model.py.
"""

import argparse
import json
import logging
import pathlib
from datetime import datetime, timezone

from src.queue_store import FileQueue
from src.sentiment_model import SentimentScorer
from src.worker import Worker

logger = logging.getLogger(__name__)


def configure_logging(log_dir: pathlib.Path) -> pathlib.Path:
    """Logs to console and to a file under log_dir named by UTC start time,
    so concurrent/repeated runs never overwrite each other's history."""
    log_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"pipeline_{started_at}.log"

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return log_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/conversations",
                         help="Directory of conversation JSON files to score")
    parser.add_argument("--work-dir", default="queue",
                         help="Where the durable queue directories live")
    parser.add_argument("--output", default="results",
                         help="Where scored results are written")
    parser.add_argument("--log-dir", default="logs",
                        help="Where per-run log files are written (one timestamped file per run)")
    args = parser.parse_args()

    log_path = configure_logging(pathlib.Path(args.log_dir))
    logger.info("Logging to %s", log_path)

    input_dir = pathlib.Path(args.input)
    work_dir = pathlib.Path(args.work_dir)
    output_dir = pathlib.Path(args.output)

    queue = FileQueue(work_dir)
    enqueued = 0
    for source_file in sorted(input_dir.glob("*.json")):
        queue.enqueue(source_file)
        enqueued += 1
    logger.info("Enqueued %d conversation file(s) from %s", enqueued, input_dir)

    logger.info("Loading sentiment model (downloads ~256MB on first run if not already cached)...")
    scorer = SentimentScorer()
    logger.info("Using backend: %s (%s @ %s)", scorer.backend_name, scorer.model_name, scorer.model_revision)

    worker = Worker(queue, output_dir, scorer)
    summary = worker.run_once()

    combined = []
    for result_file in sorted(output_dir.glob("*.json")):
        if result_file.name == "index.json":
            continue
        combined.append(json.loads(result_file.read_text(encoding="utf-8")))
    (output_dir / "results.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    dead_letters = []
    for dead_file in sorted(queue.dead_letter.glob("*.json")):
        reason_file = dead_file.parent / (dead_file.stem + ".reason.txt")
        dead_letters.append({
            "source_file": dead_file.name,
            "reason": reason_file.read_text(encoding="utf-8") if reason_file.exists() else "unknown",
        })
    (output_dir / "errors.json").write_text(
        json.dumps(dead_letters, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    logger.info("%s", json.dumps(summary, indent=2))
    logger.info("Wrote %d scored conversation(s) to %s", len(combined), output_dir / "results.json")
    logger.info("Wrote %d dead-lettered conversation(s) to %s", len(dead_letters), output_dir / "errors.json")


if __name__ == "__main__":
    main()
