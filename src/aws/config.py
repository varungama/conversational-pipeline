"""SQS/S3/DynamoDB configuration; every value has an env-var override (unset endpoint_url = real AWS)."""

import dataclasses
import math
import os

RECEIVE_BATCH_SIZE = 1  # kept at 1: batch 5 hoarded big tasks and was slower (NOTES.md 6.1)
WORST_CASE_TASK_SECONDS = 6  # expected slowest task (s); sizes the visibility timeout starting window


def worst_case_visibility_timeout(receive_batch_size: int, worst_case_task_seconds: float) -> int:
    """Visibility timeout covering a full batch of worst-case tasks; too short and in-flight work is redelivered."""
    return math.ceil(receive_batch_size * worst_case_task_seconds)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclasses.dataclass(frozen=True)
class AwsConfig:
    endpoint_url: str | None = dataclasses.field(  # falsy = real AWS
        default_factory=lambda: _env("AWS_ENDPOINT_URL", "http://localhost:4566"))
    region: str = dataclasses.field(default_factory=lambda: _env("AWS_REGION", "us-east-1"))

    conversations_bucket: str = dataclasses.field(
        default_factory=lambda: _env("SENTIMENT_CONVERSATIONS_BUCKET", "sentiment-conversations"))
    results_bucket: str = dataclasses.field(
        default_factory=lambda: _env("SENTIMENT_RESULTS_BUCKET", "sentiment-results"))

    queue_name: str = dataclasses.field(default_factory=lambda: _env("SENTIMENT_QUEUE_NAME", "sentiment-work"))
    dead_letter_queue_name: str = dataclasses.field(
        default_factory=lambda: _env("SENTIMENT_DLQ_NAME", "sentiment-work-dlq"))
    max_receive_count: int = dataclasses.field(  # after this many failed receives, SQS redrives to the DLQ
        default_factory=lambda: int(_env("SENTIMENT_MAX_RECEIVE_COUNT", "3")))
    receive_batch_size: int = dataclasses.field(
        default_factory=lambda: int(_env("SENTIMENT_RECEIVE_BATCH", str(RECEIVE_BATCH_SIZE))))
    worst_case_task_seconds: float = dataclasses.field(
        default_factory=lambda: float(_env("SENTIMENT_WORST_TASK_SECONDS", str(WORST_CASE_TASK_SECONDS))))
    visibility_timeout_seconds: int | None = dataclasses.field(  # derived when unset; env var/argument overrides
        default_factory=lambda: int(os.environ["SENTIMENT_VISIBILITY_TIMEOUT"])
        if "SENTIMENT_VISIBILITY_TIMEOUT" in os.environ else None)

    ledger_table_name: str = dataclasses.field(
        default_factory=lambda: _env("SENTIMENT_LEDGER_TABLE", "sentiment-result-index"))

    shard_expiry_days: int = dataclasses.field(  # only thing that reclaims S3ShardStore's slices/shards
        default_factory=lambda: int(_env("SENTIMENT_SHARD_EXPIRY_DAYS", "7")))

    def __post_init__(self):
        if self.visibility_timeout_seconds is None:
            object.__setattr__(self, "visibility_timeout_seconds", worst_case_visibility_timeout(
                self.receive_batch_size, self.worst_case_task_seconds))

    def client_kwargs(self) -> dict:
        kwargs = {"region_name": self.region}
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
            # LocalStack ignores these but boto3 needs some; "test"/"test" is LocalStack's own convention
            kwargs["aws_access_key_id"] = os.environ.get("AWS_ACCESS_KEY_ID", "test")
            kwargs["aws_secret_access_key"] = os.environ.get("AWS_SECRET_ACCESS_KEY", "test")
        return kwargs
