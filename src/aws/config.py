"""Configuration for the SQS/S3/DynamoDB backend.

Every value has an env-var override so the same code points at LocalStack
for a local demo and real AWS in production - only endpoint_url (unset ->
real AWS) and credentials differ.
"""

import dataclasses
import math
import os

# A worker takes this many messages per ReceiveMessage. All of them go
# invisible at once, but they are processed one after another - so the last
# one sits invisible for the whole batch, not just its own turn.
RECEIVE_BATCH_SIZE = 5
# Slowest single task, in seconds: one full shard (src/sharding.py,
# SHARD_MAX_TURNS turns). ~3s on the default model at the measured 16 turns/s
# unbatched (NOTES.md 6), doubled for headroom. Sharding is what makes this a
# bounded number - an unsharded 200-turn conversation is ~12s+ and unbounded
# in principle.
WORST_CASE_TASK_SECONDS = 6


def worst_case_visibility_timeout(receive_batch_size: int, worst_case_task_seconds: float) -> int:
    """Long enough that the last message of a full batch, behind
    `receive_batch_size - 1` worst-case tasks plus its own, is still
    invisible when its turn ends. Too short and SQS re-delivers work that
    is still in flight: duplicate scoring, and each redelivery burns one of
    the message's `max_receive_count` attempts, so healthy messages can
    reach the DLQ."""
    return math.ceil(receive_batch_size * worst_case_task_seconds)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclasses.dataclass(frozen=True)
class AwsConfig:
    # default_factory, not a plain default, so each AwsConfig() sees the
    # *current* env (a plain default is evaluated once at import, missing
    # env vars a test sets later). Falsy = talk to real AWS (boto3's normal
    # resolution); set to a LocalStack URL for the local prototype.
    endpoint_url: str | None = dataclasses.field(
        default_factory=lambda: _env("AWS_ENDPOINT_URL", "http://localhost:4566"))
    region: str = dataclasses.field(default_factory=lambda: _env("AWS_REGION", "us-east-1"))

    conversations_bucket: str = dataclasses.field(
        default_factory=lambda: _env("SENTIMENT_CONVERSATIONS_BUCKET", "sentiment-conversations"))
    results_bucket: str = dataclasses.field(
        default_factory=lambda: _env("SENTIMENT_RESULTS_BUCKET", "sentiment-results"))

    queue_name: str = dataclasses.field(default_factory=lambda: _env("SENTIMENT_QUEUE_NAME", "sentiment-work"))
    dead_letter_queue_name: str = dataclasses.field(
        default_factory=lambda: _env("SENTIMENT_DLQ_NAME", "sentiment-work-dlq"))
    # after this many failed receives, SQS's redrive policy moves it to the DLQ
    max_receive_count: int = dataclasses.field(
        default_factory=lambda: int(_env("SENTIMENT_MAX_RECEIVE_COUNT", "3")))
    receive_batch_size: int = dataclasses.field(
        default_factory=lambda: int(_env("SENTIMENT_RECEIVE_BATCH", str(RECEIVE_BATCH_SIZE))))
    worst_case_task_seconds: float = dataclasses.field(
        default_factory=lambda: float(_env("SENTIMENT_WORST_TASK_SECONDS", str(WORST_CASE_TASK_SECONDS))))
    # Left unset, this is derived from the two fields above (see
    # worst_case_visibility_timeout). Set SENTIMENT_VISIBILITY_TIMEOUT, or
    # pass a value, to override - tests pass a short one to keep DLQ runs fast.
    visibility_timeout_seconds: int | None = dataclasses.field(
        default_factory=lambda: int(os.environ["SENTIMENT_VISIBILITY_TIMEOUT"])
        if "SENTIMENT_VISIBILITY_TIMEOUT" in os.environ else None)

    ledger_table_name: str = dataclasses.field(
        default_factory=lambda: _env("SENTIMENT_LEDGER_TABLE", "sentiment-result-index"))

    def __post_init__(self):
        if self.visibility_timeout_seconds is None:
            object.__setattr__(self, "visibility_timeout_seconds", worst_case_visibility_timeout(
                self.receive_batch_size, self.worst_case_task_seconds))

    def client_kwargs(self) -> dict:
        kwargs = {"region_name": self.region}
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
            # LocalStack ignores these, but boto3 needs some credentials -
            # "test"/"test" is LocalStack's own documented convention
            kwargs["aws_access_key_id"] = os.environ.get("AWS_ACCESS_KEY_ID", "test")
            kwargs["aws_secret_access_key"] = os.environ.get("AWS_SECRET_ACCESS_KEY", "test")
        return kwargs
