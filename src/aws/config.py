"""Configuration for the SQS/S3/DynamoDB backend.

Every value has an env-var override so the same code points at LocalStack
for a local demo and real AWS in production - only endpoint_url (unset ->
real AWS) and credentials differ.
"""

import dataclasses
import os


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
    # short for a fast local demo; production should exceed the worst-case
    # processing time (a 200-turn conversation), typically 30-60s
    visibility_timeout_seconds: int = dataclasses.field(
        default_factory=lambda: int(_env("SENTIMENT_VISIBILITY_TIMEOUT", "5")))

    ledger_table_name: str = dataclasses.field(
        default_factory=lambda: _env("SENTIMENT_LEDGER_TABLE", "sentiment-result-index"))

    def client_kwargs(self) -> dict:
        kwargs = {"region_name": self.region}
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
            # LocalStack ignores these, but boto3 needs some credentials -
            # "test"/"test" is LocalStack's own documented convention
            kwargs["aws_access_key_id"] = os.environ.get("AWS_ACCESS_KEY_ID", "test")
            kwargs["aws_secret_access_key"] = os.environ.get("AWS_SECRET_ACCESS_KEY", "test")
        return kwargs
