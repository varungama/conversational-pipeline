"""Creates the S3 buckets, SQS queues and DynamoDB table; idempotent, works on LocalStack or real AWS."""

import json
import logging
import time

from botocore.exceptions import ClientError

from src.aws.clients import AwsClients
from src.aws.config import AwsConfig

logger = logging.getLogger(__name__)


def _ignore_already_exists(exc: ClientError, *codes: str):
    if exc.response.get("Error", {}).get("Code") not in codes:
        raise


def ensure_bucket(clients: AwsClients, name: str):
    try:
        clients.s3.create_bucket(Bucket=name)
        logger.info("Created S3 bucket %s", name)
    except ClientError as exc:
        _ignore_already_exists(exc, "BucketAlreadyOwnedByYou", "BucketAlreadyExists")
        logger.debug("S3 bucket %s already exists", name)


def ensure_shard_lifecycle(clients: AwsClients, bucket: str, expiry_days: int):
    """Expires `slices/` and `shards/` (S3ShardStore) after `expiry_days`; idempotent overwrite."""
    clients.s3.put_bucket_lifecycle_configuration(
        Bucket=bucket,
        LifecycleConfiguration={
            "Rules": [
                {
                    "ID": f"expire-{prefix.rstrip('/')}",
                    "Status": "Enabled",
                    "Filter": {"Prefix": prefix},
                    "Expiration": {"Days": expiry_days},
                }
                for prefix in ("slices/", "shards/")
            ],
        },
    )


def ensure_queue(clients: AwsClients, name: str, attributes: dict | None = None) -> str:
    """Returns the queue URL, creating the queue if needed."""
    try:
        response = clients.sqs.create_queue(QueueName=name, Attributes=attributes or {})
        logger.info("Created SQS queue %s", name)
        return response["QueueUrl"]
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "QueueAlreadyExists":
            raise
        logger.debug("SQS queue %s already exists", name)
        return clients.sqs.get_queue_url(QueueName=name)["QueueUrl"]


def ensure_ledger_table(clients: AwsClients, table_name: str):
    try:
        clients.dynamodb.create_table(
            TableName=table_name,
            AttributeDefinitions=[{"AttributeName": "result_key", "AttributeType": "S"}],
            KeySchema=[{"AttributeName": "result_key", "KeyType": "HASH"}],
            BillingMode="PAY_PER_REQUEST",
        )
        logger.info("Created DynamoDB table %s", table_name)
    except ClientError as exc:
        _ignore_already_exists(exc, "ResourceInUseException")
        logger.debug("DynamoDB table %s already exists", table_name)

    waiter = clients.dynamodb.get_waiter("table_exists")
    waiter.wait(TableName=table_name, WaiterConfig={"Delay": 1, "MaxAttempts": 30})


def ensure_infrastructure(clients: AwsClients, config: AwsConfig) -> dict:
    """Creates everything and returns the resource identifiers callers need (queue URLs, ARNs)."""
    ensure_bucket(clients, config.conversations_bucket)
    ensure_bucket(clients, config.results_bucket)
    ensure_shard_lifecycle(clients, config.conversations_bucket, config.shard_expiry_days)

    dlq_url = ensure_queue(clients, config.dead_letter_queue_name)
    dlq_arn = clients.sqs.get_queue_attributes(
        QueueUrl=dlq_url, AttributeNames=["QueueArn"]
    )["Attributes"]["QueueArn"]

    redrive_policy = json.dumps({
        "deadLetterTargetArn": dlq_arn,
        "maxReceiveCount": config.max_receive_count,
    })
    queue_url = ensure_queue(clients, config.queue_name, attributes={
        "VisibilityTimeout": str(config.visibility_timeout_seconds),
        "RedrivePolicy": redrive_policy,
    })

    ensure_ledger_table(clients, config.ledger_table_name)

    return {"queue_url": queue_url, "dlq_url": dlq_url, "dlq_arn": dlq_arn}
