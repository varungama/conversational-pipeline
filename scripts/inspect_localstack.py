"""Read-only look at what --backend aws left in LocalStack: S3 objects and
the DynamoDB idempotency ledger. Requires `docker compose up -d` running.

    python scripts/inspect_localstack.py
"""

import logging
import pathlib
import sys

import boto3

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.aws.config import AwsConfig

logger = logging.getLogger(__name__)


def main():
    config = AwsConfig()
    kwargs = config.client_kwargs()

    s3 = boto3.client("s3", **kwargs)
    dynamo = boto3.resource("dynamodb", **kwargs)

    logger.info("=== S3 buckets ===")
    for bucket in s3.list_buckets()["Buckets"]:
        logger.info("- %s", bucket["Name"])

    for bucket_name in (config.conversations_bucket, config.results_bucket):
        logger.info("=== Objects in %s ===", bucket_name)
        paginator = s3.get_paginator("list_objects_v2")
        count = 0
        for page in paginator.paginate(Bucket=bucket_name):
            for obj in page.get("Contents", []):
                logger.info("  %s  (%d bytes)", obj["Key"], obj["Size"])
                count += 1
        logger.info("  -> %d object(s)", count)

    logger.info("=== DynamoDB table: %s ===", config.ledger_table_name)
    table = dynamo.Table(config.ledger_table_name)
    items = table.scan().get("Items", [])
    logger.info("  -> %d item(s)", len(items))
    for item in items:
        logger.info("  %s", dict(item))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main()
