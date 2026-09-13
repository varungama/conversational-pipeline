"""boto3 client construction, in one place so tests point every client at
moto the same way run.py points them at LocalStack."""

import boto3

from src.aws.config import AwsConfig


class AwsClients:
    def __init__(self, config: AwsConfig):
        self.config = config
        kwargs = config.client_kwargs()
        self.s3 = boto3.client("s3", **kwargs)
        self.sqs = boto3.client("sqs", **kwargs)
        self.dynamodb = boto3.client("dynamodb", **kwargs)
