"""SQS as the work queue.

Contrast with src/queue_store.py's FileQueue: there, claim = os.rename and
a stuck message needs a human to notice. Here, ReceiveMessage *is* the
claim (SQS's visibility timeout hides it from others), and a failing
message is handled by AWS itself: never deleted -> reappears after
visibility timeout -> retried -> after `maxReceiveCount` attempts, SQS's
own RedrivePolicy (bootstrap.py) moves it to the DLQ. No app code involved.

Consequence: unlike the local dead_letter/*.reason.txt, the DLQ message
carries no reason - SQS only tracks receive count. The reason lives in
worker logs from each failed attempt (CloudWatch in production). See NOTES.md.
"""

import json

from src.aws.clients import AwsClients


class Message:
    def __init__(self, receipt_handle: str, body: dict):
        self.receipt_handle = receipt_handle
        self.body = body


class SqsQueue:
    def __init__(self, clients: AwsClients, queue_url: str, dlq_url: str):
        self.sqs = clients.sqs
        self.queue_url = queue_url
        self.dlq_url = dlq_url

    def enqueue(self, tenant_id: str, conversation_id: str):
        """Sends a pointer, not the transcript - see s3_store.py."""
        body = json.dumps({"tenant_id": tenant_id, "conversation_id": conversation_id})
        self.sqs.send_message(QueueUrl=self.queue_url, MessageBody=body)

    def receive_batch(self, max_messages: int = 10, wait_time_seconds: int = 1) -> list:
        response = self.sqs.receive_message(
            QueueUrl=self.queue_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=wait_time_seconds,
        )
        return [
            Message(receipt_handle=m["ReceiptHandle"], body=json.loads(m["Body"]))
            for m in response.get("Messages", [])
        ]

    def ack(self, message: Message):
        """Deletes the message - fully done, never redeliver. Used for both
        a scored conversation and a recognised duplicate; both finished."""
        self.sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=message.receipt_handle)

    def receive_dead_letters(self, max_messages: int = 10) -> list:
        response = self.sqs.receive_message(
            QueueUrl=self.dlq_url, MaxNumberOfMessages=max_messages, WaitTimeSeconds=0,
        )
        return [
            Message(receipt_handle=m["ReceiptHandle"], body=json.loads(m["Body"]))
            for m in response.get("Messages", [])
        ]

    def ack_dead_letter(self, message: Message):
        """Deletes a DLQ message once harvested into the errors report -
        otherwise it reappears after the DLQ's visibility timeout and
        double-counts."""
        self.sqs.delete_message(QueueUrl=self.dlq_url, ReceiptHandle=message.receipt_handle)

    def approximate_depth(self) -> int:
        attrs = self.sqs.get_queue_attributes(
            QueueUrl=self.queue_url, AttributeNames=["ApproximateNumberOfMessages"]
        )
        return int(attrs["Attributes"]["ApproximateNumberOfMessages"])
