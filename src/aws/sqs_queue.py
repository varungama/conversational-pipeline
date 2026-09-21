"""SQS as the work queue: receive = claim; un-acked messages reappear after the visibility timeout and go to the DLQ
after `maxReceiveCount` attempts (no app code). The DLQ carries no reason; it lives in the worker logs."""

import contextlib
import json
import logging
import threading

from src.aws.clients import AwsClients

logger = logging.getLogger(__name__)


class Message:
    def __init__(self, receipt_handle: str, body: dict):
        self.receipt_handle = receipt_handle
        self.body = body


class SqsQueue:
    def __init__(self, clients: AwsClients, queue_url: str, dlq_url: str,
                 visibility_timeout_seconds: int | None = None):
        self.sqs = clients.sqs
        self.queue_url = queue_url
        self.dlq_url = dlq_url
        # The queue's visibility timeout, needed only by `keep_alive`.
        self.visibility_timeout_seconds = visibility_timeout_seconds

    def enqueue(self, tenant_id: str, conversation_id: str, shard: dict | None = None):
        """Sends a pointer, not the transcript; `shard` selects one slice of a large conversation."""
        payload = {"tenant_id": tenant_id, "conversation_id": conversation_id}
        if shard:
            payload["shard"] = shard
        body = json.dumps(payload)
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

    @contextlib.contextmanager
    def keep_alive(self, message: Message):
        """Keeps `message` invisible while the caller works, re-extending the timeout every third of it.
        So the timeout only bounds a crashed worker's message, not the slowest task; no-op without a timeout."""
        timeout = self.visibility_timeout_seconds
        if not timeout:
            yield
            return
        stop = threading.Event()

        def beat():
            while not stop.wait(timeout / 3):
                try:
                    self.sqs.change_message_visibility(
                        QueueUrl=self.queue_url, ReceiptHandle=message.receipt_handle,
                        VisibilityTimeout=timeout)
                except Exception:  # message already deleted/expired: nothing left to protect
                    logger.warning("Could not extend visibility of a message in flight", exc_info=True)
                    return

        thread = threading.Thread(target=beat, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()

    def receive_dead_letters(self, max_messages: int = 10) -> list:
        response = self.sqs.receive_message(
            QueueUrl=self.dlq_url, MaxNumberOfMessages=max_messages, WaitTimeSeconds=0,
        )
        return [
            Message(receipt_handle=m["ReceiptHandle"], body=json.loads(m["Body"]))
            for m in response.get("Messages", [])
        ]

    def ack_dead_letter(self, message: Message):
        """Deletes a harvested DLQ message so it does not reappear and double-count."""
        self.sqs.delete_message(QueueUrl=self.dlq_url, ReceiptHandle=message.receipt_handle)

    def approximate_depth(self) -> int:
        attrs = self.sqs.get_queue_attributes(
            QueueUrl=self.queue_url, AttributeNames=["ApproximateNumberOfMessages"]
        )
        return int(attrs["Attributes"]["ApproximateNumberOfMessages"])
