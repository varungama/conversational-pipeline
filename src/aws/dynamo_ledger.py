"""The idempotency ledger, as a DynamoDB conditional write.

Replaces the local ResultIndex (src/result_index.py), which rewrites its
*entire* seen-set on every add - O(n) per write, flagged in NOTES.md
(Section 5.1) as the piece that would NOT survive 10x steady-state volume.

DynamoDB's conditional put makes "claim this key if unseen" one atomic
server-side operation - no read-then-write race, no O(n) rewrite; each
write touches exactly one item.
"""

from botocore.exceptions import ClientError

from src.aws.clients import AwsClients


class DynamoResultIndex:
    def __init__(self, clients: AwsClients, table_name: str):
        self.table = clients.dynamodb
        self.table_name = table_name

    def try_claim(self, key: str, result_pointer: str) -> bool:
        """Atomically claims `key` if unclaimed. True: proceed to score.
        False: already claimed (duplicate delivery, skip without
        re-inference). One round-trip, one item, no lock."""
        try:
            self.table.put_item(
                TableName=self.table_name,
                Item={"result_key": {"S": key}, "result_pointer": {"S": result_pointer}},
                ConditionExpression="attribute_not_exists(result_key)",
            )
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                return False
            raise

    def has(self, key: str) -> bool:
        response = self.table.get_item(TableName=self.table_name, Key={"result_key": {"S": key}})
        return "Item" in response
