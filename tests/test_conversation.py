import unittest

from src.conversation import ConversationRejected, load_conversation


class TestWholeRecordValidation(unittest.TestCase):
    def test_missing_turns_key_is_rejected(self):
        with self.assertRaises(ConversationRejected):
            load_conversation({"conversation_id": "c-1", "tenant_id": "acme"})

    def test_turns_wrong_type_is_rejected(self):
        with self.assertRaises(ConversationRejected):
            load_conversation({"conversation_id": "c-1", "tenant_id": "acme", "turns": "not a list"})

    def test_empty_turns_list_is_not_rejected(self):
        # structurally fine, no turns - "no data" downstream, not an error
        result = load_conversation({"conversation_id": "c-1", "tenant_id": "acme", "turns": []})
        self.assertEqual(result, [])


class TestPerTurnValidation(unittest.TestCase):
    def _turns(self, turns):
        return load_conversation({"conversation_id": "c-1", "tenant_id": "acme", "turns": turns})

    def test_valid_turn_passes(self):
        result = self._turns([
            {"seq": 0, "role": "customer", "ts": "2026-03-02T09:14:00Z", "text": "hello"},
        ])
        self.assertEqual(result[0]["status"], "ok")
        self.assertTrue(result[0]["has_content"])

    def test_missing_ts_is_skipped(self):
        result = self._turns([{"seq": 1, "role": "agent", "text": "sorry about that"}])
        self.assertEqual(result[0]["status"], "skipped")
        self.assertEqual(result[0]["reason"], "ts_missing_or_unparseable")

    def test_null_text_is_skipped(self):
        result = self._turns([{"seq": 2, "role": "customer", "ts": "2026-03-02T09:14:00Z", "text": None}])
        self.assertEqual(result[0]["reason"], "text_missing_or_wrong_type")

    def test_integer_text_is_skipped(self):
        result = self._turns([{"seq": 3, "role": "customer", "ts": "2026-03-02T09:14:00Z", "text": 42}])
        self.assertEqual(result[0]["reason"], "text_missing_or_wrong_type")

    def test_seq_as_string_is_skipped(self):
        result = self._turns([{"seq": "3", "role": "customer", "ts": "2026-03-02T09:14:00Z", "text": "hi"}])
        self.assertEqual(result[0]["reason"], "seq_missing_or_wrong_type")

    def test_missing_seq_is_skipped(self):
        result = self._turns([{"role": "customer", "ts": "2026-03-02T09:14:00Z", "text": "hi"}])
        self.assertEqual(result[0]["reason"], "seq_missing_or_wrong_type")

    def test_unknown_role_is_skipped(self):
        result = self._turns([{"seq": 0, "role": "shopper", "ts": "2026-03-02T09:14:00Z", "text": "hi"}])
        self.assertEqual(result[0]["reason"], "role_missing_or_unrecognised")

    def test_whitespace_only_text_has_no_content(self):
        result = self._turns([{"seq": 0, "role": "customer", "ts": "2026-03-02T09:14:00Z", "text": "  \t\n "}])
        self.assertEqual(result[0]["status"], "ok")
        self.assertFalse(result[0]["has_content"])

    def test_one_bad_turn_does_not_affect_the_others(self):
        result = self._turns([
            {"seq": 0, "role": "customer", "ts": "2026-03-02T09:14:00Z", "text": "fine turn"},
            {"seq": 1, "role": "agent", "text": "bad turn, no ts"},
            {"seq": 2, "role": "customer", "ts": "2026-03-02T09:15:20Z", "text": "another fine turn"},
        ])
        statuses = [turn["status"] for turn in result]
        self.assertEqual(statuses, ["ok", "skipped", "ok"])


if __name__ == "__main__":
    unittest.main()
