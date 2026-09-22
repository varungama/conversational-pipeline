import unittest

from src.metrics import compute_overall_sentiment, compute_sentiment_trajectory, compute_volatility


class TestOverallSentiment(unittest.TestCase):
    def test_no_customer_turns_is_none_not_zero(self):
        self.assertIsNone(compute_overall_sentiment([]))

    def test_single_turn_uses_that_turns_score(self):
        result = compute_overall_sentiment([(0.8, 0.9)])
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["label"], "positive")
        self.assertEqual(result["n_turns"], 1)

    def test_plain_mean_of_several_turns(self):
        result = compute_overall_sentiment([(0.8, 0.9), (-0.4, 0.6), (0.0, 0.1)])
        self.assertAlmostEqual(result["score"], (0.8 - 0.4 + 0.0) / 3, places=4)
        self.assertEqual(result["n_turns"], 3)

    def test_mean_near_zero_labels_neutral(self):
        result = compute_overall_sentiment([(0.05, 0.5), (-0.05, 0.5)])
        self.assertEqual(result["label"], "neutral")


class TestSentimentTrajectory(unittest.TestCase):
    def test_zero_turns_is_none(self):
        self.assertIsNone(compute_sentiment_trajectory([]))

    def test_single_turn_has_no_trajectory(self):
        # a one-turn conversation has no trajectory - it is undefined, not 0.0
        self.assertIsNone(compute_sentiment_trajectory([0.5]))

    def test_two_turns_is_technically_defined(self):
        result = compute_sentiment_trajectory([-0.6, 0.6])
        self.assertEqual(result["n_turns"], 2)
        self.assertAlmostEqual(result["score"], 0.6)  # (0.6 - -0.6) / 2
        self.assertEqual(result["label"], "improving")

    def test_bounded_regardless_of_length(self):
        # worst-case swing from -1 to +1 must still land at exactly 1.0, not 2.0
        short = compute_sentiment_trajectory([-1.0, 1.0])
        long = compute_sentiment_trajectory([-1.0, 0.1, -0.3, 0.9, 1.0])
        self.assertEqual(short["score"], 1.0)
        self.assertEqual(long["score"], 1.0)

    def test_only_endpoints_matter_not_the_middle(self):
        # a "whiplash" conversation that ends where it started is flat,
        # even though it swung wildly in between
        result = compute_sentiment_trajectory([0.5, -0.9, 0.9, -0.9, 0.5])
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["label"], "flat")

    def test_declining_trajectory(self):
        result = compute_sentiment_trajectory([0.8, 0.0, -0.8])
        self.assertLess(result["score"], 0)
        self.assertEqual(result["label"], "declining")


class TestVolatility(unittest.TestCase):
    def test_zero_or_one_turn_is_none(self):
        self.assertIsNone(compute_volatility([]))
        self.assertIsNone(compute_volatility([0.5]))

    def test_stable_positive(self):
        # small, consistent swings around a positive mean
        result = compute_volatility([0.7, 0.75, 0.7, 0.72])
        self.assertEqual(result["label"], "stable_positive")
        self.assertLess(result["score"], 0.3)

    def test_chaotic_positive(self):
        # big swings, but averaging out to solidly positive
        result = compute_volatility([0.9, 0.1, 0.95, 0.15])
        self.assertEqual(result["label"], "chaotic_positive")
        self.assertGreater(result["score"], 0.3)

    def test_stable_flat(self):
        result = compute_volatility([0.05, -0.05, 0.05, -0.05])
        self.assertEqual(result["label"], "stable_flat")

    def test_chaotic_flat(self):
        # whiplash: wild swings that cancel out to a flat mean
        result = compute_volatility([0.9, -0.9, 0.9, -0.9])
        self.assertEqual(result["label"], "chaotic_flat")
        self.assertAlmostEqual(result["score"], 1.8)

    def test_stable_negative(self):
        result = compute_volatility([-0.7, -0.75, -0.7, -0.72])
        self.assertEqual(result["label"], "stable_negative")

    def test_chaotic_negative(self):
        result = compute_volatility([-0.9, -0.1, -0.95, -0.15])
        self.assertEqual(result["label"], "chaotic_negative")

    def test_two_turns_is_defined(self):
        result = compute_volatility([-0.6, 0.6])
        self.assertEqual(result["n_turns"], 2)
        self.assertAlmostEqual(result["score"], 1.2)

    def test_independent_of_endpoints_unlike_trajectory(self):
        # ends where it started (trajectory == 0), but far from motionless
        result = compute_volatility([0.0, 0.9, -0.9, 0.9, 0.0])
        self.assertEqual(result["label"], "chaotic_flat")
        self.assertGreater(result["score"], 0.5)


if __name__ == "__main__":
    unittest.main()
